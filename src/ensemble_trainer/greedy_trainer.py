"""
Greedy training orchestration for ensemble trainer.

This module handles the orchestration of coordinated greedy concept evolution
across multiple initializations with shared feature extraction.
"""

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import src.common as common
from src.training_history import TrainingHistory

from . import data_operations
from .config import EnsembleConfig
from .greedy_concept_selector import GreedyConceptSelector
from .selection_operations import perform_greedy_selection
from .state import TrainerState


async def run_coordinated_concept_evolution(
    data_df: pd.DataFrame,
    epoch: int,
    config: EnsembleConfig,
    concept_selectors: Dict[int, GreedyConceptSelector],
    concept_generator,
    feature_extraction_manager,
    concept_tracker,
    training_histories: Dict[int, TrainingHistory],
    ensemble_state: TrainerState,
    checkpoint_callback: Optional[Callable[[], None]] = None,
) -> None:
    """
    Run coordinated concept evolution for one epoch using GreedyConceptSelector.

    This orchestrates concept replacement one at a time (or num_greedy_holdout at a time).

    Args:
        data_df: Full training dataset
        epoch: Current epoch number
        config: Ensemble configuration
        concept_selectors: Dict of GreedyConceptSelector instances
        concept_generator: Concept generator instance
        feature_extraction_manager: Shared feature extraction manager
        concept_tracker: Concept tracker instance
        training_histories: Dict of training histories by init_seed
        ensemble_state: Current trainer state for checkpointing
        checkpoint_callback: Optional callback function to save checkpoints after each concept iteration
    """
    logging.info(f"Running coordinated concept evolution for epoch {epoch + 1}")

    try:
        # Loop over concepts to replace them incrementally
        # Resume from completed iteration
        if ensemble_state.completed_concept_iteration >= 0:
            # Start from the next iteration after the last completed one
            start_j = (
                ensemble_state.completed_concept_iteration
                + config.training.num_greedy_holdout
            )
            logging.info(
                f"Resuming from concept iteration {start_j} (after completed iteration {ensemble_state.completed_concept_iteration})"
            )
        else:
            # No iterations completed yet
            start_j = 0
            logging.info("Starting fresh from concept iteration 0")

        for j in range(
            start_j,
            config.concept.goal_num_meta_concepts,
            config.training.num_greedy_holdout,
        ):
            logging.info(
                f"Epoch {epoch + 1}, replacing concepts {j} to {j + config.training.num_greedy_holdout}"
            )

            # Step 1: Extract features for CURRENT concepts across all initializations
            all_current_concepts = []
            init_current_concepts_map = {}  # Maps init_seed -> current concepts
            for init_seed in config.init_seeds:
                concept_selector = concept_selectors[init_seed]
                current_concepts = concept_selector.init_history.get_last_concepts()[
                    : config.concept.goal_num_meta_concepts
                ]
                init_current_concepts_map[init_seed] = current_concepts

                # Collect all unique current concepts
                for concept_dict in current_concepts:
                    concept = concept_dict["concept"]
                    if concept not in all_current_concepts:
                        all_current_concepts.append(concept)

            # Coordinated extraction for all current concepts
            logging.info(
                f"Extracting features for {len(all_current_concepts)} current concepts"
            )
            current_shared_extractions = (
                await feature_extraction_manager.extract_for_training(
                    data_df,  # Extract on original dataset
                    all_current_concepts,
                    max_new_tokens=config.llm.max_new_tokens,
                )
            )

            # Step 2: Generate candidate concepts for each initialization in parallel
            all_candidate_concepts = []
            init_candidate_map = {}  # Maps (init_seed, concept) -> concept_dict
            init_split_data_map = {}

            # Create data splits for all initializations
            for init_seed in config.init_seeds:
                init_split_data_map[init_seed] = data_operations.create_data_split(
                    data_df, init_seed, data_split_fraction=config.training.train_frac
                )

            # Generate candidate concepts in parallel using ThreadPoolExecutor
            # These functions perform CPU-bound operations including sklearn model training
            # for concept generation. ThreadPoolExecutor provides true parallelism across
            # multiple CPU cores, while asyncio.gather() would serialize execution.
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=len(config.init_seeds)) as executor:
                tasks = []
                for init_seed in config.init_seeds:
                    concept_selector = concept_selectors[init_seed]
                    current_concepts = init_current_concepts_map[init_seed]

                    task = loop.run_in_executor(
                        executor,
                        concept_generator.generate_candidate_concepts,
                        init_split_data_map[init_seed][0], # train split for generating concept ideas
                        concept_selector,
                        current_concepts,
                        current_shared_extractions,
                    )
                    tasks.append(task)

                # Wait for all candidate generation tasks using gather
                try:
                    all_candidate_results = await asyncio.gather(
                        *tasks, return_exceptions=False
                    )
                except Exception as e:
                    logging.error(
                        "Candidate generation failed for one or more initializations"
                    )
                    raise RuntimeError(
                        f"Parallel candidate generation failed at epoch {epoch + 1}, iteration {j}"
                    ) from e

                # Process results and track all candidates
                for init_seed, candidate_concepts in zip(
                    config.init_seeds, all_candidate_results
                ):
                    for concept_dict in candidate_concepts:
                        concept = concept_dict["concept"]
                        all_candidate_concepts.append(concept)
                        init_candidate_map[(init_seed, concept)] = concept_dict

                    logging.info(
                        f"Generated {len(candidate_concepts)} candidate concepts for init_seed {init_seed}"
                    )

            # Step 3: Coordinated feature extraction for all candidate concepts
            logging.info(
                f"Extracting features for {len(all_candidate_concepts)} candidate concepts"
            )

            candidate_shared_extractions = (
                await feature_extraction_manager.extract_for_training(
                    data_df,  # Extract on original dataset
                    all_candidate_concepts,
                )
            )

            # Step 4: Parallel greedy concept selection, then sequential updates
            logging.info(
                "Performing parallel greedy selection across all initializations"
            )

            # Phase A: Parallel greedy concept selection (the expensive operation)
            # This is the section where multiple sklearn LogisticRegression
            # models are trained per concept candidate. ThreadPoolExecutor enables true parallel
            # execution across CPU cores.
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=len(config.init_seeds)) as executor:
                selection_tasks = []
                for init_seed in config.init_seeds:
                    concept_selector = concept_selectors[init_seed]

                    task = loop.run_in_executor(
                        executor,
                        perform_greedy_selection,
                        data_df,
                        concept_selector,
                        init_candidate_map,
                        current_shared_extractions,
                        candidate_shared_extractions,
                    )
                    selection_tasks.append(task)

                # Wait for all greedy selection tasks to complete using gather
                try:
                    all_selected_concepts = await asyncio.gather(
                        *selection_tasks, return_exceptions=False
                    )
                except Exception as e:
                    logging.error(
                        "Greedy selection failed for one or more initializations"
                    )
                    raise RuntimeError(
                        f"Parallel greedy selection failed at epoch {epoch + 1}, iteration {j}"
                    ) from e

                # Create results mapping
                selection_results = {}
                for init_seed, selected_concepts in zip(
                    config.init_seeds, all_selected_concepts
                ):
                    selection_results[init_seed] = selected_concepts
                    logging.info(
                        f"Selected {len(selected_concepts)} concepts for init_seed {init_seed}"
                    )

            # Phase B: Sequential updates (lightweight operations)
            logging.info("Updating concept tracker and training histories sequentially")
            for init_seed in config.init_seeds:
                init_train_data_df = init_split_data_map[init_seed]
                selected_concepts = selection_results[init_seed]

                # Update concept tracker
                for concept_dict in selected_concepts:
                    concept_tracker.add_concept(init_seed, concept_dict["concept"])

                # Update training history
                update_training_history_after_selection(
                    init_seed,
                    init_train_data_df,
                    data_df,  # Add full dataset for evaluation
                    selected_concepts,
                    current_shared_extractions,
                    candidate_shared_extractions,
                    training_histories[init_seed],
                    config.model.final_learner_type,
                    config.training.is_greedy_metric_acc,
                    force_keep_columns=config.force_keep_columns,
                )

            logging.info("Completed parallel concept selection and sequential updates")

            # Mark this iteration as completed
            ensemble_state.completed_concept_iteration = j
            logging.debug(
                f"Marked concept iteration {j} as completed for epoch {epoch + 1}"
            )

            # Save checkpoint after marking completion
            if checkpoint_callback:
                checkpoint_callback()
                logging.info(
                    f"Checkpoint saved after completing iteration {j}, epoch {epoch + 1}"
                )

        # trainer.py handles epoch completion

    except Exception as e:
        logging.error(
            f"Concept evolution failed at epoch {epoch + 1}, "
            f"last completed iteration {ensemble_state.completed_concept_iteration}"
        )
        logging.error(f"Error: {e}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())

        # Provide context about which initialization might have failed
        logging.error(f"Active initializations: {config.init_seeds}")

        raise RuntimeError(
            f"Coordinated concept evolution failed at epoch {epoch + 1}, "
            f"last completed iteration {ensemble_state.completed_concept_iteration}"
        ) from e


def update_training_history_after_selection(
    init_seed: int,
    init_train_data_df: pd.DataFrame,
    data_df: pd.DataFrame,
    selected_concepts: List[dict],
    current_shared_extractions: Dict[str, np.ndarray],
    candidate_shared_extractions: Dict[str, np.ndarray],
    training_history: TrainingHistory,
    final_learner_type: str,
    is_greedy_metric_acc: bool,
    force_keep_columns: Optional[List[str]] = None,
):
    """
    Update training history for an initialization after concept selection.

    Args:
        init_seed: Initialization seed
        init_train_data_df: Data subset used for this initialization (unused, kept for consistency)
        data_df: Full dataset for evaluation
        selected_concepts: List of selected concept dictionaries
        current_shared_extractions: Feature extractions for current concepts
        candidate_shared_extractions: Feature extractions for candidate concepts
        training_history: TrainingHistory to update
        final_learner_type: Type of final learner (e.g., 'l1', 'l2')
        is_greedy_metric_acc: Whether to use accuracy vs AUC for evaluation
        force_keep_columns: Optional tabular columns to force keep
    """

    # Get concept extractions for full dataset evaluation
    concept_names = [c["concept"] for c in selected_concepts]
    # Selected concepts could be from current or candidate extractions
    concept_extractions = {}
    for concept in concept_names:
        if concept in current_shared_extractions:
            concept_extractions[concept] = current_shared_extractions[concept]
        elif concept in candidate_shared_extractions:
            concept_extractions[concept] = candidate_shared_extractions[concept]

    # Train and evaluate model on full dataset (matching original ConceptLearnerModel behavior)
    X_full = common.get_features(
        selected_concepts,
        concept_extractions,  # Use extractions from full dataset
        data_df,  # Use full dataset for evaluation
        force_keep_columns=force_keep_columns,
    )

    y_full = data_df["y"].to_numpy().flatten()

    model_results = common.train_LR(
        X_full,
        y_full,
        sample_weight=data_df["sample_weight"].to_numpy().flatten(),
        penalty=final_learner_type,
        use_acc=is_greedy_metric_acc,
        seed=init_seed,
    )

    # Add the new concepts and metrics to history
    training_history.update_history(selected_concepts, model_results)

    logging.info(
        f"Updated history for init_seed {init_seed}, train AUC: {model_results['auc']:.3f}"
    )
