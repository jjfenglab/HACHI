"""
Baseline training orchestration for ensemble trainer.

This module handles the orchestration of parallel baseline training across
multiple initializations with coordinated feature extraction.
"""

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import pandas as pd

import src.common as common
from src.training_history import TrainingHistory

from .config import EnsembleConfig
from .data_operations import map_extractions_to_sample


async def train_baselines_coordinated(
    data_df: pd.DataFrame,
    init_seeds: List[int],
    config: EnsembleConfig,
    generate_initial_concepts_func,
    feature_extraction_manager,
    concept_tracker,
) -> Dict[int, TrainingHistory]:
    """
    Orchestrate parallel baseline training across initializations.

    This function coordinates baseline training by:
    1. Generating concepts in parallel across initializations (split-specific)
    2. Extracting features once on the original dataset for all concepts
    3. Mapping extractions to split samples and finalizing training

    Args:
        data_df: Original training data
        init_seeds: List of initialization seeds
        config: Ensemble configuration
        generate_initial_concepts_func: Function to generate initial concepts
        feature_extraction_manager: Shared feature extraction manager
        concept_tracker: Concept tracker for managing concept assignments

    Returns:
        Dict mapping init_seed to TrainingHistory
    """
    logging.info(
        f"Training baselines for {len(init_seeds)} initializations with coordination"
    )

    # Phase 1: Parallel concept generation (split-specific)
    logging.info("Phase 1: Generating concepts from split-specific residual models")
    init_concepts = {}
    split_samples = {}

    # Using ThreadPoolExecutor instead of asyncio.gather()since these functions perform CPU-bound sklearn model training (LogisticRegression)
    # that benefits from multiple OS threads. While asyncio would provide cleaner code,
    # it would serialize execution on a single thread.
    # NumPy/scikit-learn release Python's GIL during computations, enabling real
    # parallel execution across multiple CPU cores with ThreadPoolExecutor.
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=len(init_seeds)) as executor:
        # Create all tasks at once
        tasks = [
            loop.run_in_executor(
                executor, generate_initial_concepts_func, init_seed, data_df
            )
            for init_seed in init_seeds
        ]

        # Wait for all tasks to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(
                "Baseline concept generation failed for one or more initializations"
            )
            logging.error(f"Error: {e}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            raise RuntimeError("Parallel baseline concept generation failed") from e

        # Process results
        for init_seed, (concepts, split_data) in zip(init_seeds, results):
            init_concepts[init_seed] = concepts
            split_samples[init_seed] = split_data

            # Track concepts for this initialization
            for concept_dict in concepts:
                concept_tracker.add_concept(init_seed, concept_dict["concept"])

            logging.info(
                f"Generated {len(concepts)} concepts for init_seed {init_seed}"
            )

    # Phase 2: Coordinated feature extraction on original dataset
    logging.info("Phase 2: Coordinated feature extraction on original dataset")
    all_baseline_concepts = []
    for concepts in init_concepts.values():
        for concept_dict in concepts:
            concept = concept_dict["concept"]
            if concept not in all_baseline_concepts:
                all_baseline_concepts.append(concept)

    logging.info(
        f"Extracting features for {len(all_baseline_concepts)} unique concepts"
    )

    # Extract features on original dataset (not split samples)
    original_extractions = await feature_extraction_manager.extract_features_batch(
        data_df,  # Original dataset
        all_baseline_concepts,
        max_new_tokens=config.llm.max_new_tokens,
    )

    # Phase 3: Map extractions to split samples and finalize training
    logging.info(
        "Phase 3: Mapping extractions to split samples and finalizing training"
    )
    baseline_histories = {}
    for init_seed in init_seeds:
        try:
            history = finalize_baseline_training(
                init_seed,
                init_concepts[init_seed],
                split_samples[init_seed][0],
                split_test_data=split_samples[init_seed][1],
                original_extractions=original_extractions,
                config=config,
            )
            baseline_histories[init_seed] = history
            logging.info(f"Baseline training completed for init_seed {init_seed}")
        except Exception as e:
            logging.error(
                f"Baseline finalization failed for init_seed {init_seed}: {e}"
            )
            raise

    logging.info(
        f"Coordinated baseline training completed for all {len(init_seeds)} initializations"
    )
    return baseline_histories


def finalize_baseline_training(
    init_seed: int,
    concept_dicts: List[dict],
    split_train_data: pd.DataFrame,
    split_test_data: pd.DataFrame,
    original_extractions: Dict[str, np.ndarray],
    config: EnsembleConfig,
) -> TrainingHistory:
    """
    Finalize baseline training by mapping original extractions to split sample.

    Args:
        init_seed: Initialization seed
        concept_dicts: List of concept dictionaries for this initialization
        split_data: split sample data for this initialization
        original_extractions: Feature extractions from original dataset
        config: Ensemble configuration

    Returns:
        TrainingHistory with baseline training results
    """
    logging.info(f"Finalizing baseline training for init_seed {init_seed}")

    # Create training history
    history = TrainingHistory(force_keep_cols=config.force_keep_columns)

    # Map extractions from original dataset to split sample positions
    split_extractions = map_extractions_to_sample(
        original_extractions, split_train_data, concept_dicts
    )

    # Train logistic regression model on split sample with mapped features
    X_train = common.get_features(
        concept_dicts,
        split_extractions,
        split_train_data,  # Use split data for training
        force_keep_columns=config.force_keep_columns,
    )
    y_train = split_train_data["y"].to_numpy().flatten()
    sample_weight = split_train_data["sample_weight"].to_numpy().flatten()

    split_extractions = map_extractions_to_sample(
        original_extractions, split_test_data, concept_dicts
    )
    X_test = common.get_features(
        concept_dicts,
        split_extractions,
        split_test_data,  # Use split data for testing
        force_keep_columns=config.force_keep_columns,
    )
    y_test = split_test_data["y"].to_numpy().flatten()
    test_weight = split_test_data["sample_weight"].to_numpy().flatten()

    # First figure out which features are top from lasso
    raw_model_results = common.train_LR_max_features(
        X_train,
        y_train,
        sample_weight=sample_weight,
        X_test=X_test,
        y_test=y_test,
        test_weight=test_weight,
        num_meta_concepts=config.concept.goal_num_meta_concepts,
        seed=init_seed,
    )
    
    # Then refit and filter down to the first set of concepts
    selected_idxs = np.where(raw_model_results["coef"].flatten() != 0)[0]
    non_selected_idxs = np.where(raw_model_results["coef"].flatten() == 0)[0]
    selected_concept_dicts = [concept_dicts[i] for i in selected_idxs]
    logging.info(f"Baseline model selected {[c['concept'] for c in selected_concept_dicts]}")
    if selected_idxs.size < config.concept.goal_num_meta_concepts:
        added_idxs = non_selected_idxs[:config.concept.goal_num_meta_concepts - selected_idxs.size]
        selected_idxs = np.concatenate([selected_idxs, added_idxs])
        selected_concept_dicts += [concept_dicts[i] for i in added_idxs]
        logging.info(f"Adding concept to baseline {[concept_dicts[i]['concept'] for i in added_idxs]}")
    
    model_results = common.train_LR(
        X_train[:,selected_idxs],
        y_train,
        sample_weight=sample_weight,
        X_test=X_test[:,selected_idxs],
        y_test=y_test,
        test_weight=test_weight,
        penalty=None,
        use_acc=config.model.use_acc,
        seed=init_seed,
    )
    
    # Add results to history
    history.update_history(selected_concept_dicts, model_results)

    logging.info(
        f"Baseline training completed for init_seed {init_seed}, train AUC: {model_results['auc']:.3f}"
    )

    return history
