"""
Main ensemble trainer for coordinated concept learning across multiple initializations.

This module contains the EnsembleTrainer class that orchestrates training across
multiple initializations while sharing expensive LLM operations.
"""

import asyncio
import logging
import os
import pickle
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import src.common as common
from src.llm_utils import create_llm_clients
from src.training_history import TrainingHistory

from . import data_operations, selection_operations
from .baseline_trainer import train_baselines_coordinated
from .checkpoint_manager import CheckpointManager
from .concept_generator import ConceptGeneratorFactory
from .concept_tracker import ConceptTracker
from .config import EnsembleConfig, TrainingPhase
from .evidence_enhancement import SemanticCacheManager
from .feature_extraction_manager import FeatureExtractionManager
from .greedy_concept_selector import GreedyConceptSelector
from .greedy_trainer import run_coordinated_concept_evolution
from .state import TrainerState


class EnsembleTrainer:
    """
    Main ensemble trainer that coordinates training across multiple initializations.
    """

    def __init__(
        self,
        config: EnsembleConfig,
        output_dir: str,
        seed: int = 0,
        **kwargs,
    ):
        # Store the main configurations
        self.config = config
        self.output_dir = output_dir

        # Evidence-span enhancement parameters
        self.summaries_df = None

        # Store additional arguments for baseline and greedy training
        self.kwargs = kwargs

        # Initialize trackers
        self.concept_tracker = ConceptTracker(self.config.init_seeds)
        self.training_histories: Dict[int, TrainingHistory] = {}

        # GreedyConceptSelector instances for each initialization
        self.concept_selectors: Dict[int, GreedyConceptSelector] = {}

        # Initialize LLM clients and components
        self.llm_dict = None
        self.concept_generator = None
        self.feature_extraction_manager = None

        # State management and checkpointing
        self.ensemble_state: Optional[TrainerState] = None
        self.checkpoint_file: str = os.path.join(
            output_dir, "ensemble_state_checkpoint.pkl"
        )
        self.interrupted: bool = False
        self._setup_signal_handlers()

        np.random.seed(seed)

    @property
    def num_epochs(self) -> int:
        """Get number of epochs from config."""
        return self.config.training.num_epochs

    @property
    def max_meta_concepts(self) -> int:
        """Get number of meta concepts from config."""
        return self.config.concept.max_meta_concepts

    @property
    def num_greedy_holdout(self) -> int:
        """Get number of greedy holdout from config."""
        return self.config.training.num_greedy_holdout

    @property
    def train_frac(self) -> float:
        """Get training fraction from config."""
        return self.config.training.train_frac

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful interruption."""

        def signal_handler(signum, frame):
            logging.warning(
                f"Received signal {signum}, initiating graceful shutdown..."
            )
            self.interrupted = True
            if self.ensemble_state:
                self.ensemble_state.interruption_count += 1
                logging.info("Saving checkpoint due to interruption...")
                self._save_checkpoint()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_llm_clients(self):
        """Setup LLM clients for the ensemble training."""

        logger = logging.getLogger()
        self.llm_dict = create_llm_clients(self.config.llm, logger)
        self.feature_extraction_manager = FeatureExtractionManager(
            self.config, self.llm_dict, self.concept_tracker
        )
        # Restore shared extractions if resuming from checkpoint
        if self.ensemble_state and self.ensemble_state.shared_extractions:
            self.feature_extraction_manager.set_shared_extractions(
                self.ensemble_state.shared_extractions
            )
            logging.info(
                f"Restored {len(self.ensemble_state.shared_extractions)} shared extractions from checkpoint"
            )

        # Restore state from checkpoint if resuming
        if (
            self.ensemble_state.current_phase
            != self.ensemble_state.current_phase.NOT_STARTED
        ):
            concept_tracker_state, training_histories = (
                CheckpointManager.restore_from_checkpoint(
                    self.ensemble_state,
                    self.config.init_seeds,
                )
            )
            self.concept_tracker.restore_state(concept_tracker_state)
            self.training_histories.update(training_histories)

        # Create components
        self.concept_generator = ConceptGeneratorFactory.create_generator(
            self.config, self.llm_dict, self.summaries_df
        )

    def _save_checkpoint(self):
        """Helper method to save checkpoint with current state."""
        CheckpointManager.save_checkpoint(
            self.ensemble_state,
            self.checkpoint_file,
            self.output_dir,
            self.concept_tracker,
            self.feature_extraction_manager,
            self.config.init_seeds,
            self.training_histories,
        )

    def _create_greedy_concept_selectors(self):
        """Create GreedyConceptSelector instances for each initialization."""
        for init_seed in self.config.init_seeds:
            # Get the training history for this initialization
            if init_seed in self.training_histories:
                init_history = self.training_histories[init_seed]
            else:
                # Create empty history if not available yet
                init_history = TrainingHistory(
                    force_keep_cols=self.config.force_keep_columns
                )

            # Create GreedyConceptSelector for this initialization
            concept_selector = GreedyConceptSelector(
                init_seed=init_seed,
                init_history=init_history,
                llm_iter=self.llm_dict["iter"],
                num_classes=self.config.training.num_classes,
                num_meta_concepts=self.config.concept.goal_num_meta_concepts,
                prompt_iter_file=self.config.concept.prompt_iter_file,
                config=self.config,
                residual_model_type=self.config.model.residual_model_type,
                final_model_type=self.config.model.final_model_type,
                num_greedy_holdout=self.config.training.num_greedy_holdout,
                is_greedy_metric_acc=self.config.training.is_greedy_metric_acc,
                force_keep_columns=self.config.force_keep_columns,
                num_top=self.config.concept.num_top_residual_words,
                cv=self.config.model.cv,
            )

            self.concept_selectors[init_seed] = concept_selector

    async def fit(self, data_df: pd.DataFrame) -> Dict[int, TrainingHistory]:
        """
        Fit the ensemble trainer from scratch (sklearn-like API).

        Args:
            data_df: Training data

        Returns:
            Dict mapping init_seed to final TrainingHistory

        Note: TODO: have users provide X column(s) and Y column
        """
        try:
            # Always start fresh
            self._initialize_state()
            self._setup_training_environment(data_df)

            # Run both phases
            await self._run_baseline_phase(data_df)
            if self.num_epochs > 0:
                await self._run_greedy_phase(data_df)

            # Finalize
            self._finalize_training()

            return self.training_histories

        except Exception as e:
            # Log detailed error information for debugging
            logging.error(f"Training failed with error: {e}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())

            # Log current state for debugging
            if self.ensemble_state:
                logging.error(
                    f"Failed at phase: {self.ensemble_state.current_phase.value}"
                )
                logging.error(f"Working epoch: {self.ensemble_state.working_epoch}")
                logging.error(
                    f"Completed concept iteration: {self.ensemble_state.completed_concept_iteration}"
                )

            # Re-raise with context about where we failed
            raise RuntimeError(
                f"EnsembleTrainer.fit failed at {self.ensemble_state.current_phase.value if self.ensemble_state else 'initialization'}"
            ) from e

    async def continue_training(
        self,
        data_df: pd.DataFrame,
        num_additional_epochs: int = None,
    ) -> Dict[int, TrainingHistory]:
        """
        Continue training from current state.

        IMPORTANT: Baseline training must be complete before calling this method.
        If baseline training was not completed (e.g., due to interruption), use fit() instead.

        Args:
            data_df: Training data
            num_additional_epochs: Additional epochs to train (if None, use config default)

        Returns:
            Dict mapping init_seed to final TrainingHistory

        Raises:
            RuntimeError: If baseline training was not completed
        """
        try:
            # Load state if needed
            if not self._has_state():
                self._load_checkpoint_if_exists()

            if not self._has_state():
                raise ValueError(
                    "No state to continue from. Use fit() for fresh training or load from checkpoint."
                )

            # Setup training environment
            self._setup_training_environment(data_df)

            # Optionally extend training epochs
            if num_additional_epochs is not None:
                old_epochs = self.config.training.num_epochs
                self.config.training.num_epochs += num_additional_epochs

                # Update state to reflect new epoch count and reset greedy completion
                self.ensemble_state.num_epochs = self.config.training.num_epochs

                # Reset greedy completion if we're adding epochs beyond what was already completed
                for init_seed in self.config.init_seeds:
                    self.ensemble_state.greedy_complete[init_seed] = False

                # Set phase back to greedy running since we need more training
                self.ensemble_state.current_phase = TrainingPhase.GREEDY_RUNNING

                next_epoch = (
                    (self.ensemble_state.working_epoch + 1)
                    if self.ensemble_state.working_epoch >= 0
                    else 0
                )
                logging.info(
                    f"Extended training from {old_epochs} to {self.config.training.num_epochs} epochs. "
                    f"Resuming greedy training from epoch {next_epoch + 1}."
                )

            # Check whether baseline training was completed successfully
            if not self.ensemble_state.is_baseline_complete():
                raise RuntimeError(
                    "Cannot continue training: baseline phase was not completed. "
                    "Baseline training is atomic and cannot be resumed from partial state. "
                    "Please use fit() to start training from scratch, or ensure baseline "
                    "completes before attempting to continue training."
                )

            if self.num_epochs > 0 and not self.ensemble_state.is_greedy_complete():
                await self._run_greedy_phase(data_df)

            # Finalize
            self._finalize_training()

            return self.training_histories

        except Exception as e:
            # Log detailed error information for debugging
            logging.error(f"Continue training failed with error: {e}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())

            # Log current state for debugging
            if self.ensemble_state:
                logging.error(
                    f"Failed at phase: {self.ensemble_state.current_phase.value}"
                )
                logging.error(f"Working epoch: {self.ensemble_state.working_epoch}")
                logging.error(
                    f"Completed concept iteration: {self.ensemble_state.completed_concept_iteration}"
                )

            # Re-raise with context about where we failed
            raise RuntimeError(
                f"EnsembleTrainer.continue_training failed at {self.ensemble_state.current_phase.value if self.ensemble_state else 'initialization'}"
            ) from e

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: EnsembleConfig = None,
        output_dir: str = None,
        setup_for_training: bool = False,
    ) -> "EnsembleTrainer":
        """
        Create trainer from saved checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Optional config to override saved config
            output_dir: Optional output directory to override saved path
            setup_for_training: If True, set up components needed for continuing training.
                              If False (default), set up minimal components for prediction only.

        Returns:
            EnsembleTrainer instance restored from checkpoint
        """
        # Load state from checkpoint
        state = CheckpointManager.load_state(checkpoint_path)

        # Use saved config or override
        if config is None:
            if hasattr(state, "config") and state.config is not None:
                config = state.config
            else:
                raise ValueError("No config in checkpoint and none provided")

        # Use saved output_dir or override
        if output_dir is None:
            if hasattr(state, "output_dir") and state.output_dir is not None:
                output_dir = state.output_dir
            else:
                output_dir = os.path.dirname(checkpoint_path)

        # Create trainer and restore state
        trainer = cls(config, output_dir)
        trainer._restore_state(state, setup_for_training=setup_for_training)
        return trainer

    def _initialize_state(self):
        """Initialize fresh training state."""
        self.ensemble_state = TrainerState(
            init_seeds=self.config.init_seeds,
            num_epochs=self.num_epochs,
            max_meta_concepts=self.max_meta_concepts,
            num_greedy_holdout=self.num_greedy_holdout,
            config=self.config,
            output_dir=self.output_dir,
        )

    def _has_state(self) -> bool:
        """Check if trainer has existing state."""
        return self.ensemble_state is not None or os.path.exists(self.checkpoint_file)

    def _restore_state(self, state: TrainerState, setup_for_training: bool = False):
        """
        Restore trainer from loaded state.

        Args:
            state: TrainerState to restore from
            setup_for_training: If True, set up components for training continuation.
                              If False, set up minimal components for prediction only.
        """
        self.ensemble_state = state

        # Update config if provided in state
        if hasattr(state, "config") and state.config is not None:
            self.config = state.config

        # Load training histories
        self._load_training_histories()

        if setup_for_training:
            # Set up full LLM clients and components for training continuation
            self._setup_llm_clients()
            # Note: _setup_llm_clients already calls _create_greedy_concept_selectors
            # and sets up the concept generator
        else:
            # Setup minimal LLM clients for prediction only
            self._setup_for_prediction()

    def _load_training_histories(self):
        """Load training histories from checkpoint files."""
        if not self.ensemble_state:
            return

        # Use CheckpointManager to load training histories
        concept_tracker_state, training_histories = (
            CheckpointManager.restore_from_checkpoint(
                self.ensemble_state,
                self.config.init_seeds,
            )
        )
        self.concept_tracker.restore_state(concept_tracker_state)
        self.training_histories.update(training_histories)

        logging.info(
            f"Loaded {len(self.training_histories)} training histories from checkpoint"
        )

    def _setup_for_prediction(self):
        """Setup minimal LLM clients and feature extraction needed for predictions."""
        if self.llm_dict is None:
            logger = logging.getLogger()
            self.llm_dict = create_llm_clients(self.config.llm, logger)
            logging.info("Initialized LLM clients for prediction")

        # Create feature extraction manager for predictions
        if self.feature_extraction_manager is None:
            self.feature_extraction_manager = FeatureExtractionManager(
                self.config, self.llm_dict, self.concept_tracker
            )
            # Restore shared extractions if available in ensemble state
            if self.ensemble_state and self.ensemble_state.shared_extractions:
                self.feature_extraction_manager.set_shared_extractions(
                    self.ensemble_state.shared_extractions
                )
            logging.info("Initialized FeatureExtractionManager for prediction")

    def save_checkpoint(self, checkpoint_path: str = None):
        """
        Save current state to checkpoint.

        Args:
            checkpoint_path: Optional path override (defaults to self.checkpoint_file)
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_file

        if self.ensemble_state is not None:
            # Update state with current config and output_dir
            self.ensemble_state.config = self.config
            self.ensemble_state.output_dir = self.output_dir
            CheckpointManager.save_state(self.ensemble_state, checkpoint_path)

    def get_training_histories(self) -> Dict[int, TrainingHistory]:
        """Get training histories for all initializations."""
        return self.training_histories.copy()

    async def get_annotations(
        self, data_df: pd.DataFrame, use_posterior_iters: int = None, is_baseline = False
    ) -> pd.DataFrame:
        # Collect concepts from all initializations
        all_concepts = self._collect_concepts_for_prediction(use_posterior_iters, is_baseline)

        # Extract features once for all concepts
        all_extracted_features = (
            await self.feature_extraction_manager.extract_for_prediction(
                data_df, all_concepts
            )
        )

        all_extracted_features = {k: v.flatten() for k, v in all_extracted_features.items()}
        annotated_df = pd.DataFrame(all_extracted_features)
        return annotated_df

    async def predict(
        self, data_df: pd.DataFrame, use_posterior_iters: int = None, is_baseline = False
    ) -> pd.DataFrame:
        """
        Get ensemble predictions by averaging across all initializations.

        Args:
            data_df: Data to predict on
            use_posterior_iters: If specified, use only last N iterations per init
                               (similar to evaluate_bayesian_ensemble.py behavior)

        Returns:
            DataFrame with prediction probabilities averaged across all models
        """
        if not self.training_histories:
            raise ValueError("No training histories available. Train the model first.")

        # Collect concepts from all initializations
        all_concepts = self._collect_concepts_for_prediction(use_posterior_iters, is_baseline)

        # Extract features once for all concepts
        all_extracted_features = (
            await self.feature_extraction_manager.extract_for_prediction(
                data_df, all_concepts
            )
        )

        # Get predictions from each initialization
        all_predictions = []
        for init_seed in self.config.init_seeds:
            if init_seed in self.training_histories:
                init_predictions = self._get_init_predictions(
                    init_seed, data_df, all_extracted_features, use_posterior_iters, is_baseline
                )
                all_predictions.extend(init_predictions)

        if not all_predictions:
            raise ValueError("No predictions could be generated")

        # Average predictions (ensemble)
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        # Return as DataFrame for convenience
        if ensemble_pred.ndim == 1:
            # Binary classification
            pred_df = pd.DataFrame(
                {"prob_0": 1 - ensemble_pred, "prob_1": ensemble_pred}
            )
        else:
            # Multi-class classification
            pred_df = pd.DataFrame(
                ensemble_pred,
                columns=[f"prob_{i}" for i in range(ensemble_pred.shape[1])],
            )

        return pred_df

    async def predict_all(
        self, data_df: pd.DataFrame, use_posterior_iters: int = None
    ) -> Dict[int, pd.DataFrame]:
        """
        Get predictions from each initialization separately.

        Args:
            data_df: Data to predict on
            use_posterior_iters: If specified, use only last N iterations per init

        Returns:
            Dict mapping init_seed to prediction DataFrames
        """
        if not self.training_histories:
            raise ValueError("No training histories available. Train the model first.")

        # Collect concepts from all initializations
        all_concepts = self._collect_concepts_for_prediction(use_posterior_iters)

        # Extract features once for all concepts
        all_extracted_features = (
            await self.feature_extraction_manager.extract_for_prediction(
                data_df, all_concepts
            )
        )

        # Get predictions from each initialization separately
        predictions_by_init = {}
        for init_seed in self.config.init_seeds:
            if init_seed in self.training_histories:
                init_predictions = self._get_init_predictions(
                    init_seed, data_df, all_extracted_features, use_posterior_iters
                )

                if init_predictions:
                    # Average across iterations for this initialization
                    init_ensemble = np.mean(init_predictions, axis=0)

                    # Convert to DataFrame
                    if init_ensemble.ndim == 1:
                        pred_df = pd.DataFrame(
                            {"prob_0": 1 - init_ensemble, "prob_1": init_ensemble}
                        )
                    else:
                        pred_df = pd.DataFrame(
                            init_ensemble,
                            columns=[
                                f"prob_{i}" for i in range(init_ensemble.shape[1])
                            ],
                        )

                    predictions_by_init[init_seed] = pred_df

        return predictions_by_init

    def _collect_concepts_for_prediction(
            self, use_posterior_iters: int = None, is_baseline: bool = False
    ) -> List[Dict]:
        """Collect unique concepts from all initializations for prediction."""
        all_concepts_to_extract = []

        for init_seed, history in self.training_histories.items():
            if use_posterior_iters is not None:
                start_iter = max(0, history.num_iters - use_posterior_iters)
            else:
                start_iter = 0

            if is_baseline:
                concepts = history._concepts[0:1][0]
            else:
                concepts = [
                    concept_dict
                    for concept_dicts in history._concepts[start_iter : history.num_iters]
                    for concept_dict in concept_dicts
                ]

            all_concepts_to_extract.extend(concepts)

        # Remove duplicates based on concept text
        unique_concepts = {}
        for concept in all_concepts_to_extract:
            concept_key = concept["concept"]
            if concept_key not in unique_concepts:
                unique_concepts[concept_key] = concept

        return list(unique_concepts.values())

    def _get_init_predictions(
        self,
        init_seed: int,
        data_df: pd.DataFrame,
        all_extracted_features: Dict,
        use_posterior_iters: int = None,
        is_baseline: bool = False
    ) -> List[np.ndarray]:
        """Get predictions for a specific initialization."""
        history = self.training_histories[init_seed]
        predictions = []

        if use_posterior_iters is not None:
            start_iter = max(0, history.num_iters - use_posterior_iters)
            end_iter = history.num_iters
        else:
            start_iter = 0
            end_iter = history.num_iters

        if is_baseline:
            start_iter = 0
            end_iter = 1

        for i in range(start_iter, end_iter):
            concept_dicts = history._concepts[i]

            # Get force_keep_columns and verify test data has required columns
            force_keep_columns = (
                history.force_keep_cols
                if hasattr(history, "force_keep_cols")
                and history.force_keep_cols is not None
                else None
            )

            # Verify test data has required columns if force_keep_columns is specified
            if force_keep_columns:
                missing_cols = set(force_keep_columns) - set(data_df.columns)
                if missing_cols:
                    raise ValueError(
                        f"Test data missing required tabular features: {missing_cols}. "
                        f"Available columns: {list(data_df.columns)}. "
                        f"Required columns: {force_keep_columns}"
                    )

            extracted_features = self.feature_extraction_manager.get_features_for_model(
                concept_dicts=concept_dicts,
                all_extracted_features=all_extracted_features,
                data_df=data_df,
                force_keep_columns=force_keep_columns,
            )

            model = history.get_model(index=i)
            pred_prob_i = model.predict_proba(extracted_features)
            predictions.append(pred_prob_i)

        return predictions

    def _load_checkpoint_if_exists(self):
        """Load checkpoint if it exists and no state is currently loaded."""
        if os.path.exists(self.checkpoint_file):
            state = CheckpointManager.load_state(self.checkpoint_file)
            self._restore_state(state)
            logging.info(f"Auto-loaded checkpoint from {self.checkpoint_file}")

    def _setup_training_environment(self, data_df: pd.DataFrame):
        """Setup LLM clients, evidence-span summaries, and semantic cache."""
        # Setup LLM clients
        self._setup_llm_clients()

    async def _run_baseline_phase(self, data_df: pd.DataFrame):
        """
        Execute baseline training phase.
        """
        if not self.ensemble_state.is_baseline_complete():
            logging.info("Phase 1: Parallel baseline training")
            self.ensemble_state.current_phase = TrainingPhase.BASELINE_RUNNING
            self.ensemble_state.phase_start_times["baseline"] = time.time()

            baseline_histories = await train_baselines_coordinated(
                data_df,
                self.config.init_seeds,
                self.config,
                self.concept_generator.generate_initial_concepts,
                self.feature_extraction_manager,
                self.concept_tracker,
            )
            # Update training histories and mark transition to greedy phase
            self.training_histories.update(baseline_histories)
            for init_seed, history in baseline_histories.items():
                history.mark_phase_transition("greedy", history.num_iters)

            # Mark baseline complete and save checkpoint
            for init_seed in self.config.init_seeds:
                self.ensemble_state.baseline_complete[init_seed] = True
            self.ensemble_state.current_phase = TrainingPhase.BASELINE_COMPLETE


            # Log progress summary
            completed_baseline = sum(self.ensemble_state.baseline_complete.values())
            total_baseline = len(self.config.init_seeds)
            runtime_seconds = (
                time.time() - self.ensemble_state.start_time
                if self.ensemble_state.start_time
                else 0
            )

            logging.info(
                f"Baseline phase completed: {completed_baseline}/{total_baseline} seeds, runtime: {runtime_seconds:.1f}s"
            )

            self._save_checkpoint()

            # Save training histories after baseline completes
            for init_seed, history in self.training_histories.items():
                init_output_dir = os.path.join(
                    self.output_dir, f"init_seed_{init_seed}"
                )
                os.makedirs(init_output_dir, exist_ok=True)

                # Save complete training history (consistent with greedy phase behavior)
                history_file = os.path.join(init_output_dir, "training_history.pkl")
                history.save(history_file)
                logging.info(
                    f"Saved training history for init_seed {init_seed} to {history_file}"
                )

                # Note: AUC plots are created in _finalize_training() via plot_training_aucs()

        else:
            logging.info("Baseline training already complete, skipping...")

    async def _run_greedy_phase(self, data_df: pd.DataFrame):
        """Execute greedy training phase."""
        self._create_greedy_concept_selectors()
        if not self.ensemble_state.is_greedy_complete():
            logging.info("Phase 2: Coordinated greedy training")
            self.ensemble_state.current_phase = TrainingPhase.GREEDY_RUNNING
            self.ensemble_state.phase_start_times["greedy"] = time.time()
            self._save_checkpoint()

            await self.train_greedy_coordinated(data_df, self.output_dir)

            # Mark greedy complete
            for init_seed in self.config.init_seeds:
                self.ensemble_state.greedy_complete[init_seed] = True
            self.ensemble_state.current_phase = TrainingPhase.COMPLETE


            # Log progress summary
            completed_greedy = sum(self.ensemble_state.greedy_complete.values())
            total_greedy = len(self.config.init_seeds)
            total_concepts = (
                len(self.ensemble_state.shared_extractions)
                if self.ensemble_state.shared_extractions
                else 0
            )
            runtime_seconds = (
                time.time() - self.ensemble_state.start_time
                if self.ensemble_state.start_time
                else 0
            )
            logging.info(
                f"Greedy phase completed: {completed_greedy}/{total_greedy} seeds, {total_concepts} concepts extracted, runtime: {runtime_seconds:.1f}s"
            )

            self._save_checkpoint()

            # Save final histories and plot AUCs
            for init_seed, history in self.training_histories.items():
                init_output_dir = os.path.join(
                    self.output_dir, f"init_seed_{init_seed}"
                )

                # Save complete training history
                final_file = os.path.join(init_output_dir, "training_history.pkl")
                history.save(final_file)
                logging.info(
                    f"Saved final history for init_seed {init_seed} to {final_file}"
                )

                # Note: AUC plots are created in _finalize_training() via plot_training_aucs()
        else:
            logging.info("Greedy training already complete, skipping...")

    def _finalize_training(self):
        """Save extraction files and finalize training."""
        # Save extraction files for each initialization (for plotting compatibility)
        self._save_extraction_files(self.output_dir)

        logging.info("Training completed successfully")

    async def train_greedy_coordinated(self, data_df: pd.DataFrame, output_dir: str):
        """
        Coordinated greedy training with shared concept extraction using GreedyConceptSelector.

        Instead of each initialization doing
        concept extraction separately, we batch extraction across all initializations.
        """
        logging.info("Starting coordinated greedy training with GreedyConceptSelector")

        # Store output_dir for use in _update_training_history
        self._current_output_dir = output_dir

        # Verify we have all training histories
        if self.ensemble_state.current_phase == TrainingPhase.BASELINE_COMPLETE:
            for init_seed in self.config.init_seeds:
                if init_seed not in self.training_histories:
                    raise ValueError(
                        f"Missing training history for init_seed {init_seed}. Available: {list(self.training_histories.keys())}"
                    )

        # Update GreedyConceptSelector with the training history (always do this)
        for init_seed in self.config.init_seeds:
            if init_seed in self.training_histories:
                self.concept_selectors[init_seed].init_history = (
                    self.training_histories[init_seed]
                )

        # Main coordinated training loop - resume from working epoch if needed
        if self.ensemble_state and self.ensemble_state.working_epoch >= 0:
            # Check if the working epoch was actually completed
            if self.ensemble_state.working_epoch > self.ensemble_state.completed_epoch:
                # Epoch was started but not completed, retry it
                start_epoch = self.ensemble_state.working_epoch
                logging.info(f"Retrying incomplete epoch {start_epoch + 1}")
            else:
                # Epoch was completed, move to next
                start_epoch = self.ensemble_state.working_epoch + 1
                logging.info(f"Resuming from epoch {start_epoch + 1}")
        else:
            start_epoch = 0


        for epoch in range(start_epoch, self.num_epochs):
            logging.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Update state to mark this epoch as being worked on
            self.ensemble_state.working_epoch = epoch

            # Reset concept iteration for each epoch start
            if epoch > self.ensemble_state.completed_epoch:
                # Starting a new epoch, reset iteration counter
                self.ensemble_state.completed_concept_iteration = -1
                logging.debug(f"Starting fresh epoch {epoch + 1}, reset concept iterations")
            else:
                # Retrying an incomplete epoch, preserve iteration state
                logging.debug(
                    f"Retrying epoch {epoch + 1}, preserving concept iteration state "
                    f"(completed={self.ensemble_state.completed_concept_iteration})"
                )

            # Validate state before saving
            try:
                self.ensemble_state.validate_state()
            except ValueError as e:
                logging.error(f"State validation failed before epoch {epoch + 1}: {e}")
                raise

            # Save checkpoint after marking work as started
            self._save_checkpoint()

            # Run coordinated concept evolution for this epoch
            await run_coordinated_concept_evolution(
                data_df,
                epoch,
                self.config,
                self.concept_selectors,
                self.concept_generator,
                self.feature_extraction_manager,
                self.concept_tracker,
                self.training_histories,
                self.ensemble_state,
                checkpoint_callback=self._save_checkpoint,
            )

            # Mark epoch as completed
            self.ensemble_state.completed_epoch = epoch
            self.ensemble_state.completed_concept_iteration = -1  # Reset for next epoch

            logging.info(
                f"Completed epoch {epoch + 1}/{self.num_epochs} "
                f"(working_epoch={self.ensemble_state.working_epoch}, "
                f"completed_epoch={self.ensemble_state.completed_epoch})"
            )

            # Validate state after epoch completion
            try:
                self.ensemble_state.validate_state()
            except ValueError as e:
                logging.error(f"State validation failed after epoch {epoch + 1}: {e}")
                raise

            # Save checkpoint after each epoch
            self._save_checkpoint()

            # Log epoch progress
            total_concepts = (
                len(self.ensemble_state.shared_extractions)
                if self.ensemble_state.shared_extractions
                else 0
            )
            logging.info(
                f"Completed epoch {epoch + 1}/{self.num_epochs}, {total_concepts} total concepts extracted"
            )

        logging.info("Coordinated greedy training completed")

    def _save_extraction_files(self, output_dir: str):
        """
        Save extraction files for each initialization for plotting compatibility.

        This saves the shared extractions from the annotation manager to extraction.pkl
        files in each init_seed_X directory, matching the format expected by plotting scripts.
        """
        if (
            not self.feature_extraction_manager
            or not self.feature_extraction_manager.shared_extractions
        ):
            logging.warning("No shared extractions available to save")
            return

        # Save extraction files for each initialization
        for init_seed in self.config.init_seeds:
            init_output_dir = os.path.join(output_dir, f"init_seed_{init_seed}")

            # Create directory if it doesn't exist
            os.makedirs(init_output_dir, exist_ok=True)

            extraction_file = os.path.join(init_output_dir, "extraction.pkl")

            # Save the shared extractions dictionary
            # This matches the format created by extract_features_by_llm_grouped
            with open(extraction_file, "wb") as f:
                pickle.dump(self.feature_extraction_manager.shared_extractions, f)

            logging.info(
                f"Saved extraction file for init_seed {init_seed} to {extraction_file} "
                f"({len(self.feature_extraction_manager.shared_extractions)} concepts)"
            )
