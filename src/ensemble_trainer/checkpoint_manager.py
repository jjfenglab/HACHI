"""
Checkpoint management for ensemble training.

This module handles state persistence, restoration, and validation for
coordinated ensemble training across multiple initializations.
"""

import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional

from .concept_tracker import ConceptTracker
from .feature_extraction_manager import FeatureExtractionManager
from .state import TrainerState


class CheckpointManager:
    """Manages checkpoint operations for ensemble training state."""

    @staticmethod
    def save_state(state: TrainerState, checkpoint_file: str):
        """Save TrainerState to checkpoint file."""
        checkpoint_data = {"state": state, "version": "1.0", "timestamp": time.time()}

        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

        # Save with atomic write
        temp_file = checkpoint_file + ".tmp"
        try:
            with open(temp_file, "wb") as f:
                pickle.dump(checkpoint_data, f)
            os.rename(temp_file, checkpoint_file)
            state.last_checkpoint_time = time.time()
            logging.info(f"Saved trainer state checkpoint to {checkpoint_file}")
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e

    @staticmethod
    def load_state(checkpoint_file: str) -> TrainerState:
        """Load TrainerState from checkpoint file."""
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)

        state = checkpoint_data["state"]
        version = checkpoint_data.get("version", "unknown")

        logging.info(
            f"Loaded trainer state checkpoint from {checkpoint_file} (version {version})"
        )
        return state

    @staticmethod
    def initialize_or_resume_state(
        init_seeds: List[int],
        num_epochs: int,
        num_meta_concepts: int,
        num_greedy_holdout: int,
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None,
    ) -> tuple[TrainerState, str, bool]:
        """
        Initialize new state or resume from checkpoint.

        Returns:
            tuple: (ensemble_state, checkpoint_file, is_resuming)
        """
        checkpoint_file = os.path.join(output_dir, "ensemble_state_checkpoint.pkl")

        # Check for resume request or existing checkpoint
        if resume_from_checkpoint:
            checkpoint_path = resume_from_checkpoint
        elif os.path.exists(checkpoint_file):
            checkpoint_path = checkpoint_file
        else:
            checkpoint_path = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            ensemble_state = CheckpointManager.load_state(checkpoint_path)
            logging.info(f"Resuming from checkpoint: {checkpoint_path}")
            CheckpointManager.validate_checkpoint_compatibility(
                ensemble_state, init_seeds, num_epochs, num_meta_concepts
            )
            return ensemble_state, checkpoint_file, True

        # Initialize new state
        ensemble_state = TrainerState(
            init_seeds=init_seeds,
            num_epochs=num_epochs,
            max_meta_concepts=num_meta_concepts,
            num_greedy_holdout=num_greedy_holdout,
        )
        logging.info("Starting fresh ensemble training")
        return ensemble_state, checkpoint_file, False

    @staticmethod
    def validate_checkpoint_compatibility(
        state: TrainerState,
        init_seeds: List[int],
        num_epochs: int,
        num_meta_concepts: int,
    ):
        """Validate that checkpoint is compatible with current configuration."""
        if state.init_seeds != init_seeds:
            raise ValueError(
                f"Checkpoint init_seeds {state.init_seeds} don't match current {init_seeds}"
            )

        if state.num_epochs != num_epochs:
            logging.warning(
                f"Checkpoint num_epochs {state.num_epochs} differs from current {num_epochs}"
            )

        if state.max_meta_concepts != num_meta_concepts:
            raise ValueError(
                f"Checkpoint max_meta_concepts {state.max_meta_concepts} doesn't match current {num_meta_concepts}"
            )

        logging.info("Checkpoint compatibility validated")

    @staticmethod
    def restore_from_checkpoint(
        ensemble_state: TrainerState,
        init_seeds: List[int],
    ) -> tuple[Dict, Dict[int, Any]]:
        """
        Restore trainer state from checkpoint.

        Returns:
            tuple: (concept_tracker_state, training_histories)
        """
        # Get concept tracker state
        concept_tracker_state = ensemble_state.concept_tracker_state or {}

        # Restore training histories
        training_histories = {}
        if not (
            hasattr(ensemble_state, "training_histories_files")
            and ensemble_state.training_histories_files
        ):
            raise ValueError(
                "Checkpoint does not contain unified training history files."
            )

        for init_seed in init_seeds:
            if init_seed in ensemble_state.training_histories_files:
                history_file = ensemble_state.training_histories_files[init_seed]
                with open(history_file, "rb") as f:
                    training_histories[init_seed] = pickle.load(f)
                logging.info(
                    f"Restored training history for init_seed {init_seed} from {history_file}"
                )

        logging.info(
            f"Resumed from {ensemble_state.current_phase.value} at working epoch {ensemble_state.working_epoch}"
        )

        return concept_tracker_state, training_histories

    @staticmethod
    def save_checkpoint(
        ensemble_state: TrainerState,
        checkpoint_file: str,
        output_dir: str,
        concept_tracker: ConceptTracker,
        feature_extraction_manager: Optional[FeatureExtractionManager],
        init_seeds: List[int],
        training_histories: Dict,
    ):
        """Save current state to checkpoint."""
        assert ensemble_state, "save_checkpoint called with None ensemble_state"

        # Update state with current progress
        ensemble_state.concept_tracker_state = concept_tracker.get_state()

        # Update shared extractions
        if feature_extraction_manager:
            ensemble_state.shared_extractions = (
                feature_extraction_manager.shared_extractions
            )

        # Update unified training history file paths
        # Initialize training_histories_files if it doesn't exist
        if (
            not hasattr(ensemble_state, "training_histories_files")
            or ensemble_state.training_histories_files is None
        ):
            ensemble_state.training_histories_files = {}

        for init_seed in init_seeds:
            if init_seed in training_histories:
                # Store path to unified training history
                file_path = os.path.join(
                    output_dir,
                    f"init_seed_{init_seed}",
                    "training_history.pkl",
                )
                ensemble_state.training_histories_files[init_seed] = file_path

                # Actually save the training history to disk
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                training_histories[init_seed].save(file_path)

        # Save checkpoint
        CheckpointManager.save_state(ensemble_state, checkpoint_file)
