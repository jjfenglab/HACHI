"""
State management and checkpointing for ensemble training.

This module contains the TrainerState class for managing and persisting
the state of ensemble training processes.
"""

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import EnsembleConfig, TrainingPhase


@dataclass
class TrainerState:
    """
    Comprehensive state management for ensemble training with resume support.

    This class tracks all aspects of ensemble training state to enable
    resuming from interruptions at any point in the training process.
    """

    # Training configuration
    init_seeds: List[int]
    num_epochs: int
    max_meta_concepts: int
    num_greedy_holdout: int

    # Configuration and paths (added in refactoring)
    config: Optional["EnsembleConfig"] = None
    output_dir: Optional[str] = None

    # Current progress
    current_phase: TrainingPhase = TrainingPhase.NOT_STARTED

    # Simplified state tracking - only track completed progress
    working_epoch: int = -1  # Current epoch being worked on (-1 means none started)
    completed_epoch: int = -1  # -1 means no epochs completed yet
    completed_concept_iteration: int = (
        -1
    )  # Last completed iteration (-1 means none completed)

    # Phase completion tracking per initialization
    baseline_complete: Dict[int, bool] = None
    greedy_complete: Dict[int, bool] = None

    # Concept tracking state
    concept_tracker_state: Dict[str, Any] = None

    # Shared extractions cache (concept -> features)
    shared_extractions: Dict[str, Any] = None

    # Training histories per initialization
    training_histories_files: Dict[int, str] = None

    # Timestamps for tracking
    start_time: float = None
    last_checkpoint_time: float = None
    phase_start_times: Dict[str, float] = None

    # Error recovery
    interruption_count: int = 0
    last_error: Optional[str] = None

    def __post_init__(self):
        """Initialize nested dictionaries if they weren't provided."""
        if self.baseline_complete is None:
            self.baseline_complete = {seed: False for seed in self.init_seeds}
        if self.greedy_complete is None:
            self.greedy_complete = {seed: False for seed in self.init_seeds}
        if self.concept_tracker_state is None:
            self.concept_tracker_state = {}
        if self.shared_extractions is None:
            self.shared_extractions = {}
        if self.training_histories_files is None:
            self.training_histories_files = {}
        if self.phase_start_times is None:
            self.phase_start_times = {}
        if self.start_time is None:
            self.start_time = time.time()

    def is_baseline_complete(self) -> bool:
        """Check if baseline training is complete for all initializations."""
        return all(self.baseline_complete.values())

    def is_greedy_complete(self) -> bool:
        """Check if greedy training is complete for all initializations."""
        return all(self.greedy_complete.values())

    def validate_state(self) -> bool:
        """
        Validate that state invariants are maintained.

        Returns:
            bool: True if state is valid, raises ValueError otherwise

        Raises:
            ValueError: If state invariants are violated
        """
        # Epoch invariants
        if self.completed_epoch > self.working_epoch:
            raise ValueError(
                f"Invalid state: completed_epoch ({self.completed_epoch}) > "
                f"working_epoch ({self.working_epoch})"
            )

        # Check valid ranges
        if self.working_epoch >= self.num_epochs:
            raise ValueError(
                f"Invalid state: working_epoch ({self.working_epoch}) >= "
                f"num_epochs ({self.num_epochs})"
            )

        # Concept iteration range check
        # if self.completed_concept_iteration >= self.num_meta_concepts:
        #     raise ValueError(
        #         f"Invalid state: completed_concept_iteration ({self.completed_concept_iteration}) >= "
        #         f"num_meta_concepts ({self.num_meta_concepts})"
        #     )

        return True

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return (
            f"TrainerState(phase={self.current_phase.value}, "
            f"working_epoch={self.working_epoch}, "
            f"completed_iteration={self.completed_concept_iteration}, "
            f"seeds={len(self.init_seeds)})"
        )
