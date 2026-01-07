"""Shared pytest fixtures for ensemble trainer tests."""

import logging
import os
import tempfile
import types

import pytest

from src.ensemble_trainer import (
    CheckpointManager,
    ConceptGeneratorFactory,
    EnsembleTrainer,
    FeatureExtractionManager,
)
from tests.fixtures.mock_llm import RealisticMockLLMApi
from tests.fixtures.test_configs import (
    create_minimal_config,
    create_small_config,
    create_standard_config,
)
from tests.fixtures.test_data import (
    create_minimal_test_data,
    create_realistic_test_data,
    create_small_test_data,
)


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_data():
    """Standard test dataset (150 samples)."""
    return create_realistic_test_data(n_samples=150)


@pytest.fixture
def small_test_data():
    """Small test dataset for faster tests (50 samples)."""
    return create_small_test_data(n_samples=50)


@pytest.fixture
def minimal_test_data():
    """Minimal test dataset for unit tests (20 samples)."""
    return create_minimal_test_data(n_samples=20)


@pytest.fixture
def mock_llm():
    """Realistic mock LLM API."""
    return RealisticMockLLMApi(seed=42)


@pytest.fixture
def minimal_config(temp_dir):
    """Minimal configuration for unit tests."""
    return create_minimal_config(temp_dir)


@pytest.fixture
def small_config(temp_dir):
    """Small configuration for integration tests."""
    return create_small_config(temp_dir)


@pytest.fixture
def standard_config(temp_dir):
    """Standard configuration for integration tests."""
    return create_standard_config(temp_dir)


@pytest.fixture
def setup_trainer_with_mocks(temp_dir):
    """
    Factory fixture that creates a trainer with mock LLM.

    Returns a function that accepts config and returns configured trainer.
    """

    def _setup_trainer(config):
        """Create a test ensemble trainer with mock LLM."""
        trainer = EnsembleTrainer(config=config, output_dir=temp_dir)

        # Disable signal handlers for testing
        trainer._setup_signal_handlers = lambda: None

        # Mock LLM clients
        mock_llm = RealisticMockLLMApi()
        trainer.llm_dict = {
            "iter": mock_llm,
            "extraction": mock_llm,
        }

        # Mock feature extraction manager - use the ConceptTracker already created by trainer
        trainer.feature_extraction_manager = FeatureExtractionManager(
            trainer.config, trainer.llm_dict, trainer.concept_tracker
        )

        # Initialize trainer state for testing
        trainer._initialize_state()

        # Monkey patch _setup_llm_clients to preserve our mock
        def mock_setup_llm_clients(self):
            """Mock version of _setup_llm_clients that preserves the mock LLM."""
            # Keep existing llm_dict and feature_extraction_manager (our mocks)
            # Just handle the checkpoint restoration logic
            if self.ensemble_state and self.ensemble_state.shared_extractions:
                self.feature_extraction_manager.set_shared_extractions(
                    self.ensemble_state.shared_extractions
                )
                logging.info(
                    f"Restored {len(self.ensemble_state.shared_extractions)} shared extractions from checkpoint"
                )

            # Restore state from checkpoint if resuming
            if (
                self.ensemble_state
                and self.ensemble_state.current_phase
                != self.ensemble_state.current_phase.NOT_STARTED
            ):
                CheckpointManager.restore_from_checkpoint(
                    self.ensemble_state,
                    self.config.init_seeds,
                )

            # Don't create selectors here - let the trainer create them
            # when actually needed (after baseline training)

        # Bind the mock method to the trainer instance
        trainer._setup_llm_clients = types.MethodType(mock_setup_llm_clients, trainer)

        return trainer

    return _setup_trainer


@pytest.fixture
def setup_trainer_for_baseline(setup_trainer_with_mocks):
    """
    Factory fixture that creates a trainer ready for baseline phase.

    Returns a function that accepts config and returns trainer with concept generator.
    """

    async def _setup_for_baseline(config):
        """Create trainer and setup concept generator."""
        trainer = setup_trainer_with_mocks(config)

        # Create concept generator
        trainer.concept_generator = ConceptGeneratorFactory.create_generator(
            trainer.config, trainer.llm_dict, trainer.summaries_df
        )

        return trainer

    return _setup_for_baseline


@pytest.fixture
def setup_trainer_for_greedy(setup_trainer_for_baseline, test_data):
    """
    Factory fixture that creates a trainer ready for greedy phase.

    Returns a function that accepts config, runs baseline, and returns trainer.
    """

    async def _setup_for_greedy(config):
        """Create trainer, run baseline, and setup for greedy."""
        trainer = await setup_trainer_for_baseline(config)

        # Run baseline phase
        await trainer._run_baseline_phase(test_data)

        # Create concept selectors now that baseline is available
        trainer._create_greedy_concept_selectors()

        return trainer

    return _setup_for_greedy
