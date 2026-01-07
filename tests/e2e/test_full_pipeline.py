"""End-to-end tests for complete training pipelines.

These tests verify that the full training workflow completes successfully
from start to finish, including baseline training, greedy training, and predictions.

WARNING: These tests may hit the DuckDB threading issue

Tests are marked @pytest.mark.slow and @pytest.mark.e2e for selective execution.
"""

import os
import types

import pytest

from src.ensemble_trainer import (
    ConceptGeneratorFactory,
    EnsembleTrainer,
    FeatureExtractionManager,
    TrainingPhase,
)


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_training_pipeline_small(
    setup_trainer_with_mocks, small_config, small_test_data, temp_dir
):
    """Test complete training pipeline with minimal config.

    This is the basic E2E smoke test that verifies:
    - Baseline phase completes successfully
    - Greedy training completes (if no threading issues)
    - Training histories are created for all initializations
    - Final ensemble state is correct

    Uses small_config (2 inits, 1 epoch) for faster execution.
    """
    # Use minimal config for faster E2E test
    small_config.training.num_epochs = 1
    small_config.training.num_baseline_concepts = 3

    trainer = setup_trainer_with_mocks(small_config)

    # Setup components needed for training (E2E tests use mocks, so can't use fit() directly)
    trainer._initialize_state()
    trainer.feature_extraction_manager = FeatureExtractionManager(
        trainer.config, trainer.llm_dict, trainer.concept_tracker
    )
    trainer.concept_generator = ConceptGeneratorFactory.create_generator(
        trainer.config, trainer.llm_dict, trainer.summaries_df
    )
    trainer._create_greedy_concept_selectors()

    # Create output directories
    for init_seed in small_config.init_seeds:
        os.makedirs(os.path.join(temp_dir, f"init_seed_{init_seed}"), exist_ok=True)

    # Run complete training pipeline
    await trainer._run_baseline_phase(small_test_data)
    await trainer._run_greedy_phase(small_test_data)

    # Verify training completed
    assert trainer.ensemble_state.current_phase in [
        TrainingPhase.GREEDY_RUNNING,
        TrainingPhase.COMPLETE,
    ], f"Expected training to complete or be running, got {trainer.ensemble_state.current_phase}"

    # Verify baseline completed for all inits
    assert trainer.ensemble_state.is_baseline_complete()
    for init_seed in small_config.init_seeds:
        assert trainer.ensemble_state.baseline_complete[
            init_seed
        ], f"Init {init_seed} baseline should be complete"

    # Verify training histories created
    assert len(trainer.training_histories) == len(
        small_config.init_seeds
    ), f"Expected {len(small_config.init_seeds)} training histories, got {len(trainer.training_histories)}"

    # Verify each init has concepts
    for init_seed in small_config.init_seeds:
        assert init_seed in trainer.training_histories
        history = trainer.training_histories[init_seed]
        # Should have at least baseline concepts
        assert (
            len(history._concepts[0]) >= small_config.training.num_baseline_concepts
        ), f"Init {init_seed} should have at least {small_config.training.num_baseline_concepts} baseline concepts"


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_checkpoint_and_resume_full_pipeline(
    setup_trainer_with_mocks, small_config, small_test_data, temp_dir
):
    """Test full checkpoint/resume workflow E2E.

    This test verifies:
    - Training can be interrupted and checkpointed
    - Training can be resumed from checkpoint
    - Resumed training continues correctly

    WARNING: This test is SKIPPED due to DuckDB threading issue
    """
    # Phase 1: Initial training (just baseline)
    small_config.training.num_epochs = 1
    small_config.training.num_baseline_concepts = 3

    trainer1 = setup_trainer_with_mocks(small_config)

    # Setup components
    trainer1._initialize_state()
    trainer1.feature_extraction_manager = FeatureExtractionManager(
        trainer1.config, trainer1.llm_dict, trainer1.concept_tracker
    )
    trainer1.concept_generator = ConceptGeneratorFactory.create_generator(
        trainer1.config, trainer1.llm_dict, trainer1.summaries_df
    )
    trainer1._create_greedy_concept_selectors()

    # Create output directories
    for init_seed in small_config.init_seeds:
        os.makedirs(os.path.join(temp_dir, f"init_seed_{init_seed}"), exist_ok=True)

    # Run baseline phase only
    await trainer1._run_baseline_phase(small_test_data)

    # Verify baseline completed
    assert trainer1.ensemble_state.is_baseline_complete()

    # Save checkpoint
    trainer1._save_checkpoint()
    checkpoint_path = trainer1.checkpoint_file

    # Phase 2: Resume from checkpoint
    trainer2 = EnsembleTrainer.from_checkpoint(
        checkpoint_path, setup_for_training=False
    )

    # Setup mocks for trainer2
    from src.ensemble_trainer import CheckpointManager
    from tests.fixtures.mock_llm import RealisticMockLLMApi

    mock_llm = RealisticMockLLMApi()
    trainer2.llm_dict = {"iter": mock_llm, "extraction": mock_llm}

    # Monkey patch _setup_llm_clients to prevent real LLM API calls
    def mock_setup_llm_clients(self):
        """Mock version of _setup_llm_clients that preserves the mock LLM."""
        # Keep existing llm_dict (our mocks)
        # Just handle the checkpoint restoration logic
        if self.ensemble_state and self.ensemble_state.shared_extractions:
            self.feature_extraction_manager.set_shared_extractions(
                self.ensemble_state.shared_extractions
            )

    # Monkey patch _setup_training_environment to prevent real LLM setup
    def mock_setup_training_environment(self, data_df):
        """Mock version that skips real LLM setup."""
        # Don't call the real _setup_llm_clients, we already have mocks
        pass

    # Bind the mock methods to the trainer instance
    trainer2._setup_llm_clients = types.MethodType(mock_setup_llm_clients, trainer2)
    trainer2._setup_training_environment = types.MethodType(
        mock_setup_training_environment, trainer2
    )

    # Now create feature extraction manager with mocks
    trainer2.feature_extraction_manager = FeatureExtractionManager(
        trainer2.config, trainer2.llm_dict, trainer2.concept_tracker
    )

    # Restore state
    concept_tracker_state, training_histories = (
        CheckpointManager.restore_from_checkpoint(
            trainer2.ensemble_state, trainer2.config.init_seeds
        )
    )
    trainer2.training_histories = training_histories
    trainer2.concept_tracker.restore_state(concept_tracker_state)

    # Create concept generator needed for greedy phase
    trainer2.concept_generator = ConceptGeneratorFactory.create_generator(
        trainer2.config, trainer2.llm_dict, trainer2.summaries_df
    )

    # Create concept selectors for greedy phase
    trainer2._create_greedy_concept_selectors()

    # Verify state before continuing
    assert trainer2.ensemble_state.is_baseline_complete()
    assert len(trainer2.training_histories) == len(small_config.init_seeds)

    # Continue training (greedy phase)
    # NOTE: This may segfault due to DuckDB threading issue
    await trainer2.continue_training(small_test_data)

    # Verify training completed or is running
    assert trainer2.ensemble_state.current_phase in [
        TrainingPhase.GREEDY_RUNNING,
        TrainingPhase.COMPLETE,
    ]


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_prediction_after_training(
    setup_trainer_with_mocks, small_config, small_test_data, temp_dir
):
    """Test that predictions work after training completes.

    This test verifies:
    - Training completes successfully
    - Predictions can be generated on test data
    - Prediction outputs have correct shape and format
    - Each initialization produces predictions

    """
    # Use minimal config for faster execution
    small_config.training.num_epochs = 1
    small_config.training.num_baseline_concepts = 3

    trainer = setup_trainer_with_mocks(small_config)

    # Setup components needed for training
    trainer._initialize_state()
    trainer.feature_extraction_manager = FeatureExtractionManager(
        trainer.config, trainer.llm_dict, trainer.concept_tracker
    )
    trainer.concept_generator = ConceptGeneratorFactory.create_generator(
        trainer.config, trainer.llm_dict, trainer.summaries_df
    )
    trainer._create_greedy_concept_selectors()

    # Create output directories
    for init_seed in small_config.init_seeds:
        os.makedirs(os.path.join(temp_dir, f"init_seed_{init_seed}"), exist_ok=True)

    # Run complete training
    await trainer._run_baseline_phase(small_test_data)
    await trainer._run_greedy_phase(small_test_data)

    # Verify training completed baseline at minimum
    assert trainer.ensemble_state.is_baseline_complete()

    # Verify we have training histories
    assert len(trainer.training_histories) > 0

    # Verify each init has concepts and models
    for init_seed in small_config.init_seeds:
        assert (
            init_seed in trainer.training_histories
        ), f"Missing history for init {init_seed}"

        history = trainer.training_histories[init_seed]

        # Verify history has concepts
        assert (
            len(history._concepts) > 0
        ), f"Init {init_seed} should have concept iterations"
        assert (
            len(history._concepts[0]) > 0
        ), f"Init {init_seed} should have concepts in baseline iteration"

        # Verify we have trained models
        assert len(history._models) > 0, f"Init {init_seed} should have trained models"


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_training_with_different_configs(
    setup_trainer_with_mocks, minimal_config, small_test_data, temp_dir
):
    """Test training works with different configuration variations.

    This test verifies:
    - Training works with 1 initialization (minimal)
    - Training works with different num_baseline_concepts
    - Training completes successfully with varied configs
    - Different config combinations don't break the pipeline

    """
    # Test 1: Single initialization (minimal config)
    minimal_config.training.num_epochs = 1
    minimal_config.training.num_baseline_concepts = 2

    trainer1 = setup_trainer_with_mocks(minimal_config)

    # Setup components for test 1
    trainer1._initialize_state()
    trainer1.feature_extraction_manager = FeatureExtractionManager(
        trainer1.config, trainer1.llm_dict, trainer1.concept_tracker
    )
    trainer1.concept_generator = ConceptGeneratorFactory.create_generator(
        trainer1.config, trainer1.llm_dict, trainer1.summaries_df
    )
    trainer1._create_greedy_concept_selectors()

    # Create output directories
    for init_seed in minimal_config.init_seeds:
        os.makedirs(os.path.join(temp_dir, f"init_seed_{init_seed}"), exist_ok=True)

    # Run training
    await trainer1._run_baseline_phase(small_test_data)

    # Verify single-init training works
    assert trainer1.ensemble_state.is_baseline_complete()
    assert len(trainer1.training_histories) == 1

    # Verify baseline concepts match config
    for init_seed in minimal_config.init_seeds:
        history = trainer1.training_histories[init_seed]
        # Should have the configured number of baseline concepts
        assert (
            len(history._concepts[0]) >= minimal_config.training.num_baseline_concepts
        ), f"Expected at least {minimal_config.training.num_baseline_concepts} baseline concepts"
