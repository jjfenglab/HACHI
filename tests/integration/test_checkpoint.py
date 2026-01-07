"""Integration tests for checkpoint functionality."""

import os

import pytest

from src.ensemble_trainer import CheckpointManager


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_initialization(standard_config, temp_dir):
    """Test checkpoint state initialization."""
    state, checkpoint_file, resuming = CheckpointManager.initialize_or_resume_state(
        standard_config.init_seeds,
        standard_config.training.num_epochs,
        standard_config.concept.goal_num_meta_concepts,
        standard_config.training.num_greedy_holdout,
        temp_dir,
        None,
    )

    assert not resuming, "Should not be resuming on first run"
    assert state is not None, "State should be initialized"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_save_and_load(standard_config, temp_dir, setup_trainer_with_mocks):
    """Test checkpoint save/load cycle."""
    trainer = setup_trainer_with_mocks(standard_config)

    # Initialize state
    state, checkpoint_file, _ = CheckpointManager.initialize_or_resume_state(
        trainer.config.init_seeds,
        trainer.num_epochs,
        trainer.config.concept.goal_num_meta_concepts,
        trainer.num_greedy_holdout,
        temp_dir,
        None,
    )

    # Save checkpoint
    CheckpointManager.save_checkpoint(
        state,
        checkpoint_file,
        temp_dir,
        trainer.concept_tracker,
        trainer.feature_extraction_manager,
        trainer.config.init_seeds,
        trainer.training_histories,
    )

    assert os.path.exists(checkpoint_file), "Checkpoint file should exist"

    # Load checkpoint
    state2, _, resuming2 = CheckpointManager.initialize_or_resume_state(
        trainer.config.init_seeds,
        trainer.num_epochs,
        trainer.config.concept.goal_num_meta_concepts,
        trainer.num_greedy_holdout,
        temp_dir,
        None,
    )

    assert resuming2, "Should be resuming from existing checkpoint"
    assert state2 is not None, "Loaded state should not be None"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_compatibility_validation(temp_dir):
    """Test checkpoint compatibility validation."""
    # Create a state
    state, _, _ = CheckpointManager.initialize_or_resume_state(
        [1, 2, 3], 2, 4, 1, temp_dir, None
    )

    # Modify state to be incompatible
    state.init_seeds = [1, 2]  # Different seeds

    # Should raise ValueError for incompatible checkpoint
    with pytest.raises(ValueError):
        CheckpointManager.validate_checkpoint_compatibility(
            state, [1, 2, 3], 2, 4  # Different seeds
        )
