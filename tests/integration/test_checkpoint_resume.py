"""Enhanced integration tests for checkpoint and resume functionality.

These tests verify the critical checkpoint/resume scenarios for research workflows:
- Resume after baseline completes
- State preservation across checkpoint/load
- Epoch and iteration tracking
- Extending training configuration
- Restoring shared extractions from checkpoint

Note: Most tests focus on checkpoint/resume MECHANICS without running full
greedy training (to avoid threading/cache issues).
"""

import os

import pytest

from src.ensemble_trainer import (
    CheckpointManager,
    ConceptGeneratorFactory,
    EnsembleTrainer,
    FeatureExtractionManager,
    TrainingPhase,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_from_baseline_complete(
    setup_trainer_for_baseline, small_config, small_test_data, temp_dir
):
    """Test that state is correctly restored after baseline phase completes.

    This is a common scenario: baseline training finishes, checkpoint is saved,
    then we can resume to continue with greedy training.
    """
    # Phase 1: Run baseline training
    trainer1 = await setup_trainer_for_baseline(small_config)

    # Run baseline phase
    await trainer1._run_baseline_phase(small_test_data)

    # Verify baseline completed
    assert trainer1.ensemble_state.current_phase == TrainingPhase.BASELINE_COMPLETE
    assert trainer1.ensemble_state.is_baseline_complete()
    assert len(trainer1.training_histories) == len(small_config.init_seeds)

    # Save checkpoint
    trainer1._save_checkpoint()
    checkpoint_path = trainer1.checkpoint_file
    assert os.path.exists(checkpoint_path)

    # Phase 2: Create new trainer and resume from checkpoint
    trainer2 = EnsembleTrainer.from_checkpoint(checkpoint_path, setup_for_training=True)

    # Setup mocks for trainer2 (same as original)
    from src.ensemble_trainer import FeatureExtractionManager
    from tests.fixtures.mock_llm import RealisticMockLLMApi

    mock_llm = RealisticMockLLMApi()
    trainer2.llm_dict = {"iter": mock_llm, "extraction": mock_llm}
    trainer2.feature_extraction_manager = FeatureExtractionManager(
        trainer2.config, trainer2.llm_dict, trainer2.concept_tracker
    )

    # Restore from checkpoint
    concept_tracker_state, training_histories = (
        CheckpointManager.restore_from_checkpoint(
            trainer2.ensemble_state, trainer2.config.init_seeds
        )
    )
    trainer2.training_histories = training_histories
    trainer2.concept_tracker.restore_state(concept_tracker_state)

    # Verify state restored correctly
    assert trainer2.ensemble_state.current_phase == TrainingPhase.BASELINE_COMPLETE
    assert trainer2.ensemble_state.is_baseline_complete()
    assert len(trainer2.training_histories) == len(small_config.init_seeds)

    # Verify training histories restored
    for init_seed in small_config.init_seeds:
        assert init_seed in trainer2.training_histories
        history = trainer2.training_histories[init_seed]
        assert (
            len(history._concepts) > 0
        ), f"Init {init_seed} should have concepts from baseline"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_mid_epoch_state(
    setup_trainer_for_greedy, small_config, small_test_data, temp_dir
):
    """Test checkpoint state tracking for mid-epoch interruption.

    Verifies that working_epoch and completed_epoch are correctly preserved
    when a checkpoint is saved mid-epoch.
    """
    # Use config with 2 epochs for this test
    small_config.training.num_epochs = 2

    # Phase 1: Run baseline
    trainer1 = await setup_trainer_for_greedy(small_config)

    # Manually set state to simulate mid-epoch interruption
    trainer1.ensemble_state.working_epoch = 0
    trainer1.ensemble_state.completed_epoch = -1  # No epoch completed yet
    trainer1.ensemble_state.current_phase = TrainingPhase.GREEDY_RUNNING

    # Save checkpoint (simulating auto-checkpoint during epoch)
    trainer1._save_checkpoint()
    checkpoint_path = trainer1.checkpoint_file

    # Phase 2: Resume from checkpoint
    trainer2 = EnsembleTrainer.from_checkpoint(checkpoint_path, setup_for_training=True)

    # Verify state shows mid-epoch interruption correctly
    assert trainer2.ensemble_state.working_epoch == 0
    assert trainer2.ensemble_state.completed_epoch == -1
    assert trainer2.ensemble_state.current_phase == TrainingPhase.GREEDY_RUNNING

    # Verify resume logic would retry the epoch
    # (working_epoch > completed_epoch means retry)
    assert (
        trainer2.ensemble_state.working_epoch > trainer2.ensemble_state.completed_epoch
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_completed_epoch_state(
    setup_trainer_for_greedy, small_config, small_test_data, temp_dir
):
    """Test checkpoint state tracking after completing an epoch.

    When resuming after a completed epoch, the state should indicate
    that working_epoch == completed_epoch (ready for next epoch).
    """
    # Use config with 2 epochs
    small_config.training.num_epochs = 2

    # Phase 1: Run baseline
    trainer1 = await setup_trainer_for_greedy(small_config)

    # Simulate completion of epoch 0
    trainer1.ensemble_state.working_epoch = 0
    trainer1.ensemble_state.completed_epoch = 0  # Epoch 0 completed
    trainer1.ensemble_state.current_phase = TrainingPhase.GREEDY_RUNNING

    # Save checkpoint
    trainer1._save_checkpoint()
    checkpoint_path = trainer1.checkpoint_file

    # Phase 2: Resume from checkpoint
    trainer2 = EnsembleTrainer.from_checkpoint(checkpoint_path, setup_for_training=True)

    # Verify state shows epoch 0 completed
    assert trainer2.ensemble_state.completed_epoch == 0
    assert trainer2.ensemble_state.working_epoch == 0
    assert trainer2.ensemble_state.current_phase == TrainingPhase.GREEDY_RUNNING

    # Verify resume logic would start next epoch
    # (working_epoch == completed_epoch means move to next)
    assert (
        trainer2.ensemble_state.working_epoch == trainer2.ensemble_state.completed_epoch
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_extending_training(
    setup_trainer_for_greedy, small_config, small_test_data, temp_dir
):
    """Test checkpoint state when extending training with num_additional_epochs.

    Research scenario: Model was trained for 1 epoch, but we want to train
    for 2 more epochs. Verify epoch extension mechanism works correctly.
    """
    # Phase 1: Train for 1 epoch initially
    small_config.training.num_epochs = 1

    trainer1 = await setup_trainer_for_greedy(small_config)

    # Simulate completion of 1 epoch
    trainer1.ensemble_state.completed_epoch = 0
    trainer1.ensemble_state.working_epoch = 0
    trainer1.ensemble_state.current_phase = TrainingPhase.COMPLETE

    # Save checkpoint
    trainer1._save_checkpoint()
    checkpoint_path = trainer1.checkpoint_file

    # Phase 2: Resume with extended epochs
    trainer2 = EnsembleTrainer.from_checkpoint(checkpoint_path, setup_for_training=True)

    # Verify initial state
    assert trainer2.num_epochs == 1
    assert trainer2.ensemble_state.completed_epoch == 0

    # Simulate extending training (modifying config, what continue_training does)
    num_additional_epochs = 2
    trainer2.config.training.num_epochs += num_additional_epochs
    trainer2.ensemble_state.num_epochs = trainer2.config.training.num_epochs

    # Verify num_epochs extended correctly
    assert trainer2.num_epochs == 3  # 1 original + 2 additional
    assert trainer2.ensemble_state.num_epochs == 3
    assert trainer2.config.training.num_epochs == 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_restores_shared_extractions(
    setup_trainer_for_baseline, small_config, small_test_data, temp_dir
):
    """Test that shared extractions cache is restored from checkpoint.

    This is critical for cost savings: we don't want to re-extract features
    that were already computed before the interruption.
    """
    # Phase 1: Run baseline to generate extractions
    trainer1 = await setup_trainer_for_baseline(small_config)

    await trainer1._run_baseline_phase(small_test_data)

    # Get shared extractions before checkpoint
    original_extractions = dict(trainer1.feature_extraction_manager.shared_extractions)
    assert len(original_extractions) > 0, "Should have some extractions from baseline"

    # Save checkpoint
    trainer1._save_checkpoint()
    checkpoint_path = trainer1.checkpoint_file

    # Phase 2: Resume from checkpoint
    trainer2 = EnsembleTrainer.from_checkpoint(checkpoint_path, setup_for_training=True)

    # Setup mocks and restore extraction manager
    from src.ensemble_trainer import FeatureExtractionManager
    from tests.fixtures.mock_llm import RealisticMockLLMApi

    mock_llm = RealisticMockLLMApi()
    trainer2.llm_dict = {"iter": mock_llm, "extraction": mock_llm}
    trainer2.feature_extraction_manager = FeatureExtractionManager(
        trainer2.config, trainer2.llm_dict, trainer2.concept_tracker
    )

    # Restore extractions from checkpoint (simulating what _setup_llm_clients does)
    if trainer2.ensemble_state.shared_extractions:
        trainer2.feature_extraction_manager.set_shared_extractions(
            trainer2.ensemble_state.shared_extractions
        )

    # Verify extractions were restored
    restored_extractions = trainer2.feature_extraction_manager.shared_extractions
    assert len(restored_extractions) > 0, "Should have restored extractions"

    # Verify same concepts are present (keys match)
    assert set(restored_extractions.keys()) == set(original_extractions.keys())

    # Verify extraction values match (at least for first concept)
    first_concept = list(original_extractions.keys())[0]
    import numpy as np

    np.testing.assert_array_equal(
        restored_extractions[first_concept], original_extractions[first_concept]
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_checkpoint_preserves_training_progress(
    setup_trainer_for_greedy, small_config, small_test_data, temp_dir
):
    """Test that all training progress is preserved across checkpoint/resume.

    Verifies that:
    - Training histories are preserved
    - Concept tracker state is preserved
    - Phase markers are correct
    - Epoch/iteration tracking is accurate
    """
    # Phase 1: Complete baseline
    trainer1 = await setup_trainer_for_greedy(small_config)

    # Record initial state
    num_inits = len(small_config.init_seeds)
    baseline_concepts = {}
    for init_seed in small_config.init_seeds:
        baseline_concepts[init_seed] = len(
            trainer1.training_histories[init_seed]._concepts[0]
        )

    # Save checkpoint after baseline
    trainer1._save_checkpoint()
    checkpoint_path = trainer1.checkpoint_file

    # Phase 2: Resume and verify everything preserved
    trainer2 = EnsembleTrainer.from_checkpoint(checkpoint_path, setup_for_training=True)

    # Setup mocks
    from src.ensemble_trainer import CheckpointManager, FeatureExtractionManager
    from tests.fixtures.mock_llm import RealisticMockLLMApi

    mock_llm = RealisticMockLLMApi()
    trainer2.llm_dict = {"iter": mock_llm, "extraction": mock_llm}
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

    # Verify training histories preserved
    assert len(trainer2.training_histories) == num_inits
    for init_seed in small_config.init_seeds:
        assert init_seed in trainer2.training_histories
        # Baseline concepts should match
        restored_concepts = len(trainer2.training_histories[init_seed]._concepts[0])
        assert restored_concepts == baseline_concepts[init_seed]

    # Verify phase tracking
    assert trainer2.ensemble_state.current_phase == TrainingPhase.BASELINE_COMPLETE
    assert trainer2.ensemble_state.is_baseline_complete()

    # Verify all baseline initializations marked complete
    for init_seed in small_config.init_seeds:
        assert trainer2.ensemble_state.baseline_complete[init_seed]
