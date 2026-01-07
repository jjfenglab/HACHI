"""End-to-end integration tests for training pipeline."""

import logging
import os

import pytest

from src.ensemble_trainer import ConceptGeneratorFactory, TrainingPhase


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_end_to_end_training(test_data, temp_dir, small_config, setup_trainer_with_mocks):
    """Test complete end-to-end training pipeline."""
    # Use small configuration for faster testing
    config = small_config
    config.init_seeds = [1, 2]
    config.training.num_epochs = 1
    config.concept.goal_num_meta_concepts = 3

    trainer = setup_trainer_with_mocks(config)

    trainer.concept_generator = ConceptGeneratorFactory.create_generator(
        trainer.config, trainer.llm_dict, trainer.summaries_df
    )

    # Setup output directory
    output_dir = os.path.join(temp_dir, "training_output")
    os.makedirs(output_dir, exist_ok=True)

    # Create init_seed directories (normally done by scons)
    for init_seed in [1, 2]:
        os.makedirs(os.path.join(output_dir, f"init_seed_{init_seed}"), exist_ok=True)

    trainer.output_dir = output_dir
    trainer.checkpoint_file = os.path.join(output_dir, "ensemble_state_checkpoint.pkl")

    # Run baseline phase
    await trainer._run_baseline_phase(test_data)

    # Create concept selectors
    trainer._create_greedy_concept_selectors()

    # Set phase to baseline complete
    trainer.ensemble_state.current_phase = TrainingPhase.BASELINE_COMPLETE

    # Run greedy phase
    await trainer._run_greedy_phase(test_data)
    final_histories = trainer.training_histories

    # Verify results
    assert len(final_histories) == len(trainer.config.init_seeds), "Wrong number of final histories"

    for init_seed in trainer.config.init_seeds:
        assert init_seed in final_histories, f"Missing final history for seed {init_seed}"
        history = final_histories[init_seed]
        assert len(history._concepts) > 0, f"No concepts in final history for seed {init_seed}"
        assert len(history._aucs) > 0, f"No AUCs in final history for seed {init_seed}"

    # Check output files
    for init_seed in trainer.config.init_seeds:
        seed_dir = os.path.join(output_dir, f"init_seed_{init_seed}")
        assert os.path.exists(seed_dir), f"Missing output directory for seed {init_seed}"

        logging.info(f"Files in {seed_dir}: {os.listdir(seed_dir)}")
        final_file = os.path.join(seed_dir, "training_history.pkl")

        assert os.path.exists(
            final_file
        ), f"Missing history files for seed {init_seed}. {os.listdir(seed_dir)}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_baseline_phase_completion(test_data, temp_dir, small_config, setup_trainer_with_mocks):
    """Test that baseline phase completes successfully."""
    config = small_config
    config.init_seeds = [1, 2]

    trainer = setup_trainer_with_mocks(config)
    trainer.concept_generator = ConceptGeneratorFactory.create_generator(
        trainer.config, trainer.llm_dict, trainer.summaries_df
    )

    # Create output directories
    for init_seed in config.init_seeds:
        os.makedirs(os.path.join(temp_dir, f"init_seed_{init_seed}"), exist_ok=True)

    # Run baseline phase
    await trainer._run_baseline_phase(test_data)

    # Verify baseline completed
    assert trainer.ensemble_state.current_phase == TrainingPhase.BASELINE_COMPLETE
    assert trainer.ensemble_state.is_baseline_complete()

    # Verify training histories exist for all seeds
    assert len(trainer.training_histories) == len(config.init_seeds)
    for init_seed in config.init_seeds:
        assert init_seed in trainer.training_histories
        history = trainer.training_histories[init_seed]
        assert history.num_iters > 0
        assert len(history._concepts) > 0
