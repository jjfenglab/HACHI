"""Integration tests for concept evolution."""

import os

import pytest
import src.common as common
from src.ensemble_trainer import ConceptGeneratorFactory, TrainingPhase, data_operations, selection_operations


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concept_evolution_workflow(test_data, temp_dir, small_config, setup_trainer_with_mocks):
    """Test complete concept evolution workflow from baseline through greedy selection."""
    # Setup trainer with single initialization for determinism
    config = small_config
    config.init_seeds = [1]
    config.concept.goal_num_meta_concepts = 3
    config.training.num_epochs = 1

    trainer = setup_trainer_with_mocks(config)

    # Direct outputs to isolated directory
    concept_evo_out = os.path.join(temp_dir, "concept_evolution_output")
    os.makedirs(concept_evo_out, exist_ok=True)
    trainer.output_dir = concept_evo_out
    trainer.checkpoint_file = os.path.join(concept_evo_out, "ensemble_state_checkpoint.pkl")

    # Setup concept generator and run baseline
    trainer.concept_generator = ConceptGeneratorFactory.create_generator(
        trainer.config, trainer.llm_dict, trainer.summaries_df
    )
    await trainer._run_baseline_phase(test_data)

    # Create concept selectors after baseline
    trainer._create_greedy_concept_selectors()

    init_seed = 1
    history = trainer.training_histories[init_seed]

    # Get current concepts (limit to configured goal_num_meta_concepts)
    current_concepts = history.get_last_concepts()[: trainer.config.concept.goal_num_meta_concepts]
    current_concept_names = [c["concept"] for c in current_concepts]

    # Extract features for current concepts
    current_shared_extractions = await trainer.feature_extraction_manager.extract_for_training(
        test_data,
        current_concept_names,
        max_new_tokens=trainer.config.llm.max_new_tokens,
    )

    # Build train split for candidate generation
    init_data_df, _ = data_operations.create_data_split(
        test_data, init_seed, trainer.train_frac
    )

    concept_selector = trainer.concept_selectors[init_seed]

    # Generate candidate concepts
    candidate_concept_dicts = trainer.concept_generator.generate_candidate_concepts(
        init_data_df,
        concept_selector,
        current_concepts,
        current_shared_extractions,
    )

    assert len(candidate_concept_dicts) > 0, "No candidate concepts generated"
    assert all("concept" in c for c in candidate_concept_dicts), "Candidate concepts missing 'concept' key"

    # Extract candidate features
    candidate_names = [c["concept"] for c in candidate_concept_dicts]
    candidate_shared_extractions = await trainer.feature_extraction_manager.extract_for_training(
        test_data,
        candidate_names,
        max_new_tokens=trainer.config.llm.max_new_tokens,
    )

    # Build candidate map
    init_candidate_map = {(init_seed, c["concept"]): c for c in candidate_concept_dicts}

    # Perform greedy selection
    final_concepts = selection_operations.perform_greedy_selection(
        test_data,
        concept_selector,
        init_candidate_map,
        current_shared_extractions,
        candidate_shared_extractions,
    )

    assert len(final_concepts) == trainer.config.concept.goal_num_meta_concepts, "Final concept set has wrong size"
    assert all("concept" in c for c in final_concepts), "Final concepts missing 'concept' key"

    # Verify we can build features for the final set
    all_extractions = {
        **current_shared_extractions,
        **candidate_shared_extractions,
    }

    X_check = common.get_features(
        final_concepts,
        {k: v for k, v in all_extractions.items() if k in {c["concept"] for c in final_concepts}},
        test_data,
        force_keep_columns=trainer.config.force_keep_columns,
    )

    assert X_check.shape[0] == len(test_data), "Feature rows mismatch dataset size"
