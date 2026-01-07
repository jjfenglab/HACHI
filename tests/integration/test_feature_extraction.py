"""Integration tests for feature extraction manager.

Tests the key optimization: batched feature extraction with caching to reduce
expensive LLM API calls across multiple initializations.
"""

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_feature_extraction_batch(setup_trainer_with_mocks, small_config, small_test_data):
    """Test batch feature extraction."""
    trainer = setup_trainer_with_mocks(small_config)

    concepts = [
        "Does the patient have chest pain?",
        "Does the patient have diabetes?",
    ]

    extractions = await trainer.feature_extraction_manager.extract_features_batch(
        small_test_data.head(20),
        concepts,
        max_new_tokens=1000,
    )

    assert len(extractions) == len(concepts), "Wrong number of extractions"
    for concept in concepts:
        assert concept in extractions, f"Missing extraction for {concept}"
        assert extractions[concept].shape[0] == 20, f"Wrong extraction shape for {concept}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_feature_extraction_caching(setup_trainer_with_mocks, small_config, small_test_data):
    """Test that feature extraction caching works correctly."""
    trainer = setup_trainer_with_mocks(small_config)

    concepts = [
        "Does the patient have chest pain?",
        "Does the patient have diabetes?",
    ]

    # First extraction
    extractions1 = await trainer.feature_extraction_manager.extract_features_batch(
        small_test_data.head(20),
        concepts,
        max_new_tokens=1000,
    )

    # Second extraction - should use cache
    extractions2 = await trainer.feature_extraction_manager.extract_features_batch(
        small_test_data.head(20),
        concepts,
        max_new_tokens=1000,
    )

    # Should return same results from cache
    for concept in concepts:
        np.testing.assert_array_equal(extractions1[concept], extractions2[concept])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_caching_reduces_llm_calls(setup_trainer_with_mocks, small_config, small_test_data):
    """Test that caching reduces LLM API calls.

    This is the key optimization: extracting the same concept multiple times
    should only call the LLM once.
    """
    trainer = setup_trainer_with_mocks(small_config)

    # Track LLM call count
    from tests.fixtures.mock_llm import RealisticMockLLMApi
    mock_llm = RealisticMockLLMApi(seed=42)
    trainer.llm_dict["extraction"] = mock_llm

    concepts = [
        "Does the patient have chest pain?",
        "Does the patient have diabetes?",
    ]

    # First extraction - should call LLM
    await trainer.feature_extraction_manager.extract_features_batch(
        small_test_data.head(20),
        concepts,
        max_new_tokens=1000,
    )

    call_count_after_first = mock_llm.call_count

    # Second extraction of same concepts - should NOT call LLM
    await trainer.feature_extraction_manager.extract_features_batch(
        small_test_data.head(20),
        concepts,
        max_new_tokens=1000,
    )

    # LLM call count should NOT increase (all cached)
    assert mock_llm.call_count == call_count_after_first, \
        "Cached extraction should not call LLM again"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_partial_cache_hit(setup_trainer_with_mocks, small_config, small_test_data):
    """Test extraction when some concepts are cached and some are new.

    Realistic scenario: baseline extracted concepts A, B. Greedy adds concept C.
    Should only extract C, reuse A and B from cache.
    """
    trainer = setup_trainer_with_mocks(small_config)

    from tests.fixtures.mock_llm import RealisticMockLLMApi
    mock_llm = RealisticMockLLMApi(seed=42)
    trainer.llm_dict["extraction"] = mock_llm

    # Extract first batch of concepts
    concepts_batch1 = [
        "Does the patient have chest pain?",
        "Does the patient have diabetes?",
    ]

    await trainer.feature_extraction_manager.extract_features_batch(
        small_test_data.head(20),
        concepts_batch1,
        max_new_tokens=1000,
    )

    call_count_after_batch1 = mock_llm.call_count

    # Extract second batch with overlap (chest pain is cached, hypertension is new)
    concepts_batch2 = [
        "Does the patient have chest pain?",  # Cached
        "Does the patient have hypertension?",  # New
    ]

    extractions = await trainer.feature_extraction_manager.extract_features_batch(
        small_test_data.head(20),
        concepts_batch2,
        max_new_tokens=1000,
    )

    # Should have called LLM only for the new concept
    # (Call count increase should be less than if both were extracted)
    assert mock_llm.call_count > call_count_after_batch1, \
        "Should have called LLM for new concept"

    # Both concepts should be in results
    assert "Does the patient have chest pain?" in extractions
    assert "Does the patient have hypertension?" in extractions


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_across_initializations(
    setup_trainer_for_baseline, small_config, small_test_data
):
    """Test that feature extraction is shared across multiple initializations.

    Key optimization: When multiple inits generate the same concept, extract once
    and share across all inits.
    """
    trainer = await setup_trainer_for_baseline(small_config)

    # Run baseline phase (generates concepts for each init)
    await trainer._run_baseline_phase(small_test_data)

    # Get shared extractions
    shared_extractions = trainer.feature_extraction_manager.shared_extractions

    # Should have extractions from all initializations
    assert len(shared_extractions) > 0, "Should have shared extractions"

    # Verify extractions are available across inits
    # Each init's concepts should be in the shared cache
    for init_seed in small_config.init_seeds:
        concepts_for_init = trainer.concept_tracker.get_concepts_for_init(init_seed)
        for concept in concepts_for_init:
            assert concept in shared_extractions, \
                f"Concept {concept} from init {init_seed} should be in shared cache"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_cache_restoration_from_checkpoint(
    setup_trainer_for_baseline, small_config, small_test_data, temp_dir
):
    """Test that extraction cache is correctly restored from checkpoint.

    Critical for cost savings: after resume, we shouldn't re-extract concepts
    that were already computed before interruption.
    """
    # Phase 1: Run baseline and get extractions
    trainer1 = await setup_trainer_for_baseline(small_config)
    await trainer1._run_baseline_phase(small_test_data)

    original_cache = dict(trainer1.feature_extraction_manager.shared_extractions)
    original_concepts = set(original_cache.keys())

    # Save checkpoint
    trainer1._save_checkpoint()
    checkpoint_path = trainer1.checkpoint_file

    # Phase 2: Resume from checkpoint
    from src.ensemble_trainer import EnsembleTrainer, FeatureExtractionManager
    from tests.fixtures.mock_llm import RealisticMockLLMApi

    trainer2 = EnsembleTrainer.from_checkpoint(
        checkpoint_path,
        setup_for_training=True
    )

    # Setup mocks
    mock_llm = RealisticMockLLMApi(seed=42)
    trainer2.llm_dict = {"iter": mock_llm, "extraction": mock_llm}
    trainer2.feature_extraction_manager = FeatureExtractionManager(
        trainer2.config, trainer2.llm_dict, trainer2.concept_tracker
    )

    # Restore extraction cache from checkpoint
    if trainer2.ensemble_state.shared_extractions:
        trainer2.feature_extraction_manager.set_shared_extractions(
            trainer2.ensemble_state.shared_extractions
        )

    # Verify cache restored
    restored_cache = trainer2.feature_extraction_manager.shared_extractions
    restored_concepts = set(restored_cache.keys())

    # Should have same concepts
    assert restored_concepts == original_concepts, \
        "Restored cache should have same concepts as original"

    # Verify extraction values match (sample a few)
    for concept in list(original_concepts)[:3]:  # Check first 3
        np.testing.assert_array_equal(
            restored_cache[concept],
            original_cache[concept],
            err_msg=f"Extraction for '{concept}' doesn't match after restore"
        )

    # Verify that extracting cached concepts doesn't call LLM
    initial_call_count = mock_llm.call_count
    cached_concepts = list(restored_concepts)[:2]

    await trainer2.feature_extraction_manager.extract_features_batch(
        small_test_data.head(20),
        cached_concepts,
        max_new_tokens=1000,
    )

    # Should not have called LLM (all cached)
    assert mock_llm.call_count == initial_call_count, \
        "Extracting cached concepts after restore should not call LLM"
