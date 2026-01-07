"""Integration tests for parallel execution coordination.

Tests the ThreadPoolExecutor-based parallel execution used for:
- Baseline concept generation
- Greedy candidate generation
- Concept selection

Note: These tests verify parallel execution SETUP without running full training
to avoid threading/cache issues. Full E2E parallel tests are in test_training_pipeline.py.
"""

import pytest
from concurrent.futures import ThreadPoolExecutor


@pytest.mark.integration
def test_threadpool_executor_available():
    """Test that ThreadPoolExecutor is available and works correctly.

    The EnsembleTrainer uses ThreadPoolExecutor for CPU-bound operations
    (sklearn model training during concept generation and selection).
    """
    def simple_task(x):
        return x * 2

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(simple_task, i) for i in range(5)]
        results = [f.result() for f in futures]

    assert results == [0, 2, 4, 6, 8]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_baseline_parallel_setup(setup_trainer_for_baseline, small_config, small_test_data):
    """Test that baseline phase is set up for parallel execution.

    Verifies that:
    - Multiple initializations are configured
    - Concept generator is available for each init
    - System is ready for parallel concept generation
    """
    trainer = await setup_trainer_for_baseline(small_config)

    # Verify multiple initializations configured
    assert len(small_config.init_seeds) >= 2, \
        "Need at least 2 inits for parallel execution"

    # Verify concept generator available
    assert trainer.concept_generator is not None, \
        "Concept generator should be initialized"

    # Verify ensemble state tracks all inits
    assert len(trainer.ensemble_state.baseline_complete) == len(small_config.init_seeds)
    assert len(trainer.ensemble_state.greedy_complete) == len(small_config.init_seeds)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concept_tracker_across_initializations(
    setup_trainer_for_baseline, small_config, small_test_data
):
    """Test that ConceptTracker correctly tracks concepts across multiple inits.

    The ConceptTracker is critical for parallel execution: it tracks which
    concepts belong to which initialization during parallel generation.
    """
    trainer = await setup_trainer_for_baseline(small_config)

    # Add concepts for different initializations
    for init_seed in small_config.init_seeds:
        concept = f"Test concept for init {init_seed}"
        trainer.concept_tracker.add_concept(init_seed, concept)

    # Verify each init has its own concepts
    for init_seed in small_config.init_seeds:
        concepts = trainer.concept_tracker.get_concepts_for_init(init_seed)
        expected_concept = f"Test concept for init {init_seed}"
        assert expected_concept in concepts, \
            f"Init {init_seed} should have its concept"

        # Verify other inits don't have this concept
        for other_seed in small_config.init_seeds:
            if other_seed != init_seed:
                other_concepts = trainer.concept_tracker.get_concepts_for_init(other_seed)
                assert expected_concept not in other_concepts, \
                    f"Init {other_seed} shouldn't have init {init_seed}'s concept"
