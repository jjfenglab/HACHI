"""Integration tests for data operations."""

import numpy as np
import pytest

from src.ensemble_trainer import data_operations


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_split_for_multiple_seeds(test_data):
    """Test data splitting for multiple seeds."""
    for init_seed in [1, 2, 3]:
        train_data, test_split = data_operations.create_data_split(
            test_data, init_seed, 0.7
        )
        assert len(train_data) > 0, f"Empty train split for seed {init_seed}"
        assert len(test_split) > 0, f"Empty test split for seed {init_seed}"
        assert len(train_data) + len(test_split) == len(test_data), "Split sizes don't sum to original"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_split_stratification(test_data):
    """Test that data split maintains class balance."""
    train_data, test_split = data_operations.create_data_split(test_data, 42, 0.8)

    # Check that split has expected size
    assert len(train_data) == int(0.8 * len(test_data)), "Wrong train split size"

    # Check stratification maintained approximate class balance
    orig_balance = test_data["y"].value_counts(normalize=True)
    train_balance = train_data["y"].value_counts(normalize=True)

    # Class proportions should be similar (within 10%)
    for cls in orig_balance.index:
        assert abs(orig_balance[cls] - train_balance[cls]) < 0.1, f"Class {cls} imbalanced"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extraction_mapping(test_data):
    """Test extraction mapping to bootstrap samples."""
    # Create mock extractions
    mock_extractions = {
        "concept1": np.random.rand(len(test_data), 1),
        "concept2": np.random.rand(len(test_data), 1),
    }
    mock_concepts = [{"concept": "concept1"}, {"concept": "concept2"}]

    # Map extractions
    mapped = data_operations.map_extractions_to_sample(
        mock_extractions, test_data, mock_concepts
    )

    assert len(mapped) == 2, "Wrong number of mapped extractions"
    assert "concept1" in mapped
    assert "concept2" in mapped
