"""Unit tests for data operations."""

import numpy as np
import pandas as pd
import pytest

from src.ensemble_trainer import data_operations


@pytest.mark.unit
def test_create_data_split_deterministic():
    """Test that data splits are deterministic with same seed."""
    data = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})  # Balanced classes for stratification

    train1, test1 = data_operations.create_data_split(data, init_seed=42, data_split_fraction=0.7)
    train2, test2 = data_operations.create_data_split(data, init_seed=42, data_split_fraction=0.7)

    # Should be identical with same seed (ignoring _orig_index column)
    train1_clean = train1.drop(columns=["_orig_index"])
    train2_clean = train2.drop(columns=["_orig_index"])
    pd.testing.assert_frame_equal(train1_clean, train2_clean)


@pytest.mark.unit
def test_create_data_split_different_seeds():
    """Test that different seeds produce different splits."""
    data = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})

    train1, _ = data_operations.create_data_split(data, init_seed=1, data_split_fraction=0.7)
    train2, _ = data_operations.create_data_split(data, init_seed=2, data_split_fraction=0.7)

    # Should be different (with high probability)
    train1_clean = train1.drop(columns=["_orig_index"])
    train2_clean = train2.drop(columns=["_orig_index"])
    assert not train1_clean.equals(train2_clean)


@pytest.mark.unit
def test_create_data_split_size():
    """Test that data split has correct size."""
    data = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})

    train, test = data_operations.create_data_split(data, init_seed=42, data_split_fraction=0.7)

    # Should have correct size (fraction of original)
    assert len(train) == int(0.7 * len(data))
    assert len(test) == len(data) - int(0.7 * len(data))


@pytest.mark.unit
def test_create_data_split_returns_train_and_test():
    """Test that create_data_split returns both train and test DataFrames."""
    data = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})

    train, test = data_operations.create_data_split(data, init_seed=42, data_split_fraction=0.7)

    # Both should be DataFrames
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    # Combined length should equal original
    assert len(train) + len(test) == len(data)

    # Both should have _orig_index column
    assert "_orig_index" in train.columns
    assert "_orig_index" in test.columns


@pytest.mark.unit
def test_map_extractions_to_sample_basic():
    """Test basic extraction mapping functionality."""
    # Create original data
    data = pd.DataFrame({"x": range(10), "y": range(10)})

    # Create sample with _orig_index column (simulating bootstrap)
    sample = pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 2, 3],
        "y": [0, 1, 2, 3, 4, 2, 3],
        "_orig_index": [0, 1, 2, 3, 4, 2, 3]  # Includes duplicates
    })

    # Create mock extractions (one per original row)
    extractions = {
        "concept1": np.arange(10).reshape(-1, 1).astype(float),
        "concept2": np.arange(10, 20).reshape(-1, 1).astype(float),
    }

    concepts = [{"concept": "concept1"}, {"concept": "concept2"}]

    # Map extractions
    mapped = data_operations.map_extractions_to_sample(extractions, sample, concepts)

    # Should have all concepts
    assert len(mapped) == 2
    assert "concept1" in mapped
    assert "concept2" in mapped

    # Should have correct number of rows (matching sample size including duplicates)
    assert mapped["concept1"].shape[0] == len(sample)
    assert mapped["concept2"].shape[0] == len(sample)


@pytest.mark.unit
def test_map_extractions_to_sample_preserves_values():
    """Test that extraction mapping preserves original values."""
    data = pd.DataFrame({"x": range(5), "y": range(5)})

    # Create sample with explicit _orig_index
    sample = pd.DataFrame({
        "x": [0, 2, 4],
        "y": [0, 2, 4],
        "_orig_index": [0, 2, 4]
    })

    # Create extractions with known values
    extractions = {
        "concept1": np.array([[0], [1], [2], [3], [4]], dtype=float),
    }

    concepts = [{"concept": "concept1"}]

    # Map extractions
    mapped = data_operations.map_extractions_to_sample(extractions, sample, concepts)

    # Values should correspond to the _orig_index values
    expected = np.array([[0], [2], [4]], dtype=float)
    np.testing.assert_array_equal(mapped["concept1"], expected)


@pytest.mark.unit
def test_create_data_split_stratifies():
    """Test that data split uses stratified sampling."""
    # Create imbalanced data
    data = pd.DataFrame({
        "x": range(100),
        "y": [0] * 80 + [1] * 20  # 80% class 0, 20% class 1
    })

    train, test = data_operations.create_data_split(data, init_seed=42, data_split_fraction=0.7)

    # Both splits should maintain approximate class balance
    train_balance = train["y"].value_counts(normalize=True)
    test_balance = test["y"].value_counts(normalize=True)

    # Class 1 should be approximately 20% in both splits
    assert 0.15 < train_balance[1] < 0.25
    assert 0.15 < test_balance[1] < 0.25


@pytest.mark.unit
def test_create_data_split_edge_cases():
    """Test edge cases for data splits."""
    data = pd.DataFrame({"x": range(20), "y": [0, 1] * 10})

    # Test with small train fraction
    train_small, test_small = data_operations.create_data_split(
        data, init_seed=42, data_split_fraction=0.3
    )
    assert len(train_small) == 6

    # Test with large train fraction
    train_large, test_large = data_operations.create_data_split(
        data, init_seed=42, data_split_fraction=0.9
    )
    assert len(train_large) == 18


@pytest.mark.unit
def test_map_extractions_to_sample_with_duplicates():
    """Test extraction mapping handles duplicate indices correctly."""
    data = pd.DataFrame({"x": range(5), "y": range(5)})

    # Create sample with duplicates using _orig_index
    sample = pd.DataFrame({
        "x": [0, 0, 1],
        "y": [0, 0, 1],
        "_orig_index": [0, 0, 1]  # First index repeated twice
    })

    extractions = {
        "concept1": np.array([[10], [20], [30], [40], [50]], dtype=float),
    }

    concepts = [{"concept": "concept1"}]

    mapped = data_operations.map_extractions_to_sample(extractions, sample, concepts)

    # Should have 3 rows (matching sample with duplicates)
    assert mapped["concept1"].shape[0] == 3

    # First two rows should be identical (both index 0)
    np.testing.assert_array_equal(mapped["concept1"][0], mapped["concept1"][1])
    # Both should have value 10 (index 0 in original extractions)
    np.testing.assert_array_equal(mapped["concept1"][0], np.array([10.]))


@pytest.mark.unit
def test_create_data_split_indices():
    """Test create_data_split_indices returns correct indices."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Balanced for stratification

    train_indices, test_indices = data_operations.create_data_split_indices(
        y, init_seed=42, data_split_fraction=0.7
    )

    # Should have correct sizes
    assert len(train_indices) == 7
    assert len(test_indices) == 3

    # Indices should be disjoint
    assert len(set(train_indices) & set(test_indices)) == 0

    # All indices should be valid
    assert all(0 <= idx < len(y) for idx in train_indices)
    assert all(0 <= idx < len(y) for idx in test_indices)
