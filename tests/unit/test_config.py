"""Unit tests for configuration system."""

import os

import pytest

from src.ensemble_trainer import (
    ConceptConfig,
    DataConfig,
    EnsembleConfig,
    LLMConfig,
    ModelConfig,
    TrainingConfig,
)


@pytest.mark.unit
def test_model_config_defaults():
    """Test ModelConfig initialization with defaults."""
    config = ModelConfig(
        model="l2",
        final_model_type="l2",
        learner_type="count_l2",
    )

    assert config.model == "l2"
    assert config.final_model_type == "l2"
    assert config.learner_type == "count_l2"
    assert config.use_acc is False  # Default


@pytest.mark.unit
def test_llm_config_initialization(temp_dir):
    """Test LLMConfig initialization."""
    cache_file = os.path.join(temp_dir, "test_cache.db")

    config = LLMConfig(
        llm_model="gpt-4o-mini",
        max_tokens=1000,
        max_new_tokens=5000,
        cache_file=cache_file,
    )

    assert config.llm_model == "gpt-4o-mini"
    assert config.max_tokens == 1000
    assert config.max_new_tokens == 5000
    assert config.cache_file == cache_file


@pytest.mark.unit
def test_data_config_defaults():
    """Test DataConfig initialization with defaults."""
    config = DataConfig(
        text_summary_column="sentence",
        concept_column="sentence",
    )

    assert config.text_summary_column == "sentence"
    assert config.concept_column == "sentence"
    assert config.min_prevalence == 0  # Default
    assert config.is_image is False  # Default
    assert config.max_section_length is None  # Default


@pytest.mark.unit
def test_training_config_defaults():
    """Test TrainingConfig initialization with defaults."""
    config = TrainingConfig(
        num_epochs=2,
    )

    assert config.num_epochs == 2
    assert config.train_frac == 0.5  # Default
    assert config.num_greedy_holdout == 1  # Default
    assert config.batch_size == 20  # Default


@pytest.mark.unit
def test_concept_config_initialization(temp_dir):
    """Test ConceptConfig initialization."""
    prompt_file = os.path.join(temp_dir, "prompt.txt")
    with open(prompt_file, "w") as f:
        f.write("test prompt")

    config = ConceptConfig(
        goal_num_meta_concepts=6,
        prompt_concepts_file=prompt_file,
    )

    assert config.goal_num_meta_concepts == 6
    assert config.prompt_concepts_file == prompt_file


@pytest.mark.unit
def test_ensemble_config_initialization(minimal_config):
    """Test EnsembleConfig initialization."""
    config = minimal_config

    assert len(config.init_seeds) > 0
    assert config.model is not None
    assert config.llm is not None
    assert config.data is not None
    assert config.training is not None
    assert config.concept is not None


@pytest.mark.unit
def test_ensemble_config_nested_access(minimal_config):
    """Test that nested config attributes are accessible."""
    config = minimal_config

    # Access nested model config
    assert config.model.model == "l2"

    # Access nested training config
    assert config.training.num_epochs >= 1

    # Access nested concept config
    assert config.concept.goal_num_meta_concepts >= 1


@pytest.mark.unit
def test_ensemble_config_with_overrides(temp_dir):
    """Test EnsembleConfig with custom values."""
    from tests.fixtures.test_configs import create_minimal_config

    config = create_minimal_config(
        temp_dir,
        init_seeds=[1, 2, 3, 4],
        num_epochs=5,
        num_meta_concepts=10,
    )

    assert config.init_seeds == [1, 2, 3, 4]
    assert config.training.num_epochs == 5
    assert config.concept.goal_num_meta_concepts == 10


@pytest.mark.unit
def test_config_immutability_check(minimal_config):
    """Test that config values can be modified (they're not frozen)."""
    # Configs are mutable for dynamic changes during training
    original_epochs = minimal_config.training.num_epochs
    minimal_config.training.num_epochs = original_epochs + 1

    assert minimal_config.training.num_epochs == original_epochs + 1


@pytest.mark.unit
def test_model_config_with_all_parameters():
    """Test ModelConfig with all parameters specified."""
    config = ModelConfig(
        model="l2",
        final_model_type="l2",
        learner_type="count_l2",
        count_vectorizer="count",
        use_acc=True,
        final_learner_type="l2",
    )

    assert config.model == "l2"
    assert config.final_model_type == "l2"
    assert config.learner_type == "count_l2"
    assert config.count_vectorizer == "count"
    assert config.use_acc is True
    assert config.final_learner_type == "l2"


@pytest.mark.unit
def test_training_config_with_all_parameters():
    """Test TrainingConfig with all parameters specified."""
    config = TrainingConfig(
        num_epochs=3,
        num_greedy_epochs=2,
        batch_size=16,
        batch_concept_size=20,
        batch_obs_size=2,
        train_frac=0.7,
        num_greedy_holdout=2,
    )

    assert config.num_epochs == 3
    assert config.num_greedy_epochs == 2
    assert config.batch_size == 16
    assert config.batch_concept_size == 20
    assert config.batch_obs_size == 2
    assert config.train_frac == 0.7
    assert config.num_greedy_holdout == 2


@pytest.mark.unit
def test_ensemble_config_sampling_methods(temp_dir):
    """Test EnsembleConfig with different sampling methods."""
    from tests.fixtures.test_configs import create_minimal_config

    # Test bootstrap
    config_bootstrap = create_minimal_config(temp_dir, sampling_method="bootstrap")
    assert config_bootstrap.training.sampling_method == "bootstrap"

    # Test data_split
    config_split = create_minimal_config(temp_dir, sampling_method="data_split")
    assert config_split.training.sampling_method == "data_split"
