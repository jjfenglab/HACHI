"""Test configuration builders."""

import os
from typing import Dict

from src.ensemble_trainer import (
    ConceptConfig,
    DataConfig,
    EnsembleConfig,
    LLMConfig,
    ModelConfig,
    TrainingConfig,
)


def create_test_prompt_files(temp_dir: str) -> Dict[str, str]:
    """Create test configuration files for prompts."""

    # Create concept extraction prompt
    concept_prompt = """
Please analyze the following medical note and answer these questions:

{prompt_questions}

For each question, provide a numerical answer (0 for No, 1 for Yes) and brief reasoning.

Format your response as JSON:
{
    "extractions": [
        {
            "question": 1,
            "answer": 0,
            "reasoning": "Brief explanation"
        }
    ]
}

Medical note: {sentence}
"""

    # Create baseline initialization prompt
    baseline_prompt = """
Based on the top predictive features from logistic regression analysis, generate {num_meta_concepts} medical concepts for patient classification:

Top Features:
{top_features_df}

Generate concepts as binary questions (answerable with 0/1) that could help classify patients.

Format as JSON:
{
    "concepts": [
        {
            "concept": "Does the patient have chest pain?",
            "reasoning": "Chest pain is a key diagnostic indicator"
        }
    ]
}
"""

    # Create iterative improvement prompt
    iter_prompt = """
Given the current model performance and concept set, suggest improvements:

Current concepts: {current_concepts}
Model performance: {model_performance}
Residual analysis: {residual_analysis}

Suggest {num_new_concepts} new or improved concepts for better classification.

Format as JSON:
{
    "concepts": [
        {
            "concept": "Does the patient have a specific condition?",
            "reasoning": "Explanation for this concept"
        }
    ]
}
"""

    # Create prior concept generation prompt
    prior_prompt = """
Generate {num_meta_concepts} medical concepts for patient classification based on clinical knowledge:

Focus on concepts that are:
1. Clinically relevant
2. Answerable from medical notes
3. Predictive of patient outcomes

Format as JSON:
{
    "concepts": [
        {
            "concept": "Does the patient have cardiovascular risk factors?",
            "reasoning": "Cardiovascular risk factors are key predictors"
        }
    ]
}
"""

    # Write all prompt files
    files = {
        "concept_prompt_file": os.path.join(temp_dir, "concept_prompt.txt"),
        "baseline_prompt_file": os.path.join(temp_dir, "baseline_prompt.txt"),
        "iter_prompt_file": os.path.join(temp_dir, "iter_prompt.txt"),
        "prior_prompt_file": os.path.join(temp_dir, "prior_prompt.txt"),
    }

    with open(files["concept_prompt_file"], "w") as f:
        f.write(concept_prompt)

    with open(files["baseline_prompt_file"], "w") as f:
        f.write(baseline_prompt)

    with open(files["iter_prompt_file"], "w") as f:
        f.write(iter_prompt)

    with open(files["prior_prompt_file"], "w") as f:
        f.write(prior_prompt)

    return files


def create_minimal_config(temp_dir: str, **overrides) -> EnsembleConfig:
    """Create minimal configuration for fast unit tests."""
    prompt_files = create_test_prompt_files(temp_dir)

    model_config = ModelConfig(
        model=overrides.get("model", "l2"),
        final_model_type="l2",
        learner_type="count_l2",
        count_vectorizer="count",
        use_acc=False,
        final_learner_type="l2",
    )

    llm_config = LLMConfig(
        llm_model="gpt-4o-mini",
        max_tokens=600,
        max_new_tokens=5000,
        cache_file=os.path.join(temp_dir, "test_cache.db"),
    )

    data_config = DataConfig(
        text_summary_column="sentence",
        concept_column="sentence",
        min_prevalence=0,
        is_image=False,
        max_section_length=None,
    )

    training_config = TrainingConfig(
        num_epochs=overrides.get("num_epochs", 1),
        num_greedy_epochs=1,
        batch_size=8,
        batch_concept_size=10,
        batch_obs_size=1,
        train_frac=0.8,
        num_greedy_holdout=overrides.get("num_greedy_holdout", 1),
        sampling_method=overrides.get("sampling_method", "bootstrap"),
    )

    concept_config = ConceptConfig(
        goal_num_meta_concepts=overrides.get("num_meta_concepts", 2),
        baseline_init_file=prompt_files["baseline_prompt_file"],
        prompt_concepts_file=prompt_files["concept_prompt_file"],
        prompt_iter_file=prompt_files["iter_prompt_file"],
        prompt_prior_file=prompt_files["prior_prompt_file"],
    )

    return EnsembleConfig(
        init_seeds=overrides.get("init_seeds", [1]),
        model=model_config,
        llm=llm_config,
        data=data_config,
        training=training_config,
        concept=concept_config,
    )


def create_small_config(temp_dir: str, **overrides) -> EnsembleConfig:
    """Create small configuration for integration tests."""
    defaults = {
        "init_seeds": [1, 2],
        "num_epochs": 1,
        "num_meta_concepts": 3,
        "num_greedy_holdout": 1,
    }
    defaults.update(overrides)
    return create_minimal_config(temp_dir, **defaults)


def create_standard_config(temp_dir: str, **overrides) -> EnsembleConfig:
    """Create standard configuration for integration tests (similar to original test)."""
    prompt_files = create_test_prompt_files(temp_dir)

    model_config = ModelConfig(
        model="l2",
        final_model_type="l2",
        learner_type="count_l2",
        count_vectorizer="count",
        use_acc=False,
        final_learner_type="l2",
    )

    llm_config = LLMConfig(
        llm_model="gpt-4o-mini",
        max_tokens=600,
        max_new_tokens=5000,
        cache_file=os.path.join(temp_dir, "test_cache.db"),
    )

    data_config = DataConfig(
        text_summary_column="sentence",
        concept_column="sentence",
        min_prevalence=0,
        is_image=False,
        max_section_length=None,
    )

    training_config = TrainingConfig(
        num_epochs=overrides.get("num_epochs", 2),
        num_greedy_epochs=1,
        batch_size=8,
        batch_concept_size=10,
        batch_obs_size=1,
        train_frac=0.8,
        num_greedy_holdout=overrides.get("num_greedy_holdout", 1),
        sampling_method=overrides.get("sampling_method", "bootstrap"),
    )

    concept_config = ConceptConfig(
        goal_num_meta_concepts=overrides.get("num_meta_concepts", 4),
        baseline_init_file=prompt_files["baseline_prompt_file"],
        prompt_concepts_file=prompt_files["concept_prompt_file"],
        prompt_iter_file=prompt_files["iter_prompt_file"],
        prompt_prior_file=prompt_files["prior_prompt_file"],
    )

    return EnsembleConfig(
        init_seeds=overrides.get("init_seeds", [1, 2, 3]),
        model=model_config,
        llm=llm_config,
        data=data_config,
        training=training_config,
        concept=concept_config,
    )
