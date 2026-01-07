"""
Configuration classes and enums for ensemble training.

This module contains configuration classes and enums used throughout the ensemble training system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for model parameters."""

    model: str = "l2"
    learner_type: str = "count_l2"
    count_vectorizer: str = "count"
    residual_model_type: str = "l2"
    final_model_type: str = "l1"
    inverse_penalty_param: float = 20000.0
    use_acc: bool = False
    cv: int = 5
    final_learner_type: str = "l2"


@dataclass
class LLMConfig:
    """Configuration for LLM parameters."""

    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 600
    max_new_tokens: int = 5000
    cache_file: str = "cache.db"
    use_api: bool = True

    # Optional fields for different LLM types (for backward compatibility)
    llm_model_type: Optional[str] = None  # Will use llm_model if not specified
    llm_iter_type: Optional[str] = None
    llm_extraction_type: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data processing parameters."""

    text_summary_column: str
    concept_column: str
    min_prevalence: int = 0
    num_top_attributes: int = 40
    keep_x_cols: Optional[List[str]] = None
    is_image: bool = False
    max_section_length: Optional[int] = None
    join_column: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    num_epochs: int = 2
    num_greedy_epochs: int = 2
    batch_size: int = 20
    batch_concept_size: int = 20
    batch_obs_size: int = 1
    sampling_method: str = "data_split"
    train_frac: float = 0.5
    num_greedy_holdout: int = 1
    is_greedy_metric_acc: bool = False
    num_classes: int = 2
    cv: int = 5
    do_coef_check: bool = False


@dataclass
class ConceptConfig:
    """Configuration for concept generation parameters."""

    goal_num_meta_concepts: int = 10
    max_meta_concepts: int = 20
    num_top_residual_words: int = 40
    prompt_iter_file: str = ""
    prompt_concepts_file: str = ""
    prompt_prior_file: str = ""
    baseline_init_file: str = ""
    config_dict: Optional[Dict[str, str]] = None

@dataclass
class EnsembleConfig:
    """Main configuration container for ensemble training."""
    init_seeds: List[int]
    model: ModelConfig
    llm: LLMConfig
    data: DataConfig
    training: TrainingConfig
    concept: ConceptConfig
    force_keep_columns: Optional[List[str]] = None


class ConfigBuilder:
    """Factory for creating configuration objects from various sources."""

    @staticmethod
    def from_args(args) -> EnsembleConfig:
        """Create EnsembleConfig from command-line arguments."""

        model_config = ModelConfig(
            model=getattr(args, "model", "l2"),
            learner_type=getattr(args, "learner_type", "count_l2"),
            count_vectorizer=getattr(args, "count_vectorizer", "count"),
            residual_model_type=getattr(args, "residual_model_type", "l2"),
            final_model_type=getattr(args, "final_model_type", "l1_sklearn"),
            inverse_penalty_param=getattr(args, "inverse_penalty_param", 20000.0),
            use_acc=getattr(args, "use_acc", False),
            cv=getattr(args, "cv", 5),
            final_learner_type=getattr(args, "final_learner_type", "l2"),
        )

        llm_config = LLMConfig(
            llm_model=getattr(args, "llm_model", "gpt-4o-mini"),
            max_tokens=getattr(args, "max_tokens", 600),
            max_new_tokens=getattr(args, "max_new_tokens", 5000),
            cache_file=getattr(args, "cache_file", "cache.db"),
            use_api=getattr(args, "use_api", True),
            llm_model_type=getattr(args, "llm_model_type", None),
            llm_iter_type=getattr(args, "llm_iter_type", None),
            llm_extraction_type=getattr(args, "llm_extraction_type", None),
        )

        data_config = DataConfig(
            text_summary_column=getattr(args, "text_summary_column", "llm_output"),
            concept_column=getattr(args, "concept_column", "llm_output"),
            min_prevalence=getattr(args, "min_prevalence", 0),
            num_top_attributes=getattr(args, "num_top_attributes", 40),
            keep_x_cols=getattr(args, "keep_x_cols", None),
            is_image=getattr(args, "is_image", False),
            max_section_length=getattr(args, "max_section_length", None),
            join_column=getattr(args, "join_column", None),
        )

        training_config = TrainingConfig(
            num_epochs=getattr(args, "num_epochs", 2),
            num_greedy_epochs=getattr(args, "num_greedy_epochs", 2),
            batch_size=getattr(args, "batch_size", 20),
            batch_concept_size=getattr(args, "batch_concept_size", 20),
            batch_obs_size=getattr(args, "batch_obs_size", 1),
            train_frac=getattr(args, "train_frac", 0.5),
            num_greedy_holdout=getattr(args, "num_greedy_holdout", 1),
            do_coef_check=bool(getattr(args, "do_coef_check", False)),
            is_greedy_metric_acc=getattr(args, "is_greedy_metric_acc", False),
            cv=getattr(args, "cv", 5),
            num_classes=getattr(args, "num_classes", 2),
        )

        concept_config = ConceptConfig(
            goal_num_meta_concepts=getattr(args, "goal_num_meta_concepts", 6),
            max_meta_concepts=getattr(args, "max_meta_concepts", 20),
            num_top_residual_words=getattr(args, "num_top_residual_words", 40),
            prompt_iter_file=getattr(args, "prompt_iter_file", ""),
            prompt_concepts_file=getattr(args, "prompt_concepts_file", ""),
            prompt_prior_file=getattr(args, "prompt_prior_file", ""),
            baseline_init_file=getattr(args, "baseline_init_file", ""),
            config_dict=getattr(args, "config_dict", None),
        )

        return EnsembleConfig(
            init_seeds=getattr(args, "init_seeds", [1, 2, 3]),
            force_keep_columns=getattr(args, "force_keep_columns", None),
            model=model_config,
            llm=llm_config,
            data=data_config,
            training=training_config,
            concept=concept_config,
        )


class TrainingPhase(Enum):
    """Enumeration of training phases."""

    NOT_STARTED = "not_started"
    BASELINE_RUNNING = "baseline_running"
    BASELINE_COMPLETE = "baseline_complete"
    GREEDY_RUNNING = "greedy_running"
    GREEDY_COMPLETE = "greedy_complete"
    COMPLETE = "complete"
