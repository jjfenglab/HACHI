"""
Ensemble trainer module for coordinated concept learning across multiple initializations.

This module provides classes for training concept-based models with shared feature extraction
and state management across multiple initialization seeds.
"""


from . import data_operations, selection_operations
from .checkpoint_manager import CheckpointManager
from .concept_generator import (
    BaseConceptGenerator,
    ConceptGeneratorFactory,
    StandardConceptGenerator,
)
from .concept_tracker import ConceptTracker
from .config import (
    ConceptConfig,
    ConfigBuilder,
    DataConfig,
    EnsembleConfig,
    LLMConfig,
    ModelConfig,
    TrainingConfig,
    TrainingPhase,
)
from .evidence_enhancement import SemanticCacheManager
from .feature_extraction_manager import FeatureExtractionManager
from .state import TrainerState
from .trainer import EnsembleTrainer

__all__ = [
    "ConfigBuilder",
    "ConceptConfig",
    "DataConfig",
    "EnsembleConfig",
    "LLMConfig",
    "ModelConfig",
    "TrainingConfig",
    "TrainingPhase",
    "ConceptTracker",
    "TrainerState",
    "EnsembleTrainer",
    "data_operations",
    "selection_operations",
    "SemanticCacheManager",
    "CheckpointManager",
    "BaseConceptGenerator",
    "ConceptGeneratorFactory",
    "StandardConceptGenerator",
    "FeatureExtractionManager",
]
