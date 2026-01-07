# Ensemble Trainer Module

A sophisticated ensemble training framework for concept-based models that coordinates training across multiple initializations while sharing expensive LLM operations.

## Overview

The Ensemble Trainer module provides:
- **Parallel training** across multiple initialization seeds
- **Shared LLM operations** to reduce API costs and improve efficiency
- **Checkpoint/resume functionality** for fault tolerance
- **Coordinated concept evolution** with greedy selection

## Architecture

```
ensemble_trainer/
├── __init__.py                 # Module exports
├── trainer.py                  # Main EnsembleTrainer class (orchestration)
├── config.py                   # Configuration classes (EnsembleConfig, TrainingPhase)
├── state.py                    # State management (TrainerState)
├── managers.py                 # Concept tracking and annotation management
├── data_operations.py          # Data splitting and extraction mapping
├── concept_evolution.py        # Candidate generation and semantic search
├── checkpoint_manager.py       # Checkpoint save/load/resume
├── greedy_concept_selector.py  # Streamlined greedy concept selection
└── semantic_search.py          # Cached semantic similarity search
```

## Key Components

### EnsembleTrainer (trainer.py)
The main orchestrator that manages the complete training pipeline:
- Uses modular `EnsembleConfig` for all configuration parameters
- Provides `fit()`, `predict()`, and checkpoint methods
- Coordinates baseline and greedy training phases across multiple initializations
- Manages state with `TrainerState` for resume functionality
- Handles parallel processing with asyncio
- Supports including evidence spans for concept generation

### Data Operations (data_operations.py)
Static utilities for data manipulation:
- `create_data_split()`: Bootstrap or data split sampling
- `create_train_test_split()`: Train/test partitioning
- `map_extractions_to_bootstrap()`: Maps features to bootstrap samples

### Extraction Utils (extraction_utils.py)
Unified utilities for concept extraction handling:
- `ExtractionHandler.prepare_concept_features()`: Streamlined extraction preparation with optional mapping
- Handles both full dataset and bootstrap sample scenarios
- Provides consistent interface for feature extraction across the codebase

### Concept Evolution (concept_evolution.py)
Manages concept generation with (optional) evidence span inclusion:
- `generate_candidate_concepts()`: Creates new concept candidates with option for adding additional contextual information
- `perform_greedy_selection()`: Selects best concepts using greedy algorithm
- `SemanticCacheManager`: Manager for semantic search functionality to use when there aren't exact matches for a keyword in the evidence string
- `load_evidence_mappings()`: Loads evidence mappings for enhanced context

### Checkpoint Manager (checkpoint_manager.py)
Handles state persistence and recovery:
- `initialize_or_resume_state()`: Start new or resume from checkpoint
- `save_checkpoint()`: Save current training state
- `restore_from_checkpoint()`: Restore histories and state
- `validate_checkpoint_compatibility()`: Ensure checkpoint matches config

### Greedy Concept Selector (greedy_concept_selector.py)
Streamlined concept selection (replaces ConceptLearnerModel):
- `GreedyConceptSelector`: Focused greedy selection algorithm for ensemble training
- Removes Bayesian sampling
- Maintains compatibility with existing training pipelines

### Semantic Search (semantic_search.py)
Cached semantic similarity search for enhanced concept context:
- `CachedSemanticSearch`: Pre-computed embeddings for efficient semantic search
- Thread-safe lazy initialization for parallel training environments
- Finds conceptually similar content when exact keyword matches fail

## Configuration System

The module uses a modular configuration approach with separate config classes:

```python
from src.ensemble_trainer import (
    EnsembleConfig, ModelConfig, LLMConfig, DataConfig, 
    TrainingConfig, ConceptConfig, DualTrackConfig, ConfigBuilder
)

# Create configuration from command-line arguments
config = ConfigBuilder.from_args(args)

# Or build manually with modular configs
config = EnsembleConfig(
    init_seeds=[1, 2, 3],
    sampling_method="data_split",
    model=ModelConfig(residual_model_type="l2", use_acc=False),
    llm=LLMConfig(llm_model="gpt-4o-mini", cache_file="cache.db"),
    data=DataConfig(text_summary_column="llm_output", is_image=False),
    training=TrainingConfig(num_epochs=3, batch_size=20),
    concept=ConceptConfig(num_meta_concepts=10, baseline_init_file="prompts/baseline.txt"),
    dual_track=DualTrackConfig(concept_generation_mode="standard"),
)
```

## Usage Example

```python
from src.ensemble_trainer import EnsembleTrainer, EnsembleConfig

# Create ensemble trainer with new API
trainer = EnsembleTrainer(config=config, output_dir="output/ensemble")

# Train ensemble
histories = await trainer.fit(data_df=train_data, plot_aucs=True)

# Continue training from checkpoint
trainer = EnsembleTrainer.from_checkpoint("checkpoint.pkl")
histories = await trainer.continue_training(train_data, num_additional_epochs=2)

# Make predictions with averaged probabilities or by each init
predictions = trainer.predict(test_data)
predictions_by_init = trainer.predict_all(test_data)
```

## API Methods

### Training Methods
- `fit(data_df, plot_aucs=True)`: Train ensemble from scratch
- `continue_training(data_df, num_additional_epochs=None, plot_aucs=True)`: Continue from current state
- `from_checkpoint(checkpoint_path, config=None, output_dir=None)`: Create trainer from checkpoint

### Prediction Methods  
- `predict(data_df, use_posterior_iters=None)`: Get ensemble predictions (averaged across initializations)
- `predict_all(data_df, use_posterior_iters=None)`: Get predictions from each initialization separately

### State Management
- `save_checkpoint(checkpoint_path=None)`: Save current state
- `get_training_histories()`: Get training histories for all initializations

## Parallelization Strategy

1. **Baseline Training**: All initialization seeds generate concepts in parallel
2. **Candidate Generation**: Parallel generation across all seeds using ThreadPoolExecutor
3. **Greedy Selection**: Parallel selection using asyncio.gather()
4. **Feature Extraction**: Shared across all initializations

All parallel operations use `asyncio.gather(return_exceptions=False)` to raise errors immediately.

## Extraction Mapping Logic

The module handles two types of data scenarios that require different extraction handling:

### When Extraction Mapping is Needed

1. **Bootstrap/Split Samples**: When working with samples that have different indices than the original dataset
   - Examples: `generate_candidate_concepts()`, baseline training finalization
   - Shared extractions are computed on the full dataset but need to be used on bootstrap samples
   - Bootstrap samples can have duplicate rows or different ordering
   - Use `needs_mapping=True` in `ExtractionHandler.prepare_concept_features()`

2. **Full Dataset**: When working directly with the original dataset
   - Examples: `perform_greedy_selection()`, final model evaluation
   - No index translation needed since extractions match dataset indices
   - Use `needs_mapping=False` in `ExtractionHandler.prepare_concept_features()`

### Unified Interface

The `ExtractionHandler.prepare_concept_features()` method provides a consistent interface:

```python
features, concepts, extractions = ExtractionHandler.prepare_concept_features(
    concepts=current_concepts,
    shared_extractions=shared_extractions,
    data_df=target_dataset,
    num_holdout=num_greedy_holdout,
    needs_mapping=True/False,  # Depends on dataset type
    force_keep_columns=force_keep_columns
)
```

This eliminates code duplication and ensures consistent extraction handling across the codebase.


## Checkpoint System

The checkpoint system uses `TrainerState` for fault tolerance:

### Checkpoint Contents
- Complete `EnsembleConfig` and output directory paths
- Training phase and progress (current epoch, concept iteration)
- Per-initialization completion status (baseline_complete, greedy_complete)
- Concept tracker state and shared extractions cache
- Training history file paths for all initializations
- Timing information and error recovery metadata

### Resume Behavior
- Automatically detects existing checkpoints via `from_checkpoint()`
- Validates compatibility with current configuration
- Resumes from exact point of interruption (mid-epoch, mid-concept-iteration)
- Preserves all shared extractions to avoid expensive re-computation
- Restores complete training state including histories and models

### Interruption Handling
- Graceful shutdown on SIGINT/SIGTERM with signal handlers
- Automatically saves checkpoint before exiting
- Atomic checkpoint saves to prevent corruption

## Configuration

### Required Files
- `baseline_init_file`: Prompt for initial concept generation
- `prompt_concepts_file`: Prompt for feature extraction
- `prompt_iter_file`: Prompt for iterative improvement
- `prompt_prior_file`: Prompt for prior concept generation

### Key Parameters (via EnsembleConfig)
- `init_seeds`: List of initialization seeds to train
- `num_meta_concepts`: Number of concepts to maintain  
- `num_epochs`: Number of greedy training epochs
- `batch_size`: Batch size for LLM operations
- `num_greedy_holdout`: Concepts to replace per iteration
- `sampling_method`: "data_split" or "bootstrap" for initialization diversity
- `concept_generation_mode`: "standard" or "dual_track" for enhanced generation

**Note**: `BaselineConfig` is kept for temporary backward compatibility but will be removed soon.


