<h1 align="center"> 🤖🤝🧑🏾 HACHI 🧑🏻🤝🤖 </h1>
<p align="center"> <b>Human-AI Co-design for Clinical Prediction Models</b>  (<a href="https://arxiv.org/abs/2601.09072">Feng et al. 2026, Under review</a>). 
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-green.svg">
  <img src="https://img.shields.io/badge/python-3.7+-blue">
</p>  

<img width="1017" height="307" alt="HACHI Method Overview" src="https://github.com/user-attachments/assets/9055db26-1c3f-4420-a37a-b1c08f3a7197" />

---

HACHI is a framework for developing interpretable clinical prediction models through iterative human-AI collaboration. It combines the reasoning capabilities of LLMs with clinical domain expertise to create models that are both accurate and interpretable.

---

## Quick Start

The fastest way to understand HACHI is through the [demo notebook](https://github.com/jjfenglab/HACHI/blob/main/notebooks/demo.ipynb):

The demo walks through:
- Generating synthetic clinical notes
- Running the HACHI agent loop
- Reviewing and interpreting the learned concepts
- Simulating human feedback rounds

*Note: The demo includes pre-computed outputs so it can run without API access.*

---

## Method Overview

HACHI implements a two-loop co-design process for building interpretable clinical prediction models:

### Outer Loop: Human Feedback
Clinical AI teams review model outputs and provide high-level guidance:
- Identify clinically irrelevant or redundant concepts
- Suggest domain-specific considerations
- Validate model interpretability for clinical use

### Inner Loop: AI Agent
An LLM-powered agent iteratively refines the concept space:
1. **Initialization**: Generate initial candidate concepts from clinical notes
2. **Proposal**: Create new concept candidates based on model performance
3. **Evaluation**: Extract concept features and evaluate predictive utility
4. **Selection**: Greedily select the best-performing concepts

### Concepts as Interpretable Features
HACHI represents clinical concepts as yes/no questions (e.g., "Does this patient have signs of acute kidney injury?"). These concepts are:
- Extracted from clinical notes using LLMs
- Combined into linear models for interpretability
- Iteratively refined based on predictive performance

---

## Installation

### Requirements
- Python 3.9+
- Access to an LLM API (OpenAI, Anthropic, or compatible)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jjfenglab/HACHI.git
cd HACHI
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure LLM API access:

Create a `.env` file in the project root with your API credentials:
```bash
# For OpenAI
OPENAI_API_KEY=your-api-key-here

# For Anthropic
ANTHROPIC_API_KEY=your-api-key-here
```

The framework uses the `llm-api` package for LLM interactions. See the [llm documentation](https://github.com/jjfenglab/llm-api) for additional configuration options.

---

## Usage Guide

### Running HACHI on Your Data

#### 1. Prepare Your Data
Create a CSV file with at minimum:
- A text column containing clinical notes
- A binary outcome column (0/1)

```python
import pandas as pd
data = pd.read_csv("your_data.csv")
# Required columns: 'note_text', 'outcome' (or configure via DataConfig)
```

#### 2. Configure the Training

```python
from src.ensemble_trainer import (
    EnsembleTrainer, EnsembleConfig, ConfigBuilder,
    ModelConfig, LLMConfig, DataConfig, TrainingConfig, ConceptConfig
)

config = EnsembleConfig(
    init_seeds=[1, 2, 3],  # Multiple seeds for ensemble diversity
    sampling_method="data_split",
    model=ModelConfig(residual_model_type="l2"),
    llm=LLMConfig(llm_model="gpt-4o-mini", cache_file="cache.db"),
    data=DataConfig(text_summary_column="note_text"),
    training=TrainingConfig(num_epochs=3, batch_size=20),
    concept=ConceptConfig(
        num_meta_concepts=10,
        baseline_init_file="prompts/baseline_init.txt",
        prompt_iter_file="prompts/bayesian_iter.txt"
    ),
)
```

#### 3. Train the Model

```python
trainer = EnsembleTrainer(config=config, output_dir="output/my_experiment")
histories = await trainer.fit(data_df=train_data, plot_aucs=True)
```

#### 4. Make Predictions

```python
# Ensemble predictions (averaged across initializations)
predictions = trainer.predict(test_data)

# Individual initialization predictions
predictions_by_init = trainer.predict_all(test_data)
```

### Customizing Prompts

HACHI uses prompt files to guide concept generation. Key prompts include:

| Prompt File | Purpose |
|-------------|---------|
| `baseline_init.txt` | Initial concept generation from notes |
| `bayesian_iter.txt` | Iterative concept refinement |
| `concept_questions.txt` | Feature extraction from notes |

See `exp_aki/prompts/` and `exp_tbi/prompts/` for examples.

To customize prompts:
1. Copy an existing prompt directory
2. Modify the prompt text while keeping the expected output format
3. Point your config to the new prompt files

### Using the Web UI

The `ui/` directory contains a standalone HTML viewer for reviewing HACHI outputs:

1. Export your results:
```bash
cd ui
python export_standalone.py --config your_config.json
python build_standalone.py
```

2. Open the generated HTML file in a browser

The UI allows you to:
- View clinical notes alongside LLM-generated summaries
- See which concepts were assigned to each observation
- Filter and search results
- Export annotations

See `ui/README.md` for detailed instructions.

### Using SCons Pipelines

For reproducible experiments, HACHI supports SCons-based pipelines. See `exp_aki/` and `exp_tbi/` for examples.

To run an experiment:
```bash
scons -f exp_aki/sconscript
```

The sconscripts demonstrate:
- Configuring nested parameter sweeps
- Managing multiple initialization seeds
- Organizing outputs systematically

---

## Repository Structure

```
HACHI/
├── src/                          # Core HACHI implementation
│   ├── ensemble_trainer/         # Main ensemble training module
│   │   ├── trainer.py           # EnsembleTrainer orchestrator
│   │   ├── config.py            # Configuration classes
│   │   ├── baseline_trainer.py  # Initial concept generation
│   │   ├── greedy_trainer.py    # Iterative concept refinement
│   │   └── ...                  # Supporting modules
│   ├── common.py                # Shared utilities
│   └── ...                      # Additional modules
│
├── scripts/                      # Command-line scripts
│   ├── train_ensemble.py        # Main training script
│   ├── predict_ensemble.py      # Prediction script
│   └── evaluate_ensemble.py     # Evaluation script
│
├── exp_aki/                      # AKI prediction experiment template
│   ├── sconscript               # SCons pipeline configuration
│   └── prompts/                 # Experiment-specific prompts
│
├── exp_tbi/                      # TBI prediction experiment template
│   ├── sconscript               # SCons pipeline configuration
│   └── prompts/                 # Experiment-specific prompts
│
├── ui/                           # Web UI for result review
│   ├── README.md                # UI documentation
│   ├── export_standalone.py     # Data export script
│   └── ...                      # UI components
│
├── tests/                        # Test suite
├── notebooks/                    # Demo notebooks
├── requirements.txt              # Python dependencies
└── SConstruct                    # SCons build entry point
```

---

## Citation

If you use HACHI in your research, please cite:

```bibtex
@misc{feng2026hachi,
      title={Human-AI Co-design for Clinical Prediction Models}, 
      author={Jean Feng and Avni Kothari and Patrick Vossler and Andrew Bishara and Lucas Zier and Newton Addo and Aaron Kornblith and Yan Shuo Tan and Chandan Singh},
      year={2026},
      eprint={2601.09072},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.09072}, 
}
```

---

## License

This project is licensed under the GLP V3 - see the [LICENSE](LICENSE) file for details.
