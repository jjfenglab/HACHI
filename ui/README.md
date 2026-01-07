# Standalone HTML Viewer for Clinical Data

This directory contains tools to create a self-contained HTML viewer for clinical data that can be shared with collaborators without requiring any server setup or technical expertise.

## Overview

The standalone viewer allows clinicians to:
- View clinical notes and hospital course summaries
- See concepts assigned by multiple model initializations
- Filter and search through observations
- Select and annotate text passages
- Export their annotations and filtered data

All processing happens locally in the browser - no data is sent to any server.

## Creating a Standalone Package

### Step 1: Export Data

First, export your data from the existing system into a format the standalone viewer can read:

```bash
python export_standalone.py \
  --llm-summaries /path/to/llm_summaries.csv \
  --initializations init_seed_1:/path/to/init_seed_1/,init_seed_2:/path/to/init_seed_2/ \
  --config config.json \
  --train-test-split /path/to/train_test_indices.csv \
  --output-dir ./export/

```




**Parameters:**
- `--llm-summaries`: Path to the CSV file containing notes and LLM-generated summaries
- `--initializations`: Comma-separated list of initialization names and paths (format: `name:path`)
- `--config`: Path to your original config.json file
- `--train-test-split`: (Optional) Path to CSV with train/test partition information
- `--output-dir`: Directory where exported files will be saved

This creates:
- `export/data.csv`: Combined data with notes, metadata, summaries, and concepts from all initializations
- `export/config.json`: Configuration file for the standalone viewer (enhanced with method information)

### Step 2: Build Standalone HTML

Next, create the self-contained HTML file:

```bash
python build_standalone.py --output ./export/standalone.html
```

This combines the HTML template and JavaScript into a single file.

### Step 3: Share with Collaborators

Share these three files with your collaborators:
1. `standalone.html` - The viewer application
2. `data.csv` - The clinical data
3. `config.json` - Configuration settings

## Configuration Requirements

For the Overview tab to work properly, your original `config.json` must include:

### Required for Method Overview

1. **Prompts Directory**: The export script will automatically load prompt files
```json
{
  "prompts": {
    "prompts_directory": "/path/to/your/prompts/",
    "summary_prompts": ["hospital_course_summary.txt"],
    "extraction_prompts": ["sepsis_freeform_extract_question.txt"]
  }
}
```

2. **Training History Files**: Each initialization directory should contain:
   - `training_history.pkl` or `baseline_history.pkl` - Contains final concepts with coefficients
   - `extraction.pkl` or `extractions.pkl` - Concept feature extractions (required for full analysis)
   - `concepts.csv` - List of concept names

3. **Train/Test Split**: For full concept analysis, provide:
   - `--train-test-split` parameter pointing to CSV with `idx` and `partition` columns
   - This enables coefficient computation on training data only

### Directory Structure Expected

```
your_experiment/
├── init_seed_1/
│   ├── training_history.pkl    # Final model with concepts and coefficients
│   ├── extraction.pkl          # Concept extractions
│   └── concepts.csv           # Concept names
├── init_seed_2/
│   ├── training_history.pkl
│   ├── extraction.pkl
│   └── concepts.csv
└── prompts/
    ├── sepsis_freeform_extract_question.txt
    ├── hospital_course_summary.txt
    └── bayesian_iter.txt
```

### Enhanced Config Output

The export script automatically creates an enhanced config with:

```json
{
  "method_info": {
    "prompts": {
      "summary": [{"filename": "hospital_course_summary.txt", "content": "..."}],
      "extraction": [{"filename": "sepsis_freeform_extract_question.txt", "content": "..."}],
      "generation": [{"filename": "bayesian_iter.txt", "content": "..."}]
    },
    "training_histories": {
      "init_seed_1": {
        "final_auc": 0.823,
        "concepts": [
          {"text": "Does the patient have sepsis?", "coefficient": 0.75},
          {"text": "Does the patient require vasopressors?", "coefficient": -0.42}
        ],
        "num_iterations": 5,
        "auc_history": [0.65, 0.72, 0.78, 0.81, 0.823]
      }
    }
  }
}
```

## Using the Standalone Viewer

### For Clinicians

1. **Open the Viewer**: Double-click `standalone.html` or open it in a web browser
2. **Load Configuration**: Click "Choose File" under Step 1 and select `config.json`
3. **Load Data**: Click "Choose File" under Step 2 and select `data.csv`
4. **Use the Interface**:
   - Browse observations in the left panel
   - Click an observation to view its full note and summary
   - See assigned concepts from each initialization in the right panel
   - Select text to create annotations
   - Export your work using the export buttons

### Features

#### Encounters Tab
- **Search**: Use the search box to find observations by diagnosis code or note content
- **Filters**: Filter by train/test partition or by specific concepts
- **Multiple Initializations**: View concepts from different model runs side by side
- **Text Selection**: Select important passages from notes or summaries
- **Export Options**:
  - Export filtered observation list as CSV
  - Export individual encounter details as JSON
  - Export all annotations with or without full note content

#### Method Overview Tab
- **Training Process Diagram**: Visual flow showing the 4-step CBM training process
- **Performance Summary**: AUC comparison between CBM models (5 concepts) and full models (all concepts)
- **Concept Clustering**: Hierarchical dendrograms showing concept relationships within each model
- **Full Concept Analysis**: All concepts ranked by predictive power using L2-regularized logistic regression
- **Prompts Display**: View the exact prompts used for summary generation, concept extraction, and concept generation

##### Full Concept Analysis Features
- **L2-Regularized Coefficients**: Uses the same methodology as `train_LR` in `src/common.py` for consistency
- **Predictive Ranking**: All concepts sorted by absolute coefficient value showing true predictive power
- **Model Performance Comparison**: Compare CBM performance (5 concepts) vs. full model performance (all concepts)
- **Training Set Analysis**: Coefficients computed on training data only to avoid overfitting
- **Color-Coded Display**: Green for positive coefficients, red for negative, with bold text for high-impact (|coef| > 0.5)
- **Multi-Sort Options**: Sort by predictive power, coefficient value, or alphabetically

##### Technical Methodology
The full concept analysis fits L2-regularized logistic regression models using all concepts that were generated during training:
- **Data Source**: Uses same feature extraction pipeline as evaluation scripts (`evaluate_bayesian.py`)
- **Model Training**: `LogisticRegressionCV` with cross-validation for optimal regularization parameter
- **Training Partition**: Analysis performed only on training data to maintain proper evaluation practices
- **Coefficient Interpretation**: Shows which concepts are most predictive of the outcome, validating CBM's concept selection

## Data Privacy

- All data processing happens locally in your web browser
- No data is sent to any external server
- The HTML file can be used completely offline
- Annotations are stored only in browser memory (not saved between sessions)

## Browser Compatibility

The standalone viewer works best in modern browsers:
- Chrome (recommended)
- Firefox
- Safari
- Edge

## Troubleshooting

**Large Files**: For CSV files over 100MB, loading may take a few moments. The interface will show a loading indicator.

**Memory Issues**: Very large datasets (>10,000 observations) may cause performance issues. Consider splitting the data or filtering before export.

**Concepts Not Showing**: Ensure the initialization paths in the export command point to directories containing both `extraction.pkl` (or `extractions.pkl`) and `concepts.csv` files.

**Overview Tab Issues**:
- **No Method Overview**: Check that your config has a `prompts` section with `prompts_directory` pointing to valid prompt files
- **Missing Concept Coefficients**: Ensure each initialization directory contains `training_history.pkl` or `baseline_history.pkl` 
- **Empty Prompts Section**: Verify the prompts directory exists and contains `.txt` files
- **No AUC Scores**: Training history files must be from completed training runs with valid AUC scores
- **Clustering Errors**: Dendrograms require at least 2 concepts per model and valid coefficient data
- **"Coefficient analysis not available"**: This appears when:
  - Missing `extraction.pkl` or `extractions.pkl` files in initialization directories
  - No train/test split provided (required for training partition filtering)
  - Insufficient training data or label diversity
  - Common module import errors (ensure `src/common.py` is accessible)

## Example Workflow

```bash
# 1. Export data for a specific experiment
python export_standalone.py \
  --llm-summaries exp_los/_output/seed_0/llm_summaries.csv \
  --initializations init_1:exp_los/_output/seed_0/init_seed_1/,init_2:exp_los/_output/seed_0/init_seed_2/ \
  --config config.json \
  --output-dir ./clinician_review/

# 2. Build the HTML viewer
python build_standalone.py --output ./clinician_review/viewer.html

# 3. The clinician_review/ folder now contains everything needed:
# - viewer.html
# - data.csv  
# - config.json

# 4. Zip and share the folder
zip -r clinician_review.zip clinician_review/
```

## Technical Details

The standalone viewer uses:
- **Papa Parse** for CSV parsing
- **Bootstrap 5** for UI components
- **Local Storage** for temporary annotation storage
- **File API** for loading local files

All dependencies are loaded from CDN, so an internet connection is required for the initial page load, but not for using the application with loaded data.