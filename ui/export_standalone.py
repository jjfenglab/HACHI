"""
Export data for standalone HTML viewer.

This script combines data from multiple sources:
- LLM summaries CSV (contains notes and metadata)
- Concept extractions from multiple initializations
- Train/test split information

The output is a single CSV file that can be loaded by the standalone HTML viewer.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import scipy.stats
from matplotlib import pyplot as plt

# Add project paths for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent  # HACHI repo root (parent of ui/)
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))


def load_extractions(extraction_path: str) -> Dict[str, Dict[int, float]]:
    """Load extraction dictionary from pickle file."""
    with open(extraction_path, "rb") as f:
        return pickle.load(f)


def load_concepts(concepts_csv_path: str) -> List[str]:
    """Load concept names from CSV file."""
    df = pd.read_csv(concepts_csv_path)
    # Assuming the CSV has a 'concept' column
    if "concept" in df.columns:
        return df["concept"].tolist()
    # Otherwise return first column
    return df.iloc[:, 0].tolist()


def compute_model_specific_coefficients(
    init_path: str,
    train_data_df: pd.DataFrame,
    original_train_indices: List[int],
    use_concept_cards: bool = False,
) -> Optional[dict]:
    """Compute coefficients for this model's concepts only using L2-regularized logistic regression."""
    try:
        sys.path.append(str(project_root))
        sys.path.append(str(project_root / "src"))
        import src.common as common
        from sklearn.metrics import roc_auc_score
        from training_history import TrainingHistory

        # Load training history
        possible_files = ["training_history.pkl", "baseline_history.pkl", "history.pkl"]

        history = None
        for filename in possible_files:
            history_path = os.path.join(init_path, filename)
            if os.path.exists(history_path):
                history = TrainingHistory().load(history_path)
                break

        if history is None or history.num_iters == 0:
            print(f"Warning: No valid training history found in {init_path}")
            return None

        # Extract concepts from THIS model's training history only
        all_concepts_with_dupes = [
            concept_dict
            for concept_dicts in history._concepts
            for concept_dict in concept_dicts
        ]

        # Deduplicate concepts within this model only (not across models)
        seen_concepts = set()
        all_concepts = []
        for concept_dict in all_concepts_with_dupes:
            concept_text = concept_dict["concept"]
            if concept_text not in seen_concepts:
                seen_concepts.add(concept_text)
                all_concepts.append(concept_dict)

        print(
            f"  Model-specific analysis: Found {len(all_concepts_with_dupes)} total concepts, {len(all_concepts)} unique concepts"
        )

        if len(all_concepts) == 0:
            print(f"Warning: No concepts found in training history at {init_path}")
            return None

        # Load existing extractions
        extraction_path = os.path.join(init_path, "extraction.pkl")
        if not os.path.exists(extraction_path):
            extraction_path = os.path.join(init_path, "extractions.pkl")

        if not os.path.exists(extraction_path):
            print(
                f"Warning: No extraction file found for coefficient analysis at {init_path}"
            )
            return None

        with open(extraction_path, "rb") as f:
            all_extracted_features_dict = pickle.load(f)

        # Filter extractions to only include the original training indices
        # The extraction dictionary uses original dataset indices
        filtered_extractions = {}
        for concept_name, concept_extractions in all_extracted_features_dict.items():
            # Extract only the training indices from the original dataset
            if isinstance(concept_extractions, np.ndarray):
                filtered_extractions[concept_name] = concept_extractions[
                    original_train_indices
                ]
            else:
                print(
                    f"Warning: Unexpected extraction format for concept '{concept_name}'"
                )
                continue

        # Now create feature matrix using the filtered extractions
        # Manually build feature matrix to ensure proper alignment
        concept_features = []
        for concept_dict in all_concepts:
            concept_name = concept_dict["concept"]
            if concept_name in filtered_extractions:
                features = filtered_extractions[concept_name]
                if features.ndim == 1:
                    features = features.reshape(-1, 1)
                concept_features.append(features)
            else:
                print(f"Warning: Missing extractions for concept '{concept_name}'")
                return None

        if not concept_features:
            print(f"Warning: No concept features found for {init_path}")
            return None

        feature_matrix = np.concatenate(concept_features, axis=1)

        if feature_matrix.shape[0] == 0 or feature_matrix.shape[1] == 0:
            print(
                f"Warning: Empty feature matrix for coefficient analysis at {init_path}"
            )
            return None

        # Fit L2-regularized model (matching train_LR exactly)
        y_train = train_data_df["y"].values

        # Verify dimensions match
        if feature_matrix.shape[0] != len(y_train):
            print(
                f"Warning: Feature matrix shape {feature_matrix.shape} doesn't match "
                f"y_train length {len(y_train)} at {init_path}"
            )
            return None

        if len(np.unique(y_train)) < 2:
            print(
                f"Warning: Insufficient label diversity for coefficient analysis at {init_path}"
            )
            return None

        lr_results = common.train_LR(
            X_train=feature_matrix,
            y_train=y_train,
            penalty="l2",  # Use L2 to match train_LR
            use_acc=False,  # Use AUC
            seed=42,
        )

        # Extract coefficients for each concept
        concept_coefficients = []
        coefficients = (
            lr_results["coef"][0]
            if len(lr_results["coef"].shape) > 1
            else lr_results["coef"]
        )

        for i, concept_dict in enumerate(all_concepts):
            if i < len(coefficients):
                coef_value = float(coefficients[i])
                concept_text = concept_dict["concept"]

                if use_concept_cards:
                    # For concept cards, infer value type from the concept text
                    if ">" in concept_text or "≥" in concept_text:
                        if "h" in concept_text or "d" in concept_text:
                            value_type = "duration_days"
                        else:
                            value_type = "count"
                    else:
                        value_type = "binary"

                    concept_coefficients.append(
                        {
                            "text": concept_text,
                            "coefficient": coef_value,
                            "abs_coefficient": abs(coef_value),
                            "value_type": value_type,
                            "is_concept_card": True,
                        }
                    )
                else:
                    # Original logic for standard concepts
                    concept_coefficients.append(
                        {
                            "text": concept_text,
                            "coefficient": coef_value,
                            "abs_coefficient": abs(coef_value),
                            "is_concept_card": False,
                        }
                    )

        # Sort by absolute coefficient (predictive power)
        concept_coefficients.sort(key=lambda x: x["abs_coefficient"], reverse=True)

        # Add ranks
        for rank, concept_data in enumerate(concept_coefficients):
            concept_data["predictive_rank"] = rank + 1

        return {
            "model_specific_auc": float(lr_results["auc"]),
            "concepts": concept_coefficients,
            "num_model_concepts": len(all_concepts),
            "model_intercept": (
                float(lr_results["intercept"][0])
                if hasattr(lr_results["intercept"], "__len__")
                else float(lr_results["intercept"])
            ),
        }

    except Exception as e:
        print(
            f"Warning: Could not compute model-specific coefficients for {init_path}: {e}"
        )
        import traceback

        print(f"  Debug info: {traceback.format_exc()}")
        return None


def clean_concept_text(concept_text: str, use_concept_cards: bool = False) -> str:
    """Clean concept text exactly like make_clean_str in plot_exp_los.py"""
    if use_concept_cards:
        # Concept cards are already clean and actionable
        return concept_text

    # First apply clean_str logic
    concept = concept_text.replace(
        "Does the note mention that the patient", "Does the note mention the patient"
    )

    # Then apply make_clean_str logic
    concept = (
        concept.replace("Does the note mention ", "")
        .replace("that the patient ", "")
        .replace("the patient ", "")
    )
    concept = (
        concept.replace("engaging in ", "")
        .replace("experiencing ", "")
        .replace("having a ", "")
        .replace("having ", "")
        .replace("undergoing ", "")
    )
    concept = concept.replace("requiring ", "").replace("needing ", "")
    return concept.replace("?", "")


def correlation_distance(x, y):
    """Distance function matching plot_exp_los.py exactly"""
    try:
        import numpy as np
        import scipy.stats

        # Ensure inputs are finite
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            return 1.0  # Maximum distance for invalid inputs

        # Check for constant vectors (which would cause pearsonr to fail)
        if np.var(x) == 0 or np.var(y) == 0:
            return 1.0 if not np.array_equal(x, y) else 0.0

        corr, _ = scipy.stats.pearsonr(x, y)
        return 1 - np.abs(corr) if np.isfinite(corr) else 1.0
    except Exception:
        return 1.0  # Return maximum distance if correlation fails


def generate_dendrogram_image(
    init_path: str, model_name: str, output_dir: str, use_concept_cards: bool = False
) -> Optional[str]:
    """Generate dendrogram image using plot_exp_los.py logic"""
    try:
        from training_history import TrainingHistory

        # Create images directory
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Load training history
        possible_files = ["training_history.pkl", "baseline_history.pkl", "history.pkl"]
        history = None
        for filename in possible_files:
            history_path = os.path.join(init_path, filename)
            if os.path.exists(history_path):
                history = TrainingHistory().load(history_path)
                break

        if history is None or history.num_iters == 0:
            print(
                f"Warning: No valid training history found for dendrogram at {init_path}"
            )
            return None

        # Load extraction features
        extraction_path = os.path.join(init_path, "extraction.pkl")
        if not os.path.exists(extraction_path):
            extraction_path = os.path.join(init_path, "extractions.pkl")

        if not os.path.exists(extraction_path):
            print(f"Warning: No extraction file found for dendrogram at {init_path}")
            return None

        with open(extraction_path, "rb") as f:
            all_extracted_features = pickle.load(f)

        # Get concepts and their posterior probabilities (matching plot_exp_los.py exactly)
        num_posterior_iters = min(5, history.num_iters)
        start_iter = max(0, history.num_iters - num_posterior_iters)

        concepts_to_embed = [
            concept_dict["concept"]
            for concept_dicts in history._concepts[start_iter : history.num_iters]
            for concept_dict in concept_dicts
        ]
        concepts_to_embed = pd.Series(concepts_to_embed)
        concepts_to_embed_df = concepts_to_embed.value_counts() / (
            history.num_iters - start_iter
        )

        posterior_probs = concepts_to_embed_df.to_numpy()
        concepts_df = pd.DataFrame(
            {
                concepts_to_embed_df.index[c_idx]: [posterior_probs[c_idx]]
                for c_idx in np.arange(concepts_to_embed_df.index.size)
            }
        )

        # Create hierarchical clustering plot
        plt.figure(figsize=(12, 6))
        concepts_df_t = concepts_df.T

        # Filter out concepts that don't have valid embeddings (matching plot_exp_los.py)
        valid_concepts = []
        valid_embeddings = []
        valid_weights = []

        for i, concept in enumerate(concepts_df_t.index):
            if concept not in all_extracted_features:
                continue
            elif len(all_extracted_features[concept]) == 0:
                continue
            else:
                embedding = (
                    all_extracted_features[concept][:, 0]
                    if all_extracted_features[concept].ndim > 1
                    else all_extracted_features[concept]
                )
                # Check for finite values
                if not np.all(np.isfinite(embedding)):
                    continue
                elif np.all(embedding == 0):
                    continue
                else:
                    valid_concepts.append(concept)
                    valid_embeddings.append(embedding)
                    valid_weights.append(concepts_df_t.iloc[i, 0])

        if len(valid_embeddings) < 2:
            print(
                f"Warning: Only {len(valid_embeddings)} concepts have valid embeddings for {model_name}. Skipping dendrogram."
            )
            plt.text(
                0.5,
                0.5,
                f"Only {len(valid_embeddings)} valid concepts\nSkipping hierarchical clustering",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=12,
            )
            plt.title(
                f"Hierarchical Clustering of LOS Prediction Concepts - {model_name} (Insufficient Data)"
            )
        else:
            all_embeddings = np.array(valid_embeddings)

            # Ensure all embeddings have the same shape
            if all_embeddings.ndim == 1:
                all_embeddings = all_embeddings.reshape(-1, 1)

            Z = scipy.cluster.hierarchy.linkage(
                all_embeddings, "single", metric=correlation_distance
            )
            scipy.cluster.hierarchy.dendrogram(
                Z,
                labels=[
                    clean_concept_text(txt, use_concept_cards) + f" ({weight:.2f})"
                    for txt, weight in zip(valid_concepts, valid_weights)
                ],
                leaf_rotation=0,
                orientation="right",
            )
            plt.title(
                f"Hierarchical Clustering of LOS Prediction Concepts - {model_name}"
            )

        plt.tight_layout()

        # Save the image
        image_filename = f"dendrogram_{model_name.lower().replace(' ', '_')}.png"
        image_path = os.path.join(images_dir, image_filename)
        plt.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close()  # Important: close the figure to free memory

        # Return relative path from output directory
        return f"images/{image_filename}"

    except Exception as e:
        print(f"Warning: Could not generate dendrogram image for {model_name}: {e}")
        import traceback

        print(f"  Debug info: {traceback.format_exc()}")
        return None


def create_dendrogram_clustering(
    init_path: str,
    all_concepts: List[str],
    posterior_probs: dict,
    use_concept_cards: bool = False,
) -> Optional[dict]:
    """Create hierarchical clustering data exactly like plot_exp_los.py"""
    try:
        import pickle

        import numpy as np
        import scipy.cluster.hierarchy

        # Load extraction features
        extraction_path = os.path.join(init_path, "extraction.pkl")
        if not os.path.exists(extraction_path):
            extraction_path = os.path.join(init_path, "extractions.pkl")

        if not os.path.exists(extraction_path):
            print(f"Warning: No extraction file found for clustering at {init_path}")
            return None

        with open(extraction_path, "rb") as f:
            all_extracted_features = pickle.load(f)

        # Filter concepts that have valid embeddings (matching plot_exp_los.py logic)
        valid_concepts = []
        valid_embeddings = []
        valid_weights = []

        for concept in all_concepts:
            if concept not in all_extracted_features:
                continue
            elif len(all_extracted_features[concept]) == 0:
                continue
            else:
                embedding = (
                    all_extracted_features[concept][:, 0]
                    if all_extracted_features[concept].ndim > 1
                    else all_extracted_features[concept]
                )
                # Check for finite values
                if not np.all(np.isfinite(embedding)):
                    continue
                elif np.all(embedding == 0):
                    continue
                else:
                    valid_concepts.append(concept)
                    valid_embeddings.append(embedding)
                    valid_weights.append(posterior_probs.get(concept, 0.0))

        if len(valid_embeddings) < 2:
            print(
                f"Warning: Only {len(valid_embeddings)} concepts have valid embeddings. Skipping clustering."
            )
            # Return simple data for single/no concepts
            return {
                "concepts": [
                    {
                        "name": clean_concept_text(concept, use_concept_cards),
                        "full_name": concept,
                        "posterior_probability": posterior_probs.get(concept, 0.0),
                    }
                    for concept in valid_concepts
                ],
                "linkage": None,
                "has_clustering": False,
            }

        # Prepare embeddings array
        all_embeddings = np.array(valid_embeddings)
        if all_embeddings.ndim == 1:
            all_embeddings = all_embeddings.reshape(-1, 1)

        # Perform hierarchical clustering
        Z = scipy.cluster.hierarchy.linkage(
            all_embeddings, "single", metric=correlation_distance
        )

        # Prepare concept data
        concept_data = []
        for i, (concept, weight) in enumerate(zip(valid_concepts, valid_weights)):
            concept_data.append(
                {
                    "name": clean_concept_text(concept, use_concept_cards),
                    "full_name": concept,
                    "posterior_probability": weight,
                    "index": i,
                }
            )

        return {
            "concepts": concept_data,
            "linkage": Z.tolist(),  # Convert numpy array to list for JSON serialization
            "has_clustering": True,
        }

    except Exception as e:
        print(f"Warning: Could not create clustering data for {init_path}: {e}")
        return None


def load_training_history(
    init_path: str, use_concept_cards: bool = False
) -> Optional[dict]:
    """Load training history from pkl file."""
    try:
        from training_history import TrainingHistory

        # Try different common names for training history files
        possible_files = ["training_history.pkl", "baseline_history.pkl", "history.pkl"]

        for filename in possible_files:
            history_path = os.path.join(init_path, filename)
            if os.path.exists(history_path):
                history = TrainingHistory().load(history_path)

                # Extract final concepts and coefficients
                if history.num_iters > 0:
                    final_concepts = history.get_last_concepts()
                    final_auc = history.get_last_auc()
                    final_coefs = history._coefs[-1] if history._coefs else None

                    # Calculate posterior probabilities exactly like plot_exp_los.py
                    # Use more iterations to get meaningful probability variation
                    num_posterior_iters = min(
                        5, history.num_iters
                    )  # Look at last 5 iterations or all available
                    start_iter = max(0, history.num_iters - num_posterior_iters)

                    # Get ALL concepts from all iterations (matching plot_exp_los.py exactly)
                    concepts_to_embed = []
                    for iter_idx in range(start_iter, history.num_iters):
                        iter_concepts = history._concepts[iter_idx]
                        for concept_dict in iter_concepts:
                            concepts_to_embed.append(concept_dict["concept"])

                    # Calculate posterior probabilities using value_counts (exact match to original)
                    import pandas as pd

                    concepts_series = pd.Series(concepts_to_embed)
                    posterior_probs_series = concepts_series.value_counts() / (
                        history.num_iters - start_iter
                    )
                    posterior_probs = posterior_probs_series.to_dict()

                    # Get ALL unique concepts for dendrogram (not just final selected ones)
                    all_unique_concepts = list(posterior_probs.keys())

                    # Format final selected concepts with coefficients and posterior probabilities
                    concepts_with_coefs = []
                    for i, concept_dict in enumerate(final_concepts):
                        # Handle coefficient extraction safely
                        coefficient = 0.0
                        if final_coefs is not None and i < len(final_coefs):
                            coef_value = final_coefs[i]
                            # Handle numpy arrays, scalars, and lists
                            if (
                                hasattr(coef_value, "shape")
                                and len(coef_value.shape) > 0
                            ):
                                # It's a numpy array, take the first element
                                coefficient = float(coef_value.flat[0])
                            elif hasattr(coef_value, "__len__") and not isinstance(
                                coef_value, str
                            ):
                                # It's a list or similar, take first element
                                coefficient = (
                                    float(coef_value[0]) if len(coef_value) > 0 else 0.0
                                )
                            else:
                                # It's a scalar
                                coefficient = float(coef_value)

                        concept_text = concept_dict.get("concept", "")
                        concept_data = {
                            "text": concept_text,
                            "coefficient": coefficient,
                            "posterior_probability": posterior_probs.get(
                                concept_text, 0.0
                            ),
                        }
                        concepts_with_coefs.append(concept_data)

                    # Create clustering data for dendrogram
                    clustering_data = create_dendrogram_clustering(
                        init_path,
                        all_unique_concepts,
                        posterior_probs,
                        use_concept_cards,
                    )

                    return {
                        "final_auc": float(final_auc),
                        "concepts": concepts_with_coefs,
                        "dendrogram_data": clustering_data,
                        "num_iterations": history.num_iters,
                        "auc_history": [float(auc) for auc in history._aucs],
                    }

        print(f"Warning: No training history file found in {init_path}")
        return None

    except Exception as e:
        print(f"Warning: Could not load training history from {init_path}: {e}")
        # Add more debugging info
        import traceback

        print(f"  Debug info: {traceback.format_exc()}")
        return None


def load_prompt_files(original_config: dict) -> dict:
    """Load prompt file contents based on CBM training steps."""
    prompts_data = {}

    try:
        prompts_config = original_config.get("prompts", {})
        prompts_dir = Path(prompts_config.get("prompts_directory", "prompts"))

        if not prompts_dir.exists():
            print(f"Warning: Prompts directory not found: {prompts_dir}")
            return prompts_data

        # Load prompts organized by CBM training steps
        step_prompt_types = {
            "step1_summary": prompts_config.get("step1_summary", []),
            "step2_extraction": prompts_config.get("step2_extraction", []),
            "step3_initialization": prompts_config.get("step3_initialization", []),
            "step4_generation": prompts_config.get("step4_generation", []),
        }

        for step_type, prompt_files in step_prompt_types.items():
            for prompt_file in prompt_files:
                prompt_path = prompts_dir / prompt_file
                if prompt_path.exists():
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    if step_type not in prompts_data:
                        prompts_data[step_type] = []

                    prompts_data[step_type].append(
                        {"filename": prompt_file, "content": content}
                    )
                else:
                    print(f"Warning: Prompt file not found: {prompt_path}")

    except Exception as e:
        print(f"Warning: Could not load prompt files: {e}")

    return prompts_data


def process_initialization(
    init_name: str,
    init_path: str,
    data_df: pd.DataFrame,
    id_column: str,
    use_concept_cards: bool = False,
) -> Tuple[pd.DataFrame, List[str], Optional[dict]]:
    """
    Process a single initialization's data.

    Returns:
        Updated dataframe with concept columns
        List of concept names
        Training history data
    """
    # Load extraction data
    extraction_path = os.path.join(init_path, "extraction.pkl")
    if not os.path.exists(extraction_path):
        extraction_path = os.path.join(init_path, "extractions.pkl")

    if not os.path.exists(extraction_path):
        print(f"Warning: No extraction file found for {init_name} at {init_path}")
        return data_df, []

    extractions = load_extractions(extraction_path)

    # Load concept names
    concepts_path = os.path.join(init_path, "concepts.csv")
    if not os.path.exists(concepts_path):
        print(f"Warning: No concepts file found for {init_name} at {init_path}")
        return data_df, []

    concept_names = load_concepts(concepts_path)

    # Create concept columns for this initialization
    for i, concept_name in enumerate(concept_names):
        col_name = f"{init_name}_concept_{i}"
        data_df[col_name] = 0.0  # Initialize with zeros

        # Fill in extraction values
        if concept_name in extractions:
            concept_extractions = extractions[concept_name]
            # print(concept_extractions)

            # Handle both dictionary and numpy array cases
            if hasattr(concept_extractions, "items"):
                # Dictionary case
                for idx, value in concept_extractions.items():
                    if idx in data_df.index:
                        data_df.loc[idx, col_name] = value
            else:
                # Numpy array case
                for idx, value in enumerate(concept_extractions):
                    if idx in data_df.index:
                        data_df.loc[idx, col_name] = value

    # Create assigned concepts column (concepts with value > 0.5)
    assigned_col = f"assigned_concepts_{init_name}"
    concept_cols = [f"{init_name}_concept_{i}" for i in range(len(concept_names))]

    def get_assigned_concepts(row):
        assigned = []
        for i, col in enumerate(concept_cols):
            if col in row and row[col] > 0.5:
                assigned.append(concept_names[i])
        return ", ".join(assigned)

    data_df[assigned_col] = data_df.apply(get_assigned_concepts, axis=1)

    # Load training history
    training_history = load_training_history(init_path, use_concept_cards)

    return data_df, concept_names, training_history


def create_standalone_config(
    original_config: dict,
    initializations: List[Tuple[str, List[str], Optional[dict]]],
    prompts_data: dict,
    output_path: str,
    use_concept_cards: bool = False,
) -> None:
    """Create configuration file for standalone viewer."""

    # Build initialization info and method info
    init_info = []
    training_histories = {}

    for i, (init_name, concept_names, training_history) in enumerate(initializations):
        init_data = {
            "id": i + 1,
            "name": f"Model {i + 1}",
            "concept_prefix": init_name,
            "concepts": concept_names,
        }

        # Add training history if available
        if training_history:
            init_data.update(
                {
                    "final_auc": training_history["final_auc"],
                    "num_iterations": training_history["num_iterations"],
                    "auc_history": training_history["auc_history"],
                }
            )
            # Add dendrogram image path if available
            if "dendrogram_image" in training_history:
                init_data["dendrogram_image"] = training_history["dendrogram_image"]

            training_histories[init_name] = training_history

        init_info.append(init_data)

    standalone_config = {
        "ui": {
            "title": original_config.get("ui", {}).get(
                "title", "Clinical Data Analysis"
            ),
            "encounter_display_name": original_config.get("ui", {}).get(
                "encounter_display_name", "Encounter"
            ),
            "column_display_names": original_config.get("ui", {}).get(
                "column_display_names", {}
            ),
        },
        "dataset": {
            "id_column": original_config.get("dataset", {}).get(
                "id_column", "pat_enc_csn_id_surrogate"
            ),
            "note_column": "sentence",  # From llm_summaries.csv
            "summary_column": "llm_summary",
            "metadata_columns": [
                "pat_id_surrogate",
                "parent_diagnosis_code",
                "length_of_stay_days",
                "global_median_los",
                "y",
            ],
        },
        "initializations": init_info,
        "features": {
            "show_summaries": True,
            "show_concepts": len(init_info) > 0,
            "train_test_split": False,  # Will be set based on whether we have split data
            "use_concept_cards": use_concept_cards,
        },
        "method_info": {
            "prompts": prompts_data,
            "training_histories": training_histories,
        },
    }

    with open(output_path, "w") as f:
        json.dump(standalone_config, f, indent=2)


def export_example_summaries(summaries_csv_path: str, output_path: str) -> None:
    """
    Export example summaries from CSV to JSON format for standalone viewer.

    Args:
        summaries_csv_path: Path to CSV file with example summaries
        output_path: Path for output JSON file
    """
    df = pd.read_csv(summaries_csv_path)

    # Expected columns: id, note, summary
    # Optional columns: length_of_stay, outcome
    required_columns = ["pat_enc_csn_id_surrogate", "sentence", "llm_summary"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Error: Missing required columns in summaries CSV: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Convert to list of dictionaries
    summaries_data = []
    for _, row in df.iterrows():
        summary_dict = {
            "id": str(row["pat_enc_csn_id_surrogate"]),
            # "note": str(row["sentence"]),
            "summary": str(row["llm_summary"]),
        }

        # Add optional fields if available
        if "length_of_stay" in df.columns:
            summary_dict["length_of_stay"] = row["length_of_stay"]
        if "outcome" in df.columns:
            summary_dict["outcome"] = str(row["outcome"])

        summaries_data.append(summary_dict)

    # Write to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries_data, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(summaries_data)} example summaries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export data for standalone HTML viewer"
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Main export command
    export_parser = subparsers.add_parser(
        "export", help="Export main data and config for standalone viewer"
    )
    export_parser.add_argument(
        "--llm-summaries", required=True, help="Path to LLM summaries CSV file"
    )
    export_parser.add_argument(
        "--initializations",
        required=True,
        help="Comma-separated list of init_name:path pairs (e.g., init_seed_1:/path/to/init1/,init_seed_2:/path/to/init2/)",
    )
    export_parser.add_argument(
        "--config",
        default="config.json",
        help="Path to original config file (default: config.json)",
    )
    export_parser.add_argument(
        "--train-test-split", help="Optional path to train/test split CSV"
    )
    export_parser.add_argument(
        "--output-dir", required=True, help="Output directory for exported files"
    )
    export_parser.add_argument(
        "--use-concept-cards",
        action="store_true",
        default=False,
        help="Specify if concept cards format is being used (includes thresholds and clinical intent)",
    )

    # Example summaries command
    summaries_parser = subparsers.add_parser(
        "summaries", help="Convert example summaries CSV to JSON"
    )
    summaries_parser.add_argument(
        "--input",
        required=True,
        help="Path to example summaries CSV file (columns: id, note, summary)",
    )
    summaries_parser.add_argument(
        "--output", required=True, help="Path for output JSON file"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "summaries":
        # Handle example summaries conversion
        export_example_summaries(args.input, args.output)
        return 0

    # Handle main export command
    if args.command == "export":
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load original config
        with open(args.config, "r") as f:
            original_config = json.load(f)

        # Load LLM summaries as base dataframe
        print(f"Loading LLM summaries from {args.llm_summaries}")
        data_df = pd.read_csv(args.llm_summaries)

        # Filter out observations with NA summaries (these were dropped during processing)
        original_count = len(data_df)
        data_df = data_df[data_df["llm_summary"].notna()].copy()
        filtered_count = len(data_df)
        if original_count != filtered_count:
            print(
                f"  Filtered out {original_count - filtered_count} observations with missing LLM summaries"
            )
        print(f"  Remaining observations: {filtered_count}")

    # Reset index to use row numbers
    data_df.reset_index(drop=True, inplace=True)

    # Get ID column from config
    id_column = original_config.get("dataset", {}).get(
        "id_column", "pat_enc_csn_id_surrogate"
    )

    # Parse initialization paths
    init_pairs = []
    for pair in args.initializations.split(","):
        if ":" in pair:
            init_name, init_path = pair.split(":", 1)
            init_pairs.append((init_name.strip(), init_path.strip()))

    # Process each initialization
    all_initializations = []
    for init_name, init_path in init_pairs:
        print(f"\nProcessing initialization: {init_name}")
        print(f"  Path: {init_path}")

        data_df, concept_names, training_history = process_initialization(
            init_name, init_path, data_df, id_column, args.use_concept_cards
        )

        if concept_names:
            # Generate dendrogram image for this model
            model_display_name = f"Model {len(all_initializations) + 1}"
            print(f"  Generating dendrogram image for {model_display_name}...")
            dendrogram_image_path = generate_dendrogram_image(
                init_path, model_display_name, args.output_dir, args.use_concept_cards
            )

            # Add dendrogram path to training history if generated successfully
            if training_history and dendrogram_image_path:
                training_history["dendrogram_image"] = dendrogram_image_path
                print(f"  Saved dendrogram: {dendrogram_image_path}")

            all_initializations.append((init_name, concept_names, training_history))
            print(f"  Added {len(concept_names)} concepts")
            if training_history:
                print(f"  Final AUC: {training_history['final_auc']:.3f}")
        else:
            print(f"  No concepts found")

    data_df["partition"] = "train"
    data_df.reset_index(drop=True, inplace=True)

    # Now get the training indices after reset (these will be in 0-based range)
    original_train_indices = data_df[data_df["partition"] == "train"].index.tolist()

    # Update config to indicate train/test split is available
    original_config["features"] = original_config.get("features", {})
    original_config["features"]["train_test_split"] = True
    # # Add train/test split if provided and filter data
    # original_train_indices = None  # Initialize for later use
    # if args.train_test_split and os.path.exists(args.train_test_split):
    #     print(f"\nProcessing train/test split from {args.train_test_split}")
    #     split_df = pd.read_csv(args.train_test_split)

    #     # Merge on index
    #     if "idx" in split_df.columns and "partition" in split_df.columns:
    #         data_df["partition"] = "unknown"
    #         for idx, row in split_df.iterrows():
    #             if row["idx"] < len(data_df):
    #                 data_df.loc[row["idx"], "partition"] = row["partition"]

    #         # Filter to only include train and test observations
    #         original_length = len(data_df)
    #         data_df = data_df[data_df["partition"].isin(["train", "test"])].copy()
    #         filtered_length = len(data_df)

    #         print(
    #             f"  Filtered from {original_length} to {filtered_length} observations"
    #         )
    #         print(
    #             f"  Removed {original_length - filtered_length} observations without train/test labels"
    #         )

    #         # Reset index after filtering but save the mapping
    #         # The extraction arrays are aligned with the 495 filtered observations
    #         # So we need indices in the 0-494 range for training data
    #         data_df.reset_index(drop=True, inplace=True)

    #         # Now get the training indices after reset (these will be in 0-based range)
    #         original_train_indices = data_df[data_df["partition"] == "train"].index.tolist()

    #         # Update config to indicate train/test split is available
    #         original_config["features"] = original_config.get("features", {})
    #         original_config["features"]["train_test_split"] = True

    # else:
    #     print("\nWarning: No train/test split file provided or file not found")
    #     print(
    #         "All observations will be included, but viewer may not work properly without partition labels"
    #     )

    # Save exported data
    output_csv = os.path.join(args.output_dir, "data.csv")
    print(f"\nSaving exported data to {output_csv}")
    data_df.to_csv(output_csv, index=False)
    print(f"  Saved {len(data_df)} rows with {len(data_df.columns)} columns")

    # Compute full concept coefficients if train/test split is available
    print(f"\nComputing full concept coefficients...")
    for i, (init_name, concept_names, training_history) in enumerate(
        all_initializations
    ):
        if (
            training_history
            and "partition" in data_df.columns
            and original_train_indices is not None
        ):
            # Filter to training data only
            train_data_df = data_df[data_df["partition"] == "train"].copy()
            if len(train_data_df) > 0:
                # Get the corresponding init_path
                init_path = None
                for name, path in init_pairs:
                    if name == init_name:
                        init_path = path
                        break

                if init_path:
                    print(
                        f"  Computing coefficients for {init_name} with {len(train_data_df)} training samples..."
                    )
                    model_coefficients = compute_model_specific_coefficients(
                        init_path,
                        train_data_df,
                        original_train_indices,
                        args.use_concept_cards,
                    )
                    if model_coefficients:
                        # Update the training history with coefficient data
                        training_history["model_concept_analysis"] = model_coefficients
                        all_initializations[i] = (
                            init_name,
                            concept_names,
                            training_history,
                        )
                        print(
                            f"    Model-specific AUC: {model_coefficients['model_specific_auc']:.3f}"
                        )
                        print(
                            f"    Analyzed {model_coefficients['num_model_concepts']} concepts"
                        )
                    else:
                        print(f"    Could not compute coefficients for {init_name}")
                else:
                    print(f"    Could not find init_path for {init_name}")
            else:
                print(f"    No training data available for {init_name}")
        else:
            if not training_history:
                print(f"    No training history for {init_name}")
            else:
                print(
                    f"    No partition information available for coefficient analysis"
                )

    # Load prompt files
    print(f"\nLoading prompt files...")
    prompts_data = load_prompt_files(original_config)
    print(
        f"  Loaded {sum(len(prompts) for prompts in prompts_data.values())} prompt files"
    )

    # Create standalone config
    config_path = os.path.join(args.output_dir, "config.json")
    create_standalone_config(
        original_config,
        all_initializations,
        prompts_data,
        config_path,
        args.use_concept_cards,
    )
    print(f"\nCreated standalone config at {config_path}")

    # Print summary
    print("\n=== Export Summary ===")
    print(f"Total observations: {len(data_df)}")
    print(f"Initializations: {len(all_initializations)}")
    for init_name, concepts, training_history in all_initializations:
        auc_info = (
            f" (AUC: {training_history['final_auc']:.3f})" if training_history else ""
        )
        print(f"  {init_name}: {len(concepts)} concepts{auc_info}")
    print(f"\nExported files:")
    print(f"  - {output_csv}")
    print(f"  - {config_path}")
    print("\nNext step: Build standalone HTML viewer using build_standalone.py")


if __name__ == "__main__":
    main()
