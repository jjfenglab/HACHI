"""
Model-specific utilities for ensemble training.

This module contains model-related functions that are primarily used
within the ensemble trainer module, particularly for residual analysis
and prompt generation.
"""

import json
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import the shared utilities we still need from common
from src.common import TABULAR_PREFIX, train_LR


def fit_residual(
    model,
    word_names,
    X_extracted,
    X_words,
    y_train,
    sample_weight,
    penalty_downweight_factor: float,
    is_multiclass: bool,
    num_top: int,
    use_acc: bool = False,
    seed: int = None,
    force_keep_columns: list = None,
    downweight_all: bool = True,
    cv: int = 5
):
    """
    Fit residual model to identify top features not explained by current concepts.

    Args:
        model: Model type string (e.g., 'l1', 'l2')
        word_names: Feature names
        X_extracted: Extracted concept features
        X_words: Word count features
        y_train: Target labels
        penalty_downweight_factor: Factor to downweight existing concepts
        is_multiclass: Whether this is multiclass classification
        num_top: Number of top features to return
        use_acc: Whether to use accuracy metric
        seed: Random seed
        force_keep_columns: Columns to force keep
        downweight_all: Whether to downweight all existing features

    Returns:
        DataFrame with top residual features and their coefficients
    """
    if X_extracted is None:
        num_fixed = 0
        word_resid_X = X_words
    else:
        if not downweight_all and force_keep_columns is not None:
            # Separate control features from existing concepts
            num_control_features = len(force_keep_columns)
            control_features = X_extracted[:, :num_control_features]
            existing_concepts = X_extracted[:, num_control_features:]

            # Apply penalty only to existing concepts
            penalized_concepts = existing_concepts * penalty_downweight_factor

            # Combine penalized concepts with control features
            X_extracted_penalized = np.concatenate(
                [control_features, penalized_concepts], axis=1
            )

            num_fixed = X_extracted.shape[1]
            word_resid_X = np.concatenate([X_extracted_penalized, X_words], axis=1)
        else:
            num_fixed = X_extracted.shape[1]
            word_resid_X = np.concatenate(
                [X_extracted * penalty_downweight_factor, X_words], axis=1
            )
    results = train_LR(word_resid_X, y_train, sample_weight=sample_weight, penalty=model, use_acc=use_acc, seed=seed, cv=cv)
    logging.info(f"MODEL ACC AUC {results['acc']} {results['auc']}")
    logging.info(f"residual fit AUC: {results['auc']}")
    logging.info(f"residual fit ACC: {results['acc']}")
    logging.info(f"COEFS fixed {results['coef'][:, :num_fixed]}")
    logging.info(f"COEFS words {np.sort(results['coef'][:, num_fixed:])}")
    word_coefs = results["coef"][:, num_fixed:]

    # display only top features from the residual model
    if not is_multiclass:
        df = pd.DataFrame(
            {
                "feature_name": word_names,
                "freq": X_words.mean(axis=0),
                "coef": word_coefs[0],
                "abs_coef": np.abs(word_coefs[0]),
            }
        ).sort_values(["abs_coef"], ascending=False)
        top_df = df[df.abs_coef > 0].reset_index().iloc[:num_top]
    else:
        df = pd.DataFrame(
            {
                "feature_name": word_names,
                "freq": X_words.mean(axis=0),
                "coef": np.abs(word_coefs).max(axis=0),
            }
        ).sort_values(["coef"], ascending=False)
        top_df = df[df.coef > 0].reset_index().iloc[:num_top]
    logging.info("top df %s", top_df)
    logging.info("freq sort %s", df.sort_values("freq", ascending=False).iloc[:40])

    return top_df


def generate_prior_prompt(
    data_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    word_names: Union[np.ndarray, List[str]],
    seed: int,
    max_meta_concepts: int,
    keep_x_cols: Optional[List[str]],
    model: str,
    baseline_init_file: str,
    config_file: Optional[str] = None,
    num_top: int = 40,
    use_acc: bool = False,
    cv: int = 5,
) -> str:
    """
    Generate prior prompt for concept initialization without args dependency.

    This is the new, cleaner version that takes explicit parameters instead of
    a namespace object
    """
    word_names = (
        word_names.tolist() if isinstance(word_names, np.ndarray) else word_names
    )
    is_multiclass = np.unique(y_train).size > 2
    X_keep = None
    if keep_x_cols is not None:
        print("KEEP COLS", keep_x_cols)
        print("TABULAR FEATURES:", [TABULAR_PREFIX + col for col in keep_x_cols])
        X_keep = data_df[keep_x_cols].to_numpy()

    top_df = fit_residual(
        model,
        word_names,
        X_keep,
        X_train,
        y_train,
        sample_weight=sample_weight,
        penalty_downweight_factor=100,
        is_multiclass=is_multiclass,
        num_top=num_top,
        use_acc=use_acc,
        seed=seed,
        force_keep_columns=keep_x_cols,
        cv=cv,
    )

    normalization_factor = np.max(np.abs(top_df.coef))
    top_df["coef"] = (
        top_df.coef / normalization_factor if normalization_factor > 0 else top_df.coef
    )

    print("---------------------------")
    with open(baseline_init_file, "r") as file:
        prompt_template = file.read()
        prompt_template = prompt_template.replace(
            "{top_features_df}",
            top_df[["feature_name", "coef"]].to_csv(index=False, float_format="%.3f"),
        )
        prompt_template = prompt_template.replace(
            "{max_meta_concepts}", str(max_meta_concepts)
        )

    if config_file is not None:
        with open(config_file, "r") as f:
            config_dict = json.load(f)
            for k, v in config_dict.items():
                prompt_template = prompt_template.replace(k, v)
    print(prompt_template)
    return prompt_template
