"""
Script for making predictions with trained EnsembleTrainer on test data.

This script loads a trained EnsembleTrainer from checkpoint and makes predictions
on test data that has been preprocessed with summaries.
"""

import argparse
import asyncio
import logging
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, RocCurveDisplay
from dotenv import load_dotenv

import matplotlib.pyplot as plt


sys.path.append(os.getcwd())

import src.common as common
import src.ensemble_trainer.data_operations as data_operations
from src.ensemble_trainer import EnsembleTrainer

def get_ci(data):
    mean = np.mean(data)
    sem = st.sem(data) # Standard error of the mean
    ci = st.norm.interval(0.95, loc=mean, scale=sem)
    return f"{mean:.2f} ({ci[0]:.2f}, {ci[1]:.2f})"

def get_bootstrap_ci(y_true, y_pred, sample_weight, eval_func, n_bootstraps=1000):
    """Compute bootstrap confidence intervals for evaluation metrics."""
    eval_value = eval_func(y_true, y_pred, sample_weight=sample_weight)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        try:
            score = eval_func(y_true[indices], y_pred[indices], sample_weight=sample_weight[indices])
            bootstrapped_scores.append(score)
        except Exception:
            continue
    return eval_value, (
        np.quantile(bootstrapped_scores, [0.025, 0.975])
        if bootstrapped_scores
        else None
    )


def to_table_str(bootstrap_res):
    """Format bootstrap results as table string."""
    if bootstrap_res[1] is None:
        return f"{bootstrap_res[0]:.3f} (CI unavailable)"
    return (
        f"{bootstrap_res[0]:.3f} ({bootstrap_res[1][0]:.3f}, {bootstrap_res[1][1]:.3f})"
    )


def get_specificity_at_threshold(y_test, y_scores, desired_sensitivity=.98):
    """Compute specificity at threshold"""
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    specificity = 1 - fpr
    optimal_threshold_idx = np.where(tpr >= desired_sensitivity)[0][0]
    threshold_at_sensitivity = thresholds[optimal_threshold_idx]
    return specificity[optimal_threshold_idx]

def evaluate_predictions(y_true, pred_probabs, sample_weight, desired_sensitivity=0.98):
    """Evaluate predictions with bootstrap confidence intervals.

    Args:
        y_true: True labels
        pred_probabs: Prediction probabilities (n_samples, n_classes)

    Returns:
        Tuple of (auc, accuracy, brier, log_lik) bootstrap results
    """
    pred_probabs = common.get_safe_prob(pred_probabs)
    y_pred_positive = pred_probabs[:, -1]  # Positive class probabilities

    auc_bootstrap = get_bootstrap_ci(y_true, y_pred_positive, sample_weight, roc_auc_score)
    specificity_bootstrap = get_bootstrap_ci(
        y_true, y_pred_positive, sample_weight=sample_weight, eval_func=lambda y_t, y_p, sample_weight: get_specificity_at_threshold(y_t, y_p, desired_sensitivity=desired_sensitivity)
    )
    return auc_bootstrap, specificity_bootstrap


def collect_metrics(model_name, y_true, pred_probabs, sample_weight, desired_sensitivity=0.98):
    """Collect all metrics for a given model's predictions.

    Args:
        model_name: Name identifier for the model
        y_true: True labels
        pred_probabs: Prediction probabilities

    Returns:
        Dict of metrics for this model
    """
    auc_bootstrap, specificity_bootstrap = (
        evaluate_predictions(y_true, pred_probabs, sample_weight, desired_sensitivity=desired_sensitivity)
    )

    return {
        "model": model_name,
        "auc": auc_bootstrap[0],
        "auc_str": to_table_str(auc_bootstrap),
        "specificity": specificity_bootstrap[0],
        "specificity_str": to_table_str(specificity_bootstrap),
        "n_samples": len(y_true),
        "positive_rate": y_true.mean(),
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Input/output arguments
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to ensemble_state_checkpoint.pkl file from training",
    )
    parser.add_argument(
        "--in-dataset-file",
        type=str,
        required=True,
        help="Path to dataset with summaries for prediction",
    )
    parser.add_argument(
        "--indices-csv",
        type=str,
        # required=True,
        help="Path to train/test indices file",
    )
    parser.add_argument(
        "--model-in-dataset-file",
        type=str,
        help="Path to the model's training dataset",
    )
    parser.add_argument(
        "--model-indices-csv",
        type=str,
        help="Path to the model's train/test indices file",
    )
    parser.add_argument(
        "--model-obs-id-column",
        type=str,
        help="Path to the model's obs id column",
    )
    parser.add_argument(
        "--excluded-concepts",
        type=str,
        nargs="+",
        default=[],
        help="concepts to exclude",
    )
    parser.add_argument(
        "--partition",
        type=str,
        # required=True,
        help="Partition to evaluate on",
    )
    parser.add_argument(
        "--output-predictions",
        type=str,
        required=True,
        help="Path to save ensemble predictions CSV",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Path to save prediction metrics CSV (optional)",
    )

    # Prediction parameters
    parser.add_argument(
        "--use-posterior-iters",
        type=int,
        default=None,
        help="Use only last N iterations per initialization (default: all)",
    )
    parser.add_argument(
        "--save-per-init",
        action="store_true",
        default=False,
        help="Save predictions for each initialization separately",
    )
    parser.add_argument(
        "--is-baseline",
        action="store_true",
        default=False,
        help="set to true if you only want to evaluate baseline",
    )
    parser.add_argument(
        "--concept-generation-mode",
        type=str,
        default="standard",
        choices=["standard", "evidence_span"],
        help="Concept generation mode: standard or evidence_span with summary enhancement",
    )
    parser.add_argument(
        "--desired-sensitivity",
        type=float,
        default=0.9,
    )

    # Data parameters
    parser.add_argument(
        "--max-obs",
        type=int,
        default=0,
        help="Maximum number of observations to use (0 = all)",
    )
    parser.add_argument(
        "--age-column",
        type=str,
        default=None,
        help="If we want to evaluate across age groups",
    )
    parser.add_argument(
        "--age-cutoff",
        type=int,
        default=None,
        help="If we want to evaluate across age groups",
    )
    parser.add_argument(
        "--domain-column",
        type=str,
        default=None,
        help="Column name for domain",
    )
    parser.add_argument(
        "--concept-column",
        type=str,
        default="llm_output",
        help="Column name for concepts",
    )
    parser.add_argument(
        "--text-summary-column",
        type=str,
        default="llm_summary",
        help="Column name for text summary",
    )
    parser.add_argument(
        "--join-column",
        type=str,
        default=None,
        help="Column name to join on",
    )

    # Logging
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file",
    )
    parser.add_argument(
        "--image-file",
        type=str,
        help="Path to image file for ROC curve",
    )
    parser.add_argument(
        "--annotations-csv",
        type=str,
        default=None,
        required=False,
        help="Path to saving the concept annotations if needed",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    return parser.parse_args()


async def evaluate_model(
    ensemble_trainer,
    data_df,
    use_posterior_iters=None,
    is_baseline=False,
    save_per_init=False,
    compute_metrics=True,
    desired_sensitivity=0.98,
):
    """Evaluate ensemble model on a data subset.

    Args:
        ensemble_trainer: Trained EnsembleTrainer instance
        data_df: DataFrame with data to evaluate on (must contain 'y' column)
        use_posterior_iters: Number of posterior iterations to use (default: all)
        is_baseline: Whether to evaluate baseline only
        save_per_init: Whether to generate per-initialization predictions
        compute_metrics: Whether to compute and return metrics

    Returns:
        Tuple of (result_df, metrics_df, per_init_predictions)
        - result_df: DataFrame with predictions
        - metrics_df: DataFrame with metrics (None if compute_metrics=False)
        - per_init_predictions: Dict of per-init predictions (None if save_per_init=False)
    """
    logging.info(f"Evaluating model on data with shape: {data_df.shape}")

    # Make ensemble predictions
    logging.info("Making ensemble predictions...")
    ensemble_predictions = await ensemble_trainer.predict(
        data_df, use_posterior_iters=use_posterior_iters, is_baseline=is_baseline
    )

    # Add metadata to predictions
    result_df = data_df[["y"]].copy()
    for col in ensemble_predictions.columns:
        result_df[f"ensemble_{col}"] = ensemble_predictions[col]

    # Save per-initialization predictions if requested
    per_init_predictions = None
    if save_per_init:
        # NOTE: predict_all() returns separate predictions for each initialization,
        # where each init's prediction is averaged across its iterations
        # This allows comparison of individual bootstrap chains
        logging.info("Making per-initialization predictions...")
        per_init_predictions = await ensemble_trainer.predict_all(
            data_df, use_posterior_iters=use_posterior_iters
        )

        for init_seed, pred_df in per_init_predictions.items():
            for col in pred_df.columns:
                result_df[f"init_{init_seed}_{col}"] = pred_df[col]

    # Compute metrics if requested
    metrics_df = None
    if compute_metrics:
        logging.info("Computing prediction metrics...")

        sample_weight = data_df["sample_weight"].values
        y_true = data_df["y"].values
        ensemble_pred_probs = ensemble_predictions.values  # Full probability matrix
        print(y_true.shape, ensemble_pred_probs.shape)

        # Collect ensemble metrics
        ensemble_metrics = collect_metrics("ensemble", y_true, ensemble_pred_probs, sample_weight, desired_sensitivity=desired_sensitivity)
        metrics_list = [ensemble_metrics]

        # Add per-initialization metrics if available
        if save_per_init and per_init_predictions:
            for init_seed, pred_df in per_init_predictions.items():
                init_pred_probs = pred_df.values  # Full probability matrix
                init_metrics = collect_metrics(
                    f"init_{init_seed}", y_true, init_pred_probs, sample_weight=sample_weight, desired_sensitivity=desired_sensitivity
                )
                metrics_list.append(init_metrics)

        metrics_df = pd.DataFrame(metrics_list)
        print(metrics_df)

        # Log key metrics
        logging.info(f"Ensemble AUC: {ensemble_metrics['auc']:.4f}")
        # logging.info(f"Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f}")
        # logging.info(f"Ensemble Brier Score: {ensemble_metrics['brier']:.4f}")
        # logging.info(f"Ensemble Log-likelihood: {ensemble_metrics['log_lik']:.4f}")

    return result_df, metrics_df, per_init_predictions


async def main():
    """Main function for ensemble prediction."""
    load_dotenv()
    args = parse_args()

    # Setup logging
    if args.log_file:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
            filename=args.log_file,
            level=logging.INFO,
            force=True,
        )
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info(f"Starting ensemble prediction with args: {args}")

    # Set random seed
    np.random.seed(args.seed)

    # Load trained ensemble
    logging.info(f"Loading EnsembleTrainer from checkpoint: {args.checkpoint_path}")
    ensemble_trainer = EnsembleTrainer.from_checkpoint(args.checkpoint_path)
    ensemble_trainer.feature_extraction_manager.config.data.text_summary_column = args.text_summary_column

    # remove any excluded concepts by setting coef to zero
    print(args.excluded_concepts)
    for init_seed, hist in ensemble_trainer.training_histories.items():
        print("===========", init_seed)
        for i in range(hist.num_iters - 1, hist.num_iters):
            print(hist._concepts[i])
            print(hist.get_model(i).coef_)
            for concept_idx, concept_dict in enumerate(hist._concepts[i]):
                if concept_dict["concept"] in args.excluded_concepts:
                    hist.get_model(i).coef_[0,concept_idx] = 0
                    print(hist.get_model(i).coef_)

    # Load test data with summaries
    logging.info("Loading data...")
    data_df = data_operations.load_data_partition(
        in_dataset_file=args.in_dataset_file,
        max_obs=args.max_obs,
        init_concepts_file=None,  # Not needed for prediction
        indices_csv=args.indices_csv,
        partition=args.partition,
        concept_column=args.concept_column,
        text_summary_column=args.text_summary_column,
        join_column=args.join_column,
    )
    
    if args.model_in_dataset_file:
        # if model training dataset provided, make sure the test data does not overlap with training data
        model_train_data_df = data_operations.load_data_partition(
            in_dataset_file=args.model_in_dataset_file,
            max_obs=args.max_obs,
            init_concepts_file=None,
            indices_csv=args.model_indices_csv,
            partition="train",
            concept_column=args.concept_column,
            text_summary_column=args.text_summary_column,
            join_column=args.join_column,
        )
        print("ORIG DATA SHAPE", data_df.shape)
        data_df = data_df[~data_df[args.model_obs_id_column].isin(model_train_data_df[args.model_obs_id_column])]
        print("NEW DATA SHAPE", data_df.shape)

    logging.info(f"Loaded data with shape: {data_df.shape}")

    # Ensure required columns exist
    if args.text_summary_column not in data_df.columns:
        raise ValueError(
            f"Text summary column '{args.text_summary_column}' not found in data"
        )
    if "y" not in data_df.columns:
        raise ValueError("Target column 'y' not found in data")

    # Generate annotations if requested
    if args.annotations_csv:
        annotated_df = await ensemble_trainer.get_annotations(
            data_df, use_posterior_iters=args.use_posterior_iters, is_baseline=args.is_baseline
        )
        # get prevalences
        attributes = list(annotated_df.columns)
        prevalences = []
        for col_name in annotated_df.columns:
            prevalences.append(get_ci(annotated_df[col_name].to_numpy()))
        # AKI summaries
        # prevalences.append(get_ci(data_df.age.to_numpy()))
        # prevalences.append(get_ci(data_df.gender.to_numpy() == "Male"))
        # prevalences.append(get_ci(data_df.aki_stage.to_numpy() == 1))
        # prevalences.append(get_ci(data_df.aki_stage.to_numpy() == 2))
        # prevalences.append(get_ci(data_df.aki_stage.to_numpy() == 3))
        # attributes += ["age", "is_male", "aki_stage_1", "aki_stage_2", "aki_stage_3"]

        # TBI summaries
        # prevalences.append(get_ci(data_df.age_y.to_numpy()))
        # attributes += ["age"]

        prevalences_df = pd.DataFrame({
            "attribute": attributes,
            "summary": prevalences,
        })
        prevalences_df.to_csv(args.annotations_csv)

    # Evaluate model on the data
    result_df, metrics_df, _ = await evaluate_model(
        ensemble_trainer=ensemble_trainer,
        data_df=data_df,
        use_posterior_iters=args.use_posterior_iters,
        is_baseline=args.is_baseline,
        save_per_init=args.save_per_init,
        compute_metrics=args.output_metrics is not None,
        desired_sensitivity=args.desired_sensitivity,
    )
    metrics_df["domain"] = "all"
    if args.domain_column is not None:
        for domain_val in data_df[args.domain_column].unique():
            print("DOMAIN", domain_val)
            domain_result_df, domain_metrics_df, _ = await evaluate_model(
                ensemble_trainer=ensemble_trainer,
                data_df=data_df[data_df[args.domain_column] == domain_val],
                use_posterior_iters=args.use_posterior_iters,
                is_baseline=args.is_baseline,
                save_per_init=args.save_per_init,
                compute_metrics=args.output_metrics is not None,
                desired_sensitivity=args.desired_sensitivity,
            )
            domain_metrics_df["domain"] = domain_val
            metrics_df = pd.concat([metrics_df, domain_metrics_df])
            print(domain_metrics_df)
    
    if args.age_column is not None:
        domain_result_df, domain_metrics_df, _ = await evaluate_model(
            ensemble_trainer=ensemble_trainer,
            data_df=data_df[data_df[args.age_column] < args.age_cutoff],
            use_posterior_iters=args.use_posterior_iters,
            is_baseline=args.is_baseline,
            save_per_init=args.save_per_init,
            compute_metrics=args.output_metrics is not None,
            desired_sensitivity=args.desired_sensitivity,
        )
        domain_metrics_df["domain"] = f"age_below_{args.age_cutoff}"
        metrics_df = pd.concat([metrics_df, domain_metrics_df])
        domain_result_df, domain_metrics_df, _ = await evaluate_model(
            ensemble_trainer=ensemble_trainer,
            data_df=data_df[data_df[args.age_column] >= args.age_cutoff],
            use_posterior_iters=args.use_posterior_iters,
            is_baseline=args.is_baseline,
            save_per_init=args.save_per_init,
            compute_metrics=args.output_metrics is not None,
            desired_sensitivity=args.desired_sensitivity,
        )
        domain_metrics_df["domain"] = f"age_above_{args.age_cutoff}"
        metrics_df = pd.concat([metrics_df, domain_metrics_df])
        print(domain_metrics_df)

    # Save predictions
    result_df.to_csv(args.output_predictions, index=False)
    logging.info(f"Saved predictions to {args.output_predictions}")

    # Save metrics if computed
    if args.output_metrics and metrics_df is not None:
        metrics_df.to_csv(args.output_metrics, index=False)
        logging.info(f"Saved metrics to {args.output_metrics}")
    
    if args.image_file:
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(result_df["y"], result_df["ensemble_prob_1"], ax=ax, name="Ensemble")
        RocCurveDisplay.from_predictions(result_df["y"], result_df["init_1_prob_1"], ax=ax, name="Init 1")
        RocCurveDisplay.from_predictions(result_df["y"], result_df["init_2_prob_1"], ax=ax, name="Init 2")
        plt.savefig(args.image_file)

if __name__ == "__main__":
    asyncio.run(main())
