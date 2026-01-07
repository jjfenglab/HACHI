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
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.getcwd())
sys.path.append("llm-api-main")

from src.ensemble_trainer import EnsembleTrainer
import src.ensemble_trainer.data_operations as data_operations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Input/output arguments
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to trained EnsembleTrainer checkpoint file",
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
        required=True,
        help="Path to train/test indices file",
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
        "--partition",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Which partition to predict on",
    )
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
    
    # Data parameters
    parser.add_argument(
        "--max-obs",
        type=int,
        default=0,
        help="Maximum number of observations to use (0 = all)",
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    return parser.parse_args()


async def main():
    """Main function for ensemble prediction."""
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
    
    # Load test data with summaries
    logging.info(f"Loading {args.partition} data...")
    data_df = data_operations.load_data_partition(
        in_dataset_file=args.in_dataset_file,
        indices_csv=args.indices_csv,
        partition=args.partition,
        max_obs=args.max_obs,
        init_concepts_file=None,  # Not needed for prediction
        concept_column=args.concept_column,
        text_summary_column=args.text_summary_column,
        join_column=args.join_column,
        dataset_already_filtered=True,  # test_llm_summaries.csv contains only test data
    )
    
    logging.info(f"Loaded {args.partition} data with shape: {data_df.shape}")
    
    # Ensure required columns exist
    if args.text_summary_column not in data_df.columns:
        raise ValueError(f"Text summary column '{args.text_summary_column}' not found in data")
    if "y" not in data_df.columns:
        raise ValueError("Target column 'y' not found in data")
    
    # Make ensemble predictions
    logging.info("Making ensemble predictions...")
    ensemble_predictions = await ensemble_trainer.predict(
        data_df, use_posterior_iters=args.use_posterior_iters
    )
    
    # Add metadata to predictions
    result_df = data_df[["y"]].copy()
    for col in ensemble_predictions.columns:
        result_df[f"ensemble_{col}"] = ensemble_predictions[col]
    
    # Save per-initialization predictions if requested
    if args.save_per_init:
        logging.info("Making per-initialization predictions...")
        per_init_predictions = await ensemble_trainer.predict_all(
            data_df, use_posterior_iters=args.use_posterior_iters
        )
        
        for init_seed, pred_df in per_init_predictions.items():
            for col in pred_df.columns:
                result_df[f"init_{init_seed}_{col}"] = pred_df[col]
    
    # Save predictions
    result_df.to_csv(args.output_predictions, index=False)
    logging.info(f"Saved predictions to {args.output_predictions}")
    
    # Compute and save metrics if requested
    if args.output_metrics:
        logging.info("Computing prediction metrics...")
        
        y_true = data_df["y"].values
        y_pred_ensemble = ensemble_predictions.iloc[:, 1].values  # prob_1 column
        
        # Compute ensemble metrics
        auc_ensemble = roc_auc_score(y_true, y_pred_ensemble)
        acc_ensemble = accuracy_score(y_true, y_pred_ensemble > 0.5)
        
        metrics_data = {
            "model": ["ensemble"],
            "auc": [auc_ensemble],
            "accuracy": [acc_ensemble],
            "n_samples": [len(y_true)],
            "positive_rate": [y_true.mean()],
        }
        
        # Add per-initialization metrics if available
        if args.save_per_init:
            for init_seed, pred_df in per_init_predictions.items():
                y_pred_init = pred_df.iloc[:, 1].values  # prob_1 column
                auc_init = roc_auc_score(y_true, y_pred_init)
                acc_init = accuracy_score(y_true, y_pred_init > 0.5)
                
                metrics_data["model"].append(f"init_{init_seed}")
                metrics_data["auc"].append(auc_init)
                metrics_data["accuracy"].append(acc_init)
                metrics_data["n_samples"].append(len(y_true))
                metrics_data["positive_rate"].append(y_true.mean())
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(args.output_metrics, index=False)
        logging.info(f"Saved metrics to {args.output_metrics}")
        
        # Log key metrics
        logging.info(f"Ensemble AUC: {auc_ensemble:.4f}")
        logging.info(f"Ensemble Accuracy: {acc_ensemble:.4f}")
    
    logging.info("Prediction completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
