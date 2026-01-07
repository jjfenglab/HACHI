"""
Script for ensemble training with coordinated concept learning across multiple initializations.

This script provides a command-line interface for ensemble training:
1. Running baseline training in parallel for multiple initializations
2. Coordinating greedy training to batch concept extraction across all initializations
3. Maintaining separate training histories for each initialization while sharing expensive operations
"""

import argparse
import asyncio
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

sys.path.append(os.getcwd())

import src.ensemble_trainer.data_operations as data_operations
from src.ensemble_trainer import ConfigBuilder, EnsembleTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Core parameters
    parser.add_argument(
        "--init-seeds",
        type=int,
        nargs="+",
        default=[1],
        help="List of initialization seeds",
    )
    # Training parameters
    parser.add_argument(
        "--goal-num-meta-concepts",
        type=int,
        default=5,
        help="Goal max number of meta concepts",
    )
    parser.add_argument(
        "--max-meta-concepts",
        type=int,
        default=6,
        help="Number of meta concepts to brainstorm initially",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument(
        "--num-additional-epochs",
        type=int,
        default=0,
        help="Number of training epochs to add",
    )
    parser.add_argument(
        "--do-coef-check",
        action="store_true",
        default=False,
        help="Do coefficient check with prior knowledge?",
    )
    parser.add_argument(
        "--num-greedy-epochs", type=int, default=2, help="Number of greedy epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Batch size for LLM queries"
    )
    parser.add_argument(
        "--batch-concept-size",
        type=int,
        default=20,
        help="Batch size for concept extraction",
    )
    parser.add_argument(
        "--batch-obs-size",
        type=int,
        default=1,
        help="Batch size for observation processing",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.5, help="Fraction of data for training"
    )

    # Data and model parameters
    parser.add_argument(
        "--in-dataset-file", type=str, required=True, help="Path to input dataset file"
    )
    parser.add_argument(
        "--indices-csv", type=str, required=True, help="Path to train/test indices file"
    )
    parser.add_argument(
        "--init-concepts-file",
        type=str,
        required=True,
        help="Path to initial concepts file",
    )
    parser.add_argument(
        "--prompt-concepts-file",
        type=str,
        required=True,
        help="Path to concept extraction prompt file",
    )
    parser.add_argument(
        "--baseline-init-file",
        type=str,
        required=True,
        help="Path to baseline initialization prompt file",
    )

    # LLM parameters
    parser.add_argument(
        "--llm-model", type=str, default="gpt-4o-mini", help="LLM model to use"
    )
    parser.add_argument(
        "--cache-file", type=str, default="cache.db", help="Path to cache file"
    )
    parser.add_argument(
        "--use-api", action="store_true", default=True, help="Use API for LLM calls"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="_output",
        help="Output directory for training histories",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file",
    )
    parser.add_argument(
        "--save-legacy-files",
        action="store_true",
        default=True,
        help="Save separate baseline_history.pkl files for backward compatibility (default: True)",
    )
    parser.add_argument(
        "--no-save-legacy-files",
        dest="save_legacy_files",
        action="store_false",
        help="Don't save separate baseline_history.pkl files (use unified format only)",
    )

    # Checkpointing and resume parameters
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        default=False,
        help="Ignore existing checkpoints and start fresh training",
    )

    # Additional parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--max-obs", type=int, default=0, help="Maximum number of observations to use"
    )
    parser.add_argument(
        "--learner-type", type=str, default="count_l2", help="Learner type"
    )
    parser.add_argument(
        "--concept-column",
        type=str,
        default="llm_output",
        help="Column name for concepts",
    )
    parser.add_argument(
        "--join-column",
        type=str,
        default=None,
        help="Column name to join on",
    )
    parser.add_argument(
        "--text-summary-column",
        type=str,
        default="llm_summary",
        help="Column name for text summary",
    )
    parser.add_argument(
        "--is-image",
        action="store_true",
        default=False,
        help="Whether data contains images",
    )
    parser.add_argument(
        "--max-section-length", type=int, default=None, help="Maximum section length"
    )

    # Baseline configuration parameters
    parser.add_argument(
        "--model", type=str, default="l2", help="Model type for baseline"
    )
    parser.add_argument(
        "--final-model-type", type=str, default="l2", help="Final model type"
    )
    parser.add_argument(
        "--count-vectorizer", type=str, default="count", help="Count vectorizer type"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=600, help="Maximum tokens for LLM"
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="CV for tuning penalty params"
    )
    parser.add_argument(
        "--use-acc", action="store_true", default=False, help="Use accuracy metric"
    )
    parser.add_argument(
        "--min-prevalence", type=int, default=0, help="Minimum prevalence"
    )
    parser.add_argument(
        "--num-top-attributes", type=int, default=40, help="Number of top attributes"
    )
    parser.add_argument(
        "--force-keep-columns",
        type=str,
        nargs="+",
        default=None,
        help="Columns to force keep as control features",
    )
    parser.add_argument(
        "--keep-x-cols",
        type=str,
        nargs="+",
        default=None,
        help="Column names to include in residual model for concept generation",
    )

    # ConceptLearnerModel parameters for greedy training
    parser.add_argument(
        "--prompt-iter-file",
        type=str,
        default="",
        help="Path to iteration prompt file for concept generation",
    )
    parser.add_argument(
        "--prompt-prior-file",
        type=str,
        default="",
        help="Path to prior prompt file for greedy training",
    )
    parser.add_argument(
        "--num-greedy-holdout",
        type=int,
        default=1,
        help="Number of concepts to hold out for greedy selection",
    )
    parser.add_argument(
        "--is-greedy-metric-acc",
        action="store_true",
        default=False,
        help="Use accuracy instead of AUC for greedy selection",
    )
    parser.add_argument(
        "--residual-model-type",
        type=str,
        default="l2",
        help="Model type for residual analysis",
    )
    parser.add_argument(
        "--inverse-penalty-param",
        type=float,
        default=20000,
        help="Inverse penalty parameter for logistic regression",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5000,
        help="Maximum new tokens for concept generation",
    )
    parser.add_argument(
        "--num-top-residual-words",
        type=int,
        default=40,
        help="Number of top residual words for concept generation",
    )

    # Evidence-span enhancement arguments
    parser.add_argument(
        "--concept-generation-mode",
        type=str,
        default="standard",
        choices=["standard", "evidence_span"],
        help="Concept generation mode: standard or evidence_span with summary enhancement",
    )

    parser.add_argument(
        "--summaries-file",
        type=str,
        default=None,
        help="CSV file containing clinical summaries for evidence-span mode",
    )

    parser.add_argument(
        "--evidence-file",
        type=str,
        default=None,
        help="JSON file containing evidence mappings for concept enhancement",
    )

    parser.add_argument(
        "--use-semantic-search",
        action="store_true",
        default=True,
        help="Enable semantic search fallback for context extraction",
    )

    parser.add_argument(
        "--no-semantic-search",
        dest="use_semantic_search",
        action="store_false",
        help="Disable semantic search fallback",
    )

    parser.add_argument(
        "--evidence-span-debug-dir",
        type=str,
        default=None,
        help="Directory to save debug files for evidence-span mode",
    )

    # Additional arguments for config completeness
    parser.add_argument(
        "--config-dict",
        type=dict,
        default=None,
        help="Configuration dictionary for concept learner",
    )

    return parser.parse_args()


async def main():
    """Main function for ensemble training."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
        filename=args.log_file,
        level=logging.INFO,
        force=True,
    )

    logging.info(f"Starting ensemble training with args: {args}")
    load_dotenv()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info(f"Test random number: {np.random.random()}")

    # Load data
    logging.info("Loading data...")

    data_df = data_operations.load_data_partition(
        in_dataset_file=args.in_dataset_file,
        indices_csv=args.indices_csv,
        partition="train",
        max_obs=args.max_obs,
        init_concepts_file=args.init_concepts_file,
        text_summary_column=args.text_summary_column,
        concept_column=args.concept_column,
        join_column=args.join_column,
    )

    logging.info(
        f"Loaded data with shape: {data_df.shape} and columns: {data_df.columns}"
    )

    # Create ensemble configuration using the builder
    config = ConfigBuilder.from_args(args)

    # Handle different training scenarios with new API
    if args.force_restart:
        # Force fresh training - remove any existing checkpoint
        checkpoint_file = os.path.join(args.output_dir, "ensemble_state_checkpoint.pkl")
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logging.info("Removed existing checkpoint file for fresh start")

        # Create trainer and start fresh training
        ensemble_trainer = EnsembleTrainer(
            config=config,
            output_dir=args.output_dir,
            save_legacy_files=args.save_legacy_files,
            seed=args.seed,
        )
        logging.info("Starting fresh ensemble training...")
        final_histories = await ensemble_trainer.fit(data_df)

    elif args.resume_from_checkpoint:
        # Resume from specific checkpoint
        logging.info(
            f"Resuming training from checkpoint: {args.resume_from_checkpoint}"
        )
        ensemble_trainer = EnsembleTrainer.from_checkpoint(
            checkpoint_path=args.resume_from_checkpoint,
            output_dir=args.output_dir,  # Can override output location
        )
        final_histories = await ensemble_trainer.continue_training(
            data_df, num_additional_epochs=args.num_additional_epochs
        )

    else:
        # Auto-resume from output directory if checkpoint exists
        checkpoint_file = os.path.join(args.output_dir, "ensemble_state_checkpoint.pkl")
        print(checkpoint_file)
        if os.path.exists(checkpoint_file):
            logging.info("Found existing checkpoint, resuming training...")
            ensemble_trainer = EnsembleTrainer.from_checkpoint(checkpoint_file)
            final_histories = await ensemble_trainer.continue_training(
                data_df, num_additional_epochs=args.num_additional_epochs
            )
        else:
            logging.info("No checkpoint found, starting fresh training...")
            ensemble_trainer = EnsembleTrainer(
                config=config,
                output_dir=args.output_dir,
                save_legacy_files=args.save_legacy_files,
            )
            final_histories = await ensemble_trainer.fit(data_df)

    logging.info(
        f"Ensemble training completed. Final histories saved for {len(final_histories)} initializations."
    )


if __name__ == "__main__":
    asyncio.run(main())
