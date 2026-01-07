"""
Plotting utilities for ensemble training results.

This module provides standalone plotting functions for visualizing training histories
and AUC curves from ensemble training runs.
"""

import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt

from ..training_history import TrainingHistory


def plot_simple_aucs(
    aucs: List[float], save_path: str, title: str = "All extracted concepts AUC"
) -> None:
    """
    Simple AUC plot for a single training history.

    This replaces the TrainingHistory.plot_aucs method.

    Args:
        aucs: List of AUC values
        save_path: Path to save the plot
        title: Plot title
    """
    plt.clf()
    plt.plot(aucs, linestyle="--")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_individual_auc_with_phases(
    history: TrainingHistory, save_path: str, title_suffix: str = ""
) -> None:
    """
    Plot AUC curves for a single training history.

    Args:
        history: TrainingHistory object
        save_path: Path to save the plot
        title_suffix: Additional text for plot title
    """
    plt.figure(figsize=(10, 6))

    all_aucs = history._aucs
    baseline_iters = history.get_baseline_iterations()

    greedy_aucs = all_aucs[baseline_iters:]
    plt.plot(
        range(baseline_iters, len(all_aucs)),
        greedy_aucs,
        linestyle="-",
        label="Greedy phase",
    )

    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title(f"Training AUCs{title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_ensemble_aucs_combined(
    training_histories: Dict[int, TrainingHistory],
    init_seeds: List[int],
    save_path: str,
) -> None:
    """
    Plot combined AUC curves for all initializations on one figure.

    Args:
        training_histories: Dict mapping init_seed to TrainingHistory
        init_seeds: List of initialization seeds
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))

    for init_seed in init_seeds:
        if init_seed in training_histories:
            history = training_histories[init_seed]
            all_aucs = history._aucs
            baseline_iters = history.get_baseline_iterations()

            # Plot baseline phase
            if baseline_iters > 0:
                baseline_aucs = all_aucs[:baseline_iters]
                plt.plot(
                    range(len(baseline_aucs)),
                    baseline_aucs,
                    linestyle="--",
                    alpha=0.7,
                    label=f"Baseline init_{init_seed}",
                )

            # Plot greedy phase
            if len(all_aucs) > baseline_iters:
                greedy_aucs = all_aucs[baseline_iters:]
                plt.plot(
                    range(baseline_iters, len(all_aucs)),
                    greedy_aucs,
                    linestyle="-",
                    label=f"Greedy init_{init_seed}",
                )

    plt.xlabel("Iteration")
    plt.ylabel("AUC")
    plt.title("Training AUCs - All Initializations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_aucs(
    training_histories: Dict[int, TrainingHistory],
    init_seeds: List[int],
    output_dir: str,
) -> None:
    """
    Create comprehensive AUC plots for ensemble training results.

    This creates:
    1. A combined plot showing all initializations
    2. Individual plots for each initialization (baseline subset)
    3. Individual plots for each initialization (complete history)

    Args:
        training_histories: Dict mapping init_seed to TrainingHistory
        init_seeds: List of initialization seeds
        output_dir: Base directory for plots
    """
    # Combined plot for all initializations
    combined_plot_file = os.path.join(output_dir, "all_aucs.png")
    plot_ensemble_aucs_combined(training_histories, init_seeds, combined_plot_file)
    logging.info(f"Saved combined AUC plot to {combined_plot_file}")

    # Individual plots for each initialization
    for init_seed in init_seeds:
        if init_seed in training_histories:
            history = training_histories[init_seed]
            init_dir = os.path.join(output_dir, f"init_seed_{init_seed}")
            os.makedirs(init_dir, exist_ok=True)

            # Plot baseline AUCs (subset)
            baseline_subset = history.get_baseline_subset()
            if baseline_subset.num_iters > 0:
                baseline_plot_file = os.path.join(init_dir, "baseline_aucs.png")
                plot_simple_aucs(
                    baseline_subset._aucs,
                    baseline_plot_file,
                    f"Baseline AUCs - init_seed {init_seed}",
                )
                logging.info(
                    f"Saved baseline AUC plot for init_seed {init_seed} to {baseline_plot_file}"
                )

            # Plot complete training history (includes greedy)
            complete_plot_file = os.path.join(init_dir, "greedy_aucs.png")
            plot_individual_auc_with_phases(
                history, complete_plot_file, f" - init_seed {init_seed}"
            )
            logging.info(
                f"Saved complete training AUC plot for init_seed {init_seed} to {complete_plot_file}"
            )
