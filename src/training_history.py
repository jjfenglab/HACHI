import os
import pickle
import logging
from typing import List, Optional

import numpy as np
import pandas as pd


class TrainingHistory:
    def __init__(self, force_keep_cols: Optional[List[str]] = None):
        self.force_keep_cols = force_keep_cols
        self._concepts = []
        self._models = []
        self._aucs = []
        self._test_aucs = []
        self._coefs = []
        self._intercepts = []
        self._log_liks = []
        # Phase tracking for unified history
        self._phase_markers = []  # List of {"iteration": int, "phase": str}

    @property
    def num_iters(self):
        return len(self._concepts)

    def get_last_concepts(self) -> list[dict]:
        return self._concepts[-1]

    def get_last_model(self):
        return self._models[-1]

    def get_last_auc(self) -> float:
        return self._aucs[-1]

    def get_last_log_liks(self) -> list:
        if len(self._log_liks):
            return self._log_liks[-1]
        else:
            return None

    def load(self, file_name: str):
        if os.path.exists(file_name):
            with open(file_name, "rb") as file:
                self = pickle.load(file)
        else:
            raise FileNotFoundError("FILE?", file_name)

        return self

    def add_auc(self, auc: float):
        self._aucs.append(auc)

    def add_test_auc(self, auc: float):
        self._test_aucs.append(auc)

    def add_log_liks(self, log_liks: list):
        self._log_liks.append(log_liks)

    def add_coef(self, coef: float):
        self._coefs.append(coef)

    def add_model(self, model):
        self._models.append(model)

    def add_intercept(self, intercept: float):
        self._intercepts.append(intercept)

    def update_history(self, concepts: list[dict], metrics: dict):
        coef = metrics["coef"][1] if metrics["coef"].shape[0] == 2 else metrics["coef"].flatten()
        self.add_auc(metrics["auc"])
        if "test_auc" in metrics:
            self.add_test_auc(metrics["test_auc"])
        self.add_coef(coef)
        self.add_intercept(metrics["intercept"])
        self.add_model(metrics["model"])
        self.add_concepts(concepts)
        concept_coef_df = pd.DataFrame({
            "concept": [c["concept"] for c in concepts],
            "coef": coef})
        logging.info(f"history update {concept_coef_df}")

    def get_model(self, index: int):
        return self._models[index]

    def add_concepts(self, concepts: list[dict]):
        self._concepts.append(concepts)

    def save(self, file_name: str):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)
        pd.DataFrame({
            "concept": [c['concept'] for c in self.get_last_concepts()],
            "coef": self._coefs[-1].flatten(),
        }).sort_values('coef').to_csv(file_name.replace(".pkl", "_concepts.csv"))

    def mark_phase_transition(self, phase: str, iteration: int = None):
        """Mark a phase transition in the training history.

        Args:
            phase: Name of the new phase (e.g., 'baseline', 'greedy')
            iteration: Iteration number (defaults to current num_iters)
        """
        if iteration is None:
            iteration = self.num_iters
        self._phase_markers.append({"iteration": iteration, "phase": phase})

    def get_baseline_iterations(self) -> int:
        """Get the number of baseline iterations.

        Returns:
            Number of iterations that were part of baseline training
        """
        for marker in self._phase_markers:
            if marker["phase"] == "greedy":
                return marker["iteration"]
        # If no bayesian phase marker, all iterations are baseline
        return self.num_iters

    def get_baseline_subset(self) -> "TrainingHistory":
        """Create a new TrainingHistory with only baseline iterations.

        Returns:
            A new TrainingHistory containing only the baseline phase data
        """
        baseline_iters = self.get_baseline_iterations()

        subset = TrainingHistory(self.force_keep_cols)
        subset._concepts = self._concepts[:baseline_iters]
        subset._models = self._models[:baseline_iters]
        subset._aucs = self._aucs[:baseline_iters]
        subset._coefs = self._coefs[:baseline_iters]
        subset._intercepts = self._intercepts[:baseline_iters]
        subset._log_liks = self._log_liks[:baseline_iters]

        # Copy phase markers up to baseline
        subset._phase_markers = [
            m for m in self._phase_markers if m["iteration"] < baseline_iters
        ]

        return subset
