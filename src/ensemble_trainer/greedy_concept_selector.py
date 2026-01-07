"""
Streamlined greedy concept selector for ensemble training.

This module contains a stripped-down version of ConceptLearnerModel
that only includes the functionality needed for greedy concept selection,
removing all Bayesian sampling and posterior inference code.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import src.common as common
from src.llm_response_types import CandidateConcepts
from src.training_history import TrainingHistory

from .model_utils import fit_residual
from . import data_operations
from .config import EnsembleConfig

class GreedyConceptSelector:
    """
    Streamlined greedy concept selector for ensemble training.

    Contains only the greedy concept selection functionality without
    Bayesian sampling or posterior inference.
    """

    default_prior = 0.1
    penalty_downweight_factor = 1000

    def __init__(
        self,
        init_seed: int,
        init_history: TrainingHistory,
        llm_iter,
        num_classes: int,
        num_meta_concepts: int,
        prompt_iter_file: str,
        config: EnsembleConfig,
        residual_model_type: str,
        final_model_type: str,
        num_greedy_holdout: int = 1,
        is_greedy_metric_acc: bool = False,
        force_keep_columns=None,
        num_top: int = 40,
        cv: int = 5,
    ):
        """
        Initialize greedy concept selector.

        Args:
            init_history: Training history object
            llm_iter: LLM API client for iteration prompts
            num_classes: Number of target classes
            num_meta_concepts: Number of meta concepts to maintain
            prompt_iter_file: Path to iteration prompt template file
            config: Configuration dictionary for prompt templating
            residual_model_type: Type of residual model (e.g., 'l1', 'l2')
            num_greedy_holdout: Number of concepts to replace per iteration
            is_greedy_metric_acc: Whether to use accuracy (vs AUC) for greedy selection
            force_keep_columns: Columns to force keep in feature extraction
            num_top: Number of top residual features to consider
        """
        self.init_seed = init_seed
        self.init_history = init_history
        self.llm_iter = llm_iter
        self.num_classes = num_classes
        self.is_multiclass = num_classes > 2
        self.num_meta_concepts = num_meta_concepts
        self.prompt_iter_file = prompt_iter_file
        self.config = config
        self.residual_model_type = residual_model_type
        self.final_model_type = final_model_type
        self.num_greedy_holdout = num_greedy_holdout
        self.is_greedy_metric_acc = is_greedy_metric_acc
        self.force_keep_columns = force_keep_columns
        self.num_top = num_top
        self.cv = cv

    def scrub_vectorized_sentences(self, X_features, feat_names, concept_dicts: list):
        """
        Remove words that are too correlated with concepts from residual model inputs.

        Args:
            X_features: Feature matrix
            feat_names: Feature names
            concept_dicts: List of concept dictionaries

        Returns:
            Tuple of (scrubbed_features, scrubbed_names)
        """
        # Remove the words that are too correlated with the concepts from the residual model's inputs
        words_to_scrub = [
            w
            for c in concept_dicts
            if not common.is_tabular(c["concept"])
            for w in c["words"]
            if len(w) > 2
        ]
        keep_mask = [
            ~np.any([scrub_word in w for scrub_word in words_to_scrub])
            or common.is_tabular(w)
            for w in feat_names
        ]
        return X_features[:, keep_mask], feat_names[keep_mask]

    def make_new_concept_prompt(
        self,
        X_extracted,
        X_words,
        y,
        sample_weight,
        data_df,
        extract_feature_names,
        feat_names,
        num_replace: int = 1,
    ):
        """
        Generate prompt to ask LLM for candidate concepts.

        Args:
            X_extracted: Extracted concept features
            X_words: Word count features
            y: Target labels
            data_df: Data DataFrame
            extract_feature_names: Names of existing concepts
            feat_names: Feature names
            num_replace: Number of concepts to replace

        Returns:
            Tuple of (prompt, meta_concepts_text, top_features_text, top_feat_names)
        """
        with open(self.prompt_iter_file, "r") as file:
            prompt_template = file.read()

        top_df = fit_residual(
            self.residual_model_type,
            feat_names.tolist(),
            X_extracted,
            X_words,
            y,
            sample_weight=sample_weight,
            penalty_downweight_factor=self.penalty_downweight_factor,
            is_multiclass=self.is_multiclass,
            num_top=self.num_top,
            use_acc=self.is_greedy_metric_acc,
            cv=self.cv,
            force_keep_columns=self.force_keep_columns,
        )

        # normalize the coefficients just to make it a bit easier to read for the LLM
        normalization_factor = np.max(np.abs(top_df.coef))
        top_df["coef"] = (
            top_df.coef / normalization_factor
            if normalization_factor > 0
            else top_df.coef
        )
        # Generate the prompt with the top features
        top_features_text = top_df[["feature_name", "coef"]].to_csv(
            index=False, float_format="%.3f"
        )
        prompt_template = prompt_template.replace(
            "{top_features_df}", top_features_text
        )

        prompt_template = prompt_template.replace(
            "{num_concepts}", str(self.num_meta_concepts)
        )
        meta_concepts_text = ""
        for i, feat_name in enumerate(extract_feature_names):
            meta_concepts_text += f"* X{i} = {feat_name} \n"
        prompt_template = prompt_template.replace("{meta_concepts}", meta_concepts_text)
        prompt_template = prompt_template.replace(
            "{num_concepts_fixed}", str(self.num_meta_concepts - num_replace)
        )
        prompt_template = prompt_template.replace(
            "{num_attributes}", str(top_df.shape[0])
        )

        # prompt_template = self.fill_config(prompt_template)
        return (
            prompt_template,
            meta_concepts_text,
            top_features_text,
            top_df.feature_name,
        )

    def query_for_new_cand(
        self, iter_llm_prompt, top_feat_names, times_to_retry=1, max_new_tokens=5000
    ):
        """
        Query LLM for new candidate concepts.

        Args:
            iter_llm_prompt: Iteration prompt string
            top_feat_names: Top feature names
            times_to_retry: Number of retry attempts (unused)
            max_new_tokens: Maximum tokens for LLM response

        Returns:
            List of candidate concept dictionaries
        """
        llm_response = self.llm_iter.get_output(
            iter_llm_prompt,
            max_new_tokens=max_new_tokens,
            response_model=CandidateConcepts,
        )
        candidate_concept_dicts = llm_response.to_dicts(
            default_prior=self.default_prior
        )
        candidate_concept_dicts += [
            {"concept": feat_name, "prior": self.default_prior}
            for feat_name in top_feat_names
            if common.is_tabular(feat_name)
        ]
        return candidate_concept_dicts

    def do_greedy_step(
        self,
        dataset_df,
        extracted_features,  # extracted concepts MINUS the existing concept being replaced
        candidate_concept_dicts,
        all_extracted_feat_dict,
        existing_concept_dicts,
    ):
        """
        Perform greedy concept selection step.

        Args:
            dataset_df: Dataset DataFrame
            extracted_features: Features from existing concepts (minus held-out)
            y: Target labels
            candidate_concept_dicts: Candidate concept dictionaries
            all_extracted_feat_dict: All extracted features dictionary
            existing_concept_dicts: Existing concept dictionaries being replaced

        Returns:
            List of selected concept dictionaries
        """
        y = dataset_df["y"].to_numpy().flatten()
        sample_weight = dataset_df["sample_weight"].to_numpy().flatten()
        train_idxs, test_idxs = data_operations.create_data_split_indices(
            y,
            self.init_seed,
            self.config.training.train_frac,
        )

        all_concept_dicts = existing_concept_dicts + candidate_concept_dicts
        logging.info(
            "concepts (greedy search) %s",
            [cdict["concept"] for cdict in all_concept_dicts],
        )
        num_orig_features = extracted_features.shape[1]
        selected_concepts = []
        # do greedy step-wise selection
        for i in range(self.num_greedy_holdout):
            concept_scores = []
            for concept_dict in all_concept_dicts:
                extracted_candidate = common.get_features(
                    [concept_dict], all_extracted_feat_dict, dataset_df
                )
                aug_extract = np.concatenate(
                    [extracted_features, extracted_candidate], axis=1
                )
                candidate_result_dict = common.train_LR(
                    X_train=aug_extract[train_idxs],
                    y_train=y[train_idxs],
                    sample_weight=sample_weight[train_idxs],
                    X_test=aug_extract[test_idxs],
                    y_test=y[test_idxs],
                    test_weight=sample_weight[test_idxs],
                    penalty=self.final_model_type,
                    use_acc=self.is_greedy_metric_acc,
                )
                
                candidate_coef = candidate_result_dict["coef"][0][-1]
                signs_agree = (candidate_coef > 0 and concept_dict["is_risk_factor"]) or (candidate_coef < 0 and not concept_dict["is_risk_factor"])
                if self.config.training.do_coef_check and not signs_agree:
                    candidate_score = 0
                elif self.is_greedy_metric_acc:
                    candidate_score = candidate_result_dict["test_acc"]
                else:
                    candidate_score = candidate_result_dict["test_auc"]
                concept_scores.append(candidate_score)
            max_idxs_options = np.where(
                np.isclose(concept_scores, np.max(concept_scores))
            )[0]
            max_idx = np.random.choice(max_idxs_options)

            extracted_features = np.concatenate(
                [
                    extracted_features,
                    common.get_features(
                        [all_concept_dicts[max_idx]],
                        all_extracted_feat_dict,
                        dataset_df,
                    ),
                ],
                axis=1,
            )

            logging.info("concepts (greedy all test auc) %s", concept_scores)
            logging.info("concepts (greedy best test auc) %s", np.max(concept_scores))
            logging.info(
                "selected concept (greedy) %s",
                all_concept_dicts[max_idx]["concept"],
            )
            logging.info("greedy-accept %s", max_idx >= len(existing_concept_dicts))

            selected_concepts.append(all_concept_dicts[max_idx])
            all_concept_dicts.pop(max_idx)
        return selected_concepts

    def fill_config(self, template_str: str):
        """
        Fill configuration placeholders in template string.

        Args:
            template_str: Template string with config placeholders

        Returns:
            Template string with config values filled in
        """
        for k, v in self.config.items():
            template_str = template_str.replace(k, v)
        return template_str
