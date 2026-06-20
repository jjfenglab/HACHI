"""
Concept generation functionality for ensemble training.

This module contains classes for generating initial and candidate concepts,
with separate implementations for standard and evidence-span enhanced generation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import src.common as common

from . import data_operations
from .config import EnsembleConfig
from .extraction_utils import ExtractionHandler
from .text_processing import get_word_count_data


class BaseConceptGenerator(ABC):
    """Abstract base class for concept generation."""

    def __init__(self, config: EnsembleConfig, llm_dict: Dict):
        self.config = config
        self.llm_dict = llm_dict

    @abstractmethod
    def generate_initial_concepts(
        self, init_seed: int, data_df: pd.DataFrame, concept_selector
    ) -> Tuple[List[dict], pd.DataFrame]:
        """Generate initial concepts for baseline training."""
        pass

    @abstractmethod
    def generate_candidate_concepts(
        self,
        init_data_df: pd.DataFrame,
        concept_selector,
        current_concepts: List[dict],
        current_extractions: Dict[str, np.ndarray],
    ) -> List[dict]:
        """Generate candidate concepts for greedy iteration."""
        pass


class StandardConceptGenerator(BaseConceptGenerator):
    """Standard concept generation."""

    def generate_initial_concepts(
        self, init_seed: int, data_df: pd.DataFrame, concept_selector
    ) -> Tuple[List[dict], pd.DataFrame]:
        """
        Generate initial concepts for a single initialization using bootstrap-specific residual model.

        Returns:
            Tuple of (concept_dicts, bootstrap_data)
        """
        np.random.seed(init_seed)
        torch.manual_seed(init_seed)

        logging.info(f"Generating initial concepts for init_seed {init_seed}")

        sampled_data, test_data = data_operations.create_data_split(
            data_df,
            init_seed,
            self.config.training.train_frac,
        )

        X_words_train, word_names = get_word_count_data(
            sampled_data,
            self.config.model.count_vectorizer,
            text_summary_column=self.config.data.text_summary_column,
            min_prevalence=self.config.data.min_prevalence,
        )
        y_train = sampled_data["y"].to_numpy().flatten()
        sample_weight = sampled_data["sample_weight"].to_numpy().flatten()

        init_llm_prompt, _, top_feat_names = concept_selector.make_initial_concept_prompt(
            X_words_train,
            y_train,
            sample_weight,
            sampled_data,
            word_names,
        )

        logging.debug(
            f"[CONCEPT GEN] Using standard concept generation for init_seed {init_seed}"
        )

        candidate_concept_dicts = concept_selector.query_for_new_cand(
            init_llm_prompt,
            top_feat_names,
            max_new_tokens=self.config.llm.max_new_tokens,
        )

        concept_dicts = candidate_concept_dicts[: self.config.concept.max_meta_concepts]

        logging.info(
            f"Generated {len(concept_dicts)} concepts for init_seed {init_seed}"
        )

        return concept_dicts, (sampled_data, test_data)

    def generate_candidate_concepts(
        self,
        init_data_df: pd.DataFrame,
        concept_selector,
        current_concepts: List[dict],
        current_extractions: Dict[str, np.ndarray],
    ) -> List[dict]:
        """Generate candidate concepts for greedy iteration."""
        # Get word count data from the train split
        X_words_train, word_names = get_word_count_data(
            init_data_df,
            self.config.model.count_vectorizer,
            text_summary_column=self.config.data.text_summary_column,
            min_prevalence=self.config.data.min_prevalence,
        )
        y_train = init_data_df["y"].to_numpy().flatten()

        # Prepare concept features using unified handler
        num_greedy_holdout = self.config.training.num_greedy_holdout
        extracted_features_subset, concept_subset_dicts, concept_subset_extractions = (
            ExtractionHandler.prepare_concept_features(
                concepts=current_concepts,
                shared_extractions=current_extractions,
                data_df=init_data_df,
                num_holdout=num_greedy_holdout,
                needs_mapping=True,  # Working with bootstrap/split sample
                force_keep_columns=concept_selector.force_keep_columns,
            )
        )

        # Scrub vectorized sentences (remove words too correlated with concepts)
        X_scrubbed, feat_names_scrubbed = concept_selector.scrub_vectorized_sentences(
            X_words_train, word_names, concept_subset_dicts
        )

        # Generate prompt for new concepts using concept selector method
        (
            iter_llm_prompt,
            meta_concepts_text,
            top_features_text,
            top_feat_names,
        ) = concept_selector.make_new_concept_prompt(
            X_extracted=extracted_features_subset,
            X_words=X_scrubbed,
            y=y_train,
            sample_weight=init_data_df["sample_weight"].to_numpy().flatten(),
            data_df=init_data_df,
            extract_feature_names=[c["concept"] for c in concept_subset_dicts],
            feat_names=feat_names_scrubbed,
            num_replace=num_greedy_holdout,
        )

        # Query LLM for candidate concepts using concept selector method
        candidate_concept_dicts = concept_selector.query_for_new_cand(
            iter_llm_prompt,
            top_feat_names,
        )

        return candidate_concept_dicts


class ConceptGeneratorFactory:
    """Factory for creating concept generators based on configuration."""

    @staticmethod
    def create_generator(
        config: EnsembleConfig,
        llm_dict: Dict,
        summaries_df: Optional[pd.DataFrame] = None,
    ) -> BaseConceptGenerator:
        """Create the appropriate concept generator based on config."""
        return StandardConceptGenerator(config, llm_dict)
