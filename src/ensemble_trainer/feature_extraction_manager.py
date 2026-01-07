"""
Feature extraction manager for ensemble training.

This module consolidates the functionality from FeatureExtractor and SharedAnnotationManager
into a single, cleaner FeatureExtractionManager class with internal caching.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import src.common as common

from .config import EnsembleConfig


class _ExtractionCache:
    """Internal cache for extracted features."""

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    def has(self, concept: str) -> bool:
        """Check if concept exists in cache."""
        return concept in self._cache

    def get(self, concept: str) -> np.ndarray:
        """Get features for a concept."""
        return self._cache[concept]

    def add(self, concept: str, features: np.ndarray):
        """Add features for a concept."""
        self._cache[concept] = features

    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all cached extractions."""
        return self._cache.copy()

    def update(self, new_extractions: Dict[str, np.ndarray]):
        """Update cache with new extractions."""
        self._cache.update(new_extractions)


class FeatureExtractionManager:
    """Manages feature extraction with internal caching for ensemble training."""

    def __init__(
        self,
        config: EnsembleConfig,
        llm_dict: Dict,
        concept_tracker,
    ):
        self.config = config
        self.llm_dict = llm_dict
        self.concept_tracker = concept_tracker
        self._cache = _ExtractionCache()

    @property
    def shared_extractions(self) -> Dict[str, np.ndarray]:
        """Get all cached extractions (for checkpoint compatibility)."""
        return self._cache.get_all()

    def set_shared_extractions(self, extractions: Dict[str, np.ndarray]):
        """Set cached extractions (for checkpoint restoration)."""
        self._cache._cache = extractions.copy()

    async def extract_features_batch(
        self,
        data_df: pd.DataFrame,
        all_concepts: List[str],
        max_new_tokens: int = 8000,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for all concepts in batches with caching.

        This is the core extraction method that uses the SharedAnnotationManager logic
        with optimized batch processing and caching.
        """
        # Filter out concepts we already have
        concepts_to_extract = [c for c in all_concepts if not self._cache.has(c)]

        if not concepts_to_extract:
            return self._cache.get_all()

        logging.info(
            f"Extracting features for {len(concepts_to_extract)} concepts across all initializations"
        )

        # Create concept dicts for extraction
        concept_dicts = [{"concept": concept} for concept in concepts_to_extract]

        try:
            # Use async version to avoid asyncio.run() within running event loop
            new_extractions = await common.extract_features_by_llm_grouped_async(
                self.llm_dict["extraction"],
                data_df,
                concept_dicts,
                prompt_file=self.config.concept.prompt_concepts_file,
                all_extracted_features_dict=self._cache.get_all(),
                batch_size=self.config.training.batch_size,
                batch_concept_size=self.config.training.batch_concept_size,
                max_new_tokens=max_new_tokens,
                is_image=self.config.data.is_image,
                group_size=self.config.training.batch_obs_size,
                max_section_length=self.config.data.max_section_length,
                sentence_column=self.config.data.text_summary_column,
            )
        except Exception as e:
            logging.error(
                f"Feature extraction failed for concepts: {concepts_to_extract[:3]}..."
            )
            logging.error(f"Error: {e}")
            import traceback

            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            raise RuntimeError(
                f"Failed to extract features for {len(concepts_to_extract)} concepts"
            ) from e

        # Validate extractions
        if not new_extractions:
            raise ValueError("Feature extraction returned empty results")

        for concept, features in new_extractions.items():
            if features is None or not isinstance(features, np.ndarray):
                raise ValueError(f"Invalid features returned for concept '{concept}'")
            if features.shape[0] != len(data_df):
                raise ValueError(
                    f"Feature shape mismatch for concept '{concept}': "
                    f"expected {len(data_df)} samples, got {features.shape[0]}"
                )

        # Update cache
        self._cache.update(new_extractions)

        return self._cache.get_all()

    def get_extractions_for_init(self, init_seed: int) -> Dict[str, np.ndarray]:
        """Get extractions for concepts belonging to a specific initialization."""
        init_concepts = self.concept_tracker.get_concepts_for_init(init_seed)
        return {
            concept: self._cache.get(concept)
            for concept in init_concepts
            if self._cache.has(concept)
        }

    async def extract_for_training(
        self,
        data_df: pd.DataFrame,
        concepts: List[str],
        max_new_tokens: int = 8000,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for training using coordinated feature extraction.

        This method handles coordinated feature extraction across multiple initializations.
        """
        return await self.extract_features_batch(
            data_df=data_df,
            all_concepts=concepts,
            max_new_tokens=max_new_tokens,
        )

    def get_features_for_model(
        self,
        concept_dicts: List[dict],
        all_extracted_features: Dict[str, np.ndarray],
        data_df: pd.DataFrame,
        force_keep_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Get features for model training/prediction.

        This is a wrapper around common.get_features that handles the feature
        extraction and combination with tabular features.
        """
        return common.get_features(
            concept_dicts=concept_dicts,
            all_extracted_features=all_extracted_features,
            dset=data_df,
            force_keep_columns=force_keep_columns,
        )

    def collect_concepts_for_prediction(
        self, training_histories: Dict, use_posterior_iters: Optional[int] = None
    ) -> List[Dict]:
        """Collect unique concepts from all initializations for prediction."""
        all_concepts_to_extract = []

        for init_seed, history in training_histories.items():
            if use_posterior_iters is not None:
                start_iter = max(0, history.num_iters - use_posterior_iters)
            else:
                start_iter = 0

            concepts = [
                concept_dict
                for concept_dicts in history._concepts[start_iter : history.num_iters]
                for concept_dict in concept_dicts
            ]
            all_concepts_to_extract.extend(concepts)

        # Remove duplicates based on concept text
        unique_concepts = {}
        for concept in all_concepts_to_extract:
            concept_key = concept["concept"]
            if concept_key not in unique_concepts:
                unique_concepts[concept_key] = concept

        return list(unique_concepts.values())

    async def extract_for_prediction(
        self, data_df: pd.DataFrame, concepts: List[Dict]
    ) -> Dict:
        """Extract features for prediction on new data."""
        if not self.llm_dict:
            raise ValueError("LLM clients not initialized. Set up clients first.")

        # Extract concept strings from concept dicts
        concept_strings = [c["concept"] for c in concepts]

        # Bypass extraction cache and extract from the provided data
        concept_dicts = [{"concept": concept} for concept in concept_strings]

        logging.info(
            f"Extracting features for {len(concept_dicts)} concepts from {len(data_df)} prediction samples"
        )

        # Extract features directly from the prediction data
        new_extractions = await common.extract_features_by_llm_grouped_async(
            self.llm_dict["extraction"],
            data_df,  # Use the actual prediction data
            concept_dicts,
            prompt_file=self.config.concept.prompt_concepts_file,
            all_extracted_features_dict={},  # Empty dict - no cache
            batch_size=self.config.training.batch_size,
            batch_concept_size=self.config.training.batch_concept_size,
            max_new_tokens=8000,
            is_image=self.config.data.is_image,
            group_size=self.config.training.batch_obs_size,
            max_section_length=self.config.data.max_section_length,
            sentence_column=self.config.data.text_summary_column,
        )

        return new_extractions

    def get_init_predictions_features(
        self,
        init_seed: int,
        data_df: pd.DataFrame,
        all_extracted_features: Dict,
        training_histories: Dict,
        use_posterior_iters: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Get prediction features for a specific initialization.

        Returns a list of feature arrays, one for each iteration.
        """
        history = training_histories[init_seed]
        features_list = []

        if use_posterior_iters is not None:
            start_iter = max(0, history.num_iters - use_posterior_iters)
        else:
            start_iter = 0

        for i in range(start_iter, history.num_iters):
            concept_dicts = history._concepts[i]

            # Get force_keep_columns and verify test data has required columns
            force_keep_columns = (
                history.force_keep_cols
                if hasattr(history, "force_keep_cols")
                and history.force_keep_cols is not None
                else None
            )

            # Verify test data has required columns if force_keep_columns is specified
            if force_keep_columns:
                missing_cols = set(force_keep_columns) - set(data_df.columns)
                if missing_cols:
                    raise ValueError(
                        f"Test data missing required tabular features: {missing_cols}. "
                        f"Available columns: {list(data_df.columns)}. "
                        f"Required columns: {force_keep_columns}"
                    )

            extracted_features = self.get_features_for_model(
                concept_dicts=concept_dicts,
                all_extracted_features=all_extracted_features,
                data_df=data_df,
                force_keep_columns=force_keep_columns,
            )

            features_list.append(extracted_features)

        return features_list
