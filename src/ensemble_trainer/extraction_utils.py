"""
Extraction utilities for ensemble training.

This module provides unified utilities for handling concept extraction mapping
and feature preparation across bootstrap samples and full datasets.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import src.common as common

from . import data_operations


class ExtractionHandler:
    """Unified handler for concept extraction operations."""

    @staticmethod
    def prepare_concept_features(
        concepts: List[dict],
        shared_extractions: Dict[str, np.ndarray],
        data_df: pd.DataFrame,
        num_holdout: int = 0,
        needs_mapping: bool = False,
        force_keep_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[dict], Dict[str, np.ndarray]]:
        """
        Unified method for preparing concept features with/without holdout.

        Args:
            concepts: List of concept dictionaries
            shared_extractions: Dictionary of concept extractions from full dataset
            data_df: Target dataset (bootstrap sample or full dataset)
            num_holdout: Number of concepts to hold out from the end
            needs_mapping: Whether to map extractions to bootstrap sample indices
            force_keep_columns: Optional columns to force keep in features

        Returns:
            Tuple of (features, concept_subset, extractions)

        Extraction Mapping Guide:
        - Use needs_mapping=True when working with bootstrap/split samples
        - Use needs_mapping=False when working with full dataset
        - Shared extractions are always computed on full dataset
        - Mapping handles index translation for bootstrap samples with duplicates/reordering
        """
        # Remove held-out concepts if specified
        if num_holdout > 0:
            concept_subset = concepts[:-num_holdout]
        else:
            concept_subset = concepts

        # Map extractions if working with bootstrap sample
        if needs_mapping:
            extractions = data_operations.map_extractions_to_sample(
                shared_extractions, data_df, concept_subset
            )
        else:
            extractions = {
                c["concept"]: shared_extractions[c["concept"]]
                for c in concept_subset
                if c["concept"] in shared_extractions
            }

        # Get features using the common utility
        features = common.get_features(
            concept_subset, extractions, data_df, force_keep_columns
        )

        return features, concept_subset, extractions
