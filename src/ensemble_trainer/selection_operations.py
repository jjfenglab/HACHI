"""
Selection operations for ensemble training.

This module contains greedy concept selection utilities for coordinated
ensemble training across multiple initializations.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .extraction_utils import ExtractionHandler
from .greedy_concept_selector import GreedyConceptSelector
from .config import ConceptConfig, TrainingConfig


def perform_greedy_selection(
    data_df: pd.DataFrame,
    concept_selector: GreedyConceptSelector,
    init_candidate_map: Dict[Tuple[int, str], dict],
    current_shared_extractions: Dict[str, np.ndarray],
    candidate_shared_extractions: Dict[str, np.ndarray],
) -> List[dict]:
    """Perform greedy concept selection for a single initialization."""

    # Get current concepts and setup
    current_concepts = concept_selector.init_history.get_last_concepts()[
        :concept_selector.config.concept.goal_num_meta_concepts
    ]

    # Get candidate concepts for this initialization
    candidate_concepts = [
        concept_dict
        for (seed, concept), concept_dict in init_candidate_map.items()
        if seed == concept_selector.init_seed
    ]

    # Get concepts to replace
    concepts_to_replace = current_concepts[-concept_selector.config.training.num_greedy_holdout:]

    # Prepare features using unified handler
    full_extracted_features, concept_subset_dicts, full_concept_subset_extractions = (
        ExtractionHandler.prepare_concept_features(
            concepts=current_concepts,
            shared_extractions=current_shared_extractions,
            data_df=data_df,
            num_holdout=concept_selector.config.training.num_greedy_holdout,
            needs_mapping=True,
            force_keep_columns=None,
        )
    )

    # Combine all extractions for full dataset
    full_all_extracted_features = {
        **full_concept_subset_extractions,
        **candidate_shared_extractions,  # Already on full dataset
    }

    # Add concepts_to_replace extractions for full dataset
    for concept_dict in concepts_to_replace:
        concept = concept_dict["concept"]
        if concept in current_shared_extractions:
            full_all_extracted_features[concept] = current_shared_extractions[concept]

    # Perform greedy selection on full dataset
    selected_concept_dicts = concept_selector.do_greedy_step(
        data_df,
        extracted_features=full_extracted_features,
        candidate_concept_dicts=candidate_concepts,
        all_extracted_feat_dict=full_all_extracted_features,
        existing_concept_dicts=concepts_to_replace,
    )

    # Combine selected concepts with concept subset
    final_concepts = selected_concept_dicts + concept_subset_dicts

    return final_concepts
