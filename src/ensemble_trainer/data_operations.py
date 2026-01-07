"""
Data operations for ensemble training.

This module contains data splitting, mapping, and preprocessing utilities
for coordinated ensemble training across multiple initializations.
"""

import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_data_split_indices(
    y: np.ndarray,
    init_seed: int,
    data_split_fraction: float,
) -> pd.DataFrame:
    """Create data split for a specific initialization (used for baseline training)."""
    # Data splitting without replacement
    split_size = int(y.size * data_split_fraction)
    split_indices, test_indices = train_test_split(np.arange(y.size), train_size=split_size, random_state=init_seed, shuffle=True, stratify=y)
    return split_indices, test_indices

def create_data_split(
    data_df: pd.DataFrame,
    init_seed: int,
    data_split_fraction: float,
) -> pd.DataFrame:
    """Create data split for a specific initialization (used for baseline training)."""
    # Data splitting without replacement
    split_size = int(len(data_df) * data_split_fraction)
    split_indices, test_indices = train_test_split(np.arange(data_df.shape[0]), train_size=split_size, random_state=init_seed, shuffle=True, stratify=data_df["y"])
    sampled_df = data_df.copy()
    # Store the original indices for later mapping
    sampled_df["_orig_index"] = np.arange(len(data_df))
    return sampled_df.iloc[split_indices], sampled_df.iloc[test_indices]

# def create_train_test_split(
#     data_df: pd.DataFrame, init_seed: int, epoch: int, train_frac: float
# ) -> pd.DataFrame:
#     """
#     Create train split for bayesian training (matches ConceptLearnerModel.fit() behavior).

#     This matches the behavior in ConceptLearnerModel.fit() where it does train_test_split
#     for proper train/test separation during concept evolution. Uses the same random state
#     progression as the original implementation.
#     """
#     # Use seed that varies by both initialization and epoch
#     split_seed = init_seed * 3000 + epoch * 100 + 2
#     np.random.seed(split_seed)

#     # Get target variable for stratification
#     y = data_df["y"].to_numpy().flatten()

#     # Create train/test split (use only train portion for concept evolution)
#     train_size = int(len(data_df) * train_frac)
#     train_indices, test_indices = train_test_split(
#         np.arange(len(data_df)),
#         train_size=train_size,
#         stratify=y,
#         random_state=split_seed,
#     )

#     return data_df.iloc[train_indices], data_df.iloc[test_indices]


def map_extractions_to_sample(
    original_extractions: Dict[str, np.ndarray],
    bootstrap_data: pd.DataFrame,
    concept_dicts: List[dict],
) -> Dict[str, np.ndarray]:
    """
    Map feature extractions from original dataset to bootstrap sample positions.

    This preserves the bootstrap sampling effects on training while avoiding
    redundant feature extraction.
    """
    bootstrap_extractions = {}

    # Get the stored indices from sampling
    if "_orig_index" in bootstrap_data.columns:
        # Use the stored original indices from bootstrap/split sampling
        original_indices = bootstrap_data["_orig_index"].values
        logging.info(
            f"Using original indices for mapping (length: {len(original_indices)})"
        )
    else:
        # Fallback to using DataFrame index (for "none" sampling or legacy behavior)
        original_indices = np.arange(len(bootstrap_data))
        logging.info(
            f"Using sequential indices for mapping (length: {len(original_indices)})"
        )

    for concept_dict in concept_dicts:
        concept = concept_dict["concept"]

        if concept in original_extractions:
            # Get original feature values
            original_values = original_extractions[concept]

            # Map to bootstrap positions using original indices
            # original_values is indexed by position in original_data_df
            # We want to get the values for the rows that appear in bootstrap_data
            bootstrap_values = original_values[original_indices]

            bootstrap_extractions[concept] = bootstrap_values
        else:
            # If concept not found, create zero features
            bootstrap_extractions[concept] = np.zeros((len(bootstrap_data), 1))
            logging.warning(f"Concept '{concept}' not found in original extractions")

    return bootstrap_extractions


def load_data_partition(
    in_dataset_file: str,
    indices_csv: str,
    partition: str,
    max_obs: int,
    init_concepts_file: Union[str, None] = None,
    concept_column: str = "llm_output",
    text_summary_column: str = "llm_summary",
    join_column: Union[str, None] = None,
    dataset_already_filtered: bool = False,
):
    # Read notes and summaries
    dset = pd.read_csv(in_dataset_file)
    dataset_already_filtered |= indices_csv is None

    # if we have the join column then we will join the dset_partition with dset_init_concepts
    # otherwise we will align them by index
    if init_concepts_file is not None:
        dset_init_concepts = pd.read_csv(init_concepts_file)
        if join_column is not None:
            dset = dset.drop(columns=[concept_column])
            dset = dset.merge(
                dset_init_concepts[[join_column, concept_column]],
                on=join_column,
                how="inner",
            )
            dset_partition = dset
        else:
            if "sentence" in dset_init_concepts.columns:
                print(dset_init_concepts.sentence)
                print(dset.sentence)
                assert np.all(dset_init_concepts.sentence == dset.sentence)
            elif "image_path" in dset_init_concepts.columns:
                assert np.all(dset_init_concepts.image_path == dset.image_path)
            dset[concept_column] = dset_init_concepts[concept_column]

            # filter
            if dataset_already_filtered:
                dset_partition = dset.reset_index(drop=True)
                dset_partition['sample_weight'] = 1
            else:
                partition_df = pd.read_csv(indices_csv)
                dset_partition = dset.iloc[
                    partition_df[partition_df.partition == partition].idx
                ].reset_index(drop=True)
                if 'sample_weight' in partition_df:
                    dset_partition['sample_weight'] = partition_df.sample_weight[partition_df.partition == partition]
                else:
                    dset_partition['sample_weight'] = 1
    else:
        if dataset_already_filtered:
            print("FILTERED", dset.shape)
            dset_partition = dset.reset_index(drop=True)
            dset_partition['sample_weight'] = 1
        else:
            partition_df = pd.read_csv(indices_csv)
            dset_partition = dset.iloc[
                partition_df[partition_df.partition == partition].idx
            ].reset_index(drop=True)
            print(partition_df.columns)
            if 'sample_weight' in partition_df:
                dset_partition['sample_weight'] = partition_df.sample_weight[partition_df.partition == partition]
            else:
                dset_partition['sample_weight'] = 1

    if text_summary_column in dset_partition.columns:
        dset_partition = dset_partition[~dset_partition[text_summary_column].isna()]
    logging.info("DSET SIZE %s y prevalence %f", dset.shape, dset.y.mean())

    if max_obs > 0:
        dset_partition = dset_partition.iloc[:max_obs]

    logging.info(f"FINAL NON-NA DSET {dset_partition.shape}")

    return dset_partition
