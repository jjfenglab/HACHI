"""
Script for computing the frequency of concepts from each AKI model and it's comparator 
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd

def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
            "--annotations-path",
            type=str,
            required=True,
            help="path to annotations file",
            )
    parser.add_argument(
            "--column-name",
            type=str,
            required=True)
    parser.add_argument(
            "--results-csv",
            type=str,
            required=True,
            help="Path to saving results",
            )
    parser.add_argument(
            "--original-csv",
            type=str,
            required=True,
            help="Path to original data with the outcome",
            )
    parser.add_argument(
            "--indices-csv",
            type=str,
            required=True,
            help="Path to train/ test indices",
            )
    parser.add_argument(
            "--partition",
            type=str,
            required=True,
            help="train or test partition",
            )
    parser.add_argument(
            "--data-csv",
            type=str,
            required=True,
            help="data used to create CBM model",
            )
    args = parser.parse_args(args)
    return args


def calc_concepts_freq(df, original_prev, name):
    df = df.drop(columns=['y'])
    # for each concept (columns) compute the frequency. Each concept is a row and the column is "name" with the frequencies
    freq_df = df.apply(lambda x: x.sum() / len(x)).to_frame(name)
    freq_df[name] = freq_df[name] * original_prev 
    return freq_df

def main(args):
    args = parse_args(args)

    ann_df = pd.read_csv(args.annotations_path)
    cols_to_drop = [col for col in ann_df.columns if 'pred_prob' in col or "risk_class" in col or "risk_score" in col] + ['enc_id', 'y', 'sentence']
    ann_df = ann_df.drop(columns=cols_to_drop, errors='ignore')

    original_df = pd.read_csv(args.original_csv)
    original_prev = original_df[args.column_name].sum() / len(original_df)

    data_df = pd.read_csv(args.data_csv)
    partition_df = pd.read_csv(args.indices_csv)

    data_df = data_df.iloc[partition_df[partition_df.partition == args.partition].idx].reset_index(drop=True)

    df_merged = ann_df.join(data_df[['y']])

    pos_df = calc_concepts_freq(
            df_merged[df_merged.y == 1], 
            original_prev, 
            "positive"
            )
    neg_df = calc_concepts_freq(
            df_merged[df_merged.y == 0], 
            1-original_prev, 
            "negative"
            )

    # horizontally stacks frequencies  
    combined_freq = pos_df.join(neg_df)
    combined_freq['total_frequency'] = combined_freq.sum(axis=1)
    combined_freq.to_csv(args.results_csv)

if __name__ == "__main__":
    main(sys.argv[1:])
