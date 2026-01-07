"""
Script for extracting llm summary
"""

import argparse
import sys
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from src.keyphrase_config import KeyphraseConfig
from src.keyphrase import Keyphrase

def parse_args(args):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-file", type=str, default="cache.db")
    parser.add_argument(
        "--prompt-file", type=str, help="file with prompt for extracting concepts"
    )
    parser.add_argument(
        "--in-dataset-file",
        type=str,
        help="csv of the data we want to learn concepts for",
    )
    parser.add_argument("--indices-file", type=str, help="csv of training indices")
    parser.add_argument("--is-image", action="store_true", default=False)
    parser.add_argument(
        "--llm-outputs-file", type=str, help="csv file with llm concepts"
    )
    parser.add_argument("--log-file", type=str, default="_output/log_extract.txt")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="number of llm queries to run in a batch",
    )
    parser.add_argument(
        "--num-new-tokens",
        type=int,
        default=300,
        help="the number of new tokens to generate",
    )
    parser.add_argument(
        "--llm-model-type",
        type=str,
        help="LLM model type to use",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="sentence",
        help="column name containing text data",
    )
    parser.add_argument(
        "--prompt-placeholder",
        type=str,
        default="{note}",
        help="placeholder in prompt template to replace with text",
    )
    parser.add_argument(
        "--index-col",
        type=int,
        default=0,
        help="index column for CSV (None for no index)",
    )
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    load_dotenv()
    # Load data
    data_df = pd.read_csv(args.in_dataset_file, index_col=args.index_col, header=0)
    
    # Load training indices
    indices_df = pd.read_csv(args.indices_file, header=0)
    train_idxs = indices_df[indices_df.partition == "train"].idx.to_numpy()
    
    # Create config from args
    config = KeyphraseConfig.from_args(args)
    
    # Initialize and run keyphrase extractor
    extractor = Keyphrase(config)
    result_df = extractor.extract(data_df, train_idxs)
    for output_keyphrases in result_df.llm_output[result_df.llm_output != '']:
        print(output_keyphrases)
    
    print(f"Extraction complete. Non-empty outputs: {(result_df.llm_output != '').sum()}")

if __name__ == "__main__":
    main(sys.argv[1:])
