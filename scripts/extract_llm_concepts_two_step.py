"""
Script for extracting LLM concepts using a two-step process:
1. Generate a hospital course summary from clinical notes
2. Extract key concepts/phrases from the summary
"""

import argparse
import asyncio
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

sys.path.append(os.getcwd())

import src.common as common
from src.llm_response_types import (
    ProtoConceptExtract,
    ProtoConceptExtractWithEvidence,
)

sys.path.append("llm-api-main")
from lab_llm.constants import convert_to_llm_type
from lab_llm.dataset import TextDataset
from lab_llm.duckdb_handler import DuckDBHandler
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.llm_api import LLMApi
from lab_llm.llm_cache import LLMCache


def parse_args(args):
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-file", type=str, default="cache.db")
    parser.add_argument(
        "--config-file",
        type=str,
        help="file with json config for replacing strings in template",
    )
    parser.add_argument(
        "--prompt-extract-file",
        type=str,
        help="file with prompt for extracting concepts",
    )
    parser.add_argument(
        "--prompt-summary-file", type=str, help="file with prompt for summary"
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
    parser.add_argument(
        "--llm-summaries-file",
        type=str,
        help="optional csv file to save intermediate summaries",
        default=None,
    )
    parser.add_argument("--log-file", type=str, default="_output/log_extract.txt")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="number of llm queries to run in a batch",
    )
    parser.add_argument(
        "--batch-obs-size",
        type=int,
        default=1,
        help="number of observations to batch together annotation for",
    )
    parser.add_argument(
        "--num-new-tokens",
        type=int,
        default=300,
        help="the number of new tokens to generate",
    )
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--max-section-length", type=int, default=None)
    parser.add_argument(
        "--llm-model-type",
        type=str,
        default="versa-gpt-4o-2024-05-13",
        choices=[
            "versa-gpt-4o-2024-05-13",
            "gpt-4o-mini",
            "gpt-4o-2024-08-06",
            "versa-gpt-4o-2024-08-06",
            "versa-gpt-4o-mini-2024-07-18",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
        ],
    )
    parser.add_argument(
        "--use-evidence",
        action="store_true",
        default=False,
        help="Use evidence-aware extraction with ProtoConceptExtractWithEvidence model",
    )
    parser.add_argument(
        "--partition",
        type=str,
        choices=["train", "test", "all"],
        default="train",
        help="Which partition to process: train, test, or all data",
    )
    args = parser.parse_args()
    assert args.use_api
    return args


def process_evidence_outputs(llm_outputs, missing_idxs):
    """
    Process outputs from evidence-aware extraction.

    Args:
        llm_outputs: List of ProtoConceptExtractWithEvidence objects or None
        missing_idxs: Array of indices that were processed

    Returns:
        - List of keyword strings for backward compatibility
        - Dictionary of evidence mappings
    """
    llm_output_strs = [""] * len(missing_idxs)
    evidence_mappings = {}

    for idx, output in enumerate(llm_outputs):
        if output is not None and hasattr(output, "concepts"):
            # Extract all concepts as comma-separated string for backward compatibility
            concepts_list = []
            evidence_map = {}

            # Process concepts from the list
            for concept_data in output.concepts:
                concepts_list.append(concept_data.concept)

                # Map each phrase variant to the evidence
                for phrase in concept_data.concept.split(","):
                    phrase = phrase.strip()
                    if phrase:
                        evidence_map[phrase] = concept_data.evidence

            llm_output_strs[idx] = ",".join(concepts_list)
            evidence_mappings[str(missing_idxs[idx])] = evidence_map
        else:
            llm_output_strs[idx] = ""
            evidence_mappings[str(missing_idxs[idx])] = {}

    return llm_output_strs, evidence_mappings


def main(args):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.INFO
    )
    logging.info(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_df = pd.read_csv(args.in_dataset_file)
    data_df["llm_output"] = ""

    missing_idxs = np.arange(data_df.shape[0])
    logging.info("missing raw %s", missing_idxs.size)
    if args.indices_file is not None and args.partition != "all":
        indices_df = pd.read_csv(args.indices_file, header=0)
        partition_idxs = indices_df[
            indices_df.partition == args.partition
        ].idx.to_numpy()
        missing_idxs = np.intersect1d(missing_idxs, partition_idxs)
    logging.info("missing final %s %d", missing_idxs[:10], missing_idxs.size)
    print("missing", missing_idxs.size)

    new_data_df = data_df.iloc[missing_idxs]

    cur_dir = os.getcwd()
    prompt_extract_file = os.path.abspath(
        os.path.join(cur_dir, args.prompt_extract_file)
    )
    with open(prompt_extract_file, "r") as file:
        prompt_extract_template = file.read()
    prompt_summary_file = os.path.abspath(
        os.path.join(cur_dir, args.prompt_summary_file)
    )
    with open(prompt_summary_file, "r") as file:
        prompt_summary_template = file.read()

    # Setup LLM cache + API
    logger = logging.getLogger(__name__)
    load_dotenv()
    args.cache = LLMCache(DuckDBHandler(args.cache_file))
    print(args.llm_model_type)
    args.llm_model_type = convert_to_llm_type(args.llm_model_type)
    llm = LLMApi(
        args.cache,
        seed=10,
        model_type=args.llm_model_type,
        error_handler=ErrorCallbackHandler(logger),
        logging=logger,
    )

    group_ids, sentences = common.split_sentences_by_id(
        new_data_df, args.max_section_length
    )
    print("GROUP IDS", group_ids, len(new_data_df), len(group_ids))
    # First get summaries
    summary_prompts = []
    if args.max_section_length and not args.is_image:
        # Use the split sentences if max_section_length is specified
        summary_prompts = [
            prompt_summary_template.replace("{note}", s) for s in sentences
        ]
    else:
        # Otherwise use the full text from each row
        for idx, row in new_data_df.iterrows():
            tmp = prompt_summary_template
            tmp = tmp.replace("{note}", row.sentence)
            summary_prompts.append(tmp)

    logging.info(f"prompt: {summary_prompts[0]}")
    summary_dataset = TextDataset(
        summary_prompts,
    )
    # This should be a list of strings since we're not using a response model
    summary_llm_outputs = asyncio.run(
        llm.get_outputs(
            summary_dataset,
            max_new_tokens=args.num_new_tokens,
            batch_size=args.batch_size,
            is_image=args.is_image,
        )
    )

    logging.info(f"Generated {len(summary_llm_outputs)} summaries")

    summary_df = new_data_df.copy()
    if args.max_section_length and not args.is_image:
        # Handle grouped summaries
        summary_strs = [""] * len(new_data_df)
        for grp_id in np.unique(group_ids):
            match_idxs = np.where(group_ids == grp_id)[0]
            grp_summary = " ".join(
                [
                    summary_llm_outputs[idx]
                    for idx in match_idxs
                    if idx < len(summary_llm_outputs) and summary_llm_outputs[idx]
                ]
            )
            summary_strs[grp_id] = grp_summary
        summary_df["llm_summary"] = summary_strs
    else:
        summary_df["llm_summary"] = summary_llm_outputs
    summary_df.to_csv(args.llm_summaries_file, index=False)
    logging.info(f"Saved summaries to {args.llm_summaries_file}")

    # Also save a cleaner version for easier review (without llm_output column if it exists)
    clean_summary_df = summary_df.copy()
    if "sentence" in clean_summary_df.columns:
        clean_summary_df = clean_summary_df.drop(columns=["sentence"])
    clean_summaries_file = os.path.splitext(args.llm_summaries_file)[0] + "_clean.csv"
    clean_summary_df.to_csv(clean_summaries_file, index=False)
    logging.info(f"Saved clean summaries (for review) to {clean_summaries_file}")

    # create the prompt templates for the extract step
    extract_prompts = []
    for i, summary in enumerate(summary_llm_outputs):
        if summary is None or summary.strip() == "":
            logging.warning(f"Empty summary at index {i}, using empty string")
            summary = ""
        tmp = prompt_extract_template
        tmp = tmp.replace("{note}", summary)
        extract_prompts.append(tmp)

    extract_dataset = TextDataset(
        extract_prompts,
    )

    if args.use_evidence:
        response_model = ProtoConceptExtractWithEvidence
    else:
        response_model = ProtoConceptExtract

    llm_outputs = asyncio.run(
        llm.get_outputs(
            extract_dataset,
            max_new_tokens=args.num_new_tokens,
            batch_size=args.batch_size,
            is_image=args.is_image,
            response_model=response_model,
        )
    )

    # Regular extractions (with or without evidence)
    if args.use_evidence:
        # Use evidence processing function
        llm_output_strs, evidence_mappings = process_evidence_outputs(
            llm_outputs, missing_idxs
        )

        # Save evidence mappings to JSON file
        evidence_file = args.llm_outputs_file.replace(".csv", "_evidence.json")
        import json

        with open(evidence_file, "w") as f:
            json.dump(evidence_mappings, f, indent=2)
        logging.info(f"Saved evidence mappings to {evidence_file}")
        print(f"Saved evidence mappings to {evidence_file}")

    else:
        # Original logic for regular extractions
        # Handle the grouping logic if max_section_length was used
        if args.max_section_length and not args.is_image:
            llm_output_strs = [""] * len(missing_idxs)
            # Create a map from original_index (grp_id) to position_in_missing_idxs
            original_idx_to_position_map = {
                orig_idx: pos for pos, orig_idx in enumerate(missing_idxs)
            }

            for grp_id in np.unique(group_ids):
                match_idxs = np.where(group_ids == grp_id)[0]
                grp_llm_output = ",".join(
                    [
                        ",".join(llm_outputs[match_idx].keyphrases)
                        for match_idx in match_idxs
                        if match_idx < len(llm_outputs)
                        and llm_outputs[match_idx] is not None
                    ]
                )
                # Get the target position in llm_output_strs using the map
                target_pos = original_idx_to_position_map[grp_id]
                llm_output_strs[target_pos] = grp_llm_output
        else:
            # Simple case: one-to-one mapping
            llm_output_strs = [""] * len(missing_idxs)
            for idx, output in enumerate(llm_outputs):
                if output is not None:
                    llm_output_strs[idx] = ",".join(output.keyphrases)
                else:
                    logging.warning(
                        f"LLM output is None for index {idx}, using empty string"
                    )
                    llm_output_strs[idx] = ""

    # Actually update the dataframe with the extracted concepts
    data_df.loc[data_df.index[missing_idxs], "llm_output"] = llm_output_strs

    print("not na", (data_df.llm_output != "").sum())
    data_df.to_csv(args.llm_outputs_file, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
