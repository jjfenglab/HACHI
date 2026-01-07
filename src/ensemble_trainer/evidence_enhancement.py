"""
Evidence-span enhancement functionality for concept generation.

This module contains all logic related to enhancing concept generation with
clinical evidence from summaries, including semantic search and context extraction.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .semantic_search import CachedSemanticSearch


class SemanticCacheManager:
    """Manager for semantic search cache singleton."""

    _cache: Optional[CachedSemanticSearch] = None

    @classmethod
    def initialize_cache(cls, summaries_df: pd.DataFrame):
        """Initialize semantic search cache once with immediate model loading."""
        if cls._cache is None:
            logging.info("[CONCEPT GEN] Initializing semantic search cache...")
            cls._cache = CachedSemanticSearch(summaries_df)

            # Force immediate initialization to avoid lazy loading in parallel contexts
            logging.info(
                "[CONCEPT GEN] Force loading semantic model to avoid threading issues..."
            )
            cls._cache._lazy_init()

            if cls._cache._initialized:
                logging.info(
                    "[CONCEPT GEN] Semantic cache fully initialized and ready for use"
                )
            else:
                logging.warning(
                    "[CONCEPT GEN] Semantic cache initialization incomplete"
                )

    @classmethod
    def search(
        cls, query: str, max_results: int = 3, similarity_threshold: float = 0.4
    ) -> List[str]:
        """Search using the semantic cache."""
        if cls._cache and cls._cache._initialized:
            return cls._cache.search(query, max_results, similarity_threshold)
        return []

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if cache is initialized."""
        return cls._cache is not None and cls._cache._initialized


def load_evidence_mappings(evidence_file: str) -> Optional[Dict]:
    """Load evidence mappings from file."""
    if evidence_file and os.path.exists(evidence_file):
        try:
            with open(evidence_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"[CONCEPT GEN] Failed to load evidence mappings: {e}")
    return None


def extract_contexts_with_fallback(
    feature: str,
    summaries_df: pd.DataFrame,
    evidence_mappings: Optional[Dict] = None,
    keywords_column: str = "llm_output",
    summary_column: str = "llm_summary",
    max_examples: int = 3,
    use_semantic: bool = True,
) -> Tuple[List[str], str]:
    """
    Extract contexts using multiple strategies:
    1. Evidence spans (if available)
    2. Exact string matching
    3. Semantic similarity (if enabled)

    Args:
        feature: The feature/keyword to find contexts for
        summaries_df: DataFrame containing summaries and keywords
        evidence_mappings: Optional dict mapping sample indices to evidence
        keywords_column: Column name for keywords
        summary_column: Column name for summaries
        max_examples: Maximum number of examples to return
        use_semantic: Whether to use semantic search as fallback

    Returns:
        - List of context strings
        - Source of contexts ('evidence', 'exact', 'semantic', 'none')
    """
    logging.debug(f"[CONCEPT GEN] Extracting contexts for feature: '{feature}'")
    contexts = []

    # Strategy 1: Use evidence mappings
    if evidence_mappings:
        for idx, mapping in evidence_mappings.items():
            if isinstance(mapping, dict):
                # Check exact match
                if feature in mapping:
                    contexts.append(mapping[feature][:250])
                    logging.debug(
                        f"[CONCEPT GEN] Found evidence for '{feature}': {mapping[feature][:100]}..."
                    )
                # Check partial matches
                else:
                    for concept_key, evidence in mapping.items():
                        if (
                            feature.lower() in concept_key.lower()
                            or concept_key.lower() in feature.lower()
                        ):
                            contexts.append(evidence[:250])
                            logging.debug(
                                f"[CONCEPT GEN] Found partial evidence for '{feature}': {evidence[:100]}..."
                            )
                            break

            if len(contexts) >= max_examples:
                logging.debug(
                    f"[CONCEPT GEN] Found {len(contexts)} evidence contexts for '{feature}'"
                )
                return contexts[:max_examples], "evidence"

    # Strategy 2: Exact string matching
    mask = summaries_df[keywords_column].str.contains(feature, case=False, na=False)

    if mask.any():
        for _, row in summaries_df[mask].head(max_examples * 2).iterrows():
            summary = row[summary_column]
            if pd.notna(summary):
                sentences = summary.split(".")
                for sent in sentences:
                    if feature.lower() in sent.lower():
                        clean_sent = sent.strip()[:250]
                        if clean_sent and len(clean_sent) > 20:
                            contexts.append(clean_sent)
                            logging.debug(
                                f"[CONCEPT GEN] Found exact match for '{feature}': {clean_sent[:100]}..."
                            )
                            break

            if len(contexts) >= max_examples:
                logging.debug(
                    f"[CONCEPT GEN] Found {len(contexts)} exact contexts for '{feature}'"
                )
                return contexts[:max_examples], "exact"

    # Strategy 3: Semantic similarity search
    if use_semantic and SemanticCacheManager.is_initialized():
        semantic_contexts = SemanticCacheManager.search(
            feature, max_results=max_examples - len(contexts)
        )
        if semantic_contexts:
            contexts.extend(semantic_contexts)
            logging.debug(
                f"[CONCEPT GEN] Found {len(semantic_contexts)} semantic contexts for '{feature}'"
            )
            return contexts[:max_examples], "semantic"

    if contexts:
        return contexts[:max_examples], "partial"

    logging.debug(f"[CONCEPT GEN] No contexts found for feature: '{feature}'")
    return [], "none"


def enhance_concept_prompt_with_summaries(
    original_prompt: str,
    top_features_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    evidence_file: Optional[str] = None,
    mode: str = "baseline",
    use_semantic: bool = True,
) -> str:
    """
    Enhance concept generation prompt with clinical context.

    Uses evidence spans, exact matching, and semantic search.

    Args:
        original_prompt: The original prompt template
        top_features_df: DataFrame of top features from residual model
        summaries_df: DataFrame containing summaries
        evidence_file: Optional path to evidence mappings JSON
        mode: Enhancement mode - "baseline" or "iterative"
        use_semantic: Whether to enable semantic search fallback

    Returns:
        Enhanced prompt string
    """
    logging.info(f"[CONCEPT GEN] Enhancing {mode} prompt with summary context")
    logging.debug(
        f"[CONCEPT GEN] Original prompt length: {len(original_prompt)} characters"
    )

    # Load evidence mappings if available
    evidence_mappings = load_evidence_mappings(evidence_file)
    if evidence_mappings:
        logging.info(
            f"[CONCEPT GEN] Loaded evidence for {len(evidence_mappings)} samples"
        )

    # Initialize semantic search if needed and not already done
    if use_semantic and not SemanticCacheManager.is_initialized():
        SemanticCacheManager.initialize_cache(summaries_df)

    # Track extraction statistics
    extraction_stats = {
        "evidence": 0,
        "exact": 0,
        "semantic": 0,
        "partial": 0,
        "none": 0,
    }

    # Build context section
    context_section = "\n\n===== CLINICAL CONTEXT FOR TOP FEATURES =====\n"
    context_section += (
        "Here are specific clinical contexts where these predictive features appear:\n"
    )

    top_features = top_features_df.head(10)["feature_name"].tolist()
    logging.info(
        f"[CONCEPT GEN] Analyzing contexts for {len(top_features)} top features"
    )

    contexts_found = 0
    for i, feature in enumerate(top_features[:7], 1):
        contexts, source = extract_contexts_with_fallback(
            feature, summaries_df, evidence_mappings, use_semantic=use_semantic
        )

        extraction_stats[source] += 1

        if contexts:
            contexts_found += 1
            context_section += f"\n{i}. '{feature}' "
            if source not in ["evidence", "exact"]:
                context_section += f"(found via {source}) "
            context_section += "appears in:\n"

            for ctx in contexts[:2]:
                context_section += f"   • {ctx}\n"

    logging.info(f"[CONCEPT GEN] Context extraction stats: {extraction_stats}")
    logging.info(
        f"[CONCEPT GEN] Found contexts for {contexts_found}/{len(top_features[:7])} features"
    )

    enhanced_prompt = original_prompt

    # Insert context into prompt
    enhancement_successful = False
    if mode == "baseline":
        # For baseline, insert before "Answer with a list of"
        if "Answer with a list of" in enhanced_prompt:
            enhanced_prompt = enhanced_prompt.replace(
                "Answer with a list of",
                context_section
                + "\n\nBased on these specific clinical contexts (not just the keywords), answer with a list of",
            )
            enhancement_successful = True
            logging.info("[CONCEPT GEN] Successfully enhanced baseline prompt")
        else:
            logging.warning(
                "[CONCEPT GEN] Could not find 'Answer with a list of' anchor in baseline prompt"
            )
    else:  # iterative
        # For iterative, insert before "Given the residual model"
        if "Given the residual model," in enhanced_prompt:
            enhanced_prompt = enhanced_prompt.replace(
                "Given the residual model,",
                context_section
                + "\n\nGiven these specific clinical patterns and the residual model,",
            )
            enhancement_successful = True
            logging.info("[CONCEPT GEN] Successfully enhanced iterative prompt")
        else:
            logging.warning(
                "[CONCEPT GEN] Could not find 'Given the residual model,' anchor in iterative prompt"
            )

    if not enhancement_successful:
        logging.warning(
            "[CONCEPT GEN] Prompt enhancement may not have worked - adding context to end"
        )
        enhanced_prompt = enhanced_prompt + context_section

    logging.info(
        f"[CONCEPT GEN] Enhanced prompt length: {len(enhanced_prompt)} characters (added {len(enhanced_prompt) - len(original_prompt)} characters)"
    )

    return enhanced_prompt
