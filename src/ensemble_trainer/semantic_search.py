"""
Semantic search module for enhanced concept context extraction.

This module provides cached semantic search functionality for finding
conceptually similar content when exact keyword matches fail.
"""

import logging
import os
import threading
import time
from typing import List, Optional

import numpy as np
import pandas as pd


class CachedSemanticSearch:
    """
    Pre-compute and cache embeddings for efficient semantic search.

    This class provides semantic similarity search capabilities for finding
    relevant clinical contexts when exact keyword matching fails.
    """

    def __init__(self, summaries_df: pd.DataFrame, summary_column: str = "llm_summary"):
        """
        Initialize and prepare for semantic search.

        Args:
            summaries_df: DataFrame containing clinical summaries
            summary_column: Column name containing the summary text
        """
        self.model = None
        self.sentence_embeddings = None
        self.sentences = []
        self.sentence_to_doc = []
        self.summary_column = summary_column
        self._initialized = False
        self._init_lock = threading.Lock()  # Thread safety for initialization

        # Store the summaries DataFrame for lazy initialization
        self.summaries_df = summaries_df
        logging.info(f"[SEMANTIC] Initialized with {len(summaries_df)} summaries")

    def _lazy_init(self):
        """Thread-safe lazy initialization of model and embeddings."""
        if self._initialized:
            return

        # Use lock to ensure thread safety
        with self._init_lock:
            # Double-check pattern in case another thread initialized while waiting
            if self._initialized:
                return

            start_time = time.time()

            # Set tokenizer parallelism to false to avoid forking issues
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logging.info(
                "[SEMANTIC] Set TOKENIZERS_PARALLELISM=false to avoid forking issues"
            )

            # Import here to avoid dependency if semantic search not used
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logging.error(
                    "[SEMANTIC] sentence-transformers not installed. Semantic search disabled."
                )
                logging.error(
                    "[SEMANTIC] Install with: pip install sentence-transformers"
                )
                return

            logging.info("[SEMANTIC] Initializing sentence transformer model...")
            try:
                # Use a lightweight, fast model suitable for clinical text
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logging.info("[SEMANTIC] SentenceTransformer model loaded successfully")
            except Exception as e:
                logging.error(
                    f"[SEMANTIC] Failed to initialize SentenceTransformer: {e}"
                )
                return

            # Extract all sentences from summaries
            logging.info("[SEMANTIC] Extracting sentences from summaries...")
            sentence_count = 0
            for doc_idx, row in self.summaries_df.iterrows():
                summary = row.get(self.summary_column, "")
                if pd.notna(summary) and summary:
                    # Split into sentences and filter out very short ones
                    sentences = summary.split(".")
                    for sent in sentences:
                        sent = sent.strip()
                        if len(sent) > 20:  # Skip very short fragments
                            self.sentences.append(sent)
                            self.sentence_to_doc.append(doc_idx)
                            sentence_count += 1

            # Batch encode all sentences for efficiency
            if self.sentences and self.model is not None:
                logging.info(f"[SEMANTIC] Encoding {len(self.sentences)} sentences...")
                try:
                    self.sentence_embeddings = self.model.encode(
                        self.sentences,
                        batch_size=64,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,  # Normalize for cosine similarity
                    )
                    logging.info(
                        f"[SEMANTIC] Successfully encoded {len(self.sentences)} sentence embeddings"
                    )
                except Exception as e:
                    logging.error(f"[SEMANTIC] Failed to encode sentences: {e}")
                    return

            elapsed = time.time() - start_time
            logging.info(
                f"[SEMANTIC] Pre-computed {len(self.sentences)} embeddings in {elapsed:.2f}s"
            )
            self._initialized = True

    def search(
        self, query: str, max_results: int = 3, similarity_threshold: float = 0.4
    ) -> List[str]:
        """
        Search for semantically similar sentences.

        Args:
            query: Search query (e.g., a concept or keyword)
            max_results: Maximum number of results to return
            similarity_threshold: Minimum cosine similarity score (0-1)

        Returns:
            List of matching sentence contexts, truncated to 250 chars each
        """
        # Initialize if needed
        self._lazy_init()

        if not self._initialized or not self.sentences:
            logging.debug("[SEMANTIC] Search failed - not initialized or no sentences")
            return []

        logging.debug(f"[SEMANTIC] Searching for: '{query}'")

        # Encode query
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

        # Compute cosine similarities (since embeddings are normalized, dot product = cosine similarity)
        similarities = np.dot(self.sentence_embeddings, query_embedding.T).flatten()

        # Get top matches above threshold
        top_indices = np.argsort(similarities)[::-1]

        results = []
        seen_snippets = set()  # Avoid near-duplicates

        for idx in top_indices:
            similarity_score = similarities[idx]

            # Stop if similarity too low
            if similarity_score < similarity_threshold:
                break

            # Stop if we have enough results
            if len(results) >= max_results:
                break

            sentence = self.sentences[idx]
            # Truncate for display
            truncated = sentence[:250]

            # Avoid near-duplicates by checking first 50 characters
            snippet_key = truncated[:50].lower()
            if snippet_key not in seen_snippets:
                results.append(truncated)
                seen_snippets.add(snippet_key)
                logging.debug(
                    f"[SEMANTIC] Found match (sim={similarity_score:.3f}): {truncated[:100]}..."
                )

        logging.debug(
            f"[SEMANTIC] Returned {len(results)} semantic matches for '{query}'"
        )
        return results

    def is_available(self) -> bool:
        """Check if semantic search is available (sentence-transformers installed)."""
        try:
            import importlib.util

            return importlib.util.find_spec("sentence_transformers") is not None
        except ImportError:
            return False

    def get_stats(self) -> dict:
        """Get statistics about the semantic search cache."""
        return {
            "initialized": self._initialized,
            "num_documents": len(self.summaries_df),
            "num_sentences": len(self.sentences) if self._initialized else 0,
            "model_loaded": self.model is not None,
            "available": self.is_available(),
        }
