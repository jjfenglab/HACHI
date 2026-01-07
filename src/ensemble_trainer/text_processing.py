"""
Text processing utilities for ensemble training.

This module contains text and NLP-related functions used for
concept generation and word vectorization in the ensemble trainer.
"""

import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def comma_tokenizer(text):
    """A token is comma-separated phrases."""
    comma_phrases = [z.strip().lower() for z in text.split(",")]
    return comma_phrases


def get_word_count_data(
    dset,
    count_vectorizer,
    text_summary_column: str = "llm_output",
    min_prevalence: float = 0,
    vectorizer_out_file=None,
):
    """
    Get basic word count vectorized data.
    
    Args:
        dset: DataFrame containing text data
        count_vectorizer: Type of vectorizer ('count' or 'tfidf')
        text_summary_column: Column name containing text to vectorize
        min_prevalence: Minimum prevalence threshold for features
        vectorizer_out_file: Optional file path to save vectorizer
        
    Returns:
        Tuple of (vectorized_sentences, word_names)
    """
    # Get basic word count vectorized data
    if count_vectorizer == "count":
        vectorizer = CountVectorizer(
            tokenizer=comma_tokenizer,
            binary=True,
            strip_accents="ascii",
            lowercase=True,
            token_pattern=None,  # silence warning about token_pattern
        )
    elif count_vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    else:
        raise NotImplementedError("vectorizer not recognized")
    vectorized_sentences = (
        vectorizer.fit_transform(dset[text_summary_column]).toarray().astype(int)
    )
    if vectorizer_out_file:
        with open(vectorizer_out_file, "wb") as f:
            pickle.dump(vectorizer, f)
    word_names = vectorizer.get_feature_names_out()
    logging.info(f"WORD FREQ MAX {vectorized_sentences.mean(axis=0).max()}")
    logging.info(f"WORD FREQ MIN {vectorized_sentences.mean(axis=0).min()}")
    logging.info(f"WORD FREQ MEAN {vectorized_sentences.mean(axis=0).mean()}")
    logging.info(f"WORD FREQ MEDIAN {np.median(vectorized_sentences.mean(axis=0))}")

    word_prevalences = vectorized_sentences.mean(axis=0)
    mask = word_prevalences >= min_prevalence
    vectorized_sentences = vectorized_sentences[:, mask]
    word_names = word_names[mask]
    logging.info(f"FINAL X SHAPE {vectorized_sentences.shape}")

    return vectorized_sentences, word_names
