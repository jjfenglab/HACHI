"""
LLM client utilities for creating and managing LLM API instances.

This module is separate from common.py to avoid circular imports.
"""

import logging
import os
import sys

sys.path.append(os.getcwd())
sys.path.append("llm-api-main")

from lab_llm.constants import LLMModel
from lab_llm.duckdb_handler import DuckDBHandler
from lab_llm.constants import LLMModel
from lab_llm.error_callback_handler import ErrorCallbackHandler
from lab_llm.llm_api import LLMApi
from lab_llm.llm_cache import LLMCache

from src.ensemble_trainer.config import LLMConfig


def create_llm_clients(config: LLMConfig, logger=None) -> dict[str, LLMApi]:
    """
    Create LLM clients from a structured configuration object.
    """
    if logger is None:
        logger = logging.getLogger()

    cache = LLMCache(DuckDBHandler(config.cache_file))

    # Determine which model types to use
    # Priority: specific type > llm_model > default
    llm_model_type = config.llm_model_type or config.llm_model
    llm_iter_type = config.llm_iter_type
    llm_extraction_type = config.llm_extraction_type

    # Convert model names to types if needed
    if llm_model_type:
        llm_model_type = LLMModel(name=llm_model_type)
    if llm_iter_type:
        llm_model_type = LLMModel(name=llm_iter_type)
    if llm_extraction_type:
        llm_extraction_type = LLMModel(name=llm_extraction_type)
    # Create LLM clients based on configuration
    if llm_model_type and not (llm_iter_type or llm_extraction_type):
        # Single model for both iteration and extraction
        llm = LLMApi(
            cache,
            seed=10,
            model_type=llm_model_type,
            error_handler=ErrorCallbackHandler(logger),
            logging=logger,
            timeout=120,
        )
        return {"iter": llm, "extraction": llm}
    else:
        # Separate models for iteration and extraction
        llm_iter = LLMApi(
            cache,
            seed=10,
            model_type=llm_iter_type or llm_model_type,
            error_handler=ErrorCallbackHandler(logger),
            logging=logger,
            timeout=120,
        )
        llm_extraction = LLMApi(
            cache,
            seed=10,
            model_type=llm_extraction_type or llm_model_type,
            error_handler=ErrorCallbackHandler(logger),
            logging=logger,
            timeout=120,
        )
        return {"iter": llm_iter, "extraction": llm_extraction}


def load_llms(args, logger=None) -> dict[str, LLMApi]:
    """
    Legacy function for backward compatibility.

    New code should use create_llm_clients() with an LLMConfig object instead.
    """
    assert args.cache_file is not None

    # Convert args to LLMConfig object and delegate to new function
    config = LLMConfig(
        llm_model=getattr(args, "llm_model", "gpt-4o-mini"),
        cache_file=args.cache_file,
        use_api=getattr(args, "use_api", True),
        llm_model_type=getattr(args, "llm_model_type", None),
        llm_iter_type=getattr(args, "llm_iter_type", None),
        llm_extraction_type=getattr(args, "llm_extraction_type", None),
    )

    return create_llm_clients(config, logger)
