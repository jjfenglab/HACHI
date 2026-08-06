"""
LLM client utilities for creating and managing LLM API instances.

This module is separate from common.py to avoid circular imports, and must not
import from src.ensemble_trainer at runtime for the same reason.
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

import litellm
from lab_llm import (
    CachingCompletion,
    CompletionFunction,
    ErrorTracker,
    LLMApi,
    wrap_completion_function,
)
from lab_llm.versa import make_versa_claude_completion, make_versa_openai_completion
from pydantic import BaseModel
from tqdm import tqdm

if TYPE_CHECKING:
    from src.ensemble_trainer.config import LLMConfig

SYSTEM_PROMPT = "You are a helpful assistant"


def _completion_for(model: str) -> CompletionFunction:
    """Pick the provider completion function implied by the model prefix."""
    if model.startswith("azure/"):
        return make_versa_openai_completion()
    if model.startswith("bedrock/"):
        return make_versa_claude_completion()
    return litellm.completion


def create_llm_clients(
    config: "LLMConfig", logger: Optional[logging.Logger] = None
) -> dict[str, LLMApi]:
    """
    Create LLM clients from a structured configuration object.
    """
    api = LLMApi(
        wrap_completion_function(
            _completion_for(config.llm_model),
            cache=CachingCompletion(config.cache_file),
            error_tracker=ErrorTracker(logger or logging.getLogger(__name__)),
            model=config.llm_model,
            seed=10,
            timeout=120,
            num_retries=2,
        )
    )
    return {"iter": api, "extraction": api}


async def run_prompts_batched(
    llm: LLMApi,
    prompts: List[str],
    response_format: type[BaseModel],
    batch_size: int,
    max_new_tokens: int,
    desc: str = "LLM batches",
) -> List[Any]:
    """
    Run prompts through the LLM in chunks, returning one parsed response per prompt.

    Per-item failures (API errors, response validation failures) become None so
    that a single bad response does not lose the whole batch.
    """
    messages_list = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    outputs = []
    for start in tqdm(range(0, len(messages_list), batch_size), desc=desc):
        results = await llm.run_batch(
            messages_list[start : start + batch_size],
            max_parallel_jobs=batch_size,
            max_tokens=max_new_tokens,
            temperature=0,
            response_format=response_format,
            strict_response_format=True,
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logging.warning("LLM call failed, dropping response: %s", result)
                outputs.append(None)
            else:
                outputs.append(result)
    return outputs
