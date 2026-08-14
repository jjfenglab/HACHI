"""
LLM client utilities for creating and managing LLM API instances.

This module is separate from common.py to avoid circular imports, and must not
import from src.ensemble_trainer at runtime for the same reason.
"""

import base64
import logging
from pathlib import Path
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

    Uses llm_iter_type for concept proposal ("iter") and llm_extraction_type
    for concept extraction ("extraction") if specified, otherwise falls back
    to llm_model for both.
    """
    iter_model = config.llm_iter_type or config.llm_model
    extraction_model = config.llm_extraction_type or config.llm_model

    cache = CachingCompletion(config.cache_file)
    error_tracker = ErrorTracker(logger or logging.getLogger(__name__))

    iter_api = LLMApi(
        wrap_completion_function(
            _completion_for(iter_model),
            cache=cache,
            error_tracker=error_tracker,
            model=iter_model,
            seed=10,
            timeout=120,
            num_retries=1,
        )
    )

    if extraction_model == iter_model:
        extraction_api = iter_api
    else:
        extraction_api = LLMApi(
            wrap_completion_function(
                _completion_for(extraction_model),
                cache=cache,
                error_tracker=error_tracker,
                model=extraction_model,
                seed=10,
                timeout=120,
                num_retries=1,
            )
        )

    return {"iter": iter_api, "extraction": extraction_api}


def encode_image_data_uri(image_path: str) -> str:
    """Base64 data-URI for a local image file, media type inferred from the suffix."""
    suffix = Path(image_path).suffix.lower().lstrip(".")
    media_type = "jpeg" if suffix == "jpg" else suffix
    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    return f"data:image/{media_type};base64,{data}"


def _user_content(prompt: str, image_path: Optional[str]) -> Any:
    """OpenAI-style user content: plain string, or text + image content parts."""
    if image_path is None:
        return prompt
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": encode_image_data_uri(image_path)}},
    ]


async def run_prompts_batched(
    llm: LLMApi,
    prompts: List[str],
    response_format: type[BaseModel],
    batch_size: int,
    max_new_tokens: int,
    desc: str = "LLM batches",
    image_paths: Optional[List[str]] = None,
) -> List[Any]:
    """
    Run prompts through the LLM in chunks, returning one parsed response per prompt.

    If image_paths is given (one local path per prompt), each image is attached to
    its prompt as a base64 content part. Per-item failures (API errors, response
    validation failures) become None so that a single bad response does not lose
    the whole batch.
    """
    if image_paths is not None:
        assert len(image_paths) == len(prompts)
    messages_list = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(prompt, image_paths[i] if image_paths else None)},
        ]
        for i, prompt in enumerate(prompts)
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
