"""OpenAI-compatible client factory.

The provider is fully configured by ``base_url``/``model``/``api_key`` in
Settings, so the same code targets a local vLLM Gemma server, OpenAI, or any
other OpenAI-compatible endpoint.
"""

from __future__ import annotations

import openai
from hrf_shared.config import Settings


def get_llm_client(settings: Settings) -> openai.OpenAI:
    """Build an OpenAI client pointed at the configured provider."""
    return openai.OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
    )
