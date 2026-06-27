"""Call the model and parse its output into a Classification.

A single repair retry is attempted when the first response is not parseable
JSON, before giving up with a ClassificationParseError.
"""

from __future__ import annotations

from hrf_shared.config import Settings
from hrf_shared.contracts import Classification
from hrf_shared.json_utils import ClassificationParseError, parse_classification

from hrf_llm.prompt import REPAIR_INSTRUCTION, SYSTEM_PROMPT, build_prompt


def _complete(client, settings: Settings, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return response.choices[0].message.content or ""


def classify(
    chunks: list[str],
    question: str,
    *,
    client,
    settings: Settings,
) -> Classification:
    """Classify a statement/question given retrieved reference chunks."""
    user_message = build_prompt(chunks, question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    raw = _complete(client, settings, messages)
    try:
        return parse_classification(raw)
    except ClassificationParseError:
        # One repair attempt: tell the model to return JSON only.
        repair_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": REPAIR_INSTRUCTION},
        ]
        repaired = _complete(client, settings, repair_messages)
        return parse_classification(repaired)
