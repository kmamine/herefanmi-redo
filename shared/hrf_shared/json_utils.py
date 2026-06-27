"""Tolerant parsing of LLM JSON output.

Models served over the OpenAI-compatible API frequently wrap JSON in ```json
fences or surround it with prose. These helpers extract and validate the
classification object, hardening the fence-stripping originally seen in
``dataset-regenerate/regenrate.py``.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from hrf_shared.contracts import Classification


class ClassificationParseError(ValueError):
    """Raised when model output cannot be parsed into a Classification."""


def strip_code_fences(text: str) -> str:
    """Return the most likely JSON substring from a raw model response.

    Handles ```json fences, bare ``` fences, and leading/trailing prose by
    falling back to the span between the first '{' and the last '}'.
    """
    s = text.strip()

    if "```" in s:
        # Take the content of the first fenced block.
        after = s.split("```", 1)[1]
        if after.lower().startswith("json"):
            after = after[4:]
        block = after.split("```", 1)[0]
        s = block.strip()

    # Fall back to the outermost JSON object if prose still surrounds it.
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]

    return s.strip()


def parse_classification(raw: str) -> Classification:
    """Parse a raw model response into a validated :class:`Classification`."""
    cleaned = strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ClassificationParseError(f"Model output was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ClassificationParseError("Model output JSON was not an object")
    try:
        return Classification(**data)
    except ValidationError as exc:
        raise ClassificationParseError(
            f"Model output did not match the classification contract: {exc}"
        ) from exc
