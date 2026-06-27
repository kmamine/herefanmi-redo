"""TDD for hrf_shared.json_utils — tolerant parsing of model JSON output."""

import json

import pytest
from hrf_shared.contracts import Classification
from hrf_shared.json_utils import (
    ClassificationParseError,
    parse_classification,
    strip_code_fences,
)

VALID = {
    "medical": "True",
    "news": "True",
    "label": "Trustworthy",
    "reasoning": "Backed by peer-reviewed studies.",
    "sources": ["https://www.cdc.gov/x"],
}


def _json(d: dict) -> str:
    return json.dumps(d)


def test_strip_plain_json_unchanged():
    raw = _json(VALID)
    assert json.loads(strip_code_fences(raw)) == VALID


def test_strip_json_fence():
    raw = f"```json\n{_json(VALID)}\n```"
    assert json.loads(strip_code_fences(raw)) == VALID


def test_strip_bare_fence():
    raw = f"```\n{_json(VALID)}\n```"
    assert json.loads(strip_code_fences(raw)) == VALID


def test_strip_with_surrounding_prose():
    raw = f"Sure! Here is the analysis:\n{_json(VALID)}\nHope that helps."
    assert json.loads(strip_code_fences(raw)) == VALID


def test_parse_classification_valid():
    result = parse_classification(_json(VALID))
    assert isinstance(result, Classification)
    assert result.label == "Trustworthy"
    assert result.medical == "True"
    assert result.sources == ["https://www.cdc.gov/x"]


def test_parse_classification_handles_fenced_json():
    raw = f"```json\n{_json(VALID)}\n```"
    assert parse_classification(raw).label == "Trustworthy"


def test_parse_classification_normalizes_label_case():
    # Legacy's own prompt examples emit lowercase "trustworthy" (index.py:255).
    d = {**VALID, "label": "trustworthy"}
    assert parse_classification(_json(d)).label == "Trustworthy"


def test_parse_classification_normalizes_truthflag_bool_and_case():
    d = {**VALID, "medical": True, "news": "false"}
    result = parse_classification(_json(d))
    assert result.medical == "True"
    assert result.news == "False"


def test_parse_classification_rejects_bad_label():
    d = {**VALID, "label": "Maybe"}
    with pytest.raises(ClassificationParseError):
        parse_classification(_json(d))


def test_parse_classification_garbage_raises():
    with pytest.raises(ClassificationParseError):
        parse_classification("the model refused to answer")
