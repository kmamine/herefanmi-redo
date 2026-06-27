"""TDD for hrf_shared.contracts — the cross-service Pydantic models."""

import pytest
from hrf_shared.contracts import (
    Article,
    Chunk,
    Classification,
    FindSimilarRequest,
    FindSimilarResponse,
    PredictRequest,
    ScoredChunk,
)
from pydantic import ValidationError

VALID = {
    "medical": "True",
    "news": "False",
    "label": "Doubtful",
    "reasoning": "Mixed evidence.",
    "sources": [],
}


def test_classification_requires_all_five_fields():
    for missing in ("medical", "news", "label", "reasoning"):
        d = {k: v for k, v in VALID.items() if k != missing}
        with pytest.raises(ValidationError):
            Classification(**d)


def test_classification_sources_defaults_empty():
    d = {k: v for k, v in VALID.items() if k != "sources"}
    assert Classification(**d).sources == []


def test_truthflag_only_true_false_strings():
    with pytest.raises(ValidationError):
        Classification(**{**VALID, "medical": "Yes"})


def test_label_must_be_canonical():
    with pytest.raises(ValidationError):
        Classification(**{**VALID, "label": "Real"})


def test_find_similar_request_defaults_top_n_5():
    assert FindSimilarRequest(prompt="aspirin").top_n == 5


def test_predict_request_fields():
    req = PredictRequest(chunks=["a", "b"], question="is it safe?")
    assert req.chunks == ["a", "b"]
    assert req.question == "is it safe?"


def test_scored_chunk_carries_source_and_url():
    sc = ScoredChunk(text="t", score=0.9, source="cdc", url="https://cdc.gov/x")
    assert sc.source == "cdc"
    assert sc.url == "https://cdc.gov/x"
    assert sc.score == 0.9


def test_find_similar_response_holds_scored_chunks():
    resp = FindSimilarResponse(
        chunks=[ScoredChunk(text="t", score=0.5, source="nhs", url="https://nhs.uk/x")],
        query="aspirin",
    )
    assert resp.query == "aspirin"
    assert resp.chunks[0].source == "nhs"


def test_article_and_chunk_models():
    art = Article(
        title="T",
        content="C" * 200,
        url="https://cdc.gov/x",
        source="cdc",
        content_hash="abc",
    )
    assert art.published_date is None
    chunk = Chunk(
        chunk_id="cdc:0",
        text="hello",
        source="cdc",
        url="https://cdc.gov/x",
        article_id="1",
        position=0,
    )
    assert chunk.position == 0
