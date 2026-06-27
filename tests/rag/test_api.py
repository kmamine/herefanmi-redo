"""TDD for the RAG FastAPI app — synchronous /find_similar_chunks + /health.

Anti-regression: legacy RAG returned a bare 200 OK and fired forward to the LLM.
This service must return the chunks to the caller.
"""

import pytest
from fastapi.testclient import TestClient
from hrf_rag.main import app, get_collection


@pytest.fixture
def tc(seeded_collection):
    app.dependency_overrides[get_collection] = lambda: seeded_collection
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_find_similar_returns_chunks_synchronously(tc):
    resp = tc.post("/find_similar_chunks", json={"prompt": "bone health", "top_n": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "bone health"
    assert len(body["chunks"]) >= 1
    first = body["chunks"][0]
    assert {"text", "score", "source", "url"} <= set(first)


def test_find_similar_defaults_top_n_5(tc):
    resp = tc.post("/find_similar_chunks", json={"prompt": "heart disease"})
    assert resp.status_code == 200
    assert len(resp.json()["chunks"]) <= 5


def test_find_similar_400_on_empty_prompt(tc):
    resp = tc.post("/find_similar_chunks", json={"prompt": "   "})
    assert resp.status_code == 400


def test_health_reports_collection_count(tc):
    resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json()["count"] == 5
