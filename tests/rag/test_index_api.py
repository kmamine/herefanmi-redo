"""TDD for RAG POST /index and GET /stats (admin knowledge-base view)."""

import pytest
from fastapi.testclient import TestClient
from hrf_rag.main import app, get_collection


@pytest.fixture
def tc(empty_collection):
    app.dependency_overrides[get_collection] = lambda: empty_collection
    client = TestClient(app)
    client.collection = empty_collection  # type: ignore[attr-defined]
    yield client
    app.dependency_overrides.clear()


ARTICLES = [
    {
        "title": "Aspirin",
        "content": "Aspirin reduces the risk of heart attack and stroke. " * 8,
        "url": "https://cdc.gov/aspirin",
        "source": "cdc",
    },
    {
        "title": "Vitamin D",
        "content": "Vitamin D supports bone health and calcium absorption. " * 8,
        "url": "https://nhs.uk/vitamind",
        "source": "nhs",
    },
]


def test_index_adds_chunks(tc):
    resp = tc.post("/index", json={"articles": ARTICLES})
    assert resp.status_code == 200
    assert resp.json()["indexed"] >= 2
    assert tc.collection.count() >= 2


def test_index_empty_list(tc):
    resp = tc.post("/index", json={"articles": []})
    assert resp.status_code == 200
    assert resp.json()["indexed"] == 0


def test_stats_reports_counts(tc):
    tc.post("/index", json={"articles": ARTICLES})
    resp = tc.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks"] >= 2
    assert set(body["sources"]) == {"cdc", "nhs"}
