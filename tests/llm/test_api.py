"""TDD for the LLM FastAPI app — /predict and /health."""

import pytest
from fastapi.testclient import TestClient
from hrf_llm.main import app, get_client


@pytest.fixture
def client_factory(make_client):
    """Build a TestClient whose LLM client is overridden with a fake."""

    def _build(responses: list[str]) -> TestClient:
        fake = make_client(responses)
        app.dependency_overrides[get_client] = lambda: fake
        tc = TestClient(app)
        tc.fake = fake  # type: ignore[attr-defined]
        return tc

    yield _build
    app.dependency_overrides.clear()


def test_predict_returns_200_and_classification(client_factory, llm_fixture):
    tc = client_factory([llm_fixture("clean.json")])
    resp = tc.post("/predict", json={"chunks": ["c"], "question": "q?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "Fake"
    assert body["medical"] == "True"
    assert set(body) >= {"medical", "news", "label", "reasoning", "sources"}


def test_predict_422_on_missing_chunks(client_factory, llm_fixture):
    tc = client_factory([llm_fixture("clean.json")])
    resp = tc.post("/predict", json={"question": "q?"})
    assert resp.status_code == 422


def test_predict_422_on_missing_question(client_factory, llm_fixture):
    tc = client_factory([llm_fixture("clean.json")])
    resp = tc.post("/predict", json={"chunks": ["c"]})
    assert resp.status_code == 422


def test_predict_502_on_unparseable_output(client_factory, llm_fixture):
    tc = client_factory([llm_fixture("garbage.txt"), llm_fixture("garbage.txt")])
    resp = tc.post("/predict", json={"chunks": ["c"], "question": "q?"})
    assert resp.status_code == 502


def test_health_does_not_call_model(client_factory, llm_fixture):
    tc = client_factory([llm_fixture("clean.json")])
    resp = tc.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert tc.fake.calls == []  # model never invoked
