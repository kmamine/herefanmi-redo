"""TDD for the Backend FastAPI app — auth + /medicalTalk + /save + points."""

import pytest
from app.db import Repository, init_engine
from app.main import (
    app,
    get_llm_client,
    get_rag_client,
    get_repo,
    get_settings_dep,
    get_source_validator,
)
from fastapi.testclient import TestClient
from hrf_shared.contracts import Classification

from tests.backend.conftest import FakeLLM, FakeRAG, allow_all_validator, chunk

TRUSTWORTHY = Classification(
    medical="True",
    news="True",
    label="Trustworthy",
    reasoning="Backed by clinical evidence.",
    sources=[],
)
CHUNKS = [chunk("Aspirin reduces heart attack risk.", "cdc", "https://cdc.gov/aspirin")]


@pytest.fixture
def tc(tmp_path, settings):
    repo = Repository(init_engine(f"sqlite:///{tmp_path / 'api.sqlite'}"))
    app.dependency_overrides[get_settings_dep] = lambda: settings
    app.dependency_overrides[get_repo] = lambda: repo
    app.dependency_overrides[get_rag_client] = lambda: FakeRAG(CHUNKS)
    app.dependency_overrides[get_llm_client] = lambda: FakeLLM(TRUSTWORTHY)
    app.dependency_overrides[get_source_validator] = lambda: allow_all_validator
    client = TestClient(app)
    client.repo = repo  # type: ignore[attr-defined]
    yield client
    app.dependency_overrides.clear()


def _signup(tc, email="u@e.com", password="pw123456"):
    resp = tc.post("/auth/signup", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_health(tc):
    assert tc.get("/health").status_code == 200


def test_signup_then_login(tc):
    data = _signup(tc)
    assert data["points"] == 14
    assert data["access_token"]
    login = tc.post("/auth/login", json={"email": "u@e.com", "password": "pw123456"})
    assert login.status_code == 200
    assert login.json()["access_token"]


def test_login_wrong_password(tc):
    _signup(tc)
    resp = tc.post("/auth/login", json={"email": "u@e.com", "password": "nope"})
    assert resp.status_code == 401


def test_medicaltalk_requires_auth(tc):
    resp = tc.post("/medicalTalk", json={"data": "q", "opinion": "0", "backend": "x"})
    assert resp.status_code == 401


def test_medicaltalk_returns_classification(tc):
    token = _signup(tc)["access_token"]
    resp = tc.post(
        "/medicalTalk",
        json={"data": "is aspirin safe?", "opinion": "3", "backend": "HeReFaNMi LLM"},
        headers=_auth(token),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["label"] == "Trustworthy"
    assert body["source"] == ["https://cdc.gov/aspirin"]
    assert "key" in body


def test_save_rating(tc):
    token = _signup(tc)["access_token"]
    key = tc.post(
        "/medicalTalk",
        json={"data": "q", "opinion": "1", "backend": "x"},
        headers=_auth(token),
    ).json()["key"]
    resp = tc.post("/save", json={"reference": key, "rating": "5"}, headers=_auth(token))
    assert resp.status_code == 200
    assert tc.repo.get_query(key).rating == "5"


def test_pointcheck_reflects_decrement(tc):
    token = _signup(tc)["access_token"]
    tc.post(
        "/medicalTalk", json={"data": "q", "opinion": "0", "backend": "x"}, headers=_auth(token)
    )
    resp = tc.post("/pointcheck", json={}, headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["points"] == 13


def test_out_of_points_returns_403(tc):
    data = _signup(tc)
    token = data["access_token"]
    tc.repo.set_points(data["uid"], 0)
    resp = tc.post(
        "/medicalTalk", json={"data": "q", "opinion": "0", "backend": "x"}, headers=_auth(token)
    )
    assert resp.status_code == 403
