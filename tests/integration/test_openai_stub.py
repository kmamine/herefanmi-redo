"""Verify the OpenAI-compatible stub returns output our classifier can parse."""

import pytest
from fastapi.testclient import TestClient
from hrf_llm.stub import app
from hrf_shared.json_utils import parse_classification

pytestmark = pytest.mark.integration

tc = TestClient(app)


def _ask(text: str):
    resp = tc.post(
        "/v1/chat/completions",
        json={
            "model": "stub",
            "messages": [
                {"role": "system", "content": "classify"},
                {"role": "user", "content": text},
            ],
        },
    )
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    return parse_classification(content)


def test_stub_trustworthy():
    assert _ask("Vitamin D supports bone health.").label == "Trustworthy"


def test_stub_fake_for_miracle_claims():
    assert _ask("This miracle cure is 100% cure for cancer.").label == "Fake"


def test_stub_non_medical():
    assert _ask("Who won the football match?").medical == "False"
