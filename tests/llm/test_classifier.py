"""TDD for hrf_llm.classifier — calling the model and parsing the verdict."""

import pytest
from hrf_llm.classifier import classify
from hrf_shared.config import Settings
from hrf_shared.contracts import Classification
from hrf_shared.json_utils import ClassificationParseError

SETTINGS = Settings()


def test_classify_returns_classification_on_clean_json(make_client, llm_fixture):
    client = make_client([llm_fixture("clean.json")])
    result = classify(["a chunk"], "is it real?", client=client, settings=SETTINGS)
    assert isinstance(result, Classification)
    assert result.label == "Fake"


def test_classify_handles_fenced_json(make_client, llm_fixture):
    client = make_client([llm_fixture("fenced.txt")])
    result = classify(["a chunk"], "is it real?", client=client, settings=SETTINGS)
    assert result.label == "Trustworthy"


def test_classify_sends_system_then_user_message(make_client, llm_fixture):
    client = make_client([llm_fixture("clean.json")])
    classify(["chunk-XYZ"], "QUESTION-ABC", client=client, settings=SETTINGS)
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "chunk-XYZ" in messages[1]["content"]
    assert "QUESTION-ABC" in messages[1]["content"]
    assert client.calls[0]["model"] == SETTINGS.llm_model


def test_classify_repairs_on_second_attempt(make_client, llm_fixture):
    client = make_client([llm_fixture("garbage.txt"), llm_fixture("clean.json")])
    result = classify(["a chunk"], "q?", client=client, settings=SETTINGS)
    assert result.label == "Fake"
    assert len(client.calls) == 2  # one repair retry was made


def test_classify_raises_after_repair_fails(make_client, llm_fixture):
    client = make_client([llm_fixture("garbage.txt"), llm_fixture("garbage.txt")])
    with pytest.raises(ClassificationParseError):
        classify(["a chunk"], "q?", client=client, settings=SETTINGS)
    assert len(client.calls) == 2
