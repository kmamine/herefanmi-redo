"""TDD for hrf_llm.prompt — system prompt + user-message assembly."""

from hrf_llm.prompt import SYSTEM_PROMPT, build_prompt


def test_system_prompt_contains_label_definitions():
    for label in ("Trustworthy", "Doubtful", "Fake"):
        assert label in SYSTEM_PROMPT


def test_system_prompt_describes_credibility_criteria():
    low = SYSTEM_PROMPT.lower()
    assert "peer" in low  # Trustworthy = peer-reviewed
    assert "experimental" in low  # Doubtful = experimental/mixed


def test_system_prompt_demands_json_only():
    low = SYSTEM_PROMPT.lower()
    assert "json" in low
    for field in ("medical", "news", "label", "reasoning", "sources"):
        assert field in low


def test_build_prompt_joins_chunks_and_question():
    msg = build_prompt(["chunk one", "chunk two"], "Is aspirin safe?")
    assert "chunk one" in msg
    assert "chunk two" in msg
    assert "Is aspirin safe?" in msg


def test_build_prompt_handles_no_chunks():
    msg = build_prompt([], "What is cancer?")
    assert "What is cancer?" in msg
