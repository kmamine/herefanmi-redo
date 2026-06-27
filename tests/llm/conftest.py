"""Fakes for the LLM service tests — a stand-in for the OpenAI client so no
network or live model is ever required.
"""

from __future__ import annotations

import pytest


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            content = self._responses.pop(0)
        else:  # repeat the last canned response if over-called
            content = self.calls and "{}"
        return _Response(content)


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class FakeOpenAI:
    """Mimics the subset of openai.OpenAI used by the classifier."""

    def __init__(self, responses: list[str]) -> None:
        self.chat = _Chat(_Completions(responses))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


@pytest.fixture
def make_client():
    """Factory: make_client(["<raw model text>", ...]) -> FakeOpenAI."""

    def _make(responses: list[str]) -> FakeOpenAI:
        return FakeOpenAI(responses)

    return _make
