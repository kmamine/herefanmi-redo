"""Fixtures and fakes for backend tests — no live RAG/LLM/network."""

from __future__ import annotations

import httpx
import pytest
from app.db import Repository, init_engine
from app.orchestrator import Orchestrator
from hrf_shared.config import Settings
from hrf_shared.contracts import ScoredChunk


@pytest.fixture
def settings():
    return Settings(
        jwt_secret="test-secret",
        admin_uid="admin",
        admin_emails="admin@e.com",
        signup_points=14,
        rag_url="http://rag.test",
        llm_url="http://llm.test",
        scraper_url="http://scraper.test",
    )


@pytest.fixture
def repo(tmp_path):
    engine = init_engine(f"sqlite:///{tmp_path / 'backend.sqlite'}")
    return Repository(engine)


class FakeRAG:
    def __init__(self, chunks: list[ScoredChunk] | None = None):
        self.chunks = chunks if chunks is not None else []

    async def find_similar(self, prompt: str, top_n: int = 5) -> list[ScoredChunk]:
        return self.chunks


class FakeLLM:
    def __init__(self, classification=None, error: Exception | None = None):
        self.classification = classification
        self.error = error
        self.calls: list[tuple] = []

    async def predict(self, chunks, question):
        self.calls.append((chunks, question))
        if self.error:
            raise self.error
        return self.classification


async def allow_all_validator(urls):
    return list(urls)


def chunk(text, source, url, score=0.5):
    return ScoredChunk(text=text, score=score, source=source, url=url)


def make_orchestrator(repo, rag, llm, settings, validator=allow_all_validator):
    return Orchestrator(
        rag_client=rag,
        llm_client=llm,
        repo=repo,
        source_validator=validator,
        settings=settings,
    )


@pytest.fixture
def llm_down_error():
    return httpx.ConnectError("vLLM unreachable")
