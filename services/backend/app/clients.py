"""Async HTTP clients for the RAG and LLM services."""

from __future__ import annotations

import httpx
from hrf_shared.config import Settings
from hrf_shared.contracts import Classification, ScoredChunk


class RAGClient:
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.rag_url.rstrip("/")
        self.timeout = 30.0

    async def find_similar(self, prompt: str, top_n: int = 5) -> list[ScoredChunk]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/find_similar_chunks",
                json={"prompt": prompt, "top_n": top_n},
            )
            resp.raise_for_status()
            data = resp.json()
            return [ScoredChunk(**c) for c in data.get("chunks", [])]


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.llm_url.rstrip("/")
        self.timeout = settings.llm_timeout + 60.0

    async def predict(self, chunks: list[str], question: str) -> Classification:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/predict",
                json={"chunks": chunks, "question": question},
            )
            resp.raise_for_status()
            return Classification(**resp.json())
