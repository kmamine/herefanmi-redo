"""HTTP gateway the backend uses to drive the scraper + read RAG stats.

Isolated from FastAPI so it can be faked in tests via dependency override.
"""

from __future__ import annotations

import httpx
from hrf_shared.config import Settings


class AdminGateway:
    def __init__(self, settings: Settings) -> None:
        self.scraper = settings.scraper_url.rstrip("/")
        self.rag = settings.rag_url.rstrip("/")
        self.timeout = 60.0

    async def _json(self, method: str, url: str, **kw):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.request(method, url, **kw)
            resp.raise_for_status()
            return resp.json()

    async def list_sources(self) -> list[dict]:
        data = await self._json("GET", f"{self.scraper}/sources")
        return data.get("sources", [])

    async def create_source(self, body: dict) -> dict:
        return await self._json("POST", f"{self.scraper}/sources", json=body)

    async def update_source(self, name: str, body: dict) -> dict:
        return await self._json("PATCH", f"{self.scraper}/sources/{name}", json=body)

    async def delete_source(self, name: str) -> dict:
        return await self._json("DELETE", f"{self.scraper}/sources/{name}")

    async def run_scrape(self, sources: list[str] | None) -> dict:
        return await self._json("POST", f"{self.scraper}/ingest", json={"sources": sources})

    async def kb_stats(self) -> dict:
        try:
            return await self._json("GET", f"{self.rag}/stats")
        except httpx.HTTPError:
            return {"chunks": 0, "sources": {}}
