"""RAG service FastAPI app.

POST /find_similar_chunks  {prompt, top_n} -> {chunks: [ScoredChunk], query}
POST /index                {articles: [ArticleIn]} -> {indexed: n}
GET  /stats                knowledge-base counts (total + per source)
GET  /health               liveness + collection doc count
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from hrf_shared.chroma_client import get_chroma_client, get_or_create_collection
from hrf_shared.config import Settings, get_settings
from hrf_shared.contracts import (
    Article,
    ArticleIn,
    FindSimilarRequest,
    FindSimilarResponse,
)
from pydantic import BaseModel

from hrf_rag.retriever import find_similar_chunks
from ingest.indexing import index_articles

app = FastAPI(title="HeReFaNMi RAG Service")


@lru_cache
def _cached_collection() -> Any:
    settings = get_settings()
    client = get_chroma_client(settings)
    return get_or_create_collection(client, settings)


def get_collection() -> Any:
    """Dependency returning the Chroma collection (overridable in tests)."""
    return _cached_collection()


def settings_dep() -> Settings:
    return get_settings()


@app.get("/health")
def health(collection: Any = Depends(get_collection)) -> dict:
    try:
        count = collection.count()
    except Exception:  # noqa: BLE001 - health must never raise
        count = -1
    return {"status": "ok", "count": count}


@app.post("/find_similar_chunks", response_model=FindSimilarResponse)
def find_similar(
    body: FindSimilarRequest,
    collection: Any = Depends(get_collection),
    settings: Settings = Depends(settings_dep),
) -> FindSimilarResponse:
    if not body.prompt or not body.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    chunks = find_similar_chunks(body.prompt, body.top_n, collection=collection, settings=settings)
    return FindSimilarResponse(chunks=chunks, query=body.prompt)


class IndexRequest(BaseModel):
    articles: list[ArticleIn]


@app.post("/index")
def index(body: IndexRequest, collection: Any = Depends(get_collection)) -> dict:
    """Chunk + embed + upsert the given articles into the vector store."""
    articles = [
        Article(
            title=a.title,
            content=a.content,
            url=a.url,
            source=a.source,
            content_hash=a.article_id
            or hashlib.sha256(a.content.strip().encode("utf-8")).hexdigest(),
        )
        for a in body.articles
    ]
    indexed = index_articles(articles, collection)
    return {"indexed": indexed}


@app.get("/stats")
def stats(collection: Any = Depends(get_collection)) -> dict:
    """Knowledge-base stats: total chunks and a per-source breakdown."""
    try:
        total = collection.count()
        got = collection.get(include=["metadatas"])
        per_source: dict[str, int] = {}
        for meta in got.get("metadatas") or []:
            src = (meta or {}).get("source", "unknown")
            per_source[src] = per_source.get(src, 0) + 1
    except Exception:  # noqa: BLE001 - stats must not 500 the admin panel
        return {"chunks": 0, "sources": {}}
    return {"chunks": total, "sources": per_source}
