"""Upsert chunks into a ChromaDB collection (shared by ingestion and tests)."""

from __future__ import annotations

from typing import Any

from hrf_shared.contracts import Chunk


def upsert_chunks(collection: Any, chunks: list[Chunk]) -> int:
    """Upsert chunks; returns the number written. Idempotent on chunk_id."""
    if not chunks:
        return 0
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "source": c.source,
                "url": c.url,
                "article_id": c.article_id,
                "position": c.position,
            }
            for c in chunks
        ],
    )
    return len(chunks)
