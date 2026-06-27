"""Hybrid retrieval: dense (Chroma) + sparse (BM25), fused with RRF."""

from __future__ import annotations

from typing import Any

from hrf_shared.config import Settings
from hrf_shared.contracts import ScoredChunk

from hrf_rag.bm25 import build_bm25_from_collection
from hrf_rag.hybrid import fuse


def find_similar_chunks(
    prompt: str,
    top_n: int,
    *,
    collection: Any,
    settings: Settings,
) -> list[ScoredChunk]:
    """Return the top_n most relevant chunks for the prompt (hybrid search)."""
    if not prompt or not prompt.strip():
        return []
    if collection.count() == 0:
        return []

    candidates = max(settings.dense_candidates, top_n)

    # Dense retrieval via Chroma's vector index.
    dense = collection.query(query_texts=[prompt], n_results=candidates)
    dense_ids: list[str] = dense["ids"][0]

    # Sparse retrieval via BM25 over the same corpus.
    bm25 = build_bm25_from_collection(collection)
    sparse_ids = bm25.rank(prompt, top_n=candidates)

    fused = fuse(dense_ids, sparse_ids, k=settings.rrf_k, top_n=top_n)
    if not fused:
        return []

    # Fetch documents/metadata for the winning ids and preserve fused order.
    fused_ids = [doc_id for doc_id, _ in fused]
    got = collection.get(ids=fused_ids, include=["documents", "metadatas"])
    by_id = {
        doc_id: (doc, meta)
        for doc_id, doc, meta in zip(got["ids"], got["documents"], got["metadatas"], strict=False)
    }

    results: list[ScoredChunk] = []
    for doc_id, score in fused:
        if doc_id not in by_id:
            continue
        doc, meta = by_id[doc_id]
        results.append(
            ScoredChunk(
                text=doc,
                score=float(score),
                source=(meta or {}).get("source", ""),
                url=(meta or {}).get("url", ""),
            )
        )
    return results
