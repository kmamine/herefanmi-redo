"""Reciprocal Rank Fusion of dense and sparse rankings."""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Fuse multiple ranked id-lists into a combined score per id.

    RRF score for an id is the sum over rankings of 1 / (k + rank), with rank
    1-based. Ids appearing high across multiple rankings score highest.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse(
    dense_ids: list[str],
    sparse_ids: list[str],
    *,
    k: int = 60,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Fuse dense + sparse rankings and return the top_n (id, score) pairs.

    Ordered by fused score (desc); ties broken by id (asc) for determinism.
    """
    scores = reciprocal_rank_fusion([dense_ids, sparse_ids], k=k)
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top_n]
