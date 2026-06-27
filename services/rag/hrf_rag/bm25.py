"""Sparse keyword ranking (BM25) over the chunk corpus.

Deterministic tokenization (lowercase + whitespace split) keeps results stable
and testable.
"""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, ids: list[str], documents: list[str]) -> None:
        self.ids = list(ids)
        self._tokenized = [tokenize(d) for d in documents]
        # BM25Okapi requires a non-empty corpus.
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def rank(self, query: str, top_n: int | None = None) -> list[str]:
        """Return chunk ids ordered by BM25 score (desc), ties broken by id."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(
            range(len(self.ids)),
            key=lambda i: (-scores[i], self.ids[i]),
        )
        ranked = [self.ids[i] for i in order]
        return ranked[:top_n] if top_n else ranked


def build_bm25_from_collection(collection: Any) -> BM25Index:
    """Build a BM25 index from all documents currently in the collection."""
    data = collection.get(include=["documents"])
    return BM25Index(ids=data["ids"], documents=data["documents"])
