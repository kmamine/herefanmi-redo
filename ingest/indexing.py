"""Index stored Articles into a ChromaDB collection.

Shared by run_ingestion.py and seed_sample_data.py. Each article is chunked
with overlap and upserted; re-running is idempotent because chunk ids are
derived from the article id and position.
"""

from __future__ import annotations

from typing import Any

from hrf_rag.chunking import chunk_article
from hrf_rag.indexer import upsert_chunks
from hrf_shared.contracts import Article


def index_articles(
    articles: list[Article],
    collection: Any,
    *,
    max_chars: int = 800,
    overlap: int = 150,
) -> int:
    """Chunk and upsert all articles. Returns the total number of chunks."""
    total = 0
    for article in articles:
        chunks = chunk_article(article, max_chars=max_chars, overlap=overlap)
        total += upsert_chunks(collection, chunks)
    return total
