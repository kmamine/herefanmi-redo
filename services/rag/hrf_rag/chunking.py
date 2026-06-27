"""Window article text into overlapping chunks for indexing."""

from __future__ import annotations

from hrf_shared.contracts import Article, Chunk


def chunk_text(text: str, *, max_chars: int = 800, overlap: int = 150) -> list[str]:
    """Split text into windows of ``max_chars`` with ``overlap`` shared chars.

    Consecutive chunks overlap exactly so retrieval doesn't miss content that
    straddles a boundary.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    step = max_chars - overlap
    pieces: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + max_chars]
        pieces.append(piece)
        if start + max_chars >= len(text):
            break
    return pieces


def chunk_article(
    article: Article,
    *,
    article_id: str | None = None,
    max_chars: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Produce indexed Chunks from an Article, carrying source/url metadata."""
    aid = article_id or article.content_hash[:16]
    pieces = chunk_text(article.content, max_chars=max_chars, overlap=overlap)
    return [
        Chunk(
            chunk_id=f"{article.source}:{aid}:{i}",
            text=piece,
            source=article.source,
            url=article.url,
            article_id=aid,
            position=i,
        )
        for i, piece in enumerate(pieces)
    ]
