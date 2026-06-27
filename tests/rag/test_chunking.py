"""TDD for hrf_rag.chunking — windowing article text into Chunks."""

from hrf_rag.chunking import chunk_article, chunk_text
from hrf_shared.contracts import Article, Chunk


def _article(content: str) -> Article:
    return Article(
        title="T",
        content=content,
        url="https://cdc.gov/x",
        source="cdc",
        content_hash="h",
    )


def test_chunk_respects_max_size():
    text = "word " * 500  # 2500 chars
    pieces = chunk_text(text, max_chars=200, overlap=40)
    assert all(len(p) <= 200 for p in pieces)
    assert len(pieces) > 1


def test_chunk_overlap_applied():
    text = "".join(chr(ord("a") + (i % 26)) for i in range(1000))
    pieces = chunk_text(text, max_chars=200, overlap=50)
    # The tail of one chunk equals the head of the next.
    assert pieces[0][-50:] == pieces[1][:50]


def test_short_text_single_chunk():
    pieces = chunk_text("just a short sentence", max_chars=800, overlap=100)
    assert len(pieces) == 1


def test_chunk_article_carries_metadata():
    art = _article("sentence number one. " * 100)
    chunks = chunk_article(art, article_id="42", max_chars=200, overlap=40)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source == "cdc"
    assert chunks[0].url == "https://cdc.gov/x"
    assert chunks[0].article_id == "42"
    assert chunks[0].position == 0
    assert chunks[1].position == 1
    # ids are unique and namespaced
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0].startswith("cdc:42:")


def test_chunk_article_defaults_article_id_to_hash():
    art = _article("body " * 100)
    chunks = chunk_article(art, max_chars=200, overlap=40)
    assert chunks[0].article_id  # populated from content_hash fallback
