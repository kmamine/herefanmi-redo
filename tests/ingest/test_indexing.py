"""TDD for ingest.indexing — turning stored Articles into Chroma chunks."""

import uuid

import chromadb
import pytest
from hrf_shared.contracts import Article

from ingest.indexing import index_articles


@pytest.fixture(scope="module")
def embed_fn():
    from hrf_shared.chroma_client import get_embedding_function
    from hrf_shared.config import Settings

    return get_embedding_function(Settings())


@pytest.fixture
def collection(embed_fn):
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        f"ix_{uuid.uuid4().hex}",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _article(url, source, body):
    return Article(title="T", content=body, url=url, source=source, content_hash=f"hash-{url}")


def test_index_articles_chunks_and_upserts(collection):
    articles = [
        _article("https://cdc.gov/a", "cdc", "Aspirin and heart health. " * 50),
        _article("https://nhs.uk/b", "nhs", "Vitamin D and bones. " * 50),
    ]
    total = index_articles(articles, collection, max_chars=200, overlap=40)
    assert total > 2  # multiple chunks per article
    assert collection.count() == total


def test_reindex_is_idempotent(collection):
    articles = [_article("https://cdc.gov/a", "cdc", "stable body text. " * 50)]
    first = index_articles(articles, collection, max_chars=200, overlap=40)
    index_articles(articles, collection, max_chars=200, overlap=40)
    assert collection.count() == first


def test_index_empty_articles(collection):
    assert index_articles([], collection) == 0
    assert collection.count() == 0
