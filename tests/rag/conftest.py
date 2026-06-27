"""Fixtures for RAG tests — a real (CPU, deterministic) embedding function and
ephemeral Chroma collections seeded with known medical chunks.
"""

from __future__ import annotations

import pytest
from hrf_shared.contracts import Chunk

SEED_CHUNKS = [
    Chunk(
        chunk_id="c1",
        text="Aspirin lowers the risk of heart attack and stroke in high-risk patients.",
        source="cdc",
        url="https://cdc.gov/aspirin",
        article_id="a1",
        position=0,
    ),
    Chunk(
        chunk_id="c2",
        text="Vitamin D supports bone health and helps the body absorb calcium.",
        source="nhs",
        url="https://nhs.uk/vitamind",
        article_id="a2",
        position=0,
    ),
    Chunk(
        chunk_id="c3",
        text="The Mediterranean diet is associated with a lower risk of heart disease.",
        source="healthline",
        url="https://healthline.com/meddiet",
        article_id="a3",
        position=0,
    ),
    Chunk(
        chunk_id="c4",
        text="Vaccines train the immune system to recognize and fight pathogens.",
        source="medlineplus",
        url="https://medlineplus.gov/vaccines",
        article_id="a4",
        position=0,
    ),
    Chunk(
        chunk_id="c5",
        text="Regular physical exercise improves cardiovascular fitness and mental health.",
        source="webmd",
        url="https://webmd.com/exercise",
        article_id="a5",
        position=0,
    ),
]


@pytest.fixture(scope="session")
def embed_fn():
    from hrf_shared.chroma_client import get_embedding_function
    from hrf_shared.config import Settings

    return get_embedding_function(Settings())


@pytest.fixture
def empty_collection(embed_fn):
    import uuid

    import chromadb

    # EphemeralClient shares one in-memory system across calls, so a fixed
    # collection name would leak state between tests. Use a unique name.
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        f"test_{uuid.uuid4().hex}",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture
def seeded_collection(empty_collection):
    from hrf_rag.indexer import upsert_chunks

    upsert_chunks(empty_collection, SEED_CHUNKS)
    return empty_collection
