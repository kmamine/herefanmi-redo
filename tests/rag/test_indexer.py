"""TDD for hrf_rag.indexer — upserting chunks into a Chroma collection."""

from hrf_rag.indexer import upsert_chunks

from tests.rag.conftest import SEED_CHUNKS


def test_upsert_chunks_adds_documents(empty_collection):
    n = upsert_chunks(empty_collection, SEED_CHUNKS)
    assert n == len(SEED_CHUNKS)
    assert empty_collection.count() == len(SEED_CHUNKS)


def test_upsert_idempotent_on_same_chunk_id(empty_collection):
    upsert_chunks(empty_collection, SEED_CHUNKS)
    upsert_chunks(empty_collection, SEED_CHUNKS)  # same ids again
    assert empty_collection.count() == len(SEED_CHUNKS)


def test_upsert_empty_list_noop(empty_collection):
    assert upsert_chunks(empty_collection, []) == 0
    assert empty_collection.count() == 0


def test_embedding_dimension_is_384(seeded_collection):
    got = seeded_collection.get(include=["embeddings"])
    assert len(got["embeddings"][0]) == 384


def test_metadata_round_trips(seeded_collection):
    got = seeded_collection.get(ids=["c2"], include=["metadatas"])
    meta = got["metadatas"][0]
    assert meta["source"] == "nhs"
    assert meta["url"] == "https://nhs.uk/vitamind"
