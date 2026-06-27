"""TDD for hrf_rag.retriever — hybrid (dense + BM25 + RRF) retrieval."""

from hrf_rag.retriever import find_similar_chunks
from hrf_shared.config import Settings
from hrf_shared.contracts import ScoredChunk

SETTINGS = Settings(rrf_k=60, dense_candidates=20, default_top_n=5)


def test_returns_scored_chunks(seeded_collection):
    results = find_similar_chunks(
        "what helps bone health?", top_n=3, collection=seeded_collection, settings=SETTINGS
    )
    assert results
    assert all(isinstance(r, ScoredChunk) for r in results)


def test_top_n_respected(seeded_collection):
    results = find_similar_chunks("heart", top_n=2, collection=seeded_collection, settings=SETTINGS)
    assert len(results) <= 2


def test_known_query_returns_expected_chunk_first(seeded_collection):
    results = find_similar_chunks(
        "calcium absorption and bone strength",
        top_n=3,
        collection=seeded_collection,
        settings=SETTINGS,
    )
    # The vitamin D / bone-health chunk should rank first.
    assert results[0].url == "https://nhs.uk/vitamind"


def test_scored_chunks_carry_metadata(seeded_collection):
    results = find_similar_chunks(
        "aspirin heart attack", top_n=1, collection=seeded_collection, settings=SETTINGS
    )
    top = results[0]
    assert top.source in {"cdc", "nhs", "healthline", "medlineplus", "webmd"}
    assert top.url.startswith("https://")
    assert isinstance(top.score, float)


def test_empty_collection_returns_empty_list(empty_collection):
    results = find_similar_chunks(
        "anything", top_n=5, collection=empty_collection, settings=SETTINGS
    )
    assert results == []
