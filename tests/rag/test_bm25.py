"""TDD for hrf_rag.bm25 — sparse keyword ranking over the chunk corpus."""

from hrf_rag.bm25 import BM25Index, tokenize


def test_tokenize_is_deterministic_and_lowercase():
    assert tokenize("Vitamin D Supports BONES") == tokenize("vitamin d supports bones")
    assert tokenize("a b  c") == ["a", "b", "c"]


def test_bm25_ranks_exact_term_match_first():
    idx = BM25Index(
        ids=["c1", "c2", "c3"],
        documents=[
            "aspirin lowers heart attack risk",
            "vitamin d supports bone health and calcium",
            "exercise improves fitness",
        ],
    )
    ranked = idx.rank("calcium and bone health")
    assert ranked[0] == "c2"


def test_bm25_rank_respects_top_n():
    idx = BM25Index(ids=["a", "b", "c"], documents=["x y", "y z", "z w"])
    assert len(idx.rank("y", top_n=2)) == 2


def test_bm25_returns_all_ids_by_default():
    idx = BM25Index(ids=["a", "b"], documents=["x", "y"])
    assert set(idx.rank("x")) == {"a", "b"}
