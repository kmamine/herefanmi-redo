"""TDD for hrf_rag.hybrid.reciprocal_rank_fusion — pure fusion math, no embeddings."""

from hrf_rag.hybrid import fuse, reciprocal_rank_fusion


def test_rrf_combines_two_rankings():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert set(scores) == {"a", "b", "c"}


def test_rrf_rewards_agreement():
    # "b" appears near the top of both lists -> should win.
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert scores["b"] == max(scores.values())


def test_rrf_rank_one_beats_rank_two_single_list():
    scores = reciprocal_rank_fusion([["x", "y"]], k=60)
    assert scores["x"] > scores["y"]


def test_fuse_returns_ordered_top_n():
    ordered = fuse(["a", "b", "c"], ["b", "c", "a"], k=60, top_n=2)
    assert len(ordered) == 2
    assert ordered[0][0] == "b"  # agreement winner first
    assert all(isinstance(score, float) for _, score in ordered)


def test_fuse_stable_tiebreak_by_id():
    # Symmetric inputs make scores equal; ties break by id ascending.
    ordered = fuse(["a", "b"], ["b", "a"], k=60, top_n=2)
    assert [i for i, _ in ordered] == ["a", "b"]
