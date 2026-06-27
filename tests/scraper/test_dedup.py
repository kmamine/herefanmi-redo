"""TDD for hrf_scraper.dedup — content hashing and duplicate detection."""

from hrf_scraper.dedup import content_hash


def test_content_hash_is_stable():
    assert content_hash("hello world") == content_hash("hello world")


def test_content_hash_distinct_for_different_text():
    assert content_hash("a") != content_hash("b")


def test_content_hash_ignores_surrounding_whitespace():
    assert content_hash("  hello  ") == content_hash("hello")


def test_content_hash_is_hex_sha256():
    h = content_hash("x")
    assert len(h) == 64
    int(h, 16)  # raises if not hex
