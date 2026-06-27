"""TDD for hrf_scraper.storage — SQLite article store with dedup."""

import pytest
from hrf_scraper.dedup import content_hash
from hrf_scraper.storage import ArticleRepository, create_engine_and_tables
from hrf_shared.contracts import Article


def _article(url: str, content: str = "x" * 200, source: str = "cdc") -> Article:
    return Article(
        title="T",
        content=content,
        url=url,
        source=source,
        content_hash=content_hash(content),
    )


@pytest.fixture
def repo(tmp_path):
    engine = create_engine_and_tables(f"sqlite:///{tmp_path/'t.sqlite'}")
    return ArticleRepository(engine)


def test_upsert_inserts_new_article_returns_true(repo):
    assert repo.upsert(_article("https://cdc.gov/a")) is True
    assert repo.count() == 1


def test_upsert_existing_url_returns_false(repo):
    repo.upsert(_article("https://cdc.gov/a"))
    assert repo.upsert(_article("https://cdc.gov/a")) is False
    assert repo.count() == 1


def test_upsert_duplicate_content_returns_false(repo):
    repo.upsert(_article("https://cdc.gov/a", content="same body " * 30))
    # Different URL, identical content -> duplicate by content hash.
    assert repo.upsert(_article("https://cdc.gov/b", content="same body " * 30)) is False
    assert repo.count() == 1


def test_get_articles_filter_by_source(repo):
    repo.upsert(_article("https://cdc.gov/a", source="cdc"))
    repo.upsert(_article("https://nhs.uk/a", content="y" * 200, source="nhs"))
    cdc = repo.get_articles(source="cdc")
    assert len(cdc) == 1
    assert cdc[0].source == "cdc"


def test_get_all_articles(repo):
    repo.upsert(_article("https://cdc.gov/a"))
    repo.upsert(_article("https://nhs.uk/a", content="y" * 200, source="nhs"))
    assert len(repo.get_articles()) == 2
