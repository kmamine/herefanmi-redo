"""TDD for hrf_scraper.pipeline — orchestrating scrape -> clean -> dedup -> store."""

import pytest
from hrf_scraper.dedup import content_hash
from hrf_scraper.pipeline import ingest
from hrf_scraper.storage import ArticleRepository, create_engine_and_tables
from hrf_shared.contracts import Article


def _article(url, content, source):
    return Article(
        title="T",
        content=content,
        url=url,
        source=source,
        content_hash=content_hash(content),
    )


class StubScraper:
    def __init__(self, name, articles):
        self.name = name
        self._articles = articles

    async def scrape(self, limit=None):
        return list(self._articles)


@pytest.fixture
def repo(tmp_path):
    engine = create_engine_and_tables(f"sqlite:///{tmp_path/'t.sqlite'}")
    return ArticleRepository(engine)


def _factory(mapping):
    def get(name):
        return mapping[name]

    return get


async def test_ingest_persists_to_sqlite(repo):
    scrapers = {
        "cdc": StubScraper("cdc", [_article("https://cdc.gov/a", "body " * 40, "cdc")]),
    }
    report = await ingest(["cdc"], repo=repo, scraper_factory=_factory(scrapers))
    assert report.stored == 1
    assert repo.count() == 1
    assert report.per_source["cdc"]["stored"] == 1


async def test_ingest_dedups_across_sources(repo):
    same = "identical medical body text " * 10
    scrapers = {
        "cdc": StubScraper("cdc", [_article("https://cdc.gov/a", same, "cdc")]),
        "nhs": StubScraper("nhs", [_article("https://nhs.uk/a", same, "nhs")]),
    }
    report = await ingest(["cdc", "nhs"], repo=repo, scraper_factory=_factory(scrapers))
    assert report.scraped == 2
    assert report.stored == 1  # second is a content duplicate
    assert report.duplicates == 1


async def test_ingest_all_sources_when_none_specified(repo):
    scrapers = {
        "cdc": StubScraper("cdc", [_article("https://cdc.gov/a", "x " * 60, "cdc")]),
        "nhs": StubScraper("nhs", [_article("https://nhs.uk/a", "y " * 60, "nhs")]),
    }
    report = await ingest(
        None,
        repo=repo,
        scraper_factory=_factory(scrapers),
        source_names=list(scrapers),
    )
    assert report.stored == 2
