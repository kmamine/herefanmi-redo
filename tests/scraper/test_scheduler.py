"""TDD for the per-source scheduler (pure due-logic + run orchestration)."""

from datetime import UTC, datetime, timedelta

import pytest
from hrf_scraper.dedup import content_hash
from hrf_scraper.scheduler import due_sources, run_sources
from hrf_scraper.storage import (
    ArticleRepository,
    SourceRepository,
    create_engine_and_tables,
)
from hrf_shared.contracts import Article, SourceConfigModel, SourceCreate

NOW = datetime(2026, 6, 27, 12, 0, tzinfo=UTC)


def _src(name, *, enabled=True, interval=60, last_run=None):
    return SourceConfigModel(
        name=name,
        base_url="https://x.test",
        listing_url="https://x.test/news",
        enabled=enabled,
        interval_minutes=interval,
        last_run_at=last_run,
    )


def test_due_includes_never_run_enabled():
    assert due_sources([_src("a", last_run=None)], NOW) == ["a"]


def test_due_excludes_disabled():
    assert due_sources([_src("a", enabled=False, last_run=None)], NOW) == []


def test_due_excludes_recently_run():
    recent = NOW - timedelta(minutes=10)
    assert due_sources([_src("a", interval=60, last_run=recent)], NOW) == []


def test_due_includes_overdue():
    old = NOW - timedelta(minutes=120)
    assert due_sources([_src("a", interval=60, last_run=old)], NOW) == ["a"]


def test_due_mixed():
    names = due_sources(
        [
            _src("fresh", interval=60, last_run=NOW - timedelta(minutes=5)),
            _src("stale", interval=60, last_run=NOW - timedelta(minutes=90)),
            _src("paused", enabled=False, last_run=None),
            _src("new", last_run=None),
        ],
        NOW,
    )
    assert set(names) == {"stale", "new"}


# ---- run_sources orchestration ----


class StubScraper:
    def __init__(self, cfg):
        self.cfg = cfg

    async def scrape(self, limit=None):
        body = f"medical content for {self.cfg.name} " * 12
        return [
            Article(
                title="T",
                content=body,
                url=f"https://x.test/{self.cfg.name}/a",
                source=self.cfg.name,
                content_hash=content_hash(body),
            )
        ]


@pytest.fixture
def repos(tmp_path):
    engine = create_engine_and_tables(f"sqlite:///{tmp_path / 'sch.sqlite'}")
    return SourceRepository(engine), ArticleRepository(engine)


async def test_run_sources_scrapes_indexes_and_marks(repos):
    source_repo, article_repo = repos
    source_repo.create(
        SourceCreate(
            name="cdc", base_url="https://x.test", listing_url="https://x.test/n", enabled=True
        )
    )
    indexed: list = []

    async def index_cb(articles):
        indexed.extend(articles)

    report = await run_sources(
        ["cdc"],
        source_repo=source_repo,
        article_repo=article_repo,
        index_callback=index_cb,
        scraper_factory=StubScraper,
        now=NOW,
    )

    assert report["stored"] == 1
    assert len(indexed) == 1
    assert article_repo.count() == 1
    marked = source_repo.get("cdc")
    assert marked.last_run_at is not None
    assert "ok" in marked.last_status


async def test_run_sources_records_error_status(repos):
    source_repo, article_repo = repos
    source_repo.create(
        SourceCreate(
            name="bad", base_url="https://x.test", listing_url="https://x.test/n", enabled=True
        )
    )

    class Boom:
        def __init__(self, cfg):
            pass

        async def scrape(self, limit=None):
            raise RuntimeError("network down")

    async def index_cb(articles):
        pass

    report = await run_sources(
        ["bad"],
        source_repo=source_repo,
        article_repo=article_repo,
        index_callback=index_cb,
        scraper_factory=Boom,
        now=NOW,
    )
    assert "error" in report["sources"]["bad"]
    assert "error" in source_repo.get("bad").last_status
