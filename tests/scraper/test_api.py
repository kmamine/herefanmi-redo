"""TDD for the Scraper FastAPI app — source CRUD, ingest/run-now, articles."""

import pytest
from fastapi.testclient import TestClient
from hrf_scraper.dedup import content_hash
from hrf_scraper.main import (
    app,
    get_article_repo,
    get_index_callback,
    get_scraper_factory,
    get_source_repo,
)
from hrf_scraper.storage import (
    ArticleRepository,
    SourceRepository,
    create_engine_and_tables,
)
from hrf_shared.contracts import Article, SourceCreate


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
def tc(tmp_path):
    engine = create_engine_and_tables(f"sqlite:///{tmp_path / 's.sqlite'}")
    source_repo = SourceRepository(engine)
    article_repo = ArticleRepository(engine)
    indexed: list = []

    async def index_cb(articles):
        indexed.extend(articles)

    app.dependency_overrides[get_source_repo] = lambda: source_repo
    app.dependency_overrides[get_article_repo] = lambda: article_repo
    app.dependency_overrides[get_index_callback] = lambda: index_cb
    app.dependency_overrides[get_scraper_factory] = lambda: StubScraper
    client = TestClient(app)
    client.source_repo = source_repo  # type: ignore[attr-defined]
    client.article_repo = article_repo  # type: ignore[attr-defined]
    client.indexed = indexed  # type: ignore[attr-defined]
    yield client
    app.dependency_overrides.clear()


def _seed_cdc(tc, enabled=True):
    tc.source_repo.create(
        SourceCreate(
            name="cdc",
            base_url="https://www.cdc.gov",
            listing_url="https://www.cdc.gov/media",
            enabled=enabled,
        )
    )


def test_health(tc):
    assert tc.get("/health").status_code == 200


def test_create_list_patch_delete_source(tc):
    r = tc.post(
        "/sources",
        json={
            "name": "cdc",
            "base_url": "https://www.cdc.gov",
            "listing_url": "https://www.cdc.gov/m",
        },
    )
    assert r.status_code == 201
    assert tc.get("/sources").json()["sources"][0]["name"] == "cdc"

    r = tc.patch("/sources/cdc", json={"enabled": True, "interval_minutes": 30})
    assert r.status_code == 200
    assert r.json()["enabled"] is True
    assert r.json()["interval_minutes"] == 30

    assert tc.delete("/sources/cdc").status_code == 200
    assert tc.get("/sources").json()["sources"] == []


def test_create_duplicate_returns_409(tc):
    _seed_cdc(tc)
    r = tc.post(
        "/sources",
        json={
            "name": "cdc",
            "base_url": "https://www.cdc.gov",
            "listing_url": "https://www.cdc.gov/m",
        },
    )
    assert r.status_code == 409


def test_patch_unknown_returns_404(tc):
    assert tc.patch("/sources/nope", json={"enabled": True}).status_code == 404


def test_ingest_scrapes_indexes_and_stores(tc):
    _seed_cdc(tc, enabled=True)
    r = tc.post("/ingest", json={"sources": ["cdc"]})
    assert r.status_code == 200
    assert r.json()["stored"] == 1
    assert tc.article_repo.count() == 1
    assert len(tc.indexed) == 1


def test_run_now_uses_enabled_sources(tc):
    _seed_cdc(tc, enabled=True)
    tc.source_repo.create(
        SourceCreate(
            name="nhs", base_url="https://nhs.uk", listing_url="https://nhs.uk/n", enabled=False
        )
    )
    r = tc.post("/run-now")
    assert r.status_code == 200
    assert set(r.json()["sources"]) == {"cdc"}


def test_articles_readback(tc):
    _seed_cdc(tc, enabled=True)
    tc.post("/ingest", json={"sources": ["cdc"]})
    arts = tc.get("/articles", params={"source": "cdc"}).json()["articles"]
    assert len(arts) == 1
    assert arts[0]["source"] == "cdc"
