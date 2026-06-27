"""TDD for the persisted, admin-editable source config store."""

from datetime import UTC

import pytest
from hrf_scraper.storage import SourceRepository, create_engine_and_tables
from hrf_shared.contracts import SourceCreate


@pytest.fixture
def repo(tmp_path):
    engine = create_engine_and_tables(f"sqlite:///{tmp_path / 's.sqlite'}")
    return SourceRepository(engine)


def _cfg(name="cdc"):
    return SourceCreate(
        name=name,
        base_url="https://www.cdc.gov",
        listing_url="https://www.cdc.gov/media/index.html",
        listing_link_selector="a.feed-item-title",
        title_selector="h1",
        content_selector="div.syndicate",
        date_selector='meta[property="article:published_time"]',
        date_attr="content",
        interval_minutes=720,
    )


def test_create_and_get(repo):
    repo.create(_cfg())
    got = repo.get("cdc")
    assert got is not None
    assert got.content_selector == "div.syndicate"
    assert got.enabled is False
    assert got.interval_minutes == 720


def test_create_duplicate_returns_none(repo):
    repo.create(_cfg())
    assert repo.create(_cfg()) is None


def test_list_sources(repo):
    repo.create(_cfg("cdc"))
    repo.create(_cfg("nhs"))
    assert {s.name for s in repo.list()} == {"cdc", "nhs"}


def test_update_fields(repo):
    repo.create(_cfg())
    repo.update("cdc", {"enabled": True, "interval_minutes": 60})
    got = repo.get("cdc")
    assert got.enabled is True
    assert got.interval_minutes == 60


def test_update_missing_returns_none(repo):
    assert repo.update("nope", {"enabled": True}) is None


def test_delete(repo):
    repo.create(_cfg())
    assert repo.delete("cdc") is True
    assert repo.get("cdc") is None
    assert repo.delete("cdc") is False


def test_mark_run_sets_timestamp_and_status(repo):
    from datetime import datetime

    repo.create(_cfg())
    when = datetime(2026, 6, 27, 12, 0, tzinfo=UTC)
    repo.mark_run("cdc", status="ok: 3 scraped", when=when)
    got = repo.get("cdc")
    assert got.last_status == "ok: 3 scraped"
    assert got.last_run_at is not None


def test_seed_defaults_is_idempotent(repo):
    cfgs = [_cfg("cdc"), _cfg("nhs")]
    repo.seed_defaults(cfgs)
    repo.seed_defaults(cfgs)  # second call must not duplicate
    assert len(repo.list()) == 2
