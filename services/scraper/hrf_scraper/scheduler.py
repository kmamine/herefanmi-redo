"""Per-source scheduling: pure due-logic + run orchestration.

`due_sources` is pure (inject `now`) so it's trivially testable. `run_sources`
scrapes the named sources, persists new articles, hands them to an injectable
`index_callback` (default: POST to the RAG /index endpoint), and records each
source's last run + status.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from hrf_shared.contracts import Article, SourceConfigModel

from hrf_scraper.configurable import ConfigurableScraper

logger = logging.getLogger(__name__)

IndexCallback = Callable[[list[Article]], Awaitable[None]]


def _naive(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=None) if dt and dt.tzinfo else dt


def due_sources(sources: list[SourceConfigModel], now: datetime) -> list[str]:
    """Names of enabled sources that are due to run at `now`."""
    now_n = _naive(now)
    due: list[str] = []
    for s in sources:
        if not s.enabled:
            continue
        if s.last_run_at is None:
            due.append(s.name)
            continue
        if now_n - _naive(s.last_run_at) >= timedelta(minutes=s.interval_minutes):
            due.append(s.name)
    return due


async def run_sources(
    names: list[str],
    *,
    source_repo,
    article_repo,
    index_callback: IndexCallback,
    scraper_factory: Callable[[SourceConfigModel], object] = ConfigurableScraper,
    limit: int | None = None,
    now: datetime | None = None,
) -> dict:
    """Scrape the named sources, store new articles, index them, mark each run."""
    when = now or datetime.now(UTC)
    report: dict = {"sources": {}, "scraped": 0, "stored": 0, "indexed": 0}

    for name in names:
        cfg = source_repo.get(name)
        if cfg is None:
            report["sources"][name] = {"error": "unknown source"}
            continue
        try:
            scraper = scraper_factory(cfg)
            articles = await scraper.scrape(limit=limit)
            new = [a for a in articles if article_repo.upsert(a)]
            if new:
                await index_callback(new)
            status = f"ok: {len(articles)} scraped, {len(new)} new"
            source_repo.mark_run(name, status=status, when=when)
            report["sources"][name] = {"scraped": len(articles), "new": len(new)}
            report["scraped"] += len(articles)
            report["stored"] += len(new)
            report["indexed"] += len(new)
        except Exception as exc:  # noqa: BLE001 - record per-source failure, keep going
            logger.warning("Source %s failed: %s", name, exc)
            source_repo.mark_run(name, status=f"error: {exc}", when=when)
            report["sources"][name] = {"error": str(exc)}

    return report


async def run_due(
    *,
    source_repo,
    article_repo,
    index_callback: IndexCallback,
    scraper_factory: Callable[[SourceConfigModel], object] = ConfigurableScraper,
    limit: int | None = None,
    now: datetime | None = None,
) -> dict:
    """Run all sources currently due (used by the scheduler tick)."""
    when = now or datetime.now(UTC)
    names = due_sources(source_repo.list(), when)
    if not names:
        return {"sources": {}, "scraped": 0, "stored": 0, "indexed": 0}
    return await run_sources(
        names,
        source_repo=source_repo,
        article_repo=article_repo,
        index_callback=index_callback,
        scraper_factory=scraper_factory,
        limit=limit,
        now=when,
    )
