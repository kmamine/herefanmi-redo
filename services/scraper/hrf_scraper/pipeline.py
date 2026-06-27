"""Ingestion pipeline: scrape -> (clean inside parse) -> dedup -> store."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from hrf_scraper.registry import SOURCE_NAMES, get_scraper
from hrf_scraper.storage import ArticleRepository


@dataclass
class IngestReport:
    scraped: int = 0
    stored: int = 0
    duplicates: int = 0
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "scraped": self.scraped,
            "stored": self.stored,
            "duplicates": self.duplicates,
            "per_source": self.per_source,
        }


async def ingest(
    sources: list[str] | None,
    *,
    repo: ArticleRepository,
    scraper_factory: Callable[[str], object] = get_scraper,
    source_names: list[str] | None = None,
    limit: int | None = None,
) -> IngestReport:
    """Scrape the given sources (or all) and persist new articles to SQLite."""
    names = sources or source_names or SOURCE_NAMES
    report = IngestReport()

    for name in names:
        scraper = scraper_factory(name)
        articles = await scraper.scrape(limit=limit)
        stored = duplicates = 0
        for article in articles:
            report.scraped += 1
            if repo.upsert(article):
                report.stored += 1
                stored += 1
            else:
                report.duplicates += 1
                duplicates += 1
        report.per_source[name] = {
            "scraped": len(articles),
            "stored": stored,
            "duplicates": duplicates,
        }

    return report
