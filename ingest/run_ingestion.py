"""CLI: harvest sources into SQLite and index them into ChromaDB.

Examples:
    python ingest/run_ingestion.py --scrape --limit 5      # scrape then index
    python ingest/run_ingestion.py                         # index existing SQLite
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from hrf_scraper.pipeline import ingest as scrape_ingest
from hrf_scraper.storage import ArticleRepository, create_engine_and_tables
from hrf_shared.chroma_client import get_chroma_client, get_or_create_collection
from hrf_shared.config import get_settings

from ingest.indexing import index_articles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape and/or index health content.")
    parser.add_argument("--scrape", action="store_true", help="run scrapers first")
    parser.add_argument("--sources", nargs="*", default=None, help="source names")
    parser.add_argument("--limit", type=int, default=None, help="articles per source")
    args = parser.parse_args()

    settings = get_settings()
    engine = create_engine_and_tables(f"sqlite:///{settings.sqlite_path}")
    repo = ArticleRepository(engine)

    if args.scrape:
        logger.info("Scraping sources=%s limit=%s", args.sources or "ALL", args.limit)
        report = asyncio.run(scrape_ingest(args.sources, repo=repo, limit=args.limit))
        logger.info("Scrape report: %s", report.as_dict())

    articles = repo.get_articles()
    logger.info("Indexing %d articles into ChromaDB...", len(articles))

    client = get_chroma_client(settings)
    collection = get_or_create_collection(client, settings)
    total = index_articles(articles, collection)
    logger.info("Indexed %d chunks into collection '%s'", total, settings.chroma_collection)


if __name__ == "__main__":
    main()
