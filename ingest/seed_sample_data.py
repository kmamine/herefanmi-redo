"""CLI: seed bundled sample articles into SQLite and ChromaDB.

Lets the full stack run end-to-end without live scraping (CI, demos, smoke test).
"""

from __future__ import annotations

import logging

from hrf_scraper.storage import ArticleRepository, create_engine_and_tables
from hrf_shared.chroma_client import get_chroma_client, get_or_create_collection
from hrf_shared.config import get_settings

from ingest.indexing import index_articles
from ingest.sample_articles import sample_articles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed")


def main() -> None:
    settings = get_settings()
    articles = sample_articles()

    engine = create_engine_and_tables(f"sqlite:///{settings.sqlite_path}")
    repo = ArticleRepository(engine)
    stored = sum(1 for a in articles if repo.upsert(a))
    logger.info("Seeded %d/%d articles into SQLite", stored, len(articles))

    client = get_chroma_client(settings)
    collection = get_or_create_collection(client, settings)
    total = index_articles(articles, collection)
    logger.info("Indexed %d chunks into collection '%s'", total, settings.chroma_collection)


if __name__ == "__main__":
    main()
