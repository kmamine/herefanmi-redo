"""Scraper service FastAPI app.

Sources are persisted, admin-editable config rows. A per-source APScheduler tick
(optional) runs due sources; each run scrapes → SQLite → pushes new articles to
the RAG /index endpoint.

  GET    /sources            list source configs
  POST   /sources            add a source
  PATCH  /sources/{name}     edit selectors / interval / enable-pause
  DELETE /sources/{name}     remove a source
  POST   /ingest             scrape given (or all enabled) sources now
  POST   /run-now            scrape all enabled sources now
  GET    /articles           read back stored articles
  GET    /health
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from hrf_shared.config import Settings, get_settings
from hrf_shared.contracts import (
    Article,
    ArticleIn,
    SourceConfigModel,
    SourceCreate,
    SourceUpdate,
)
from pydantic import BaseModel

from hrf_scraper.configurable import ConfigurableScraper
from hrf_scraper.registry import default_source_configs
from hrf_scraper.scheduler import run_due, run_sources
from hrf_scraper.storage import (
    ArticleRepository,
    SourceRepository,
    create_engine_and_tables,
)

logger = logging.getLogger(__name__)

_scheduler = None


def _engine():
    return create_engine_and_tables(f"sqlite:///{get_settings().sqlite_path}")


def get_source_repo() -> SourceRepository:
    return SourceRepository(_engine())


def get_article_repo() -> ArticleRepository:
    return ArticleRepository(_engine())


async def rag_index_callback(articles: list[Article]) -> None:
    """Default index callback: push new articles to the RAG /index endpoint."""
    settings = get_settings()
    payload = {
        "articles": [
            ArticleIn(title=a.title, content=a.content, url=a.url, source=a.source).model_dump()
            for a in articles
        ]
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{settings.rag_url.rstrip('/')}/index", json=payload)
        resp.raise_for_status()


def get_index_callback() -> Callable[[list[Article]], Awaitable[None]]:
    return rag_index_callback


def get_scraper_factory() -> Callable[[SourceConfigModel], object]:
    return ConfigurableScraper


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fresh Settings() (not the cached get_settings) so env set by tests applies.
    settings = Settings()
    if settings.scraper_autostart:
        try:
            SourceRepository(_engine()).seed_defaults(default_source_configs())
        except Exception as exc:  # noqa: BLE001 - never block startup on seeding
            logger.warning("Source seeding skipped: %s", exc)
        if settings.scheduler_enabled:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            async def _tick():
                eng = _engine()
                await run_due(
                    source_repo=SourceRepository(eng),
                    article_repo=ArticleRepository(eng),
                    index_callback=rag_index_callback,
                    limit=settings.scrape_limit,
                )

            global _scheduler
            _scheduler = AsyncIOScheduler()
            _scheduler.add_job(_tick, "interval", seconds=settings.scheduler_tick_seconds)
            _scheduler.start()
            logger.info("Scheduler started (tick=%ss)", settings.scheduler_tick_seconds)
    yield
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)


app = FastAPI(title="HeReFaNMi Scraper Service", lifespan=lifespan)


class IngestBody(BaseModel):
    sources: list[str] | None = None
    limit: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/sources")
def list_sources(repo: SourceRepository = Depends(get_source_repo)) -> dict:
    return {"sources": jsonable_encoder(repo.list())}


@app.post("/sources", status_code=201)
def create_source(
    body: SourceCreate, repo: SourceRepository = Depends(get_source_repo)
) -> SourceConfigModel:
    created = repo.create(body)
    if created is None:
        raise HTTPException(status_code=409, detail="Source already exists")
    return created


@app.patch("/sources/{name}")
def update_source(
    name: str, body: SourceUpdate, repo: SourceRepository = Depends(get_source_repo)
) -> SourceConfigModel:
    updated = repo.update(name, body.model_dump(exclude_unset=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Unknown source")
    return updated


@app.delete("/sources/{name}")
def delete_source(name: str, repo: SourceRepository = Depends(get_source_repo)) -> dict:
    if not repo.delete(name):
        raise HTTPException(status_code=404, detail="Unknown source")
    return {"status": "deleted"}


async def _run(names, source_repo, article_repo, index_callback, scraper_factory, limit):
    return await run_sources(
        names,
        source_repo=source_repo,
        article_repo=article_repo,
        index_callback=index_callback,
        scraper_factory=scraper_factory,
        limit=limit,
    )


@app.post("/ingest")
async def ingest_endpoint(
    body: IngestBody,
    source_repo: SourceRepository = Depends(get_source_repo),
    article_repo: ArticleRepository = Depends(get_article_repo),
    index_callback=Depends(get_index_callback),
    scraper_factory=Depends(get_scraper_factory),
) -> dict:
    names = body.sources or [s.name for s in source_repo.list() if s.enabled]
    return await _run(names, source_repo, article_repo, index_callback, scraper_factory, body.limit)


@app.post("/run-now")
async def run_now(
    source_repo: SourceRepository = Depends(get_source_repo),
    article_repo: ArticleRepository = Depends(get_article_repo),
    index_callback=Depends(get_index_callback),
    scraper_factory=Depends(get_scraper_factory),
) -> dict:
    names = [s.name for s in source_repo.list() if s.enabled]
    return await _run(names, source_repo, article_repo, index_callback, scraper_factory, None)


@app.get("/articles")
def articles(
    source: str | None = None,
    limit: int | None = None,
    repo: ArticleRepository = Depends(get_article_repo),
) -> dict:
    return {"articles": jsonable_encoder(repo.get_articles(source=source, limit=limit))}
