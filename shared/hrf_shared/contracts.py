"""Cross-service Pydantic contracts — the single source of truth for the data
shapes exchanged between the Scraper, RAG, LLM, and Backend services.

The classification JSON contract is preserved verbatim from the legacy system:
    {"medical","news","label","reasoning","sources"}
with ``medical``/``news`` kept as the *string* literals "True"/"False" (not bools)
and ``label`` one of Trustworthy/Doubtful/Fake.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

TruthFlag = Literal["True", "False"]
Label = Literal["Trustworthy", "Doubtful", "Fake"]

_LABELS = {"trustworthy": "Trustworthy", "doubtful": "Doubtful", "fake": "Fake"}


def _normalize_truthflag(value: object) -> object:
    """Coerce booleans and arbitrary casing to the canonical "True"/"False"."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "false"):
            return low.capitalize()
    return value


class Classification(BaseModel):
    """The LLM's medical-news verdict — the contract returned by /predict."""

    medical: TruthFlag
    news: TruthFlag
    label: Label
    reasoning: str
    sources: list[str] = Field(default_factory=list)

    @field_validator("medical", "news", mode="before")
    @classmethod
    def _coerce_truthflag(cls, value: object) -> object:
        return _normalize_truthflag(value)

    @field_validator("label", mode="before")
    @classmethod
    def _coerce_label(cls, value: object) -> object:
        if isinstance(value, str):
            return _LABELS.get(value.strip().lower(), value)
        return value


class Article(BaseModel):
    """A normalized article harvested by a scraper, before chunking."""

    title: str
    content: str
    url: str
    source: str
    published_date: datetime | None = None
    content_hash: str


class Chunk(BaseModel):
    """A retrievable text chunk derived from an Article and indexed in Chroma."""

    chunk_id: str
    text: str
    source: str
    url: str
    article_id: str
    position: int


class ScoredChunk(BaseModel):
    """A chunk returned by RAG retrieval, with its fused relevance score."""

    text: str
    score: float
    source: str
    url: str


# ---- Request / response models ----------------------------------------------


class PredictRequest(BaseModel):
    """Body for LLM POST /predict."""

    chunks: list[str]
    question: str


class FindSimilarRequest(BaseModel):
    """Body for RAG POST /find_similar_chunks."""

    prompt: str
    top_n: int = 5


class FindSimilarResponse(BaseModel):
    """Synchronous response from RAG — the chunks, returned to the caller."""

    chunks: list[ScoredChunk]
    query: str


# ---- Source management + indexing -------------------------------------------


class SourceConfigModel(BaseModel):
    """Admin-editable scraper source (drives a ConfigurableScraper)."""

    name: str
    base_url: str
    listing_url: str
    listing_link_selector: str = "a"
    title_selector: str = "h1"
    content_selector: str = "article"
    date_selector: str | None = None
    date_attr: str | None = None
    enabled: bool = False
    interval_minutes: int = 1440
    last_run_at: datetime | None = None
    last_status: str | None = None


class SourceCreate(BaseModel):
    """Body to add a new source (selectors required, schedule optional)."""

    name: str
    base_url: str
    listing_url: str
    listing_link_selector: str = "a"
    title_selector: str = "h1"
    content_selector: str = "article"
    date_selector: str | None = None
    date_attr: str | None = None
    enabled: bool = False
    interval_minutes: int = 1440


class SourceUpdate(BaseModel):
    """Partial update for a source (any subset of fields)."""

    base_url: str | None = None
    listing_url: str | None = None
    listing_link_selector: str | None = None
    title_selector: str | None = None
    content_selector: str | None = None
    date_selector: str | None = None
    date_attr: str | None = None
    enabled: bool | None = None
    interval_minutes: int | None = None


class ArticleIn(BaseModel):
    """An article pushed to RAG POST /index for chunking + embedding."""

    title: str
    content: str
    url: str
    source: str
    article_id: str | None = None
