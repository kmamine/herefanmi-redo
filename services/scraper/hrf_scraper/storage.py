"""SQLite article store (SQLAlchemy 2.0).

Replaces the legacy MongoDB document store. Dedup is enforced on both the URL
(unique) and the content hash.
"""

from __future__ import annotations

from datetime import UTC, datetime

from hrf_shared.contracts import Article, SourceConfigModel, SourceCreate
from sqlalchemy import Boolean, DateTime, Integer, String, create_engine, func, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class ArticleRow(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String, unique=True, index=True)
    source: Mapped[str] = mapped_column(String, index=True)
    published_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    content_hash: Mapped[str] = mapped_column(String, index=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    def to_contract(self) -> Article:
        return Article(
            title=self.title,
            content=self.content,
            url=self.url,
            source=self.source,
            published_date=self.published_date,
            content_hash=self.content_hash,
        )


class SourceRow(Base):
    """Admin-editable scraper source config (drives ConfigurableScraper)."""

    __tablename__ = "sources"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    base_url: Mapped[str] = mapped_column(String)
    listing_url: Mapped[str] = mapped_column(String)
    listing_link_selector: Mapped[str] = mapped_column(String, default="a")
    title_selector: Mapped[str] = mapped_column(String, default="h1")
    content_selector: Mapped[str] = mapped_column(String, default="article")
    date_selector: Mapped[str | None] = mapped_column(String, nullable=True)
    date_attr: Mapped[str | None] = mapped_column(String, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    interval_minutes: Mapped[int] = mapped_column(Integer, default=1440)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_status: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    def to_contract(self) -> SourceConfigModel:
        return SourceConfigModel(
            name=self.name,
            base_url=self.base_url,
            listing_url=self.listing_url,
            listing_link_selector=self.listing_link_selector,
            title_selector=self.title_selector,
            content_selector=self.content_selector,
            date_selector=self.date_selector,
            date_attr=self.date_attr,
            enabled=self.enabled,
            interval_minutes=self.interval_minutes,
            last_run_at=self.last_run_at,
            last_status=self.last_status,
        )


def create_engine_and_tables(url: str) -> Engine:
    """Create the SQLite engine and ensure tables exist (idempotent)."""
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine


class ArticleRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def upsert(self, article: Article) -> bool:
        """Insert the article. Returns False if a URL/content duplicate exists."""
        with Session(self.engine) as session:
            existing = session.scalar(
                select(ArticleRow).where(
                    or_(
                        ArticleRow.url == article.url,
                        ArticleRow.content_hash == article.content_hash,
                    )
                )
            )
            if existing is not None:
                return False
            session.add(
                ArticleRow(
                    title=article.title,
                    content=article.content,
                    url=article.url,
                    source=article.source,
                    published_date=article.published_date,
                    content_hash=article.content_hash,
                )
            )
            session.commit()
            return True

    def count(self) -> int:
        with Session(self.engine) as session:
            return session.scalar(select(func.count()).select_from(ArticleRow)) or 0

    def get_articles(self, source: str | None = None, limit: int | None = None) -> list[Article]:
        stmt = select(ArticleRow).order_by(ArticleRow.id)
        if source:
            stmt = stmt.where(ArticleRow.source == source)
        if limit:
            stmt = stmt.limit(limit)
        with Session(self.engine) as session:
            return [row.to_contract() for row in session.scalars(stmt).all()]


class SourceRepository:
    """CRUD for admin-managed scraper sources."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._fields = (
            "base_url",
            "listing_url",
            "listing_link_selector",
            "title_selector",
            "content_selector",
            "date_selector",
            "date_attr",
            "enabled",
            "interval_minutes",
        )

    def create(self, cfg: SourceCreate) -> SourceConfigModel | None:
        """Add a source. Returns None if the name already exists."""
        row = SourceRow(
            name=cfg.name,
            base_url=cfg.base_url,
            listing_url=cfg.listing_url,
            listing_link_selector=cfg.listing_link_selector,
            title_selector=cfg.title_selector,
            content_selector=cfg.content_selector,
            date_selector=cfg.date_selector,
            date_attr=cfg.date_attr,
            enabled=cfg.enabled,
            interval_minutes=cfg.interval_minutes,
        )
        with Session(self.engine) as session:
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return None
            return row.to_contract()

    def get(self, name: str) -> SourceConfigModel | None:
        with Session(self.engine) as session:
            row = session.get(SourceRow, name)
            return row.to_contract() if row else None

    def list(self) -> list[SourceConfigModel]:
        with Session(self.engine) as session:
            rows = session.scalars(select(SourceRow).order_by(SourceRow.name)).all()
            return [r.to_contract() for r in rows]

    def update(self, name: str, fields: dict) -> SourceConfigModel | None:
        with Session(self.engine) as session:
            row = session.get(SourceRow, name)
            if row is None:
                return None
            for key, value in fields.items():
                if key in self._fields and value is not None:
                    setattr(row, key, value)
            session.commit()
            return row.to_contract()

    def delete(self, name: str) -> bool:
        with Session(self.engine) as session:
            row = session.get(SourceRow, name)
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def mark_run(self, name: str, *, status: str, when: datetime | None = None) -> None:
        with Session(self.engine) as session:
            row = session.get(SourceRow, name)
            if row is None:
                return
            row.last_run_at = when or datetime.now(UTC)
            row.last_status = status
            session.commit()

    def seed_defaults(self, configs: list[SourceCreate]) -> int:
        """Insert any configs whose name isn't present yet. Returns count added."""
        added = 0
        for cfg in configs:
            if self.get(cfg.name) is None and self.create(cfg) is not None:
                added += 1
        return added
