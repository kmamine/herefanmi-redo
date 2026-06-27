"""BaseScraper — the common fetch/parse/scrape contract.

The design splits I/O (``fetch``) from pure HTML parsing (``parse``) so tests
can exercise parsing against saved fixtures and never touch the network.
Per-source subclasses only declare CSS selectors; the base does the work.
"""

from __future__ import annotations

import logging
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from hrf_shared.contracts import Article

from hrf_scraper.cleaning import clean_text
from hrf_scraper.dedup import content_hash

logger = logging.getLogger(__name__)

USER_AGENT = "HeReFaNMiBot/1.0 (+https://sites.google.com/view/herefanmi/)"


class ScrapeError(Exception):
    """Raised when an article page cannot be parsed into a valid Article."""


class BaseScraper:
    """Generic scraper. Subclasses only declare CSS selectors (see sources/)."""

    # --- per-source configuration (override in subclasses) ---
    name: str = "base"
    base_url: str = ""
    title_selector: str = "h1"
    content_selector: str = "article"
    date_selector: str | None = None
    date_attr: str | None = None  # read this attribute if set, else element text
    listing_url: str | None = None
    listing_link_selector: str = "a"
    min_content_len: int = 100
    request_timeout: float = 20.0

    async def fetch(self, url: str) -> str:
        async with httpx.AsyncClient(
            timeout=self.request_timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    def parse(self, html: str, url: str) -> Article:
        soup = BeautifulSoup(html, "lxml")

        title_el = soup.select_one(self.title_selector)
        title = clean_text(title_el.get_text(" ", strip=True)) if title_el else ""

        container = soup.select_one(self.content_selector)
        paragraphs: list[str] = []
        if container is not None:
            for p in container.select("p"):
                txt = p.get_text(" ", strip=True)
                if txt:
                    paragraphs.append(txt)
        content = clean_text(" ".join(paragraphs))

        published = self._parse_date(soup)

        if not title:
            raise ScrapeError(f"{self.name}: no title found at {url}")
        if len(content) < self.min_content_len:
            raise ScrapeError(f"{self.name}: content too short ({len(content)} chars) at {url}")

        return Article(
            title=title,
            content=content,
            url=url,
            source=self.name,
            published_date=published,
            content_hash=content_hash(content),
        )

    def _parse_date(self, soup: BeautifulSoup):
        if not self.date_selector:
            return None
        el = soup.select_one(self.date_selector)
        if el is None:
            return None
        raw = el.get(self.date_attr) if self.date_attr else el.get_text(" ", strip=True)
        if not raw:
            return None
        try:
            return dateparser.parse(raw)
        except (ValueError, OverflowError, TypeError):
            return None

    async def list_article_urls(self, limit: int | None = None) -> list[str]:
        if not self.listing_url:
            return []
        html = await self.fetch(self.listing_url)
        soup = BeautifulSoup(html, "lxml")
        urls: list[str] = []
        for a in soup.select(self.listing_link_selector):
            href = a.get("href")
            if href:
                urls.append(urljoin(self.base_url, href))
        if limit:
            urls = urls[:limit]
        return urls

    async def scrape(self, limit: int | None = None) -> list[Article]:
        urls = await self.list_article_urls(limit=limit)
        articles: list[Article] = []
        for url in urls:
            try:
                html = await self.fetch(url)
                articles.append(self.parse(html, url))
            except (ScrapeError, httpx.HTTPError) as exc:
                logger.warning("Skipping %s: %s", url, exc)
        return articles
