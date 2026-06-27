"""TDD for hrf_scraper.base — the BaseScraper fetch/parse/scrape seam.

The key property: parse() is pure (no network), and scrape() uses an injectable
fetch so tests never hit live sites.
"""

import pytest
from hrf_scraper.base import BaseScraper, ScrapeError
from hrf_shared.contracts import Article

ARTICLE_HTML = """
<html><body>
  <h1>Exercise and Heart Health</h1>
  <article>
    <p>Regular physical activity reduces the risk of chronic disease.</p>
    <p>Adults should aim for 150 minutes of moderate activity each week.</p>
    <time datetime="2024-03-01">March 1, 2024</time>
  </article>
</body></html>
"""

LISTING_HTML = """
<html><body>
  <ul class="list">
    <li><a class="teaser" href="/news/a">A</a></li>
    <li><a class="teaser" href="/news/b">B</a></li>
  </ul>
</body></html>
"""


class DummyScraper(BaseScraper):
    name = "dummy"
    base_url = "https://example.test"
    title_selector = "h1"
    content_selector = "article"
    date_selector = "time"
    date_attr = "datetime"
    listing_url = "https://example.test/news"
    listing_link_selector = "a.teaser"


def test_parse_is_pure_no_network():
    art = DummyScraper().parse(ARTICLE_HTML, "https://example.test/news/x")
    assert isinstance(art, Article)
    assert art.title == "Exercise and Heart Health"
    assert "physical activity" in art.content
    assert art.source == "dummy"
    assert art.url == "https://example.test/news/x"
    assert art.published_date is not None
    assert art.published_date.year == 2024
    assert art.content_hash  # populated


def test_parse_raises_on_missing_title():
    html = "<html><body><article><p>" + "word " * 40 + "</p></article></body></html>"
    with pytest.raises(ScrapeError):
        DummyScraper().parse(html, "https://example.test/x")


def test_parse_raises_on_short_content():
    html = "<html><body><h1>Title</h1><article><p>too short</p></article></body></html>"
    with pytest.raises(ScrapeError):
        DummyScraper().parse(html, "https://example.test/x")


async def test_scrape_uses_injected_fetch():
    calls = []

    class Injected(DummyScraper):
        async def fetch(self, url: str) -> str:
            calls.append(url)
            return ARTICLE_HTML

        async def list_article_urls(self, limit=None):
            return ["https://example.test/news/a", "https://example.test/news/b"]

    articles = await Injected().scrape(limit=2)
    assert len(articles) == 2
    assert all(isinstance(a, Article) for a in articles)
    assert calls == ["https://example.test/news/a", "https://example.test/news/b"]


async def test_list_article_urls_resolves_relative_links():
    class Injected(DummyScraper):
        async def fetch(self, url: str) -> str:
            return LISTING_HTML

    urls = await Injected().list_article_urls()
    assert "https://example.test/news/a" in urls
    assert "https://example.test/news/b" in urls


async def test_scrape_skips_unparseable_articles():
    class Injected(DummyScraper):
        async def fetch(self, url: str) -> str:
            return "<html><body>no title or content</body></html>"

        async def list_article_urls(self, limit=None):
            return ["https://example.test/news/a"]

    articles = await Injected().scrape(limit=1)
    assert articles == []  # bad article skipped, no crash
