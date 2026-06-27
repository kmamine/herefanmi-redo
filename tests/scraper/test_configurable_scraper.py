"""TDD for ConfigurableScraper — a BaseScraper driven by stored source config."""

from hrf_scraper.configurable import ConfigurableScraper
from hrf_shared.contracts import SourceConfigModel


def _cdc_config():
    return SourceConfigModel(
        name="cdc",
        base_url="https://www.cdc.gov",
        listing_url="https://www.cdc.gov/media/index.html",
        listing_link_selector="a.feed-item-title",
        title_selector="h1",
        content_selector="div.syndicate",
        date_selector='meta[property="article:published_time"]',
        date_attr="content",
    )


def test_configurable_parse_uses_config_selectors(html_fixture):
    scraper = ConfigurableScraper(_cdc_config())
    art = scraper.parse(html_fixture("cdc_article.html"), "https://www.cdc.gov/x")
    assert art.source == "cdc"
    assert "flu" in art.title.lower()
    assert len(art.content) >= 100
    assert art.published_date is not None
    assert art.published_date.year == 2024


def test_configurable_inherits_base_behavior(html_fixture):
    # Wrong content selector -> too-short content -> ScrapeError (base behavior).
    from hrf_scraper.base import ScrapeError

    cfg = _cdc_config()
    cfg.content_selector = "div.does-not-exist"
    scraper = ConfigurableScraper(cfg)
    try:
        scraper.parse(html_fixture("cdc_article.html"), "https://www.cdc.gov/x")
        raised = False
    except ScrapeError:
        raised = True
    assert raised
