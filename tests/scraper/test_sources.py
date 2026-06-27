"""TDD for the per-source scrapers — each parses its fixture into an Article.

Parsing is fixture-driven; live sites are never contacted.
"""

from datetime import datetime

import pytest
from hrf_scraper.registry import SOURCE_NAMES, get_scraper

SOURCES = [
    "cdc",
    "nhs",
    "medlineplus",
    "statnews",
    "medpagetoday",
    "webmd",
    "newsmedical",
    "healthline",
]


def test_registry_has_eight_sources():
    assert set(SOURCE_NAMES) == set(SOURCES)
    assert len(SOURCE_NAMES) == 8


@pytest.mark.parametrize("name", SOURCES)
def test_source_parses_fixture(name, html_fixture):
    scraper = get_scraper(name)
    html = html_fixture(f"{name}_article.html")
    url = f"{scraper.base_url}/article/sample"

    art = scraper.parse(html, url)

    assert art.title.strip(), f"{name}: empty title"
    assert len(art.content) >= 100, f"{name}: content too short"
    assert art.source == name
    assert art.url == url
    assert art.content_hash
    assert art.published_date is None or isinstance(art.published_date, datetime)


@pytest.mark.parametrize(
    "name,expected_year",
    [
        ("cdc", 2024),
        ("nhs", 2024),
        ("statnews", 2024),
        ("medpagetoday", 2024),
        ("newsmedical", 2024),
        ("healthline", 2024),
    ],
)
def test_sources_with_dates_parse_year(name, expected_year, html_fixture):
    scraper = get_scraper(name)
    art = scraper.parse(html_fixture(f"{name}_article.html"), f"{scraper.base_url}/x")
    assert art.published_date is not None
    assert art.published_date.year == expected_year
