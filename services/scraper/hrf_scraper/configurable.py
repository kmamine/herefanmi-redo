"""A BaseScraper whose selectors come from stored config (not a code class).

Lets admins add/edit sources at runtime: any object exposing the selector
attributes (e.g. SourceConfigModel or the SourceRow) can drive a scraper.
"""

from __future__ import annotations

from hrf_scraper.base import BaseScraper


class ConfigurableScraper(BaseScraper):
    def __init__(self, config) -> None:
        self.name = config.name
        self.base_url = config.base_url
        self.listing_url = config.listing_url
        self.listing_link_selector = config.listing_link_selector
        self.title_selector = config.title_selector
        self.content_selector = config.content_selector
        self.date_selector = config.date_selector
        self.date_attr = config.date_attr
