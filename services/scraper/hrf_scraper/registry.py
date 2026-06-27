"""Source registry — maps a source name to its scraper class.

Adding or swapping a source (e.g. Healthline -> Mayo Clinic) is a one-line
change here plus the module under sources/.
"""

from __future__ import annotations

from hrf_shared.contracts import SourceCreate

from hrf_scraper.base import BaseScraper
from hrf_scraper.sources import (
    CDCScraper,
    HealthlineScraper,
    MedlinePlusScraper,
    MedPageTodayScraper,
    NewsMedicalScraper,
    NHSScraper,
    STATNewsScraper,
    WebMDScraper,
)

SOURCE_REGISTRY: dict[str, type[BaseScraper]] = {
    CDCScraper.name: CDCScraper,
    NHSScraper.name: NHSScraper,
    MedlinePlusScraper.name: MedlinePlusScraper,
    STATNewsScraper.name: STATNewsScraper,
    MedPageTodayScraper.name: MedPageTodayScraper,
    WebMDScraper.name: WebMDScraper,
    NewsMedicalScraper.name: NewsMedicalScraper,
    HealthlineScraper.name: HealthlineScraper,
}

SOURCE_NAMES: list[str] = list(SOURCE_REGISTRY)


def get_scraper(name: str) -> BaseScraper:
    """Instantiate the scraper registered under ``name``."""
    try:
        return SOURCE_REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown source '{name}'. Known: {', '.join(SOURCE_NAMES)}") from exc


def default_source_configs() -> list[SourceCreate]:
    """Seed configs for the 8 built-in sources (disabled, daily by default).

    Selectors are read from the built-in scraper classes so the persisted,
    admin-editable config starts from the known-good defaults.
    """
    configs: list[SourceCreate] = []
    for cls in SOURCE_REGISTRY.values():
        configs.append(
            SourceCreate(
                name=cls.name,
                base_url=cls.base_url,
                listing_url=cls.listing_url or cls.base_url,
                listing_link_selector=cls.listing_link_selector,
                title_selector=cls.title_selector,
                content_selector=cls.content_selector,
                date_selector=cls.date_selector,
                date_attr=cls.date_attr,
                enabled=False,
                interval_minutes=1440,
            )
        )
    return configs
