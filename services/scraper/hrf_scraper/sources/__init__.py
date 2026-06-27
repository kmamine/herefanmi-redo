"""Per-source scraper implementations for the 8 credible health sources."""

from hrf_scraper.sources.cdc import CDCScraper
from hrf_scraper.sources.healthline import HealthlineScraper
from hrf_scraper.sources.medlineplus import MedlinePlusScraper
from hrf_scraper.sources.medpagetoday import MedPageTodayScraper
from hrf_scraper.sources.newsmedical import NewsMedicalScraper
from hrf_scraper.sources.nhs import NHSScraper
from hrf_scraper.sources.statnews import STATNewsScraper
from hrf_scraper.sources.webmd import WebMDScraper

__all__ = [
    "CDCScraper",
    "NHSScraper",
    "MedlinePlusScraper",
    "STATNewsScraper",
    "MedPageTodayScraper",
    "WebMDScraper",
    "NewsMedicalScraper",
    "HealthlineScraper",
]
