from hrf_scraper.base import BaseScraper


class MedlinePlusScraper(BaseScraper):
    name = "medlineplus"
    base_url = "https://medlineplus.gov"
    title_selector = "h1"
    content_selector = "div#mplus-content"
    listing_url = "https://medlineplus.gov/healthtopics.html"
    listing_link_selector = "ul#index li a"
