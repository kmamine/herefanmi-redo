from hrf_scraper.base import BaseScraper


class HealthlineScraper(BaseScraper):
    # The 8th source: the legacy README named only 7. Swappable here (e.g. Mayo Clinic).
    name = "healthline"
    base_url = "https://www.healthline.com"
    title_selector = "h1"
    content_selector = "article"
    date_selector = "time"
    date_attr = "datetime"
    listing_url = "https://www.healthline.com/health-news"
    listing_link_selector = "a.css-1qhn6m6, ul li a"
