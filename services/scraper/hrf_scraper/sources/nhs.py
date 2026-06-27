from hrf_scraper.base import BaseScraper


class NHSScraper(BaseScraper):
    name = "nhs"
    base_url = "https://www.nhs.uk"
    title_selector = "h1"
    content_selector = "article"
    date_selector = "time"
    date_attr = "datetime"
    listing_url = "https://www.nhs.uk/news/"
    listing_link_selector = "a.nhsuk-list-panel__link, ul.nhsuk-list li a"
