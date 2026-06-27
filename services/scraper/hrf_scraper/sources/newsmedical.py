from hrf_scraper.base import BaseScraper


class NewsMedicalScraper(BaseScraper):
    name = "newsmedical"
    base_url = "https://www.news-medical.net"
    title_selector = "h1"
    content_selector = "div.article-content"
    date_selector = "span.article-meta-date"
    listing_url = "https://www.news-medical.net/medical/news"
    listing_link_selector = "div.feature-article a, h3 a"
