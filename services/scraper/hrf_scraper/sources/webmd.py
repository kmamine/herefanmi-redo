from hrf_scraper.base import BaseScraper


class WebMDScraper(BaseScraper):
    name = "webmd"
    base_url = "https://www.webmd.com"
    title_selector = "h1"
    content_selector = "div.article__body"
    listing_url = "https://www.webmd.com/news/default.htm"
    listing_link_selector = "div.tease a, h3 a"
