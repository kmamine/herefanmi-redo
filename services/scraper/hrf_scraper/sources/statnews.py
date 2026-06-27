from hrf_scraper.base import BaseScraper


class STATNewsScraper(BaseScraper):
    name = "statnews"
    base_url = "https://www.statnews.com"
    title_selector = "h1.article__title"
    content_selector = "div.article__body"
    date_selector = "time"
    date_attr = "datetime"
    listing_url = "https://www.statnews.com/category/health/"
    listing_link_selector = "article a.media__link, h3 a"
