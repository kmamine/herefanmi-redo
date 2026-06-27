from hrf_scraper.base import BaseScraper


class CDCScraper(BaseScraper):
    name = "cdc"
    base_url = "https://www.cdc.gov"
    title_selector = "h1"
    content_selector = "div.syndicate"
    date_selector = 'meta[property="article:published_time"]'
    date_attr = "content"
    listing_url = "https://www.cdc.gov/media/index.html"
    listing_link_selector = "a.feed-item-title, div.feed-item a"
