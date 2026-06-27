from hrf_scraper.base import BaseScraper


class MedPageTodayScraper(BaseScraper):
    name = "medpagetoday"
    base_url = "https://www.medpagetoday.com"
    title_selector = "h1"
    content_selector = "div.article-body"
    date_selector = 'meta[name="publishdate"]'
    date_attr = "content"
    listing_url = "https://www.medpagetoday.com/publichealthpolicy"
    listing_link_selector = "div.article a, h3 a"
