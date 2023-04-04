from clftools.web.secrawler import SearchEngineCrawler
from clftools.web.webpage import Webpage
import asyncio


def crawl_and_extract(keyword: str, depth: int = 20):
    crawler = SearchEngineCrawler('bing')
    urls = asyncio.run(crawler.crawl_by_keyword(keyword, depth))
    pages = []
    for url in urls:
        pages.append(Webpage(url))
    return pages


