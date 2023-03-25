from typing import List
import asyncio
import aiohttp
from lxml import etree


class SearchEngineCrawler:
    def __init__(self, search_engine: str):
        supported_search_engines = ['bing']
        if search_engine not in supported_search_engines:
            raise ValueError(search_engine, f'{search_engine} is not supported.')
        self.search_engine = search_engine
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Accept-Encoding": "gzip, deflate, br",
            "Cookie": 'BAIDUID=1A6EF88EE4929836C761FB37A1303522:FG=1; BIDUPSID=1A6EF88EE4929836C761FB37A1303522; PSTM=1603199415; H_PS_PSSID=32755_1459_32877_7567_31253_32706_32231_7517_32117_32845_32761_26350; BD_UPN=13314752; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; delPer=0; BD_CK_SAM=1; PSINO=5; H_PS_645EC=e4bcE4275G3zWcvH2pxYG6R32rBxb5yuey8xcioaej8V7IaJRfEq4xp4iCo; COOKIE_SESSION=45294_0_2_5_0_2_0_1_0_2_3_0_0_0_0_0_0_0_1603244844%7C5%230_0_1603244844%7C1; BA_HECTOR=2gal2h2ga58025f1vs1fov5vf0k'
        }
        if self.search_engine == 'bing':
            self.url_prefix = "https://cn.bing.com/search?q="
            self.url_suffix = "&go=%E6%90%9C%E7%B4%A2&qs=ds&first="
        elif self.search_engine == 'baidu': # not supported
            self.url_prefix = "https://www.baidu.com/s?wd="
            self.url_suffix = "&pn="

    async def crawl_by_keyword(self, keyword, depth) -> List[str]:
        async with aiohttp.ClientSession() as session:
            pages = []
            for i in range(1, depth):
                num = str(i * 10 - 1 if self.search_engine == 'bing' else i * 10)
                url = self.url_prefix + keyword + self.url_suffix + num
                print("Now crawling {}".format(url))
                try:
                    async with session.get(url, headers=self.headers) as resp:
                        r = await resp.text()
                        a = etree.HTML(r)
                        for j in range(1, 20):
                            if self.search_engine == 'bing':
                                href_xpath = f"//*[@id=\"b_results\"]/li[{j}]/div[1]/h2/a/@href"
                            elif self.search_engine == 'baidu':
                                href_xpath = f"//*[@id=\"{10 + j}\"]/div/div[1]/h3/a/@href" # not supported
                            href = a.xpath(href_xpath)
                            if not href:
                                continue
                            for each in href:
                                pages.append(each)
                except Exception:
                    print("Failure")
            return pages

if __name__ == '__main__':
    crawler = SearchEngineCrawler('bing')
    pages = asyncio.run(crawler.crawl_by_keyword('nlp', 20))
    print(len(pages))
