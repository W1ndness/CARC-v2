import re
import unicodedata
from typing import List, Union

import bs4
import requests
from bs4 import BeautifulSoup
from lxml import etree
from lxml.html.clean import Cleaner


class Webpage:
    def __init__(self, url):
        self.url = url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
            "Accept-Encoding": "gzip, deflate, br",
            "Cookie": 'BAIDUID=1A6EF88EE4929836C761FB37A1303522:FG=1; BIDUPSID=1A6EF88EE4929836C761FB37A1303522; PSTM=1603199415; H_PS_PSSID=32755_1459_32877_7567_31253_32706_32231_7517_32117_32845_32761_26350; BD_UPN=13314752; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; delPer=0; BD_CK_SAM=1; PSINO=5; H_PS_645EC=e4bcE4275G3zWcvH2pxYG6R32rBxb5yuey8xcioaej8V7IaJRfEq4xp4iCo; COOKIE_SESSION=45294_0_2_5_0_2_0_1_0_2_3_0_0_0_0_0_0_0_1603244844%7C5%230_0_1603244844%7C1; BA_HECTOR=2gal2h2ga58025f1vs1fov5vf0k'
        }
        self.encoding = 'utf-8'
        self.response = self.__get_response()
        self.html = self.response.text
        cleaner = Cleaner()
        self.html = cleaner.clean_html(self.html)
        self.dom = etree.HTML(self.html)
        self.soup = BeautifulSoup(self.html, 'lxml')

        self.all_texts = None
        self.node_texts = None
        self.node_infos = None

    @classmethod
    def clean_spaces(cls, text):
        return " ".join(re.split(r"\s+", text.strip()))

    @classmethod
    def clean_format_str(cls, text):
        """Cleans unicode control symbols, non-ascii chars, and extra blanks."""
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        text = "".join([c if ord(c) < 128 else "" for c in text])
        text = cls.clean_spaces(text)
        return text

    def __get_response(self):
        return requests.get(url=self.url, headers=self.headers)

    def get_all_texts(self) -> str:
        all_texts = self.soup.get_text()
        all_texts = Webpage.clean_spaces(all_texts)
        self.all_texts = all_texts
        return all_texts

    def get_node_texts(self) -> List[str]:
        texts = []
        for elem in self.soup.descendants:
            if isinstance(elem, bs4.NavigableString):
                texts.append(elem.get_text())
            else:
                texts.append([])
        self.node_texts = texts
        return texts

    def get_node_infos(self):
        infos = []

        def recursive(t: Union[bs4.Tag, bs4.NavigableString]):
            path_from_root = '.'.join(reversed([p.name for p in t.parentGenerator() if p]))
            node_info = (t, t.name, t.parent.name, path_from_root)
            infos.append(node_info)
            if not t.find_next():
                return
            recursive(t.find_next())
        recursive(self.soup.html)
        self.node_infos = infos
        return infos


if __name__ == '__main__':
    url = "https://www.jianshu.com/p/ff37a8524e72"
    webpage = Webpage(url)
    print(webpage.get_all_texts())
    print(webpage.get_node_texts())
    print(webpage.get_node_infos())
