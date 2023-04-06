import multiprocessing

from clftools.web.secrawler import SearchEngineCrawler
from clftools.web.webpage import Webpage
from clftools import constants
import asyncio
import os
from tqdm import tqdm
import threading
import joblib
from joblib import Parallel, delayed


def crawl_from_url(keyword: str, depth: int = 20):
    crawler = SearchEngineCrawler('bing')
    urls = asyncio.run(crawler.crawl_by_keyword(keyword, depth))
    print("Successfully finding {:d} webpages".format(len(urls)))
    # return [Webpage(url=url) for url in urls]
    return [Webpage(url=url) for url in urls]


def read_from_path(base_dir, endswith_html, preprocessing_func=None):
    page_paths = []
    for cur_dir, dirs, files in os.walk(base_dir):
        if endswith_html:
            for file in files:
                if file.endswith('.html'):
                    page_paths.append(os.path.join(cur_dir, file))
        else:
            page_paths.extend([os.path.join(cur_dir, file) for file in files])
    print("Successfully finding {:d} webpages".format(len(page_paths)))
    html_docs = []
    for path in page_paths:
        with open(path, 'r', encoding='ISO-8859-1') as fp:
            doc = fp.read()
            if preprocessing_func is not None:
                doc = preprocessing_func(doc)
            html_docs.append(doc)

    def process(doc):
        return Webpage(html_doc=doc)

    ret = Parallel(n_jobs=constants.N_JOBS,
                   backend="threading",
                   verbose=2,
                   batch_size='auto',
                   pre_dispatch='2*n_jobs')(delayed(process)(doc) for doc in html_docs)
    return ret


def load_labels(base_dir, endswith_html):
    labels = []
    for cur_dir, dirs, files in tqdm(os.walk(base_dir)):
        if endswith_html:
            for file in files:
                if file.endswith('.html'):
                    labels.append(cur_dir)
        else:
            labels.extend([cur_dir] * len(files))
    return labels
