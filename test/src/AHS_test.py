import os
import multiprocessing
from joblib import Parallel, delayed

from clftools.web.webpage import Webpage

import numpy as np

if __name__ == '__main__':
    base_dir = '../datasets/AHS'
    page_paths = []
    for cur_dir, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.html'):
                page_paths.append(os.path.join(cur_dir, file))
    print("Number of pages:", len(page_paths))
    html_docs = []
    for path in page_paths:
        with open(path, 'r') as fp:
            doc = fp.read()
            html_docs.append(doc)
    webpages = []
    num_nodes_of_webpages = []
    num_edges_of_webpages = []

    def process(doc):
        page = Webpage(html_doc=doc)
        webpages.append(page)
        num_nodes_of_webpages.append(page.dom_as_graph.number_of_nodes())
        num_edges_of_webpages.append(page.dom_as_graph.number_of_edges())


    num_cores = multiprocessing.cpu_count()
    print("CPU Cores:", num_cores)
    print("Usage:", num_cores // 2)
    Parallel(n_jobs=num_cores // 2)(delayed(process)(doc) for doc in html_docs)
    num_nodes_of_webpages = np.array(num_nodes_of_webpages)
    num_edges_of_webpages = np.array(num_edges_of_webpages)
    print(num_nodes_of_webpages.mean(), num_edges_of_webpages.mean())