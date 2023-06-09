import collections
import os
import sys

import sklearn.ensemble

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import multiprocessing

from clftools.models.classify import Classifier, SklearnClassifier, TorchClassifier
from clftools.datasets import load
from clftools.models.encoder.bert import BertEncoder
from clftools.models.encoder.fastnlp import FastNLPEncoder
from clftools.models.encoder.glove import GloVeEncoder
from clftools import constants
from clftools.models.classifier.mlpcls import MLPClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

import torch
from torch.utils.data import TensorDataset
from torch import nn
from torch.optim import AdamW

set_loky_pickler("dill")


def classify(clf: Classifier,
             encoding_method: str,
             keyword=None, data_dir=None,
             **kwargs):
    # load data in one way
    if (keyword is None and data_dir is None) or (keyword is not None and data_dir is not None):
        raise ValueError("Cannot load data in a certain way.")
    if keyword is not None:
        if 'depth' in kwargs:
            webpages = load.crawl_from_url(keyword, depth=kwargs['depth'])
        else:
            webpages = load.crawl_from_url(keyword)
    if data_dir is not None:
        if 'endswith_html' not in kwargs:
            raise ValueError("Miss argument: endswith_html")
        if not kwargs['endswith_html']:
            if 'preprocessing_func' not in kwargs:
                raise ValueError("Miss argument: preprocessing_func, cannot process files.")
            func = kwargs['preprocessing_func']
        else:
            func = None
        webpages = load.read_from_path(base_dir=data_dir,
                                       endswith_html=kwargs['endswith_html'],
                                       preprocessing_func=func)
    text_of_webpages = []
    for page in tqdm(webpages):
        text_of_webpages.append(page.get_all_texts())
    print("Webpage fetching and class <Webpage> instance initializing succeed.")

    # encoding
    supported_encoding_method = ['bert',
                                 'fastnlp-static', 'fastnlp-bert',
                                 'glove']
    if encoding_method not in supported_encoding_method:
        raise ValueError(encoding_method, f'{encoding_method} is not supported')
    if encoding_method == 'bert':
        encoder = BertEncoder(
            model_name_or_path='/Users/macbookpro/PycharmProjects/CARC-v2/clftools/models/encoder/bert-cache/bert-base-chinese'
        )

        def embed_work(text):
            return encoder.embedding(text)

        text_embedding_of_webpages = Parallel(n_jobs=constants.N_JOBS,
                                              backend="threading",
                                              verbose=2,
                                              batch_size='auto',
                                              pre_dispatch='2*n_jobs')(
            delayed(embed_work)(text) for text in text_of_webpages)
    elif encoding_method == 'glove':
        encoder = GloVeEncoder()

        def embed_work(text):
            return encoder.embedding(text)

        text_embedding_of_webpages = Parallel(n_jobs=constants.N_JOBS,
                                              backend="threading",
                                              verbose=2,
                                              batch_size='auto',
                                              pre_dispatch='2*n_jobs')(
            delayed(embed_work)(text) for text in text_of_webpages)
    elif 'fastnlp' in encoding_method:
        encoder = FastNLPEncoder()
        if encoding_method.endswith('static'):
            if 'embed_size' not in kwargs:
                raise ValueError("Using fastnlp-static embedding, missing argument \'embed_size\'")
            text_embedding_of_webpages = [encoder.static_embed(text,
                                                               # model_name='/Users/macbookpro/PycharmProjects/CARC-v2/clftools/models/encoder/bert-cache/bert-base-chinese',
                                                               kwargs['embed_size']) for text in text_of_webpages]
        elif encoding_method.endswith('bert'):
            text_embedding_of_webpages = [encoder.bert_embed(text) for text in text_of_webpages]
    print("Encoding texts from webpages succeed.")
    # print(type(text_embedding_of_webpages[0]))

    # classification
    X = np.array(text_embedding_of_webpages)
    y = load.load_labels(base_dir=data_dir, endswith_html=kwargs['endswith_html'])
    print("Loading labels succeed.")
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y)
    train_size = kwargs['train_size'] if 'train_size' in kwargs else 0.7
    test_size = kwargs['test_size'] if 'test_size' in kwargs else 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=train_size, test_size=test_size,
                                                        random_state=12)
    clf.fit(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    # model = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=42)
    # model = XGBClassifier(n_estimators=1000, learning_rate=0.01, objective='multi:softmax', gamma=.5)
    # clf = SklearnClassifier(model, labels=['1', '2'], model_name='RFclf')

    y = load.load_labels(base_dir='/Users/macbookpro/PycharmProjects/CARC-v2/test/datasets/ki-04',
                         endswith_html=True)
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y)
    num_classes = len(label_enc.classes_)
    model = MLPClassifier(768, num_classes, [(768, 256), (256, 64), (64, num_classes)], dropout=True)
    opt = AdamW(params=model.parameters(), lr=1e-3)
    clf = TorchClassifier(model, ['1', '2'], 'mlpcls',
                          TensorDataset,
                          50,
                          nn.CrossEntropyLoss(), opt,
                          batch_size=4)
    classify(clf, encoding_method='bert',
             data_dir='/Users/macbookpro/PycharmProjects/CARC-v2/test/datasets/ki-04',
             endswith_html=True)
