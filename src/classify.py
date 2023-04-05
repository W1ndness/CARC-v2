from clftools.models.classify import Classifier, SklearnClassifier, TorchClassifier
from clftools.datasets import load
from clftools.models.encoder.bert import BertEncoder
from clftools.models.encoder.fastnlp import FastNLPEncoder
from clftools.models.encoder.glove import GloVeEncoder
from clftools.models.encoder.gensim import GensimEncoder
from sklearn.svm import SVC

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
    text_of_webpages = [page.get_all_texts() for page in webpages]
    print("Webpage fetching and class <Webpage> instance initializing succeed.")
    # encoding
    supported_encoding_method = ['bert',
                                 'fastnlp-static', 'fastnlp-bert',
                                 'glove',
                                 'gensim']
    if encoding_method not in supported_encoding_method:
        raise ValueError(encoding_method, f'{encoding_method} is not supported')
    if encoding_method == 'bert':
        encoder = BertEncoder()
        text_embedding_of_webpages = [encoder.embedding(text) for text in text_of_webpages]
    elif encoding_method == 'glove':
        encoder = GloVeEncoder()
        text_embedding_of_webpages = [encoder.embedding(text) for text in text_of_webpages]
    elif 'fastnlp' in encoding_method:
        encoder = FastNLPEncoder()
        if encoding_method.endswith('static'):
            if 'embed_size' not in kwargs:
                raise ValueError("Using fastnlp-static embedding, missing argument \'embed_size\'")
            text_embedding_of_webpages = [encoder.static_embed(text,
                                                               kwargs['embed_size']) for text in text_of_webpages]
        elif encoding_method.endswith('bert'):
            text_embedding_of_webpages = [encoder.bert_embed(text) for text in text_of_webpages]
    elif 'gensim' in encoding_method:
        encoder = GensimEncoder()
    print("Encoding texts from webpages succeed.")
    print(text_embedding_of_webpages[0])
    # classification

if __name__ == '__main__':
    clf = SVC()
    classify(clf, encoding_method='bert', keyword='nlp')



