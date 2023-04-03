import torch
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from fastNLP.embeddings import ElmoEmbedding


class FastNLPEncoder:
    def __init__(self, vocab=None):
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary()
        self.embed = None
        self.embed_size = None

    def add_tokens(self, tokens):
        self.vocab.add_word_lst(tokens)

    def add_token(self, token):
        self.vocab.add_word(token)

    def static_embed(self, model_name, sentence, embed_size):
        self.embed_size = embed_size
        self.embed = StaticEmbedding(self.vocab, model_dir_or_name=model_name, embedding_dim=embed_size)
        x = torch.LongTensor([[self.vocab.to_index(word) for word in sentence.split()]])
        return self.embed(x)

    def elmo_embed(self, model_name, sentence, requires_grad=False, layers=None):
        self.embed_size = 256
        if requires_grad and layers is not None:
            self.embed_size *= len(layers.split(','))
        self.embed = ElmoEmbedding(self.vocab,
                                   model_dir_or_name=model_name,
                                   requires_grad=requires_grad,
                                   layers=layers)
        x = torch.LongTensor([[self.vocab.to_index(word) for word in sentence.split()]])
        return self.embed(x)

    def bert_embed(self, model_name, sentence, include_cls_sep=False, layers=None):
        self.embed_size = 768
        if include_cls_sep and layers is not None:
            self.embed_size *= len(layers.split(','))
        self.embed = ElmoEmbedding(self.vocab,
                                   model_dir_or_name=model_name,
                                   include_cls_sep=include_cls_sep,
                                   layers=layers)
        x = torch.LongTensor([[self.vocab.to_index(word) for word in sentence.split()]])
        return self.embed(x)
