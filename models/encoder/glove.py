from torchtext.vocab import GloVe


class GloVeEncoder:
    def __init__(self, name, embed_size, cache_dir=None):
        self.name = name
        self.embed_size = embed_size
        self.glove = GloVe(name=name, dim=embed_size, cache=cache_dir)

    def embedding(self, tokens):
        return self.glove.get_vecs_by_tokens(tokens, True)
