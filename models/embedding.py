from torch import nn
from d2l import torch as d2l
import torch


class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)


class Word2Vec(nn.Module):
    def __init__(self, vocab, embed_size):
        super().__init__()
        self.net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                              embedding_dim=embed_size))
    