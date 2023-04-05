import torch
from transformers import BertTokenizer, BertModel
import numpy as np


class BertEncoder:
    def __init__(self, model: str = 'bert-base-chinese'):
        # 得到特征矩阵
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.bert = BertModel.from_pretrained(model)
        self.word_embeddings_dim = 768  # 嵌入维度

    def embedding(self, sentence):
        input_ids = torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # 转化为id
        outputs = self.bert(input_ids)
        embeddings = outputs[0][:, 0, :]  # 取出[CLS]对应的嵌入向量
        return embeddings.detach().numpy()
