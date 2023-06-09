import torch
from transformers import BertTokenizer, BertModel
import numpy as np


class BertEncoder:
    def __init__(self,
                 model_name_or_path: str = 'bert-base-chinese'):
        # 得到特征矩阵
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.bert = BertModel.from_pretrained(model_name_or_path)
        self.word_embeddings_dim = 768  # 嵌入维度

    def embedding(self, sentence):
        x = torch.tensor(self.tokenizer.encode(sentence, add_special_tokens=True))
        # 截断处理方法存疑，或许需要修改
        x = x[:512]  # cutoff as 512-dim
        input_ids = x.unsqueeze(0)  # 转化为id
        outputs = self.bert(input_ids)
        embeddings = outputs[0][:, 0, :]  # 取出[CLS]对应的嵌入向量
        return embeddings.detach().numpy()[0]
