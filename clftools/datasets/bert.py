from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BertDataset(Dataset):
    def __init__(self,
                 maxlen,
                 sentences, labels=None, with_labels=True,
                 model: str="bert-base-chinese"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels
        self.maxlen = maxlen

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.maxlen,
                                      return_tensors='pt')

        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

        if self.with_labels:
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids