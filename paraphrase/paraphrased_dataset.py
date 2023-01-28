import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .bart import bart_paraphrase
from .t5 import t5_paraphrase


class ParaphrasedDataset(Dataset):
    def __init__(self, df, pretrained_weights, with_bart_aug=False, with_t5_aug=False, max_length=100):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, use_fast=False)
        self.paraphrasers = [lambda x: [x]]
        if with_bart_aug:
            self.paraphrasers.append(bart_paraphrase)
        if with_t5_aug:
            self.paraphrasers.append(t5_paraphrase)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        paraphrases = random.choice(self.paraphrasers)(record['text'])
        paraphrased_text = random.choice(paraphrases) if len(paraphrases) else record['text']
        return {**self.tokenize_text(paraphrased_text), 'label': torch.tensor(record['label'], dtype=torch.int64)}

    def tokenize_text(self, text):
        tokenized = self.tokenizer(text,
                                   add_special_tokens=True,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=self.max_length,
                                   return_tensors="pt")
        return {k: torch.squeeze(v) for k, v in tokenized.items()}
