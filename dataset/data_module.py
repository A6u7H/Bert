import re
import string
import spacy
import torchdata
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from omegaconf import DictConfig
# from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k

from .utils import SeqCollate, Vocabulary


class TranslateDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.data_path = config.data_path
        self.batch_size = config.batch_size

    def preprocessing_text(self, text):
        text = text.lower().strip()
        text = re.sub(f'[{string.punctuation}\n]', '', text)
        return text

    def tokenize_de(self, text):
        text = self.preprocessing_text(text)
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        text = self.preprocessing_text(text)
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def prepare_data(self) -> None:
        self.source_vocab = Vocabulary("German")
        self.target_vocab = Vocabulary("English")

        self.pad_idx = self.source_vocab.to_index("[PAD]")
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

    def setup(self, stage: str):
        self.train_data, self.valid_data, self.test_data = Multi30k(
            self.data_path
        )

        for src_text, tgt_text in self.train_data:
            self.source_vocab.add_sentence(self.tokenize_de(src_text))
            self.target_vocab.add_sentence(self.tokenize_en(tgt_text))

        self.collate_fn = SeqCollate(
            self.source_vocab,
            self.target_vocab,
            self.tokenize_de,
            self.tokenize_en,
            self.pad_idx
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )
