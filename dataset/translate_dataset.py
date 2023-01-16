import re
import string

from torch.utils.data import Dataset


class TranslateDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def preprocessing_text(text):
        text = text.lower().strip()
        text = re.sub(f'[{string.punctuation}\n]', '', text)
        return text

    def tokenize_de(text):
        text = preprocessing_text(text)
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        text = preprocessing_text(text)
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def __getitem__(self, idx):
        return 1

    def __len__(self):
        return 1


if __name__ == "__main__":
    dataset = TranslateDataset()
