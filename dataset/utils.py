import torch
from torch.nn.utils.rnn import pad_sequence

from typing import List


class Vocabulary:
    PAD_token = 0
    BOS_token = 1
    EOS_token = 2
    UNK_token = 3

    def __init__(self, name):
        self.name = name
        self.word2count = {}
        self.word2index = {
            "[PAD]": self.PAD_token,
            "[BOS]": self.BOS_token,
            "[EOS]": self.EOS_token,
            "[UNK]": self.UNK_token,
        }
        self.index2word = {
            self.PAD_token: "[PAD]",
            self.BOS_token: "[BOS]",
            self.EOS_token: "[EOS]",
            self.UNK_token: "[UNK]"
        }
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, token: str):
        if token not in self.word2index:
            self.word2index[token] = self.num_words
            self.word2count[token] = 1
            self.index2word[self.num_words] = token
            self.num_words += 1
        else:
            self.word2count[token] += 1

    def add_sentence(self, tokens: List[str]):
        sentence_len = 0
        for word in tokens:
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            self.longest_sentence = sentence_len
        self.num_sentences += 1

    def tokens_to_idx(self, tokens: List[str]):
        token_idxs = [self.BOS_token]
        for token in tokens:
            if token in self.word2index:
                token_idxs.append(self.to_index(token))
            else:
                token_idxs.append(self.UNK_token)
        token_idxs.append(self.EOS_token)
        return token_idxs

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]


class SeqCollate:
    def __init__(
        self,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        src_tokenizer,
        tgt_tokenizer,
        pad_idx: int
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_tokens_idx = []
        tgt_tokens_idx = []
        for item in batch:
            src_tokens = self.src_tokenizer(item[0])
            tgt_tokens = self.src_tokenizer(item[1])
            src_tokens_idx.append(
                torch.tensor(
                    self.src_vocab.tokens_to_idx(src_tokens),
                    dtype=torch.int64
                )
            )
            tgt_tokens_idx.append(
                torch.tensor(
                    self.tgt_vocab.tokens_to_idx(tgt_tokens),
                    dtype=torch.int64
                )
            )

        src_seq = pad_sequence(
            src_tokens_idx,
            padding_value=self.pad_idx,
            batch_first=True
        )

        tgt_seq = pad_sequence(
            tgt_tokens_idx,
            padding_value=self.pad_idx,
            batch_first=True
        )

        return src_seq, tgt_seq
