import torch
import torch.nn as nn

from torch import Tensor


class TransformerEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_length: int,
        dropout: int,
        device: str
    ) -> None:
        super(TransformerEmbedding, self).__init__()

        self.device = device
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor([hidden_dim], device=device))

    def forward(self, x: Tensor):
        batch_size, seq_len = x.shape
        pos = torch.arange(
            0, seq_len
        ).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        token_emb = self.token_emb(x) * self.scale
        pos_emb = self.pos_emb(pos)

        transformer_emb = self.dropout(token_emb + pos_emb)
        return transformer_emb
