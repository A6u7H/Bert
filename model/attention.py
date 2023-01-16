import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        device: str
    ) -> None:
        super(MultiHeadAttention, self).__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ):
        batch_size = query.shape[0]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query = query.view(
            batch_size, -1, self.n_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [batch_size, n_heads, query_len, head_dim]

        key = key.view(
            batch_size, -1, self.n_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [batch_size, n_heads, key_len, head_dim]

        value = value.view(
            batch_size, -1, self.n_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [batch_size, n_heads, value_len, head_dim]

        energy = (query @ key.transpose(2, 3)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        out = self.dropout(attention) @ value
        out = out.view(batch_size, -1, self.hid_dim)
        out = self.fc_o(out)

        return out, attention
