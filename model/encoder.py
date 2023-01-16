import torch.nn as nn

from attention import MultiHeadAttention
from layers import PFFlayer


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: int,
        device: int,
        max_length: int
    ) -> None:
        """
        :param vocab_size: vocab_size of total words
        :param hidden_dim: model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param n_heads: number of attention heads
        :param feedforward_dim
        :param dropout: dropout rate
        :param device: computation device
        """
        super(Encoder, self).__init__()

        self.device = device
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_dim,
                n_heads,
                feedforward_dim,
                dropout,
                device
            )
            for _ in range(n_layers)
        ])


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
        device: str
    ) -> None:
        super(EncoderLayer).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)

        self.self_attention = MultiHeadAttention(
            hidden_dim,
            n_heads,
            dropout,  # in official it's 0
            device
        )

        self.positionwise_feedforward = PFFlayer(
            hidden_dim,
            feedforward_dim,
            dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        '''
        :param x: [batch_size, src_len, hid_dim]
        :param mask: [batch_size, 1, 1, src_len]
        :return : [batch_size, src_len, hid_dim]
        '''
        out, _ = self.self_attention(x, x, x, mask)
        out = self.self_attn_layer_norm(x + self.dropout(out))
        out = self.positionwise_feedforward(out)
        out = self.ff_layer_norm(x + self.dropout(out))
        return out
