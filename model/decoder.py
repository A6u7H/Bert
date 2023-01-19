import torch.nn as nn

from model.layers import MultiHeadAttentionLayer, PFFlayer
from model.embedding import TransformerEmbedding


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()

        self.device = device
        self.embedding = TransformerEmbedding(
            vocab_size,
            hidden_dim,
            max_length,
            dropout,
            device
        )

        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim,
                n_heads,
                feedforward_dim,
                dropout,
                device
            )
            for _ in range(n_layers)
        ])

        self.outpur_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, src_enc, tgt_mask, src_mask):
        tgt_token_emb = self.embedding(tgt)

        for layer in self.layers:
            tgt_token_emb, attention = layer(
                tgt_token_emb,
                src_enc,
                tgt_mask,
                src_mask
            )
        output = self.outpur_layer(tgt_token_emb)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
        device: str
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)

        self.self_attention = MultiHeadAttentionLayer(
            hidden_dim,
            n_heads,
            dropout,  # in official it's 0
            device
        )

        self.cross_attention = MultiHeadAttentionLayer(
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

    def forward(self, tgt, src, tgt_mask, src_mask):
        '''
        :param tgt: [batch_size, tgt_len, hid_dim]
        :param src: [batch_size, src_len, hid_dim]
        :param tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        :param tgt_mask: [batch_size, 1, 1, src_len]
        :return : [batch_size, src_len, hid_dim]
        '''
        tgt_out, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.self_attn_layer_norm(tgt + self.dropout(tgt_out))

        tgt_out, attention = self.cross_attention(tgt, src, src, src_mask)
        tgt = self.self_attn_layer_norm(tgt + self.dropout(tgt_out))

        tgt_out = self.positionwise_feedforward(tgt)
        tgt_out = self.ff_layer_norm(tgt + self.dropout(tgt_out))
        return tgt_out, attention
