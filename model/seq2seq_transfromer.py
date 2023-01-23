import torch
import torch.nn as nn


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        pad_idx: int,
        device: str
    ) -> None:
        super(Seq2SeqTransformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def get_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def get_tgt_mask(self, tgt):
        tgt_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=self.device)
        ).bool()
        trg_mask = tgt_mask & tgt_sub_mask
        return trg_mask

    def forward(self, src, tgt):
        src_mask = self.get_src_mask(src)
        tgt_mask = self.get_tgt_mask(tgt)

        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        return output, attention
