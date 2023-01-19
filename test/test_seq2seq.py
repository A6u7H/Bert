import unittest
import torch

from model import Seq2SeqTransformer, Encoder, Decoder


class EncoderConfig:
    vocab_size = 10000
    hidden_dim = 256
    n_layers = 3
    n_heads = 8
    feedforward_dim = 1024
    dropout = 0
    device = "cuda"
    max_length = 100


class DecoderConfig:
    vocab_size = 10000
    hidden_dim = 256
    n_layers = 3
    n_heads = 8
    feedforward_dim = 1024
    dropout = 0
    device = "cuda"
    max_length = 100


class TestTransformer(unittest.TestCase):
    def test_encoder_load(self):
        enc_config = EncoderConfig()
        enc_transformer = Encoder(
            enc_config.vocab_size,
            enc_config.hidden_dim,
            enc_config.n_layers,
            enc_config.n_heads,
            enc_config.feedforward_dim,
            enc_config.dropout,
            enc_config.device,
            enc_config.max_length
        ).to(enc_config.device)
        self.assertIsInstance(enc_transformer, Encoder)
        src_seq = torch.randint(
            0,
            500,
            (4, 100),
            device=enc_config.device,
            requires_grad=False
        )
        output = enc_transformer(src_seq, None)
        self.assertEqual(output.shape, (4, 100, 256))

    def test_decoder_load(self):
        dec_config = DecoderConfig()
        dec_transformer = Decoder(
            dec_config.vocab_size,
            dec_config.hidden_dim,
            dec_config.n_layers,
            dec_config.n_heads,
            dec_config.feedforward_dim,
            dec_config.dropout,
            dec_config.device,
            dec_config.max_length
        ).to(dec_config.device)
        self.assertIsInstance(dec_transformer, Decoder)

        src_enc_seq = torch.randn(
            (4, 50, 256),
            device=dec_config.device
        )

        tgt_seq = torch.randint(
            0,
            500,
            (4, 20),
            device=dec_config.device,
            requires_grad=False
        )
        output = dec_transformer(tgt_seq, src_enc_seq, None, None)
        self.assertEqual(output[0].shape, (4, 20, dec_config.vocab_size))

    def test_seq2seq_load(self):
        enc_config = EncoderConfig()
        dec_config = DecoderConfig()

        encoder = Encoder(
            enc_config.vocab_size,
            enc_config.hidden_dim,
            enc_config.n_layers,
            enc_config.n_heads,
            enc_config.feedforward_dim,
            enc_config.dropout,
            enc_config.device,
            enc_config.max_length
        ).to(enc_config.device)

        decoder = Decoder(
            dec_config.vocab_size,
            dec_config.hidden_dim,
            dec_config.n_layers,
            dec_config.n_heads,
            dec_config.feedforward_dim,
            dec_config.dropout,
            dec_config.device,
            dec_config.max_length
        ).to(dec_config.device)

        pad_idx = 0
        translation_model = Seq2SeqTransformer(
            encoder,
            decoder,
            pad_idx,
            dec_config.device
        )

        src_seq = torch.randint(
            0,
            500,
            (4, 100),
            device=enc_config.device,
            requires_grad=False
        )

        tgt_seq = torch.randint(
            0,
            500,
            (4, 20),
            device=dec_config.device,
            requires_grad=False
        )

        output = translation_model(src_seq, tgt_seq)
        self.assertEqual(output[0].shape, (4, 20, dec_config.vocab_size))


if __name__ == '__main__':
    unittest.main()
