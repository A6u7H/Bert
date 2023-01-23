import torch
import pytorch_lightning as pl

from omegaconf import DictConfig
from torch import Tensor
from hydra.utils import instantiate


class Solver(pl.LightningModule):
    def __init__(
        self,
        config: DictConfig,
        source_vocab_size: int,
        target_vocab_size: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.generation_loss = instantiate(self.config.optimizer.loss)
        self.model = self.create_model()

    def forward(self, x):
        return self.model(x)

    def create_model(self):
        encoder_model = instantiate(self.config.encoder)
        decoder_model = instantiate(self.config.decoder)
        model = isinstance(
            self.config.seq2seq,
            encoder=encoder_model,
            decoder=decoder_model,
        )
        return model

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = instantiate(
            self.config.optimizer.optimizer,
            params=params
        )
        scheduler = instantiate(
            self.config.optimizer.scheduler,
            optimizer=optimizer
        )

        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val/loss'
        }

    def loss_fn(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.generation_loss(pred, target)

    def training_step(self, batch: Tensor, batch_idx: int):
        src_text, tgt_text = batch
        pred_text_prob = self.model(src_text, tgt_text[:, :-1])
        loss = self.loss_fn(
            pred_text_prob.view(-1, pred_text_prob.shape[-1]),
            tgt_text[:, 1:].view(-1)
        )

        self.log("train/loss", loss)
        return {
            "loss": loss
        }

    def validation_step(self, batch: Tensor, batch_idx: int):
        src_text, tgt_text = batch
        pred_text_prob = self.model(src_text, tgt_text[:, :-1])
        loss = self.loss_fn(
            pred_text_prob.view(-1, pred_text_prob.shape[-1]),
            tgt_text[:, 1:].view(-1)
        )

        self.log("val/loss", loss)
        return {
            "loss": loss
        }

    def test_step(self, batch: Tensor, batch_idx: int):
        src_text, tgt_text = batch
        pred_text_prob = self.model(src_text, tgt_text[:, :-1])
        loss = self.loss_fn(
            pred_text_prob.view(-1, pred_text_prob.shape[-1]),
            tgt_text[:, 1:].view(-1)
        )

        self.log("test/loss", loss)
        return {
            "loss": loss
        }

    def fit(self):
        self.log_artifact(".hydra/config.yaml")
        self.log_artifact(".hydra/hydra.yaml")
        self.log_artifact(".hydra/overrides.yaml")
        self.trainer.fit(
            self,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )

    def log_artifact(self, artifact_path: str):
        self.logger.experiment.log_artifact(self.logger.run_id, artifact_path)
