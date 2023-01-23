import os
import hydra
import logging

from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset.data_module import TranslateDataModule
from solver import Solver


logger = logging.getLogger(__name__)

@hydra.main(config_path="config/", config_name="config.yaml", version_base="1.2.0")
def main(config: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    logger.info("Creating model logger...")
    model_logger = instantiate(config.logger)

    logger.info("Creating datamodule")
    datamodule = TranslateDataModule(config)

    logger.info("Creating model...")
    model = Solver(config)

    output_dir = hydra_cfg['runtime']['output_dir']
    saving_weight_path = os.path.join(output_dir, "weight")

    checkpoint_callback = ModelCheckpoint(
        dirpath=saving_weight_path,
        save_top_k=5,
        monitor="val/loss"
    )

    early_stopping = EarlyStopping("val/loss")
    trainer = instantiate(
        config.trainer,
        logger=model_logger,
        callbacks=[checkpoint_callback, early_stopping]
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()