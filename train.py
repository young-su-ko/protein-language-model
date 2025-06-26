import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch

from src.lightning_modules.lit_esm import LitESM
from src.lightning_modules.data_module import ProteinDataModule

@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    torch.set_float32_matmul_precision('medium')
    # Initialize WandB logger
    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        log_model=True,
        save_dir="logs",
        name=config.logging.run_name
    )
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="esm2-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy=config.trainer.strategy,
        precision=config.trainer.precision,
        max_steps=config.training.max_steps,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        gradient_clip_val=config.training.gradient_clip_val,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=config.logging.log_every_n_steps,
    )
    
    # Initialize model and data module
    model = LitESM(config)
    data_module = ProteinDataModule(config)
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 
