import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from src.model.esm2 import ESM2
from transformers import get_linear_schedule_with_warmup

class LitESM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = ESM2(
            num_layers=config.model.num_layers,
            embed_dim=config.model.embed_dim,
            ffn_embed_dim=config.model.ffn_embed_dim,
            num_heads=config.model.num_heads,
            initializer_range=config.model.initializer_range,
            bias=config.model.bias,
            weight_tying=config.model.weight_tying,
        )
        self.objective = config.model.objective
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx)
        
    def forward(self, x, mask=None):
        return self.model(x, mask)
        
    def shared_step(self, batch, stage: str):
        input_ids = batch['input_ids']              # [B, L]
        
        if self.objective == "mlm":
            padding_mask = batch['attn_mask'] # [B, 1, 1, L]
            outputs = self(input_ids, padding_mask)
            logits = outputs['logits']
            
            logits = logits.reshape(-1, logits.shape[-1])
            labels = batch['labels'].reshape(-1)
        
        elif self.objective == "clm":
            causal_mask = batch['attn_mask'] # already in [B, 1, L, L]
            outputs = self(input_ids, causal_mask)
            logits = outputs['logits']
            
            logits = logits[:,:-1, :].reshape(-1, logits.shape[-1])
            labels = batch['labels'][:,1:].reshape(-1)
            
            # logits [bos, token1, token2, ...]
            # labels [token1, token2, ... eos]
            
        loss = self.criterion(logits, labels)
        perplexity = torch.exp(loss)

        self.log(f'{stage}_loss', loss, prog_bar=False, sync_dist=True)
        self.log(f'{stage}_perplexity', perplexity, prog_bar=False, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, stage="val")
        
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,  # peak LR
            betas=(self.config.training.beta1, self.config.training.beta2),
            eps=self.config.training.eps,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.max_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        } 