import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from myplm.esm2 import ESM2
import torch
from typing import Optional, List, Dict
from Bio import SeqIO
from myplm.tokenizer import Alphabet

def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attn_mask = torch.stack([item['attn_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attn_mask': attn_mask
    }

class ProteinDataset(Dataset):
    def __init__(self, fasta_file: str, tokenizer, max_length: int, objective: str, mask_prob: float = 0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.objective = objective

        if self.objective == "mlm":
            self.mask_prob = mask_prob
        elif self.objective == "clm":
            self.mask_prob = 0.0

        self.sequences = []

        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = "$" + str(record.seq) + "!"
            self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.sequences[idx])
        input_ids = torch.full((self.max_length,), self.tokenizer.pad_idx, dtype=torch.long)
        labels = torch.full((self.max_length,), self.tokenizer.pad_idx, dtype=torch.long)

        if len(tokens) > self.max_length: # truncate
            start = torch.randint(0, len(tokens) - self.max_length + 1, (1,)).item()
            tokens = tokens[start:start + self.max_length]
        input_ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        
        attn_mask = torch.zeros(self.max_length, dtype=torch.bool)
        attn_mask[:len(tokens)] = 1 # padding mask is 1 for all tokens

        if self.objective == "mlm":
            special_mask = self.tokenizer.special_toks_mask[input_ids]
            candidate_mask = attn_mask & ~special_mask
            mask_tokens = (torch.rand(self.max_length) < self.mask_prob) & candidate_mask
            labels[mask_tokens] = input_ids[mask_tokens]
            input_ids[mask_tokens] = self.tokenizer.mask_idx
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1) # [1, 1, L]

        elif self.objective == "clm":
            labels = input_ids.clone()
            square_mask = attn_mask.unsqueeze(0) & attn_mask.unsqueeze(1) # [L, L]
            attn_mask = torch.tril(square_mask).bool().unsqueeze(0) # [1, L, L]

        return {
            "input_ids": input_ids, # [B, L]
            "labels": labels,       # [B, L]
            "attn_mask": attn_mask  # [B, 1, 1, L] for MLM or [B, 1, L, L] for CLM
        }

class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = Alphabet()
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ProteinDataset(
                self.config.data.train_file,
                self.tokenizer,
                self.config.data.max_seq_length,
                self.config.model.objective,
                self.config.data.mask_prob
            )
            self.val_dataset = ProteinDataset(
                self.config.data.val_file,
                self.tokenizer,
                self.config.data.max_seq_length,
                self.config.model.objective,
                self.config.data.mask_prob
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.trainer.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_batch(batch),
            prefetch_factor=self.config.trainer.prefetch_factor,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.trainer.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_batch(batch),
            prefetch_factor=self.config.trainer.prefetch_factor,
            persistent_workers=True
        ) 
