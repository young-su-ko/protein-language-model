from src.constants import proteinseq_toks
import torch

class Alphabet:
    def __init__(
        self,
        standard_toks = proteinseq_toks['toks'],
        # For convenience, we will represent special tokens with a single character
        # cls = bos = $
        # pad = -
        # eos = !
        # unk = ?
        # mask = #
        special_toks = ("$", "-", "!", "?", "#")
    ):

        self.standard_toks = list(standard_toks)
        self.special_toks = list(special_toks)
        self.all_toks = self.standard_toks + self.special_toks

        self.tok_to_idx = {tok: idx for idx, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["?"]
        self.pad_idx = self.tok_to_idx["-"]
        self.bos_idx = self.tok_to_idx["$"]
        self.eos_idx = self.tok_to_idx["!"]
        self.mask_idx = self.tok_to_idx["#"]
        self.special_toks_mask = torch.zeros(len(self.all_toks), dtype=torch.bool)
        for tok in self.special_toks:
            self.special_toks_mask[self.tok_to_idx[tok]] = True

    def __len__(self) -> int:
        return len(self.all_toks)
    
    def get_idx(self, tok) -> int:
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, idx: int) -> str:
        return self.all_toks[idx]

    def tokenize(self, seq: str) -> list[str]:
        return list(seq)
    
    def encode(self, seq: str) -> list[int]:
        return [self.get_idx(tok) for tok in seq]
