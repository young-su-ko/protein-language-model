from typing import Union, Optional
from torch import Tensor
import torch.nn as nn
from src.submodules import TransformerBlock, LMHead
from src.tokenizer import Alphabet
from torch.nn import LayerNorm

class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 320,
        ffn_embed_dim: int = 1280,
        num_heads: int = 20,
        initializer_range: float = 0.02,
        bias: bool = False,
        weight_tying: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.num_heads = num_heads
        self.weight_tying = weight_tying
        
        self.alphabet = Alphabet()
        self.alphabet_size = len(self.alphabet)
        self.padding_idx = self.alphabet.pad_idx
        self.bos_idx = self.alphabet.bos_idx
        self.eos_idx = self.alphabet.eos_idx
        self.mask_idx = self.alphabet.mask_idx
        self.initializer_range = initializer_range

        self._init_submodules()
        self.apply(self._init_weights)

    def _init_submodules(self):
        self.embed_tokens = nn.Embedding(self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [TransformerBlock(self.embed_dim, self.ffn_embed_dim, self.num_heads) for _ in range(self.num_layers)])
        
        if self.weight_tying:
            self.lm_head = LMHead(self.embed_dim, len(self.alphabet), self.embed_tokens.weight)
        else:
            self.lm_head = LMHead(self.embed_dim, len(self.alphabet))

        self.last_layer_norm = LayerNorm(self.embed_dim)

    def _init_weights(self, module):
        """Initialize the weights""" # this is exactly how esm hugginface inits weights. i dont know how much it matters
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, LMHead):
            module.bias.data.zero_()

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> dict:
        assert tokens.ndim == 2, "tokens must be a 2D tensor"
    
        x = self.embed_tokens(tokens)

        for layer in self.layers:
            x = layer(x, mask=mask)
        
        x = self.last_layer_norm(x)
        last_layer_hidden_state = x # add the representation after layernorm
        x = self.lm_head(x)

        outputs = {"logits": x, "representations": last_layer_hidden_state}
        return outputs
