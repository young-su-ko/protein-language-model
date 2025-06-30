import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, RMSNorm
from myplm.multihead_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        num_heads: int,
        use_rotary_embedding: bool = True,
        norm_type: str = "layernorm",  # "layernorm" or "rmsnorm"
        bias: bool = False,
    ):
        super().__init__()
        norm_cls = {
            "layernorm": LayerNorm,
            "rmsnorm": RMSNorm
        }[norm_type]

        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.num_heads = num_heads
        self.use_rotary_embedding = use_rotary_embedding
        self.self_attn = MultiHeadAttention(
            embed_dim, 
            num_heads, 
            bias=bias, 
            )
        self.self_attn_layer_norm = norm_cls(embed_dim)
        self.final_layer_norm = norm_cls(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_embed_dim, bias=bias),
            nn.GELU(),
            nn.Linear(ffn_embed_dim, embed_dim, bias=bias),
        )


    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.self_attn_layer_norm(x) # pre-attention norm
        x = self.self_attn(x, mask=mask) # padding attn mask
        
        x += residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x += residual

        return x

class LMHead(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int, embedding_weights: Optional[Tensor] = None):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.layer_norm = LayerNorm(embed_dim)
        
        if embedding_weights is not None:
            self.embedding_weights = embedding_weights
        else:
            self.embedding_weights = None
            self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
            
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.gelu = nn.GELU()


    def forward(self, features: Tensor) -> Tensor:
        x = self.linear(features)
        x = self.gelu(x)
        x = self.layer_norm(x)

        if self.embedding_weights is not None:
            return F.linear(x, self.embedding_weights) + self.bias
        
        return self.decoder(x) + self.bias
