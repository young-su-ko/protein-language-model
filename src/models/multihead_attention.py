from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5 # we scale by the square root of the head dimension to prevent the dot product from becoming too large?

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for param in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(param.weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor: # i think attn_mask will be for padded tokens

        q = self.q_proj(query) # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, embed_dim]
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Now, split up the embed_dim into num_heads * head_dim

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Now we need to compute the attention scores with a matmul 
        attn_scores = torch.matmul(q, k.transpose(-2, -1))*self.scaling # k.transpose(-2, -1) = [batch, #head, seq, dim] -> [batch, #head, dim, seq] so that after matmul we get [batch, #head, seq, seq]
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))

        attn_out = torch.matmul(attn_weights, v) # [batch, #head, seq, seq] @ [batch, #head, seq, dim] -> [batch, #head, seq, dim]
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)") # rejoin heads

        output = self.out_proj(attn_out)
        return output




        
