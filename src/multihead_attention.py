from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange
from src.rotary_embedding import RotaryEmbedding

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, use_rotary_embedding: bool = True):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.scaling = self.head_dim ** -0.5 # we scale by the square root of the head dimension to prevent the dot product from becoming too large?
#         self.use_rotary_embedding = use_rotary_embedding
#         self.rotary_embedding = RotaryEmbedding(self.head_dim)
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

#         self.init_weights()

#     def init_weights(self):
#         for param in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
#             nn.init.xavier_uniform_(param.weight)

#     def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor: # i think attn_mask will be for padded 
#         # currently, attn_mask is [batch, seq_len], but it should be [batch, 1, seq_len, seq_len]

        
#         q = self.q_proj(query) # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, embed_dim]
#         k = self.k_proj(key)
#         v = self.v_proj(value)

#         # Now, split up the embed_dim into num_heads * head_dim

#         q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
#         k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
#         v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)


#         if self.use_rotary_embedding:
#             q, k = self.rotary_embedding(q, k)
            
#         # Now we need to compute the attention scores with a matmul 
#         attn_scores = torch.matmul(q, k.transpose(-2, -1))*self.scaling # k.transpose(-2, -1) = [batch, #head, seq, dim] -> [batch, #head, dim, seq] so that after matmul we get [batch, #head, seq, seq]
#         if attn_mask is not None:
#             attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
#         attn_weights = F.softmax(attn_scores, dim=-1) # esm got rid of dropout, but just putting here in case we want to use it
#         # if attn_mask is not None:
#         #     attn_weights = attn_weights.masked_fill(attn_mask, 0)
#         attn_out = torch.matmul(attn_weights, v) # [batch, #head, seq, seq] @ [batch, #head, seq, dim] -> [batch, #head, seq, dim]
#         attn_out = rearrange(attn_out, "b h s d -> b s (h d)") # rejoin heads
#         output = self.out_proj(attn_out)
#         return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: [batch, seq_len, embed_dim]
        qkv = self.qkv_proj(x)  # [B, L, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        # Apply RoPE
        q = self.rotary(q)
        k = self.rotary(k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )  # [batch, num_heads, seq_len, head_dim]

        # Merge heads
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')

        # Output projection
        output = self.out_proj(attn_out)
        return output


        
