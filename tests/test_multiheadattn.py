import torch
from myplm.multihead_attention import MultiHeadAttention

def test_forward_shape():
    B, L, D = 2, 5, 8
    x = torch.randn(B, L, D)
    attn = MultiHeadAttention(D, 4)
    out = attn(x)
    assert out.shape == x.shape

def test_forward_with_mlm_mask():
    B, L, D = 2, 5, 8
    x = torch.randn(B, L, D)
    attn = MultiHeadAttention(D, 4)
    attn_mask = torch.zeros(B, 1, 1, L, dtype=torch.bool)
    out = attn(x, attn_mask)
    assert out.shape == x.shape

def test_forward_with_clm_mask():
    B, L, D = 2, 5, 8
    x = torch.randn(B, L, D)
    attn = MultiHeadAttention(D, 4)
    attn_mask = torch.zeros(B, 1, L, L, dtype=torch.bool)
    out = attn(x, attn_mask)
    assert out.shape == x.shape
