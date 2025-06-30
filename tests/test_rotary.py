import torch
from myplm.rotary_embedding import rotate_half, apply_rotary_pos_emb, RotaryEmbedding

def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rotated = rotate_half(x)
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.allclose(rotated, expected)

def test_apply_rotary_pos_emb_shapes():
    B, N, L, D = 2, 5, 8, 16
    x = torch.randn(B, N, L, D)
    cos = torch.ones(1, L, D)
    sin = torch.zeros(1, L, D)
    out = apply_rotary_pos_emb(x, cos, sin)
    assert out.shape == x.shape
    assert torch.allclose(out, x)  # since sin is 0 and cos is 1

def test_rotary_embedding_forward_shape():
    B, N, L, D = 2, 5, 8, 16
    x = torch.randn(B, N, L, D)
    rotary = RotaryEmbedding(D)
    out = rotary(x)
    assert out.shape == x.shape

def test_rotary_embedding_repeat_forward():
    B, N, L, D = 2, 5, 8, 16
    rotary = RotaryEmbedding(D)
    x1 = torch.randn(B, N, L, D)
    x2 = torch.randn(B, N, L+10, D)
    _ = rotary(x1)
    cos1, _ = rotary._update_cos_sin_cache(x1)
    _ = rotary(x2)
    cos2, _ = rotary._update_cos_sin_cache(x2)
    assert cos2.shape[1] > cos1.shape[1]  # check that cache is updated for longer sequences

def test_requires_grad():
    B, N, L, D = 2, 5, 8, 16
    x = torch.randn(B, N, L, D, requires_grad=True)
    rotary = RotaryEmbedding(D)
    out = rotary(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
