import torch

from components.vit.vit_block import VitBlock


def test_vit_block_sdpa_matches_explicit_attention():
    torch.manual_seed(0)
    block = VitBlock(dim=32, num_heads=4).eval()
    x = torch.randn(2, 12, 32, requires_grad=True)
    padding_mask = torch.zeros(2, 12, dtype=torch.bool)
    padding_mask[0, -2:] = True
    padding_mask[1, -3:] = True

    actual = block(x, padding_mask=padding_mask)

    x_norm = block.norm1(x)
    qkv = block.qkv(x_norm).reshape(2, 12, 3, 4, 8).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    attn = (q @ k.transpose(-2, -1)) * (8**-0.5)
    attn = attn.masked_fill(padding_mask[:, None, None, :], torch.finfo(attn.dtype).min)
    attn = attn.masked_fill(padding_mask[:, None, :, None], 0.0)
    attn = attn.softmax(dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(2, 12, 32)
    out = out.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    expected = x + block.drop_path(block.proj(block.proj_drop(out)))
    expected = expected + block.drop_path(block.mlp(block.norm2(expected)))
    expected = expected.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    (actual_grad,) = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected.square().sum(), x)
    assert torch.allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)
