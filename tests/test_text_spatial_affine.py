"""Smoke tests for TextSpatialAffine module."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.text_spatial_affine import TextSpatialAffine


def test_output_shape_matches_input():
    """Output must have the same shape as input feature."""
    B, C, H, W = 2, 64, 32, 32
    feat = torch.randn(B, C, H, W)
    text = torch.randn(B, 512)
    m = TextSpatialAffine(text_dim=512, feat_channels=C, num_heads=4)
    out = m(feat, text)
    assert out.shape == feat.shape, f"Expected {feat.shape}, got {out.shape}"


def test_zero_init_produces_identity():
    """With zero-init gamma/beta, output ≈ feat * 1 + 0 = feat."""
    B, C, H, W = 1, 16, 8, 8
    feat = torch.randn(B, C, H, W)
    text = torch.randn(B, 512)
    m = TextSpatialAffine(text_dim=512, feat_channels=C, num_heads=4)
    # Zero-init already done in __init__
    out = m(feat, text)
    assert torch.allclose(out, feat, atol=1e-5), \
        f"Zero-init should produce identity, max diff: {(out - feat).abs().max():.6f}"


def test_attention_map_shape():
    """Optional return_attn should produce [B, num_heads, H, W] map."""
    B, C, H, W = 1, 32, 16, 16
    feat = torch.randn(B, C, H, W)
    text = torch.randn(B, 512)
    m = TextSpatialAffine(text_dim=512, feat_channels=C, num_heads=4)
    out, attn = m(feat, text, return_attn=True)
    assert attn.shape == (B, 4, H, W), f"Expected (1, 4, {H}, {W}), got {attn.shape}"


def test_gradient_flows():
    """Backward pass should populate gradients on all learnable params."""
    B, C, H, W = 1, 16, 8, 8
    feat = torch.randn(B, C, H, W, requires_grad=True)
    text = torch.randn(B, 512)
    m = TextSpatialAffine(text_dim=512, feat_channels=C, num_heads=4)
    out = m(feat, text)
    out.sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"No gradient on {name}"
    assert feat.grad is not None, "No gradient on input feat"


if __name__ == '__main__':
    test_output_shape_matches_input()
    print("PASS: test_output_shape_matches_input")
    test_zero_init_produces_identity()
    print("PASS: test_zero_init_produces_identity")
    test_attention_map_shape()
    print("PASS: test_attention_map_shape")
    test_gradient_flows()
    print("PASS: test_gradient_flows")
    print("All tests passed.")
