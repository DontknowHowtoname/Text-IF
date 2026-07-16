"""Smoke tests for hybrid TextSpatialAffine module.

The hybrid design combines channel-wise affine (main, matches FeatureWiseAffine)
with a bounded spatial gate (auxiliary, zero-init = identity). These tests
verify shape correctness, gradient flow, gate boundedness, and attention shape.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from model.text_spatial_affine import TextSpatialAffine


def test_output_shape_matches_input():
    """Output must have the same shape as input feature."""
    B, C, H, W = 2, 64, 32, 32
    feat = torch.randn(B, C, H, W)
    text = torch.randn(B, 512)
    m = TextSpatialAffine(text_dim=512, feat_channels=C, num_heads=4)
    out = m(feat, text)
    assert out.shape == feat.shape, f"Expected {feat.shape}, got {out.shape}"


def test_gate_zero_init_makes_spatial_path_identity():
    """With zero-init gate_conv, the spatial gate is exactly 1.0 everywhere.

    Note: the channel-wise gamma/beta use default init (not zero), so the full
    output is NOT identical to the input. We only test that the spatial gate
    contributes nothing at initialization, i.e. out == channel_out.
    """
    B, C, H, W = 1, 16, 8, 8
    feat = torch.randn(B, C, H, W)
    text = torch.randn(B, 512)
    m = TextSpatialAffine(text_dim=512, feat_channels=C, num_heads=4)

    # Compute channel-only output manually
    gamma_beta = m.MLP(text)
    gamma, beta = gamma_beta.chunk(2, dim=-1)
    gamma = gamma.view(B, C, 1, 1)
    beta = beta.view(B, C, 1, 1)
    channel_out = (1 + gamma) * feat + beta

    # Full forward
    out = m(feat, text)

    # Spatial gate should be exactly 1.0 at init => out == channel_out
    assert torch.allclose(out, channel_out, atol=1e-5), \
        f"Spatial gate should be identity at init, max diff: {(out - channel_out).abs().max():.6f}"


def test_gate_bounded_to_gate_scale():
    """Spatial gate must stay within [1 - gate_scale, 1 + gate_scale].

    Randomize gate_conv weights/biases to non-zero values; the tanh + scale
    bound must hold regardless.
    """
    B, C, H, W = 2, 32, 16, 16
    feat = torch.randn(B, C, H, W)
    text = torch.randn(B, 512)
    gate_scale = 0.1
    m = TextSpatialAffine(text_dim=512, feat_channels=C, num_heads=4,
                          gate_scale=gate_scale)
    # Randomize gate_conv to non-zero values
    with torch.no_grad():
        m.gate_conv.weight.normal_(0, 2.0)
        m.gate_conv.bias.normal_(0, 2.0)

    # Reconstruct gate: re-run cross-attention path manually
    N = H * W
    q = m.q_proj(text).view(B, m.num_heads, m.head_dim)
    k = m.k_proj(feat).view(B, m.num_heads, m.head_dim, N)
    attn_logits = torch.einsum('bhd,bhdn->bhn', q, k) * m.scale
    attn_probs = F.softmax(attn_logits, dim=-1)
    attn_map = attn_probs.view(B, m.num_heads, H, W)
    gate = 1.0 + torch.tanh(m.gate_conv(attn_map)) * gate_scale

    assert gate.min().item() >= 1.0 - gate_scale - 1e-6, \
        f"Gate min {gate.min().item()} below bound {1.0 - gate_scale}"
    assert gate.max().item() <= 1.0 + gate_scale + 1e-6, \
        f"Gate max {gate.max().item()} above bound {1.0 + gate_scale}"


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


def test_gradient_flows_to_attention_path():
    """q_proj and k_proj (cross-attention) must receive gradients via the gate."""
    B, C, H, W = 1, 16, 8, 8
    feat = torch.randn(B, C, H, W)
    text = torch.randn(B, 512)
    m = TextSpatialAffine(text_dim=512, feat_channels=C, num_heads=4)
    out = m(feat, text)
    out.sum().backward()
    assert m.q_proj.weight.grad is not None, "No gradient on q_proj"
    assert m.k_proj.weight.grad is not None, "No gradient on k_proj"
    assert m.gate_conv.weight.grad is not None, "No gradient on gate_conv"


if __name__ == '__main__':
    test_output_shape_matches_input()
    print("PASS: test_output_shape_matches_input")
    test_gate_zero_init_makes_spatial_path_identity()
    print("PASS: test_gate_zero_init_makes_spatial_path_identity")
    test_gate_bounded_to_gate_scale()
    print("PASS: test_gate_bounded_to_gate_scale")
    test_attention_map_shape()
    print("PASS: test_attention_map_shape")
    test_gradient_flows()
    print("PASS: test_gradient_flows")
    test_gradient_flows_to_attention_path()
    print("PASS: test_gradient_flows_to_attention_path")
    print("All tests passed.")
