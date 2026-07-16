"""Hybrid text-driven modulation: channel-wise affine + bounded spatial gate.

Design rationale (see docs/superpowers/specs/2026-07-03-flir-text-fusion-design.md):
- Channel-wise gamma/beta matches the original FeatureWiseAffine exactly,
  preserving its training stability on the Text-IF base architecture.
- A bounded spatial gate (±gate_scale, tanh-constrained, zero-init) introduces
  text-driven spatial variation gradually without risking divergence.
- The cross-attention map is always returned via return_attn=True for
  visualization (paper heatmaps showing where text attended).

Forward:
    feat: [B, C, H, W]
    text_embed: [B, text_dim]
    -> [B, C, H, W]

Modulation:
    out = ((1 + gamma) * feat + beta) * gate
    where gamma, beta are channel-wise [B, C, 1, 1] (unbounded, like base)
          gate is spatial [B, 1, H, W] bounded to [1 - s, 1 + s], zero-init = 1
"""
import torch
import torch.nn as nn


class TextSpatialAffine(nn.Module):
    """Hybrid modulation: channel affine (main) + bounded spatial gate (auxiliary).

    Args:
        text_dim: CLIP text feature dim (default 512).
        feat_channels: channels of the input feature map (C).
        num_heads: attention heads for cross-attention (default 4).
        gate_scale: bounds the spatial gate to [1 - s, 1 + s]. Default 0.1
            means ±10% multiplicative variation around the channel-modulated
            feature. Small value keeps training stable while still allowing
            text-driven spatial differentiation.

    Input:
        feat: [B, C, H, W] decoder feature map
        text_embed: [B, text_dim] CLIP text features
        return_attn: if True, also return attention map [B, num_heads, H, W]
                     for visualization

    Output:
        [B, C, H, W] modulated feature
    """

    def __init__(self, text_dim=512, feat_channels=64, num_heads=4, gate_scale=0.1):
        super().__init__()
        assert text_dim % num_heads == 0, \
            f"text_dim {text_dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gate_scale = gate_scale

        # Cross-attention projections (text query x image keys).
        # Drives the spatial gate and provides visualization.
        self.q_proj = nn.Linear(text_dim, text_dim)
        self.k_proj = nn.Conv2d(feat_channels, text_dim, 1)

        # Spatial gate: [B, num_heads, H, W] -> [B, 1, H, W]
        # Zero-init so gate = 1 + tanh(0) * scale = 1.0 (identity at start).
        # This means initial behavior is EXACTLY the channel-only modulation,
        # matching the original FeatureWiseAffine for stable warmup.
        self.gate_conv = nn.Conv2d(num_heads, 1, 1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

        # Channel-wise gamma/beta MLP (identical to original FeatureWiseAffine).
        # No bound constraint — relies on the same dynamics that made the
        # original Text-IF training stable.
        self.MLP = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(text_dim * 2, feat_channels * 2),
        )

    def forward(self, feat, text_embed, return_attn=False):
        B, C, H, W = feat.shape
        N = H * W

        # --- Channel-wise modulation (main path, matches FeatureWiseAffine) ---
        gamma_beta = self.MLP(text_embed)  # [B, C*2]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, C]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        channel_out = (1 + gamma) * feat + beta

        # --- Cross-attention (drives gate + visualization) ---
        q = self.q_proj(text_embed).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(feat).view(B, self.num_heads, self.head_dim, N)
        # attn_logits: [B, num_heads, N]
        attn_logits = torch.einsum('bhd,bhdn->bhn', q, k) * self.scale
        attn_probs = torch.softmax(attn_logits, dim=-1)  # [B, num_heads, N]
        attn_map = attn_probs.view(B, self.num_heads, H, W)

        # --- Bounded spatial gate (zero-init => identity at start) ---
        # tanh bounds the conv output to [-1, 1]; scaling by gate_scale
        # further restricts multiplicative variation to +/- gate_scale.
        gate = 1.0 + torch.tanh(self.gate_conv(attn_map)) * self.gate_scale  # [B, 1, H, W]

        # Final output: channel-modulated feature, gently gated per-pixel.
        out = channel_out * gate

        if return_attn:
            # Return softmax-normalized attention for visualization (paper figures).
            # This is the same attn_map used to drive the gate, so visualization
            # matches what actually influences the output spatially.
            return out, attn_map
        return out
