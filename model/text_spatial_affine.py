"""Cross-attention based spatial modulation module.

Replaces FeatureWiseAffine (channel-wise gamma/beta) with spatially-varying
gamma/beta driven by text-image cross-attention. Text is the query, image
patches are the keys; attention weights are reshaped to a spatial map and
converted to per-pixel gamma/beta via 1x1 convs.

Zero-init on gamma/beta convs ensures the module acts as identity at start,
so the network starts well-conditioned.
"""
import torch
import torch.nn as nn


class TextSpatialAffine(nn.Module):
    """Text-driven spatial affine modulation.

    Args:
        text_dim: CLIP text feature dim (default 512).
        feat_channels: channels of the input feature map (C).
        num_heads: attention heads (default 4). text_dim must be divisible.

    Input:
        feat: [B, C, H, W] decoder feature map
        text_embed: [B, text_dim] CLIP text features
        return_attn: if True, also return attention map for visualization

    Output:
        [B, C, H, W] modulated feature
    """

    def __init__(self, text_dim=512, feat_channels=64, num_heads=4):
        super().__init__()
        assert text_dim % num_heads == 0, \
            f"text_dim {text_dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Text query projection: [B, text_dim] -> [B, text_dim]
        self.q_proj = nn.Linear(text_dim, text_dim)

        # Image key projection: [B, C, H, W] -> [B, text_dim, H, W]
        self.k_proj = nn.Conv2d(feat_channels, text_dim, 1)

        # Spatial gamma/beta: [B, num_heads, H, W] -> [B, 1, H, W]
        self.gamma_conv = nn.Conv2d(num_heads, 1, 1)
        self.beta_conv = nn.Conv2d(num_heads, 1, 1)

        # Zero-init so initial output = feat * 1 + 0 * mean = feat (identity)
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)
        nn.init.zeros_(self.beta_conv.weight)
        nn.init.zeros_(self.beta_conv.bias)

    def forward(self, feat, text_embed, return_attn=False):
        B, C, H, W = feat.shape
        N = H * W

        # Text query: [B, text_dim] -> [B, num_heads, head_dim]
        q = self.q_proj(text_embed).view(B, self.num_heads, self.head_dim)

        # Image keys: [B, C, H, W] -> [B, text_dim, H, W] -> [B, num_heads, head_dim, N]
        k = self.k_proj(feat).view(B, self.num_heads, self.head_dim, N)

        # Cross-attention logits: q @ k -> [B, num_heads, N]
        # einsum: b h d, b h d n -> b h n
        attn_logits = torch.einsum('bhd,bhdn->bhn', q, k) * self.scale
        # Apply softmax over spatial positions so attention is normalized
        # probabilities per head (sum-to-1 over N). This makes gamma/beta
        # driven by *relative* text-image alignment, not by raw logit magnitude,
        # and keeps the visualization attention consistent with what drives
        # the modulation (spec §5.3).
        attn_probs = torch.softmax(attn_logits, dim=-1)  # [B, num_heads, N]
        # [B, num_heads, N] -> [B, num_heads, H, W]
        attn_map = attn_probs.view(B, self.num_heads, H, W)

        # Spatial gamma/beta from attention map (zero-init => identity at start)
        gamma = self.gamma_conv(attn_map)  # [B, 1, H, W]
        beta = self.beta_conv(attn_map)    # [B, 1, H, W]

        # Modulate. feat_mean acts as a per-pixel baseline for beta.
        feat_mean = feat.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        out = feat * (1 + gamma) + beta * feat_mean

        if return_attn:
            # Same normalized probabilities used for modulation, returned for
            # visualization without re-computing.
            return out, attn_map
        return out
