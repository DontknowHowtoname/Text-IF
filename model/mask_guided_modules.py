"""Mask-guided feature modulation for object-level enhancement.

Replaces FeatureWiseAffine at decoder levels 2-4 with dual-path modulation:
- Global path: original FeatureWiseAffine (text -> channel-wise gamma/beta, spatially uniform)
- Object path: mask-encoded spatial refinement (mask -> spatially-varying gamma/beta)
When mask is None, behaves identically to FeatureWiseAffine.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Text_IF_model import FeatureWiseAffine


class MaskGuidedAffine(nn.Module):
    def __init__(self, text_dim, feat_channels):
        super(MaskGuidedAffine, self).__init__()
        # Global path (preserves original FeatureWiseAffine)
        self.global_affine = FeatureWiseAffine(text_dim, feat_channels)

        # Object path: encode binary mask to spatial weight map
        self.mask_encode = nn.Sequential(
            nn.Conv2d(1, feat_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 4, feat_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # Object path: spatial refinement generates spatially-varying gamma/beta
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(feat_channels * 2, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels * 2, 3, padding=1),
        )

    def forward(self, feat, text_embed, mask=None):
        """
        Args:
            feat: [B, C, H, W] decoder feature map
            text_embed: [B, 512] CLIP text features
            mask: [B, 1, H_orig, W_orig] pre-computed object mask (optional)
        Returns:
            [B, C, H, W] modulated features
        """
        # Global modulation (always active)
        out = self.global_affine(feat, text_embed)

        if mask is not None:
            B, C, H, W = feat.shape
            mask_resized = F.interpolate(
                mask.float(), size=(H, W), mode='bilinear', align_corners=False
            )

            # Skip if mask is all zeros (no object in this image)
            if mask_resized.sum() > 0:
                # Encode mask to per-channel spatial weight
                spatial_weight = self.mask_encode(mask_resized)  # [B, C, H, W]

                # Generate spatially-varying gamma/beta from masked+bg features
                masked_feat = feat * spatial_weight
                bg_feat = feat * (1 - spatial_weight)
                refined = self.spatial_refine(
                    torch.cat([masked_feat, bg_feat], dim=1)
                )
                gamma_s, beta_s = refined.chunk(2, dim=1)  # each [B, C, H, W]

                # Object-enhanced modulation
                obj_enhanced = (1 + gamma_s) * feat + beta_s

                # Blend: object region gets enhanced, background keeps global
                out = out * (1 - mask_resized) + obj_enhanced * mask_resized

        return out
