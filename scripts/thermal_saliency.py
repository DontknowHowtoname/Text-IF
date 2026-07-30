"""Thermal saliency maps for supervising cross-attention.

Builds a soft target map highlighting where attention *should* focus:
  1. Hot regions in the IR image (top-k% brightest pixels, Gaussian-smoothed)
  2. Object regions from YOLO bbox annotations (Gaussian painted at box centers,
     sigma scaled by box dims)

The two branches are blended and min-max normalized per sample to [0, 1],
then used as an MSE target for the model's per-level attention maps.

Designed for FLIR-align-3class. The IR images are RGB-converted grayscale
(channels identical) — we average to recover a single intensity channel.
"""
import torch
import torch.nn.functional as F


def _gaussian_2d(H, W, cx, cy, sigma_x, sigma_y, device):
    """Return a [H, W] Gaussian centered at (cx, cy) in pixel coords."""
    ys = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
    return torch.exp(-((xs - cx) ** 2) / (2 * sigma_x ** 2)
                     - ((ys - cy) ** 2) / (2 * sigma_y ** 2))


def _separable_gaussian_blur(x, ksize, device):
    """Apply a separable Gaussian blur to [B, 1, H, W].

    ksize must be odd. Sigma derived from ksize (sigma = ksize / 6).
    """
    pad = ksize // 2
    sigma = ksize / 6.0
    coords = torch.arange(ksize, device=device, dtype=torch.float32) - pad
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = (g1d / g1d.sum()).view(1, 1, ksize)
    x = F.conv2d(x, g1d.view(1, 1, ksize, 1), padding=(pad, 0))
    x = F.conv2d(x, g1d.view(1, 1, 1, ksize), padding=(0, pad))
    return x


def build_thermal_saliency(ir, bboxes, top_k_pct=0.15, sigma_factor=0.3,
                           ir_gamma=0.5, bbox_gamma=0.5):
    """Build per-sample thermal saliency map.

    Args:
        ir: [B, 3, H, W] IR image (RGB-converted grayscale; channels identical).
        bboxes: length-B list; each element is a [N, 5] tensor of
                (class, cx_norm, cy_norm, w_norm, h_norm) in YOLO format,
                or a [0, 5] tensor / None when no boxes exist.
        top_k_pct: fraction of brightest pixels to keep in the IR branch.
        sigma_factor: Gaussian sigma as fraction of per-axis bbox dims.
        ir_gamma: blend weight for the IR-intensity branch.
        bbox_gamma: blend weight for the bbox-Gaussian branch.

    Returns:
        [B, H, W] saliency, min-max normalized per sample to [0, 1].
    """
    B, _, H, W = ir.shape
    device = ir.device

    # --- IR intensity branch: mean over channels -> top-k mask -> blur ---
    gray = ir.mean(dim=1)  # [B, H, W]
    k = max(1, int(H * W * top_k_pct))
    flat = gray.view(B, -1)
    _, topk_idx = flat.topk(k, dim=-1)
    mask = torch.zeros_like(flat)
    mask.scatter_(1, topk_idx, 1.0)
    mask = mask.view(B, 1, H, W)  # [B, 1, H, W]
    ksize = max(3, int(0.05 * min(H, W)) | 1)  # odd kernel ~5% of shorter side
    mask = _separable_gaussian_blur(mask, ksize, device)
    ir_saliency = mask.squeeze(1)  # [B, H, W]

    # --- BBox Gaussian branch ---
    bbox_saliency = torch.zeros(B, H, W, device=device)
    for b in range(B):
        boxes = bboxes[b]
        if boxes is None:
            continue
        if isinstance(boxes, torch.Tensor) and boxes.numel() == 0:
            continue
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.tolist()
        for box in boxes:
            _, cx_n, cy_n, w_n, h_n = box
            cx, cy = cx_n * W, cy_n * H
            sigma_x = max(1.0, w_n * W * sigma_factor)
            sigma_y = max(1.0, h_n * H * sigma_factor)
            g = _gaussian_2d(H, W, cx, cy, sigma_x, sigma_y, device)
            bbox_saliency[b] = torch.maximum(bbox_saliency[b], g)

    # --- Blend + per-sample min-max normalize ---
    saliency = ir_gamma * ir_saliency + bbox_gamma * bbox_saliency
    flat = saliency.view(B, -1)
    mn = flat.min(dim=-1, keepdim=True).values
    mx = flat.max(dim=-1, keepdim=True).values
    flat = (flat - mn) / (mx - mn + 1e-8)
    return flat.view(B, H, W)


def resize_saliency_to_attn(saliency, attn_h, attn_w):
    """Bilinear-resize saliency to match an attention map's resolution.

    Args:
        saliency: [B, H, W]
        attn_h, attn_w: target resolution

    Returns:
        [B, attn_h, attn_w]
    """
    s = saliency.unsqueeze(1)  # [B, 1, H, W]
    s = F.interpolate(s, size=(attn_h, attn_w),
                     mode='bilinear', align_corners=False)
    return s.squeeze(1)
