"""Frozen SAM ViT-B + CLIP filter for online mask generation during iterative fusion.

Given a fused image tensor, produces a binary object mask by:
1. SAM AutomaticMaskGenerator -> all candidate masks
2. CLIP cosine similarity filtering by obj_text -> keep relevant masks
3. Merge into single binary mask

All parameters are frozen (no gradient). Used inside torch.no_grad() context.
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import clip

# Add SAM to path
sam_path = os.path.join(os.path.dirname(__file__), '..', 'references', 'segment-anything')
sys.path.insert(0, os.path.abspath(sam_path))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def _get_mask_crop(image_np, mask):
    """Crop the masked region from the image for CLIP encoding."""
    bbox = mask['bbox']
    x, y, w, h = bbox
    pad = max(w, h) // 4
    x1 = max(0, int(x - pad))
    y1 = max(0, int(y - pad))
    x2 = min(image_np.shape[1], int(x + w + pad))
    y2 = min(image_np.shape[0], int(y + h + pad))

    crop = image_np[y1:y2, x1:x2]
    seg = mask['segmentation'][y1:y2, x1:x2]
    return crop * seg[:, :, np.newaxis]


def _filter_masks_by_clip(masks, image_np, text_features, clip_model, clip_preprocess,
                          device, threshold=0.22):
    """Filter SAM masks by CLIP cosine similarity with pre-computed text features."""
    if len(masks) == 0:
        return []

    filtered = []
    for mask in masks:
        if mask['area'] < 500:
            continue

        crop = _get_mask_crop(image_np, mask)
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        crop_pil = Image.fromarray(crop)
        crop_input = clip_preprocess(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(crop_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze().item()
        if similarity >= threshold:
            filtered.append(mask)

    return filtered


def _merge_masks(masks, height, width):
    """Merge multiple SAM mask dicts into a single binary mask."""
    combined = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, mask['segmentation'].astype(np.uint8) * 255)
    return combined


def _tensor_to_numpy_img(tensor):
    """Convert [C, H, W] tensor in [0,1] to [H, W, 3] uint8 numpy."""
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


class IterativeSAMFilter(nn.Module):
    """Frozen SAM ViT-B + CLIP filter for online mask generation.

    Usage:
        sam_filter = IterativeSAMFilter(
            sam_ckpt='references/segment-anything/checkpoints/sam_vit_b_01ec64.pth',
            obj_text='person',
            clip_model=clip_model,  # shared with Fusion model
            device=device
        )

        # Inside forward pass (under torch.no_grad()):
        mask = sam_filter(fused_tensor)  # [B, 1, H, W]
    """
    def __init__(self, sam_ckpt, obj_text, clip_model, clip_preprocess, device,
                 clip_threshold=0.22):
        super(IterativeSAMFilter, self).__init__()

        # Load SAM ViT-B (frozen)
        print(f"Loading SAM ViT-B for iterative filtering: {sam_ckpt}")
        sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
        sam.to(device)
        for p in sam.parameters():
            p.requires_grad = False
        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

        # CLIP model (shared, frozen)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

        # Pre-compute text features
        text_tokens = clip.tokenize([obj_text]).to(device)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tokens)
            self.register_buffer('text_features', text_feat / text_feat.norm(dim=-1, keepdim=True))

        self.clip_threshold = clip_threshold
        self.device = device

    @torch.no_grad()
    def forward(self, fused_tensor):
        """Generate object masks from fused image tensor.

        Args:
            fused_tensor: [B, 3, H, W] fused image in [0, 1]
        Returns:
            [B, 1, H, W] binary mask tensor (float, 0.0 or 1.0)
        """
        B, C, H, W = fused_tensor.shape
        masks_out = []

        for i in range(B):
            img_np = _tensor_to_numpy_img(fused_tensor[i])

            # SAM automatic mask generation
            sam_masks = self.generator.generate(img_np)

            # CLIP filter
            filtered = _filter_masks_by_clip(
                sam_masks, img_np, self.text_features,
                self.clip_model, self.clip_preprocess,
                self.device, self.clip_threshold
            )

            # Merge
            if len(filtered) > 0:
                merged = _merge_masks(filtered, H, W)
            else:
                merged = np.zeros((H, W), dtype=np.uint8)

            mask_tensor = torch.from_numpy(merged.astype(np.float32) / 255.0)
            masks_out.append(mask_tensor.unsqueeze(0))  # [1, H, W]

        return torch.stack(masks_out, dim=0).to(self.device)  # [B, 1, H, W]
