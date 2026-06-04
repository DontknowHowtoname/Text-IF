# Iterative SAM-Fusion Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an iterative fusion pipeline where SAM ViT-B segments the fused image to generate object masks, which then guide a second fusion pass for object-level enhancement. No pre-computed masks needed.

**Architecture:** Two-pass fusion with shared weights. Pass 1: global fusion (no mask). Pass 2: SAM+CLIP generates mask from fused image, then fusion runs again with mask guidance. SAM and CLIP are frozen; only the Fusion network trains.

**Tech Stack:** PyTorch, CLIP (ViT-B/32), SAM (ViT-B), segment-anything library

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `model/sam_iterative_filter.py` | CREATE | IterativeSAMFilter: SAM ViT-B + CLIP filter wrapper |
| `model/Text_IF_recon_model_4.py` | CREATE | Text_IF_Recon_v4: iterative fusion with encoder caching |
| `scripts/utils.py` | MODIFY | Add `train_one_epoch_iterative` and `evaluate_iterative` |
| `train_fusion_iterative.py` | CREATE | Training script (no pre-computed masks) |
| `evaluate_textif_obj_enhance.py` | MODIFY | Update to support v4 iterative mode |

---

### Task 1: Create IterativeSAMFilter

**Files:**
- Create: `model/sam_iterative_filter.py`

- [ ] **Step 1: Create `model/sam_iterative_filter.py`**

```python
"""Frozen SAM ViT-B + CLIP filter for online mask generation during iterative fusion.

Given a fused image tensor, produces a binary object mask by:
1. SAM AutomaticMaskGenerator → all candidate masks
2. CLIP cosine similarity filtering by obj_text → keep relevant masks
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
```

- [ ] **Step 2: Verify import**

Run:
```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF && conda run -n xpu python -c "from model.sam_iterative_filter import IterativeSAMFilter; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add model/sam_iterative_filter.py
git commit -m "feat: add IterativeSAMFilter for online SAM+CLIP mask generation"
```

---

### Task 2: Create Text_IF_Recon_v4

**Files:**
- Create: `model/Text_IF_recon_model_4.py`

- [ ] **Step 1: Create `model/Text_IF_recon_model_4.py`**

```python
"""Text_IF_Recon v4: Iterative fusion with SAM feedback.

Pass 1: Global fusion without mask (identical to v2).
Pass 2: SAM generates mask from fused_1, then fusion runs again with mask guidance.

Encoder runs once and is cached for both passes.
Only the decoder + MaskGuidedAffine path runs multiple times.

When iterations=1 or sam_filter=None, behaves identically to v3 with mask=None.
"""
import torch
import torch.nn as nn

from model.Text_IF_model import Text_IF
from model.freefusion_blocks import FFBlock, FDBlock, ReconHead
from model.mask_guided_modules import MaskGuidedAffine


class Text_IF_Recon_v4(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 iterations=2):
        super(Text_IF_Recon_v4, self).__init__()

        self.iterations = iterations

        # Original Text-IF model as submodule
        self.base = Text_IF(
            model_clip, inp_A_channels, inp_B_channels, out_channels,
            dim, num_blocks, num_refinement_blocks, heads,
            ffn_expansion_factor, bias, LayerNorm_type
        )

        # Replace prompt_guidance at levels 2-4 with MaskGuidedAffine
        self.base.prompt_guidance_4 = MaskGuidedAffine(512, dim * 2 ** 3)
        self.base.prompt_guidance_3 = MaskGuidedAffine(512, dim * 2 ** 2)
        self.base.prompt_guidance_2 = MaskGuidedAffine(512, dim * 2 ** 1)

        # FFBlock fusion at encoder levels 1-3 (same as v2/v3)
        self.ffb_1 = FFBlock(in_channels=dim, out_channels=dim)
        self.ffb_2 = FFBlock(in_channels=dim * 2, out_channels=dim * 2)
        self.ffb_3 = FFBlock(in_channels=dim * 4, out_channels=dim * 4)

        # FDBlock decoupling (same as v2/v3)
        channels_3lev = [dim, dim * 2, dim * 4]
        self.fdb_ir = FDBlock(channels_3lev)
        self.fdb_vis = FDBlock(channels_3lev)

        # Shared reconstruction head (same as v2/v3)
        self.recon_head = ReconHead(
            in_channels=[dim * 4, dim * 2, dim],
            out_channels=out_channels
        )

    def _encode(self, inp_img_A, inp_img_B):
        """Run encoder once, cache results for all fusion passes."""
        out_enc_L4_A, out_enc_L3_A, out_enc_L2_A, out_enc_L1_A = self.base.encoder_A(inp_img_A)
        out_enc_L4_B, out_enc_L3_B, out_enc_L2_B, out_enc_L1_B = self.base.encoder_B(inp_img_B)

        # FFBlock fusion at levels 1-3
        fus_L1 = self.ffb_1(out_enc_L1_A, out_enc_L1_B)
        fus_L2 = self.ffb_2(out_enc_L2_A, out_enc_L2_B)
        fus_L3 = self.ffb_3(out_enc_L3_A, out_enc_L3_B)

        # Detached encoder features for reconstruction losses
        enc_A_3lev = [out_enc_L1_A.detach(), out_enc_L2_A.detach(), out_enc_L3_A.detach()]
        enc_B_3lev = [out_enc_L1_B.detach(), out_enc_L2_B.detach(), out_enc_L3_B.detach()]

        # FDBlock decoupling (only needs to run once)
        fus_feas = [fus_L1, fus_L2, fus_L3]
        dec_ir_feas = self.fdb_ir(fus_feas, enc_A_3lev)
        dec_vis_feas = self.fdb_vis(fus_feas, enc_B_3lev)

        # Reconstructions (shared across passes)
        recon_vis = self.recon_head([enc_A_3lev[2], enc_A_3lev[1], enc_A_3lev[0]])
        recon_ir = self.recon_head([enc_B_3lev[2], enc_B_3lev[1], enc_B_3lev[0]])
        recon_dec_ir = self.recon_head([dec_ir_feas[2], dec_ir_feas[1], dec_ir_feas[0]])
        recon_dec_vis = self.recon_head([dec_vis_feas[2], dec_vis_feas[1], dec_vis_feas[0]])

        # Cached for fusion passes
        cached = {
            'out_enc_L4_A': out_enc_L4_A,
            'out_enc_L4_B': out_enc_L4_B,
            'fus_L1': fus_L1,
            'fus_L2': fus_L2,
            'fus_L3': fus_L3,
        }
        return cached, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis

    def _fusion_pass(self, cached, text_features, mask=None):
        """Run decoder path once with optional mask.

        Args:
            cached: dict with encoder outputs from _encode()
            text_features: [B, 512] CLIP text features
            mask: [B, 1, H, W] or None
        Returns:
            [B, 3, H, W] fused image
        """
        # Clone L4 features for this pass (decoder modifies them)
        out_enc_L4_A = cached['out_enc_L4_A']
        out_enc_L4_B = cached['out_enc_L4_B']
        fus_L1 = cached['fus_L1']
        fus_L2 = cached['fus_L2']
        fus_L3 = cached['fus_L3']

        out_enc_L4_A, out_enc_L4_B = self.base.cross_attention(out_enc_L4_A, out_enc_L4_B)
        out_enc_L4 = self.base.feature_fusion_4(out_enc_L4_A, out_enc_L4_B)
        out_enc_L4 = self.base.attention_spatial(out_enc_L4)
        out_enc_L4 = self.base.prompt_guidance_4(out_enc_L4, text_features, mask)

        out_dec_L4 = self.base.decoder_level4(out_enc_L4)

        inp_dec_L3 = self.base.up4_3(out_dec_L4)
        inp_dec_L3 = self.base.prompt_guidance_3(inp_dec_L3, text_features, mask)
        inp_dec_L3 = torch.cat([inp_dec_L3, fus_L3], 1)
        inp_dec_L3 = self.base.reduce_chan_level3(inp_dec_L3)
        out_dec_L3 = self.base.decoder_level3(inp_dec_L3)

        inp_dec_L2 = self.base.up3_2(out_dec_L3)
        inp_dec_L2 = self.base.prompt_guidance_2(inp_dec_L2, text_features, mask)
        inp_dec_L2 = torch.cat([inp_dec_L2, fus_L2], 1)
        inp_dec_L2 = self.base.reduce_chan_level2(inp_dec_L2)
        out_dec_L2 = self.base.decoder_level2(inp_dec_L2)

        inp_dec_L1 = self.base.up2_1(out_dec_L2)
        inp_dec_L1 = self.base.prompt_guidance_1(inp_dec_L1, text_features)
        inp_dec_L1 = torch.cat([inp_dec_L1, fus_L1], 1)
        out_dec_L1 = self.base.decoder_level1(inp_dec_L1)

        fused = self.base.output(self.base.refinement(out_dec_L1))
        return fused

    def forward(self, inp_img_A, inp_img_B, text, sam_filter=None):
        """
        Args:
            inp_img_A: [B, 3, H, W] visible image
            inp_img_B: [B, 3, H, W] infrared image
            text: CLIP tokenized text [B, 77]
            sam_filter: IterativeSAMFilter instance (optional)
        Returns:
            (fused_final, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis)
            If iterations=1: fused_final == fused_1
        """
        b = inp_img_A.shape[0]
        text_features = self.base.get_text_feature(text.expand(b, -1)).to(inp_img_A.dtype)

        # Encoder runs once
        cached, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = self._encode(
            inp_img_A, inp_img_B)

        # Pass 1: no mask (global fusion)
        fused_1 = self._fusion_pass(cached, text_features, mask=None)

        if self.iterations <= 1 or sam_filter is None:
            return fused_1, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis

        # Pass 2+: SAM generates mask from fused, then fuse again with mask
        fused_prev = fused_1
        for k in range(1, self.iterations):
            with torch.no_grad():
                mask = sam_filter(fused_prev.detach())
            fused_curr = self._fusion_pass(cached, text_features, mask=mask)
            fused_prev = fused_curr

        return fused_prev, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis
```

- [ ] **Step 2: Verify import and shape**

Run:
```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF && conda run -n xpu python -c "
import torch
import clip
from model.Text_IF_recon_model_4 import Text_IF_Recon_v4

device = 'cpu'
model_clip, _ = clip.load('ViT-B/32', device=device)
model = Text_IF_Recon_v4(model_clip, iterations=2).to(device)

x = torch.randn(1, 3, 64, 64)
text = clip.tokenize(['test']).to(device)

# Without sam_filter (iterations=2 but no filter)
out = model(x, x, text, sam_filter=None)
print(f'Outputs: {len(out)}, shapes: {[o.shape for o in out]}')
assert out[0].shape == (1, 3, 64, 64)
assert out[1].shape == (1, 3, 64, 64)
print('OK')
"
```
Expected: `Outputs: 6, shapes: [torch.Size([1, 3, 64, 64]) x 6]`, prints `OK`

- [ ] **Step 3: Commit**

```bash
git add model/Text_IF_recon_model_4.py
git commit -m "feat: add Text_IF_Recon_v4 with iterative SAM-fusion feedback"
```

---

### Task 3: Add Iterative Training/Eval Functions to utils.py

**Files:**
- Modify: `scripts/utils.py`

- [ ] **Step 1: Append `train_one_epoch_iterative` and `evaluate_iterative` to end of `scripts/utils.py`**

Add at the end of file (after the existing `evaluate_obj_enhance` function):

```python


# ====================== Iterative SAM-Fusion Training/Eval ======================

def train_one_epoch_iterative(model, model_clip, sam_filter, optimizer, lr_scheduler,
                               data_loader, device, epoch, recon_weight=1.0,
                               enhance_factor=1.5, bg_factor=0.5, mask_loss_weight=1.0,
                               pass1_weight=0.3):
    model.train()
    model_clip.eval()
    loss_function = fusion_dual_recon_mask_loss(
        recon_weight=recon_weight,
        enhance_factor=enhance_factor,
        bg_factor=bg_factor,
        mask_loss_weight=mask_loss_weight
    )

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    accu_mask_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, _, task, _, _ = data
        text_line = []

        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append(get_low_light_prompt())
            elif task[index] == "over_exposure":
                text_line.append(get_over_exposure_prompt())
            elif task[index] == "ir_low_contrast":
                text_line.append(get_ir_low_contrast_prompt())
            elif task[index] == "ir_noise":
                text_line.append(get_ir_noise_prompt())
            else:
                text_line.append("This is unknown to the image fusion task.")
        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        # v4 forward: (fused_final, fused_1, recon_ir, recon_vis, dec_ir, dec_vis)
        I_fused, I_fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
            I_A, I_B, text, sam_filter=sam_filter)

        # SAM-generated mask for loss (reuse from last iteration)
        with torch.no_grad():
            mask_for_loss = sam_filter(I_fused_1.detach())

        # Pass 2 loss (primary)
        loss_p2, loss_ssim, loss_max, loss_color, loss_text, loss_recon, loss_mask = loss_function(
            I_A_gt, I_B_gt, I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=mask_for_loss)

        # Pass 1 loss (baseline quality)
        loss_p1, _, _, _, _, _, _ = loss_function(
            I_A_gt, I_B_gt, I_fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=None)

        loss = loss_p2 + pass1_weight * loss_p1
        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text.detach()
        accu_recon_loss += loss_recon.detach()
        accu_mask_loss += loss_mask.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = ("[train epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon: {:.3f}  mask: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), accu_mask_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1),
            accu_mask_loss.item() / (step + 1), lr)


@torch.no_grad()
def evaluate_iterative(model, sam_filter, data_loader, device, epoch, lr, filefold_path,
                       recon_weight=1.0, enhance_factor=1.5, bg_factor=0.5,
                       mask_loss_weight=1.0, pass1_weight=0.3):
    loss_function = fusion_dual_recon_mask_loss(
        recon_weight=recon_weight,
        enhance_factor=enhance_factor,
        bg_factor=bg_factor,
        mask_loss_weight=mask_loss_weight
    )
    model.eval()

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_recon_loss = torch.zeros(1).to(device)
    accu_mask_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)

    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name, _ = data
        text_line = []
        for index in range(len(task)):
            if task[index] == "low_light":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the low light degradation.")
            elif task[index] == "over_exposure":
                text_line.append("This is the infrared-visible light fusion task. Visible images have the overexposure degradation.")
            elif task[index] == "ir_low_contrast":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the low contrast degradation.")
            elif task[index] == "ir_noise":
                text_line.append("This is the infrared-visible light fusion task. Infrared images have the noise degradation.")
            else:
                text_line.append("This is unknown to the image fusion task.")

        text = clip.tokenize(text_line).to(device)

        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        I_fused, I_fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
            I_A, I_B, text, sam_filter=sam_filter)

        # SAM mask for loss
        mask_for_loss = sam_filter(I_fused_1)

        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)
                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        # Pass 2 loss
        loss_p2, loss_ssim, loss_max, loss_color, loss_text, loss_recon, loss_mask = loss_function(
            I_A_gt, I_B_gt, I_fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=mask_for_loss)

        # Pass 1 loss
        loss_p1, _, _, _, _, _, _ = loss_function(
            I_A_gt, I_B_gt, I_fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis,
            task, mask=None)

        loss = loss_p2 + pass1_weight * loss_p1

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text
        accu_recon_loss += loss_recon
        accu_mask_loss += loss_mask

        data_loader.desc = ("[val epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                            "color: {:.3f}  text: {:.3f}  recon: {:.3f}  mask: {:.3f}  lr: {:.6f}").format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1),
            accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1),
            accu_recon_loss.item() / (step + 1), accu_mask_loss.item() / (step + 1), lr)

    return (accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_recon_loss.item() / (step + 1),
            accu_mask_loss.item() / (step + 1))
```

- [ ] **Step 2: Verify import**

Run:
```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF && conda run -n xpu python -c "from scripts.utils import train_one_epoch_iterative, evaluate_iterative; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/utils.py
git commit -m "feat: add iterative SAM-fusion training and evaluation functions"
```

---

### Task 4: Create Training Script

**Files:**
- Create: `train_fusion_iterative.py`

- [ ] **Step 1: Create `train_fusion_iterative.py`**

```python
"""
Training script: Text_IF_Recon v4 with iterative SAM-fusion feedback.

Pass 1: Global fusion without mask (v2 baseline).
Pass 2: SAM ViT-B generates mask from fused_1, fusion runs again with mask.

No pre-computed masks needed. SAM+CLIP runs online (frozen).
Based on train_fusion_obj_enhance.py but uses PromptDataSet (no masks).
"""
import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import clip
from data.prompt_dataset import PromptDataSetWithMask

from model.Text_IF_recon_model_4 import Text_IF_Recon_v4 as create_model
from model.sam_iterative_filter import IterativeSAMFilter
from scripts.utils import (read_data, train_one_epoch_iterative, evaluate_iterative,
                            create_lr_scheduler)
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import transforms as T


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/TextIF_iterative_{}".format(file_name)
    os.makedirs(filefold_path)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_val_loss = 1e5
    start_epoch = 0

    print("Loading IVF Fusion and Low-Light Task!")
    if args.low_light_path is not None:
        train_low_light_path_list, val_low_light_path_list = read_data(args.low_light_path)
    else:
        train_low_light_path_list = val_low_light_path_list = None

    print("Loading IVF Fusion and Over-Exposure Task!")
    if args.over_exposure_path is not None:
        train_over_exposure_path_list, val_over_exposure_path_list = read_data(args.over_exposure_path)
    else:
        train_over_exposure_path_list = val_over_exposure_path_list = None

    print("Loading IVF Fusion and ir_low_contrast Task!")
    if args.ir_low_contrast_path is not None:
        train_ir_low_contrast_path_list, val_ir_low_contrast_path_list = read_data(args.ir_low_contrast_path)
    else:
        train_ir_low_contrast_path_list = val_ir_low_contrast_path_list = None

    print("Loading IVF Fusion and ir_noise_path Task!")
    if args.ir_noise_path is not None:
        train_ir_noise_path_list, val_ir_noise_path_list = read_data(args.ir_noise_path)
    else:
        train_ir_noise_path_list = val_ir_noise_path_list = None

    data_transform = {
        "train": T.Compose([T.RandomCrop(96),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    # Use PromptDataSetWithMask for collate_fn compatibility (masks will be zeros)
    train_dataset = PromptDataSetWithMask(
        train_low_light_path_list=train_low_light_path_list,
        val_low_light_path_list=val_low_light_path_list,
        train_over_exposure_path_list=train_over_exposure_path_list,
        val_over_exposure_path_list=val_over_exposure_path_list,
        train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
        val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
        train_ir_noise_path_list=train_ir_noise_path_list,
        val_ir_noise_path_list=val_ir_noise_path_list,
        phase="train",
        transform=data_transform["train"])

    val_dataset = PromptDataSetWithMask(
        train_low_light_path_list=train_low_light_path_list,
        val_low_light_path_list=val_low_light_path_list,
        train_over_exposure_path_list=train_over_exposure_path_list,
        val_over_exposure_path_list=val_over_exposure_path_list,
        train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
        val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
        train_ir_noise_path_list=train_ir_noise_path_list,
        val_ir_noise_path_list=val_ir_noise_path_list,
        phase="val",
        transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # Load CLIP (shared between model and SAM filter)
    model_clip, clip_preprocess = clip.load("ViT-B/32", device=device)

    # Create v4 model
    model = create_model(model_clip, iterations=args.iterations).to(device)

    # Freeze CLIP
    for param in model.base.model_clip.parameters():
        param.requires_grad = False

    # Load pretrained weights with key remapping
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]

        has_base_prefix = any(k.startswith('base.') for k in weights_dict)
        if not has_base_prefix:
            weights_dict = {f'base.{k}': v for k, v in weights_dict.items()}

        # Remap prompt_guidance_X.MLP -> prompt_guidance_X.global_affine.MLP
        new_weights = {}
        for k, v in weights_dict.items():
            for level in ['2', '3', '4']:
                old_prefix = f'base.prompt_guidance_{level}.'
                new_prefix = f'base.prompt_guidance_{level}.global_affine.'
                if k.startswith(old_prefix) and 'global_affine' not in k:
                    k = k.replace(old_prefix, new_prefix)
                    break
            new_weights[k] = v

        missing, unexpected = model.load_state_dict(new_weights, strict=False)
        loaded_count = len(new_weights) - len(unexpected)
        print(f"Loaded pretrained weights from: {args.weights}")
        print(f"  Keys loaded: {loaded_count}/{len(new_weights)}")

    # Freeze encoders
    for param in model.base.encoder_A.parameters():
        param.requires_grad = False
    for param in model.base.encoder_B.parameters():
        param.requires_grad = False
    print("Encoders frozen. Training: MaskGuidedAffine, FFBlock, FDBlock, ReconHead, Decoder")

    # Create IterativeSAMFilter (frozen SAM ViT-B + shared CLIP)
    sam_filter = IterativeSAMFilter(
        sam_ckpt=args.sam_ckpt_iter,
        obj_text=args.obj_text,
        clip_model=model.base.model_clip,
        clip_preprocess=clip_preprocess,
        device=device,
        clip_threshold=args.clip_threshold
    )

    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    print(f"Training Text_IF_Recon_v4 iterative fusion "
          f"(iterations={args.iterations}, obj_text='{args.obj_text}', "
          f"pass1_weight={args.pass1_weight}, lr={args.lr})")

    for epoch in range(start_epoch, args.epochs):
        # Train
        (train_loss, train_ssim, train_max, train_color, train_text,
         train_recon, train_mask, lr) = train_one_epoch_iterative(
            model=model,
            model_clip=model_clip,
            sam_filter=sam_filter,
            optimizer=optimizer,
            data_loader=train_loader,
            lr_scheduler=lr_scheduler,
            device=device,
            epoch=epoch,
            recon_weight=args.recon_weight,
            enhance_factor=args.enhance_factor,
            bg_factor=args.bg_factor,
            mask_loss_weight=args.mask_loss_weight,
            pass1_weight=args.pass1_weight)

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_ssim_loss", train_ssim, epoch)
        tb_writer.add_scalar("train_max_loss", train_max, epoch)
        tb_writer.add_scalar("train_color_loss", train_color, epoch)
        tb_writer.add_scalar("train_text_loss", train_text, epoch)
        tb_writer.add_scalar("train_recon_loss", train_recon, epoch)
        tb_writer.add_scalar("train_mask_loss", train_mask, epoch)

        if epoch % args.val_every_epcho == 0 and epoch != 0:
            (val_loss, val_ssim, val_max, val_color, val_text,
             val_recon, val_mask) = evaluate_iterative(
                model=model,
                sam_filter=sam_filter,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
                lr=lr,
                filefold_path=file_img_path,
                recon_weight=args.recon_weight,
                enhance_factor=args.enhance_factor,
                bg_factor=args.bg_factor,
                mask_loss_weight=args.mask_loss_weight,
                pass1_weight=args.pass1_weight)

            tb_writer.add_scalar("val_total_loss", val_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss", val_ssim, epoch)
            tb_writer.add_scalar("val_max_loss", val_max, epoch)
            tb_writer.add_scalar("val_color_loss", val_color, epoch)
            tb_writer.add_scalar("val_text_loss", val_text, epoch)
            tb_writer.add_scalar("val_recon_loss", val_recon, epoch)
            tb_writer.add_scalar("val_mask_loss", val_mask, epoch)

            if val_loss < best_val_loss:
                if args.use_dp == True:
                    save_file = {"model": model.module.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
                else:
                    save_file = {"model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
                torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")
                best_val_loss = val_loss

            if args.use_dp == True:
                    save_file = {"model": model.module.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
            else:
                    save_file = {"model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
            torch.save(save_file, file_weights_path + "/" + "checkpoint_lastest.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--low_light_path', type=str, default="./dataset/EMS_lite/Low_light")
    parser.add_argument('--over_exposure_path', type=str, default="./dataset/EMS_lite/Over_exposure")
    parser.add_argument('--ir_low_contrast_path', type=str, default="./dataset/EMS_lite/IR_Low_contrast")
    parser.add_argument('--ir_noise_path', type=str, default="./dataset/EMS_lite/IR_Noise")

    parser.add_argument('--weights', type=str,
                        default='experiments/TextIF_train_20260408-185710/weights/checkpoint.pth',
                        help='textif-me pretrained weights path')
    parser.add_argument('--val_every_epcho', type=int, default=2, help='val every epcho')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default=False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')

    # Reconstruction loss parameters
    parser.add_argument('--recon_weight', type=float, default=0.05)
    parser.add_argument('--pass1_weight', type=float, default=0.3,
                        help='Weight for Pass 1 loss (default: 0.3)')

    # Mask enhancement parameters
    parser.add_argument('--enhance_factor', type=float, default=1.5)
    parser.add_argument('--bg_factor', type=float, default=0.5)
    parser.add_argument('--mask_loss_weight', type=float, default=1.0)

    # Iterative SAM parameters
    parser.add_argument('--iterations', type=int, default=2,
                        help='Number of fusion iterations (default: 2)')
    parser.add_argument('--obj_text', type=str, default='person',
                        help='Object category for CLIP filtering (default: person)')
    parser.add_argument('--sam_ckpt_iter', type=str,
                        default='references/segment-anything/checkpoints/sam_vit_b_01ec64.pth',
                        help='SAM ViT-B checkpoint for iterative mask generation')
    parser.add_argument('--clip_threshold', type=float, default=0.22,
                        help='CLIP similarity threshold for mask filtering')

    opt = parser.parse_args()
    main(opt)
```

- [ ] **Step 2: Verify syntax**

Run:
```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF && conda run -n xpu python -c "import ast; ast.parse(open('train_fusion_iterative.py').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add train_fusion_iterative.py
git commit -m "feat: add iterative SAM-fusion training script (no pre-computed masks)"
```

---

### Task 5: Update Evaluation Script for v4 Iterative Mode

**Files:**
- Modify: `evaluate_textif_obj_enhance.py`

- [ ] **Step 1: Add iterative mode support to `evaluate_textif_obj_enhance.py`**

Add new imports after the existing model import:

```python
from model.Text_IF_recon_model_4 import Text_IF_Recon_v4 as create_model_v4
from model.sam_iterative_filter import IterativeSAMFilter
```

Add a new `load_model_v4` function after the existing `load_model` function:

```python
def load_model_v4(weights_path: str, device: torch.device, iterations=2):
    """Load Text_IF_Recon_v4 with key remapping and return (model, None)."""
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model_v4(model_clip, iterations=iterations).to(device)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v

    remapped = {}
    for k, v in clean_state.items():
        for level in ['2', '3', '4']:
            old_prefix = f'base.prompt_guidance_{level}.'
            new_prefix = f'base.prompt_guidance_{level}.global_affine.'
            if k.startswith(old_prefix) and 'global_affine' not in k:
                k = k.replace(old_prefix, new_prefix)
                break
        remapped[k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"Loaded weights: {len(remapped) - len(unexpected)}/{len(remapped)} keys")

    model.eval()
    return model, model_clip
```

Add `--iterative`, `--iterations`, `--obj_text`, `--sam_ckpt_iter`, `--clip_threshold` args to the argparse section:

```python
    parser.add_argument("--iterative", action="store_true",
                        help="Use iterative v4 model with online SAM mask generation")
    parser.add_argument("--iterations", type=int, default=2,
                        help="Number of fusion iterations for v4 (default: 2)")
    parser.add_argument("--obj_text", type=str, default="person",
                        help="Object category for CLIP filtering (v4 iterative mode)")
    parser.add_argument("--sam_ckpt_iter", type=str,
                        default="references/segment-anything/checkpoints/sam_vit_b_01ec64.pth",
                        help="SAM ViT-B checkpoint for iterative mask generation")
    parser.add_argument("--clip_threshold", type=float, default=0.22,
                        help="CLIP similarity threshold for mask filtering")
```

In the `main` function, replace the model loading block with:

```python
    if args.iterative:
        print("Using iterative v4 model with online SAM mask generation")
        model, model_clip = load_model_v4(args.weights_path, device, iterations=args.iterations)
        _, clip_preprocess = clip.load("ViT-B/32", device=device)
        sam_filter = IterativeSAMFilter(
            sam_ckpt=args.sam_ckpt_iter,
            obj_text=args.obj_text,
            clip_model=model.base.model_clip,
            clip_preprocess=clip_preprocess,
            device=device,
            clip_threshold=args.clip_threshold
        )
    else:
        model = load_model(args.weights_path, device)
        sam_filter = None
```

Replace the model forward call section:

```python
            if args.iterative:
                with torch.no_grad():
                    fused, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
                        vis_tensor, ir_tensor, text, sam_filter=sam_filter)
            else:
                with torch.no_grad():
                    fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
                        vis_tensor, ir_tensor, text, mask=mask_tensor)
```

Update the `finally` block to handle the extra `fused_1` output:

```python
        finally:
            if args.iterative:
                del ir_tensor, vis_tensor, mask_tensor, fused, fused_1
            else:
                del ir_tensor, vis_tensor, mask_tensor, fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, metrics
            clear_device_cache(device)
```

Add mask usage info to the final print:

```python
    mask_info = f"Iterative SAM (obj='{args.obj_text}')" if args.iterative else ('Yes' if use_mask else 'No')
    print(f"Mask usage: {mask_info}")
```

- [ ] **Step 2: Verify syntax**

Run:
```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF && conda run -n xpu python -c "import ast; ast.parse(open('evaluate_textif_obj_enhance.py').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add evaluate_textif_obj_enhance.py
git commit -m "feat: update evaluation script to support iterative v4 model"
```

---

## Execution Order

Tasks 1-2 are sequential (v4 depends on SAM filter). Tasks 3-4 depend on Tasks 1-2. Task 5 depends on Task 1.

```
Task 1 (IterativeSAMFilter) ──┐
Task 2 (Model v4)           ───┤
Task 3 (utils iterative)    ───┤
Task 4 (training script)    ───┤
Task 5 (eval script update) ───┘── all above
```

Recommended: Task 1 → 2 → 3 → 4 → 5, sequential.
