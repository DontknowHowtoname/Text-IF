# Iterative SAM-Fusion Feedback Design

**Goal:** Replace the static pre-computed mask pipeline with an iterative approach where SAM segments the fused image to produce better masks, which then guide a second fusion pass for object-level enhancement. No pre-computed masks needed.

**Architecture:** Two-pass fusion with SAM ViT-B + CLIP filtering as a frozen online mask generator. Pass 1 produces a global fusion (no mask). Pass 2 uses SAM+CLIP on the fused image to generate object masks, then fuses again with mask guidance.

---

## Core Flow

```
Input: vis, ir, text, obj_text="person"
SAM = ViT-B (frozen), CLIP = ViT-B/32 (frozen)

Pass 1 (global fusion):
  fused_1, ... = Fusion(vis, ir, text, mask=None)

Pass 2 (object enhancement):
  mask = SAM_Automatic(fused_1) + CLIP_Filter(obj_text)
  fused_2, ... = Fusion(vis, ir, text, mask=mask)

Output: fused_2
```

- Pass 1 has no mask (identical to v2 baseline)
- SAM segments the fused image, which has clearer object boundaries than raw input
- CLIP filters masks by cosine similarity with `obj_text` (e.g. "person")
- Pass 2 uses the refined mask in MaskGuidedAffine at decoder levels 2-4
- Fusion network weights are shared across passes

---

## Key Components

### IterativeSAMFilter

Encapsulates SAM ViT-B AutomaticMaskGenerator + CLIP filtering. Initialized once, called each forward pass.

- SAM ViT-B loaded from `references/segment-anything/checkpoints/sam_vit_b_01ec64.pth`
- SAM parameters frozen (`requires_grad=False`, `torch.no_grad()`)
- CLIP text features pre-computed once from `obj_text`
- Input: `[B, 3, H, W]` tensor (fused image)
- Output: `[B, 1, H, W]` binary mask tensor (float, 0/1)
- Internally converts tensor to numpy, runs SAM Automatic, filters by CLIP, merges masks

### Text_IF_Recon_v4

Extends v3 with iterative fusion logic:

- `__init__`: same modules as v3 + `iterations` parameter (default 2)
- `_fusion_pass(vis, ir, text, mask)`: one complete fusion forward (encoder + decoder + output), identical to v3's forward
- `forward(vis, ir, text, sam_filter=None)`:
  - Pass 1: `_fusion_pass(vis, ir, text, mask=None)` → `fused_1`
  - Pass 2: `mask = sam_filter(fused_1)` → `_fusion_pass(vis, ir, text, mask=mask)` → `fused_2`
  - Returns: `(fused_final, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis)`
  - 6 outputs (adds `fused_1` for iterative improvement loss)

When `iterations=1` or `sam_filter=None`, behaves identically to v3 with mask=None.

### Training

- Encoder runs once, cached for both passes (no re-encoding)
- SAM+CLIP always under `torch.no_grad()` (frozen, no gradient)
- Pass 2's Fusion path receives gradients (MaskGuidedAffine learns to use SAM masks)
- `PromptDataSet` used instead of `PromptDataSetWithMask` (no pre-computed masks needed)
- New args: `--obj_text "person"`, `--sam_ckpt_iter` (ViT-B path), `--iterations 2`

### Loss

```
loss_p1 = loss_fn(fused_1, ...) * pass1_weight    # 0.3
loss_p2 = loss_fn(fused_2, ...) * pass2_weight    # 1.0
total   = loss_p2 + pass1_weight * loss_p1
```

- `loss_fn` = existing `fusion_dual_recon_mask_loss` (fusion + dual_recon + mask_enhance)
- Pass 1 loss ensures baseline quality
- Pass 2 loss is the primary optimization target
- mask_enhance loss in Pass 2 uses the SAM-generated mask

### Evaluation

- Same iterative flow as training
- `evaluate_textif_obj_enhance.py` updated to use v4 with `sam_filter`
- `--obj_text` arg added to evaluation script
- No `--mask_dir` needed anymore

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `model/sam_iterative_filter.py` | CREATE | IterativeSAMFilter class (SAM+CLIP wrapper) |
| `model/Text_IF_recon_model_4.py` | CREATE | Text_IF_Recon_v4 with iterative forward |
| `scripts/utils.py` | MODIFY | Add `train_one_epoch_iterative`, `evaluate_iterative` |
| `train_fusion_iterative.py` | CREATE | Training script using iterative fusion |
| `evaluate_textif_obj_enhance.py` | MODIFY | Update to support v4 iterative mode |

---

## Design Decisions

1. **No pre-computed masks**: SAM+CLIP runs online during training and inference. This removes the offline `generate_masks.py` dependency entirely.
2. **SAM ViT-B (not ViT-H) for iterations**: ViT-B is 5x faster than ViT-H, acceptable for per-forward usage. ViT-H only used if user runs offline generation.
3. **Fusion weights shared**: Same Fusion network processes both passes. The model learns to produce good Pass 1 output (so SAM can segment well) and respond to SAM masks in Pass 2.
4. **Encoder runs once**: Encoder features are computed once and reused for both passes. Only the decoder + MaskGuidedAffine run twice.
5. **Frozen SAM+CLIP**: These modules never train. All learning happens in the Fusion network.
6. **obj_text as fixed training arg**: The object category (e.g. "person") is specified at training time and used consistently for CLIP filtering.
