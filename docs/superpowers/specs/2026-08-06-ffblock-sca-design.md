# FFBlock-SCA (Spatial-Channel Joint Attention) Design

**Date**: 2026-08-06
**Status**: Approved (pending spec review)
**Baseline**: `Text_IF_recon_model_2` (v2-ft), trained via `train_fusion_full_recon_v2_ft.py`
**Goal**: Augment `FFBlock` with a lightweight, shared spatial-attention branch that works alongside the existing per-modality channel attention, improving target localization ("foreground vs background") without sacrificing modality specificity.

---

## 1. Motivation

The current `FFBlock` ([model/freefusion_blocks.py:44-73](../../../model/freefusion_blocks.py#L44-L73)) produces **per-modality channel weights** via two 1×1 convs (`channel_weights_1`, `channel_weights_2`). Channel attention answers **"what feature matters"** (thermal radiation vs reflected light) but cannot answer **"where the target is"** (foreground vs background).

The two modalities observe the **same physical scene**, so target location is shared across modalities. Modality specificity is already encoded by the per-modality channel weights — the spatial decision can therefore be **shared** to keep parameter cost negligible.

**Physical narrative** (for paper):
- Channel attention selects *physical attributes* (per-modality).
- Spatial attention locates *physical targets* (shared, scene-level).
- Their product fuses the two judgments.

---

## 2. Architecture

### 2.1 New module: `FFBlockSCA`

Lives in [model/freefusion_blocks.py](../../../model/freefusion_blocks.py), placed immediately after `FFBlock`. The original `FFBlock` is **not modified** — this guarantees the v2-ft baseline is untouched.

```
FFBlockSCA(in_channels, out_channels, use_spatial=True):

  # ─── Identical to FFBlock ─────────────────────────────────────
  conv_1_1          : 3-stage 3×3 stack on cat([en1, en2])  → sconv_1 [B, C, H, W]
  conv_1_2          : 3-stage 3×3 stack on cat([en1, en2])  → sconv_2 [B, C, H, W]
  channel_weights_1 : 1×1 conv on sconv_1                  → w_c1   [B, C, H, W]
  channel_weights_2 : 1×1 conv on sconv_2                  → w_c2   [B, C, H, W]
  conv_2            : 3×3 conv on cat([x_1, x_2])          → fus    [B, Cout, H, W]

  # ─── New: shared spatial attention branch (only if use_spatial=True) ──
  spatial_attn      : SpatialAttention()                   → w_s    [B, 1, H, W]
```

### 2.2 New submodule: `SpatialAttention`

CBAM-style spatial attention. Lives next to `FFBlockSCA` in the same file.

```python
class SpatialAttention(nn.Module):
    """CBAM-style spatial attention.

    Aggregates channel-wise max + avg pools of the input, then applies a 7x7 conv
    to produce a single-channel spatial mask in [0, 1].

    Input x may have any number of channels (typical: 2*C from cat([sconv_1, sconv_2])).
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)             # [B, 1, H, W]
        mx, _ = torch.max(x, dim=1, keepdim=True)            # [B, 1, H, W]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # [B, 1, H, W]
```

### 2.3 `FFBlockSCA.forward`

```python
def forward(self, en1, en2):
    cat_1_1 = torch.cat([en1, en2], dim=1)
    sconv_1 = self.conv_1_1(cat_1_1)
    sconv_2 = self.conv_1_2(cat_1_1)

    w_c1 = self.channel_weights_1(sconv_1)
    w_c2 = self.channel_weights_2(sconv_2)

    if self.use_spatial:
        ctx = torch.cat([sconv_1, sconv_2], dim=1)           # [B, 2C, H, W]
        w_s = self.spatial_attn(ctx)                         # [B, 1, H, W]
        x_1 = en1 * w_c1 * w_s
        x_2 = en2 * w_c2 * w_s
    else:
        x_1 = en1 * w_c1
        x_2 = en2 * w_c2

    return self.conv_2(torch.cat([x_1, x_2], dim=1))
```

### 2.4 Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Spatial context source | `cat([sconv_1, sconv_2])` | Concatenation preserves richer joint information than averaging; no extra cost (see params note). |
| Spatial mask shape | `[B, 1, H, W]` | Broadcasts across all channels — pure spatial modulation. |
| Mask applied to | `en1`, `en2` (same site as channel weights) | Topology mirrors FFBlock for clean A/B comparison. |
| Channel × spatial combination | Multiplicative (`en * w_c * w_s`) | Standard CBAM pattern, gradient flow is clean. |
| Activation | Sigmoid | Mask in [0, 1]. |
| Kernel size | 7×7 | CBAM default; larger receptive field suits foreground localization. |
| `use_spatial` flag | Default `True`; when `False`, block is numerically and structurally identical to `FFBlock` | Enables clean ablation (Experiment B). |

### 2.5 Parameter cost

`SpatialAttention.conv`: `2 × 1 × 7 × 7 + 0 (bias=False) = 98` parameters per FFBlock, **independent of channel count** (because the conv input is always 2 after the channel-wise max/avg pools).

| Level | Channels | New params |
|---|---|---|
| L1 | 48 | 98 |
| L2 | 96 | 98 |
| L3 | 192 | 98 |
| **Total added** | | **294** |

Relative to baseline (~hundreds of K params): **<0.1%**.

### 2.6 Key invariant

When `use_spatial=False`:
- `state_dict` keys are exactly `FFBlock`'s keys (no `spatial_attn.*` keys exist).
- Forward output matches `FFBlock` within floating-point tolerance.

This is the foundation of the Experiment-B structural control.

---

## 3. Scaffold

### 3.1 New model file: `model/Text_IF_recon_model_5.py`

Derived from [model/Text_IF_recon_model_2.py](../../../model/Text_IF_recon_model_2.py) by copy-then-edit. Only 3 logical changes:

1. **Import**:
   ```python
   from model.freefusion_blocks import FFBlockSCA, FDBlock, ReconHead
   ```
2. **Class name**: `Text_IF_Recon_v5` (or alias `Text_IF_Recon` at module level so training scripts can use either).
3. **Constructor signature**: add `use_spatial: bool = True` parameter.
4. **FFBlock → FFBlockSCA** (3 instantiation sites):
   ```python
   self.ffb_1 = FFBlockSCA(in_channels=dim,     out_channels=dim,     use_spatial=use_spatial)
   self.ffb_2 = FFBlockSCA(in_channels=dim * 2, out_channels=dim * 2, use_spatial=use_spatial)
   self.ffb_3 = FFBlockSCA(in_channels=dim * 4, out_channels=dim * 4, use_spatial=use_spatial)
   ```

Everything else (encoder/cross-attention/prompt-guidance/decoder/FDBlock/ReconHead wiring, `forward` returning the 5-tuple `(fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis)`) is **identical to v2**.

Docstring:
```python
"""Text_IF_Recon v5: FFBlockSCA (Spatial-Channel Joint Attention) replaces FFBlock.

Identical to v2 when use_spatial=False. When use_spatial=True, adds a shared,
CBAM-style spatial attention branch that modulates both modalities jointly,
while per-modality channel attention continues to encode modality-specific
attributes.
"""
```

### 3.2 New training script: `train_fusion_full_recon_v5_ft.py`

Derived from [train_fusion_full_recon_v2_ft.py](../../../train_fusion_full_recon_v2_ft.py). Changes:

1. **Import**: `from model.Text_IF_recon_model_5 import Text_IF_Recon_v5 as create_model`
2. **CLI arg** (new):
   ```python
   parser.add_argument('--use_spatial', type=int, default=1,
                       choices=[0, 1],
                       help='Enable FFBlockSCA spatial attention (1=on, 0=off for ablation).')
   ```
3. **Model construction**:
   ```python
   model = create_model(model_clip, use_spatial=bool(args.use_spatial)).to(device)
   ```
4. **Default output dir**:
   ```python
   filefold_path = "./experiments/TextIF_full_recon_v5_ft_{}_sp{}".format(file_name, args.use_spatial)
   ```
5. **Status print**: include `use_spatial=args.use_spatial` in the existing fine-tuning banner.

**Untouched**: data loading, transforms, loss functions, optimizer, lr scheduler, freeze logic (CLIP + encoders frozen, FFBlockSCA/FDBlock/ReconHead/decoder/cross-attn/prompt-guidance trainable), checkpoint format, validation cadence, default hyperparameters (`lr=2e-5`, `recon_weight=0.3`, `epochs=50`, `batch_size=8`).

### 3.3 New evaluation script: `evaluate_textif_full_recon_v5.py`

Derived from [evaluate_textif_full_recon_v2.py](../../../evaluate_textif_full_recon_v2.py). Changes:

1. **Import**: `from model.Text_IF_recon_model_5 import Text_IF_Recon_v5 as create_model`
2. **CLI arg**: same `--use_spatial` as the training script.
3. **Model construction**:
   ```python
   model = create_model(model_clip, use_spatial=bool(args.use_spatial)).to(device)
   ```

**Untouched**: metric set, CSV output schema, image saving logic. This ensures [sweeps/aggregate_results.py](../../../sweeps/aggregate_results.py) and [sweeps/aggregate_sweep.py](../../../sweeps/aggregate_sweep.py) work without modification.

### 3.4 Weight loading strategy

Load from a v2-ft checkpoint (which contains `FFBlock` weights) into the v5 model via `strict=False`:

```python
weights_dict = torch.load(args.weights, map_location=device)["model"]
# v2-ft ckpt keys already carry the 'base.' prefix → match v5's base.* directly.
# ffb_{1,2,3}.{conv_1_1.*, conv_1_2.*, channel_weights_1.*, channel_weights_2.*, conv_2.*}
#   match FFBlockSCA's keys exactly.
# ffb_{1,2,3}.spatial_attn.conv.weight is absent in the ckpt → falls into `missing`
#   → initialized by nn.Conv2d default (Kaiming uniform).

missing, unexpected = model.load_state_dict(weights_dict, strict=False)
```

**Expected outcomes**:
- `unexpected == []` (v5 has no fewer submodules than v2).
- `missing` contains exactly: `ffb_{1,2,3}.spatial_attn.conv.weight` (3 keys) + `base.model_clip.*` (CLIP, always reloaded separately).
- No special initialization needed — Sigmoid output ≈ 0.5 at start, so early-training signal magnitude is preserved.

### 3.5 Sanity check before training

After loading weights with `use_spatial=True`:
- All `ffb_*` keys except `spatial_attn.conv.weight` must be loaded from the ckpt (zero unexpected, three expected missing).
- A single forward pass on a dummy batch should not NaN/Inf.

---

## 4. A/B/C Experiment Plan

Three controlled experiments, identical data and seeds:

| Experiment | Model class | `use_spatial` | Training script | Tag |
|---|---|---|---|---|
| **A** (baseline) | `FFBlock` | — | `train_fusion_full_recon_v2_ft.py` | `v2-ft` |
| **B** (structural control) | `FFBlockSCA` | `False` | `train_fusion_full_recon_v5_ft.py` (`--use_spatial 0`) | `v5-ft-sa-off` |
| **C** (new method) | `FFBlockSCA` | `True` | `train_fusion_full_recon_v5_ft.py` (`--use_spatial 1`) | `v5-ft-sa-on` |

**Roles**:
- **B vs A**: structural control. Validates that the refactor from `FFBlock` → `FFBlockSCA(use_spatial=False)` introduces no behavioral change. Expected: metrics within ±1e-4 across all evaluation metrics.
- **C vs A**: primary comparison. Quantifies the contribution of adding spatial attention on top of the existing v2-ft pipeline.
- **C vs B**: pure effect of spatial attention alone (since B is the structurally-matched baseline).

**Hyperparameters** (all three runs): identical to v2-ft defaults (`lr=2e-5`, `recon_weight=0.3`, `epochs=50`, `batch_size=8`, encoders + CLIP frozen, textif-me pretrained init at `experiments/TextIF_train_20260408-185710/weights/checkpoint.pth`).

**Evaluation**: run `evaluate_textif_full_recon_v{2,5}.py` on the same test sets currently used by the sweep workflow; aggregate via existing scripts.

---

## 5. Verification Checklist

These checks must pass before training is considered valid:

- [ ] **V1 — State-dict key parity (use_spatial=False)**:
  `set(FFBlockSCA(C, C, use_spatial=False).state_dict().keys()) == set(FFBlock(C, C).state_dict().keys())` for `C ∈ {48, 96, 192}`.

- [ ] **V2 — Numerical parity (use_spatial=False)**:
  Given identical random input and identical seeded init, `max(|FFBlockSCA(use_spatial=False)(x1, x2) − FFBlock(x1, x2)|) < 1e-6` for `C ∈ {48, 96, 192}`.

- [ ] **V3 — Forward smoke test (use_spatial=True)**:
  `FFBlockSCA(C, C, use_spatial=True)` produces correctly-shaped output `[B, C, H, W]` on a dummy batch and contains no NaN/Inf.

- [ ] **V4 — Checkpoint transfer**:
  Loading a v2-ft checkpoint into the v5 model (`use_spatial=True`) yields:
  - `unexpected == []`
  - `missing == {ffb_1.spatial_attn.conv.weight, ffb_2.spatial_attn.conv.weight, ffb_3.spatial_attn.conv.weight} ∪ {base.model_clip.*}`

- [ ] **V5 — Training smoke test**:
  One full epoch of `train_fusion_full_recon_v5_ft.py` completes without error on a small data subset; training loss is finite and trends downward by the end of the epoch.

---

## 6. Out of Scope

The following are explicitly **not** part of this design:
- Modifying `FFBlock` itself.
- Changes to the encoder, decoder, cross-attention, prompt-guidance, FDBlock, or ReconHead.
- Changes to loss functions, optimizer, lr scheduler, or data pipeline.
- New sweep runner logic (existing [sweeps/run_sweep.py](../../../sweeps/run_sweep.py) supports launching new training scripts as new variants).
- Inference-time optimizations, ONNX export, or deployment work.
- Multi-mask or per-modality spatial attention variants (deferred — the shared single-mask design is the deliberate first experiment).
