# FFBlock-SCA (Spatial-Channel Joint Attention) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `FFBlockSCA` (FFBlock + shared CBAM-style spatial attention) and wire it through a new v5 model + v5 train/eval script, derived from the v2-ft baseline so A/B/C ablations can run.

**Architecture:** A new `FFBlockSCA` class subclasses the structure of `FFBlock` and adds an optional `SpatialAttention` branch (max+avg pool → 7×7 conv → sigmoid). When `use_spatial=False`, it is structurally and numerically identical to `FFBlock`, providing a clean ablation control. A new `Text_IF_recon_model_5.py` wires `FFBlockSCA` into the existing v2 architecture; `train_fusion_full_recon_v5_ft.py` and `evaluate_textif_full_recon_v5.py` are thin derivations of the v2-ft scripts with one new CLI flag.

**Tech Stack:** PyTorch 2.10.0+xpu, clip 0.2.0, Python via `D:/software/anaconda3/envs/xpu/python.exe`, pytest under `tests/`.

**Spec:** [docs/superpowers/specs/2026-08-06-ffblock-sca-design.md](../specs/2026-08-06-ffblock-sca-design.md)

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `model/freefusion_blocks.py` | Modify | Add `SpatialAttention` class and `FFBlockSCA` class after existing `FFBlock`. Original `FFBlock` unchanged. |
| `tests/test_ffblock_sca.py` | Create | Unit tests: state-dict parity, numerical parity, forward smoke. |
| `model/Text_IF_recon_model_5.py` | Create | v5 model: `FFBlockSCA` instead of `FFBlock`, plus `use_spatial` constructor arg. |
| `tests/test_text_if_recon_model_5.py` | Create | Model construction smoke tests (both `use_spatial` values). |
| `train_fusion_full_recon_v5_ft.py` | Create | v5 training script: copy of v2-ft with `--use_spatial` flag. |
| `evaluate_textif_full_recon_v5.py` | Create | v5 evaluation script: copy of v2 eval with `--use_spatial` flag. |
| `tests/test_v5_checkpoint_transfer.py` | Create | V4: load v2-ft checkpoint into v5 model, assert missing/unexpected keys. |

**Out of scope** (per spec §6): changes to `FFBlock`, encoder/decoder/FDBlock/ReconHead, loss functions, data pipeline, sweep runner.

**Conventions used throughout:**
- Python executable: `D:/software/anaconda3/envs/xpu/python.exe` (per project memory). The plan refers to it as `$PY` — set it once per session:
  ```bash
  PY="D:/software/anaconda3/envs/xpu/python.exe"
  ```
- Tests use the existing `tests/test_*.py` pattern (no conftest, `sys.path.insert` at top).
- Working directory is the repo root: `d:/StudyFiles/MachineLearning/codes/Text-IF`.

---

### Task 1: Add `SpatialAttention` and `FFBlockSCA` to `freefusion_blocks.py`

**Files:**
- Modify: `model/freefusion_blocks.py` (append after line 73, the end of `FFBlock`)
- Test: `tests/test_ffblock_sca.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ffblock_sca.py` with this exact content:

```python
"""Unit tests for FFBlockSCA: spatial-channel joint attention.

Covers spec verification V1 (state-dict key parity), V2 (numerical parity
when use_spatial=False), V3 (forward smoke when use_spatial=True).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.freefusion_blocks import FFBlock, FFBlockSCA


def _seed_all(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_state_dict_keys_parity_when_spatial_off():
    """V1: FFBlockSCA(use_spatial=False) has the same state_dict keys as FFBlock."""
    for C in [48, 96, 192]:
        keys_ffb = set(FFBlock(C, C).state_dict().keys())
        keys_sca = set(FFBlockSCA(C, C, use_spatial=False).state_dict().keys())
        assert keys_ffb == keys_sca, (
            f"Key mismatch at C={C}.\n"
            f"  Only in FFBlock: {keys_ffb - keys_sca}\n"
            f"  Only in FFBlockSCA: {keys_sca - keys_ffb}"
        )


def test_numerical_parity_when_spatial_off():
    """V2: with identical seeded init and input, FFBlockSCA(use_spatial=False)
    matches FFBlock within floating-point tolerance."""
    for C in [48, 96, 192]:
        B, H, W = 2, 16, 16
        x1 = torch.randn(B, C, H, W)
        x2 = torch.randn(B, C, H, W)

        _seed_all(123)
        ffb = FFBlock(C, C).eval()

        _seed_all(123)
        sca = FFBlockSCA(C, C, use_spatial=False).eval()

        # Copy weights so the two modules are bit-identical.
        sca.load_state_dict(ffb.state_dict())

        with torch.no_grad():
            y_ffb = ffb(x1, x2)
            y_sca = sca(x1, x2)
        max_diff = (y_ffb - y_sca).abs().max().item()
        assert max_diff < 1e-6, f"C={C}: max abs diff {max_diff} >= 1e-6"


def test_forward_shape_and_finite_when_spatial_on():
    """V3: FFBlockSCA(use_spatial=True) produces correct shape and finite output."""
    for C in [48, 96, 192]:
        B, H, W = 2, 16, 16
        x1 = torch.randn(B, C, H, W)
        x2 = torch.randn(B, C, H, W)
        sca = FFBlockSCA(C, C, use_spatial=True).eval()
        with torch.no_grad():
            y = sca(x1, x2)
        assert y.shape == (B, C, H, W), f"C={C}: expected {(B, C, H, W)}, got {tuple(y.shape)}"
        assert torch.isfinite(y).all(), f"C={C}: output contains NaN/Inf"


def test_spatial_mask_is_shared_across_modalities():
    """Spatial mask has shape [B, 1, H, W] — broadcasts to all channels."""
    C = 48
    B, H, W = 2, 16, 16
    x1 = torch.randn(B, C, H, W)
    x2 = torch.randn(B, C, H, W)
    sca = FFBlockSCA(C, C, use_spatial=True).eval()
    # Run forward once to ensure the spatial branch executes without error.
    with torch.no_grad():
        _ = sca(x1, x2)
    # The mask itself is internal; we verify by checking that the spatial_attn
    # submodule produces a 1-channel output on the expected input shape.
    cat_ctx = torch.randn(B, 2 * C, H, W)  # stand-in for cat([sconv_1, sconv_2])
    mask = sca.spatial_attn(cat_ctx)
    assert mask.shape == (B, 1, H, W), f"mask shape {tuple(mask.shape)}"
    assert (mask >= 0).all() and (mask <= 1).all(), "mask not in [0, 1]"
```

- [ ] **Step 2: Run the tests to verify they fail (module not yet defined)**

Run:
```bash
PY="D:/software/anaconda3/envs/xpu/python.exe"
cd d:/StudyFiles/MachineLearning/codes/Text-IF
$PY -m pytest tests/test_ffblock_sca.py -v
```
Expected: All 4 tests FAIL with `ImportError: cannot import name 'FFBlockSCA' from 'model.freefusion_blocks'`.

- [ ] **Step 3: Implement `SpatialAttention` and `FFBlockSCA`**

Append the following to `model/freefusion_blocks.py` (after the existing `FFBlock` class, before `FDBlock`):

```python
class SpatialAttention(nn.Module):
    """CBAM-style spatial attention producing a single-channel mask in [0, 1].

    Channel-wise max and avg pools of the input are concatenated (2 channels)
    and fed through a 7x7 conv. Parameter count is independent of input
    channels: 2 * 1 * 7 * 7 = 98 + 0 (bias=False).
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)             # [B, 1, H, W]
        mx, _ = torch.max(x, dim=1, keepdim=True)            # [B, 1, H, W]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # [B, 1, H, W]


class FFBlockSCA(nn.Module):
    """Feature Fusion Block with Spatial-Channel Joint Attention.

    Identical structure to FFBlock when use_spatial=False. When
    use_spatial=True, adds a shared CBAM-style spatial attention branch
    that modulates both modalities jointly via a single [B, 1, H, W] mask.
    Per-modality channel attention (channel_weights_1/2) is preserved, so
    modality specificity is encoded by the channel branch while the spatial
    branch captures scene-level foreground/background structure.

    Args:
        in_channels: input channel count per modality (C).
        out_channels: output channel count.
        use_spatial: when True, enable the spatial attention branch. When
            False, this module is numerically and structurally identical
            to FFBlock (clean ablation control).
    """
    def __init__(self, in_channels, out_channels, use_spatial: bool = True):
        super(FFBlockSCA, self).__init__()
        self.use_spatial = use_spatial

        # Identical to FFBlock
        self.conv_1_1 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 2, kernel_size=3, relu=True),
            BasicConv(in_channels * 2, out_channels, kernel_size=3, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, relu=True),
        )
        self.conv_1_2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 2, kernel_size=3, relu=True),
            BasicConv(in_channels * 2, out_channels, kernel_size=3, relu=True),
            BasicConv(out_channels, out_channels, kernel_size=3, relu=True),
        )
        self.channel_weights_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.channel_weights_2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_2 = BasicConv(in_channels * 2, out_channels, kernel_size=3, relu=True)

        # New: shared spatial attention branch
        if use_spatial:
            self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, en1, en2):
        cat_1_1 = torch.cat([en1, en2], dim=1)
        sconv_1 = self.conv_1_1(cat_1_1)
        sconv_2 = self.conv_1_2(cat_1_1)

        w_c1 = self.channel_weights_1(sconv_1)
        w_c2 = self.channel_weights_2(sconv_2)

        if self.use_spatial:
            ctx = torch.cat([sconv_1, sconv_2], dim=1)       # [B, 2C, H, W]
            w_s = self.spatial_attn(ctx)                     # [B, 1, H, W]
            x_1 = en1 * w_c1 * w_s
            x_2 = en2 * w_c2 * w_s
        else:
            x_1 = en1 * w_c1
            x_2 = en2 * w_c2

        return self.conv_2(torch.cat([x_1, x_2], dim=1))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
$PY -m pytest tests/test_ffblock_sca.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF
git add model/freefusion_blocks.py tests/test_ffblock_sca.py
git commit -m "$(cat <<'EOF'
feat(model): add FFBlockSCA with shared spatial-channel attention

FFBlockSCA mirrors FFBlock and adds an optional CBAM-style spatial
attention branch that derives a single [B,1,H,W] mask from the
concatenated per-modality contexts, modulating both modalities jointly.
use_spatial=False reproduces FFBlock exactly (clean ablation control).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create `Text_IF_recon_model_5.py`

**Files:**
- Create: `model/Text_IF_recon_model_5.py`
- Test: `tests/test_text_if_recon_model_5.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_text_if_recon_model_5.py`:

```python
"""Smoke tests for Text_IF_Recon v5 (FFBlockSCA-based).

Uses a stubbed CLIP model to avoid loading real CLIP weights. Verifies
that both use_spatial={False,True} construct, forward returns the expected
5-tuple, and FFBlockSCA submodules actually exist.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from model.Text_IF_recon_model_5 import Text_IF_Recon_v5


class _StubCLIP(nn.Module):
    """Minimal stand-in for the CLIP model returned by clip.load(...).

    The real Text_IF uses model_clip.visual.* and model_clip.text projection.
    This stub mirrors the attribute layout enough for model construction.
    """
    def __init__(self):
        super().__init__()
        # Text_IF_model.py references model_clip.visual.transformer.resblocks
        # and other attributes. For pure construction smoke we don't run
        # forward, so a bare Module is sufficient.
        self.dummy = nn.Parameter(torch.zeros(1))


def test_model_constructs_with_use_spatial_true():
    model = Text_IF_Recon_v5(_StubCLIP(), use_spatial=True)
    # FFBlockSCA submodules must exist with spatial_attn
    for name in ['ffb_1', 'ffb_2', 'ffb_3']:
        sub = getattr(model, name)
        assert hasattr(sub, 'spatial_attn'), f"{name} missing spatial_attn"
        assert sub.use_spatial is True


def test_model_constructs_with_use_spatial_false():
    model = Text_IF_Recon_v5(_StubCLIP(), use_spatial=False)
    for name in ['ffb_1', 'ffb_2', 'ffb_3']:
        sub = getattr(model, name)
        assert sub.use_spatial is False
        assert not hasattr(sub, 'spatial_attn'), f"{name} should not have spatial_attn"


def test_model_default_use_spatial_is_true():
    model = Text_IF_Recon_v5(_StubCLIP())
    assert model.ffb_1.use_spatial is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
$PY -m pytest tests/test_text_if_recon_model_5.py -v
```
Expected: All 3 tests FAIL with `ModuleNotFoundError: No module named 'model.Text_IF_recon_model_5'`.

- [ ] **Step 3: Create `model/Text_IF_recon_model_5.py`**

Create the file with this exact content:

```python
"""Text_IF_Recon v5: FFBlockSCA (Spatial-Channel Joint Attention) replaces FFBlock.

Identical to v2 when use_spatial=False. When use_spatial=True, adds a shared,
CBAM-style spatial attention branch that modulates both modalities jointly,
while per-modality channel attention continues to encode modality-specific
attributes.
"""
import torch
import torch.nn as nn

from model.Text_IF_model import Text_IF
from model.freefusion_blocks import FFBlockSCA, FDBlock, ReconHead


class Text_IF_Recon_v5(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 use_spatial: bool = True):
        super(Text_IF_Recon_v5, self).__init__()

        # Original Text-IF model as submodule
        self.base = Text_IF(
            model_clip, inp_A_channels, inp_B_channels, out_channels,
            dim, num_blocks, num_refinement_blocks, heads,
            ffn_expansion_factor, bias, LayerNorm_type
        )

        # FFBlockSCA fusion at encoder levels 1-3
        self.ffb_1 = FFBlockSCA(in_channels=dim,         out_channels=dim,         use_spatial=use_spatial)
        self.ffb_2 = FFBlockSCA(in_channels=dim * 2,     out_channels=dim * 2,     use_spatial=use_spatial)
        self.ffb_3 = FFBlockSCA(in_channels=dim * 4,     out_channels=dim * 4,     use_spatial=use_spatial)

        # FDBlock decoupling (2 instances)
        channels_3lev = [dim, dim * 2, dim * 4]  # [48, 96, 192]
        self.fdb_ir = FDBlock(channels_3lev)
        self.fdb_vis = FDBlock(channels_3lev)

        # Shared lightweight reconstruction head
        self.recon_head = ReconHead(
            in_channels=[dim * 4, dim * 2, dim],  # [192, 96, 48] deepest first
            out_channels=out_channels
        )

    def forward(self, inp_img_A, inp_img_B, text):
        b = inp_img_A.shape[0]
        text_features = self.base.get_text_feature(text.expand(b, -1)).to(inp_img_A.dtype)

        # ---- Encoder (run once) ----
        out_enc_L4_A, out_enc_L3_A, out_enc_L2_A, out_enc_L1_A = self.base.encoder_A(inp_img_A)
        out_enc_L4_B, out_enc_L3_B, out_enc_L2_B, out_enc_L1_B = self.base.encoder_B(inp_img_B)

        # ---- FFBlockSCA replaces feature_fusion at levels 1-3 ----
        fus_L1 = self.ffb_1(out_enc_L1_A, out_enc_L1_B)   # [B, 48, H, W]
        fus_L2 = self.ffb_2(out_enc_L2_A, out_enc_L2_B)   # [B, 96, H/2, W/2]
        fus_L3 = self.ffb_3(out_enc_L3_A, out_enc_L3_B)   # [B, 192, H/4, W/4]
        fus_feas = [fus_L1, fus_L2, fus_L3]

        # Detached encoder features for FDBlock subtraction and direct reconstruction
        enc_A_3lev = [out_enc_L1_A.detach(), out_enc_L2_A.detach(), out_enc_L3_A.detach()]  # visible
        enc_B_3lev = [out_enc_L1_B.detach(), out_enc_L2_B.detach(), out_enc_L3_B.detach()]  # infrared

        # ---- Direct reconstruction (encoder preservation constraint) ----
        recon_vis = self.recon_head([enc_A_3lev[2], enc_A_3lev[1], enc_A_3lev[0]])
        recon_ir = self.recon_head([enc_B_3lev[2], enc_B_3lev[1], enc_B_3lev[0]])

        # ---- FDBlock decoupling ----
        dec_ir_feas = self.fdb_ir(fus_feas, enc_A_3lev)    # fused - vis -> IR residual
        dec_vis_feas = self.fdb_vis(fus_feas, enc_B_3lev)  # fused - IR -> vis residual

        # ---- Decoupled reconstruction ----
        recon_dec_ir = self.recon_head([dec_ir_feas[2], dec_ir_feas[1], dec_ir_feas[0]])
        recon_dec_vis = self.recon_head([dec_vis_feas[2], dec_vis_feas[1], dec_vis_feas[0]])

        # ---- Fusion path (FFBlockSCA output used at L1-L3, cross-attention at L4) ----
        out_enc_L4_A, out_enc_L4_B = self.base.cross_attention(out_enc_L4_A, out_enc_L4_B)
        out_enc_L4 = self.base.feature_fusion_4(out_enc_L4_A, out_enc_L4_B)
        out_enc_L4 = self.base.attention_spatial(out_enc_L4)
        out_enc_L4 = self.base.prompt_guidance_4(out_enc_L4, text_features)

        out_dec_L4 = self.base.decoder_level4(out_enc_L4)

        inp_dec_L3 = self.base.up4_3(out_dec_L4)
        inp_dec_L3 = self.base.prompt_guidance_3(inp_dec_L3, text_features)
        inp_dec_L3 = torch.cat([inp_dec_L3, fus_L3], 1)
        inp_dec_L3 = self.base.reduce_chan_level3(inp_dec_L3)
        out_dec_L3 = self.base.decoder_level3(inp_dec_L3)

        inp_dec_L2 = self.base.up3_2(out_dec_L3)
        inp_dec_L2 = self.base.prompt_guidance_2(inp_dec_L2, text_features)
        inp_dec_L2 = torch.cat([inp_dec_L2, fus_L2], 1)
        inp_dec_L2 = self.base.reduce_chan_level2(inp_dec_L2)
        out_dec_L2 = self.base.decoder_level2(inp_dec_L2)

        inp_dec_L1 = self.base.up2_1(out_dec_L2)
        inp_dec_L1 = self.base.prompt_guidance_1(inp_dec_L1, text_features)
        inp_dec_L1 = torch.cat([inp_dec_L1, fus_L1], 1)
        out_dec_L1 = self.base.decoder_level1(inp_dec_L1)

        fused = self.base.output(self.base.refinement(out_dec_L1))

        return fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis


# Backwards-compatible alias.
Text_IF_Recon = Text_IF_Recon_v5
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
$PY -m pytest tests/test_text_if_recon_model_5.py -v
```
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add model/Text_IF_recon_model_5.py tests/test_text_if_recon_model_5.py
git commit -m "$(cat <<'EOF'
feat(model): add Text_IF_Recon v5 wiring FFBlockSCA

Replaces v2's FFBlock with FFBlockSCA across encoder levels 1-3. Adds
use_spatial constructor flag (default True). Otherwise structurally
identical to v2 — same forward signature returning the 5-tuple
(fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Create `train_fusion_full_recon_v5_ft.py`

**Files:**
- Create: `train_fusion_full_recon_v5_ft.py`

This is a copy-with-deltas of `train_fusion_full_recon_v2_ft.py`. No new tests — verified by `--help` and a dry-run check.

- [ ] **Step 1: Copy the v2-ft script and apply the deltas**

Run:
```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF
cp train_fusion_full_recon_v2_ft.py train_fusion_full_recon_v5_ft.py
```

Then apply exactly these edits to `train_fusion_full_recon_v5_ft.py`:

**Edit 1 — module docstring** (top of file):

Replace lines 1-8:
```
"""
Training script: Text_IF_Recon v2 fine-tuned from textif-me pretrained weights.
Key changes from train_fusion_full_recon.py:
  - Uses Text_IF_recon_model_2 (FFBlock replaces feature_fusion)
  - Loads textif-me pretrained weights
  - Freezes encoder to preserve generalization
  - Lower recon_weight (0.05) and learning rate (2e-5)
"""
```
with:
```
"""
Training script: Text_IF_Recon v5 (FFBlockSCA) fine-tuned from textif-me pretrained weights.
Derived from train_fusion_full_recon_v2_ft.py. Differences:
  - Uses Text_IF_recon_model_5 (FFBlockSCA replaces FFBlock at levels 1-3)
  - New CLI flag --use_spatial (default 1; set 0 for ablation control)
  - Default output dir reflects v5 + use_spatial value
  - Status banner prints use_spatial
Hyperparameters (lr, recon_weight, freeze policy, epochs, batch_size) match v2-ft.
"""
```

**Edit 2 — model import** (line ~18):

Replace:
```python
from model.Text_IF_recon_model_2 import Text_IF_Recon as create_model
```
with:
```python
from model.Text_IF_recon_model_5 import Text_IF_Recon_v5 as create_model
```

**Edit 3 — model construction** (currently line ~124):

Replace:
```python
    model = create_model(model_clip).to(device)
```
with:
```python
    model = create_model(model_clip, use_spatial=bool(args.use_spatial)).to(device)
```

**Edit 4 — encoder-freeze status banner** (currently line ~151):

Replace:
```python
    print("Encoders frozen. Training: FFBlock, FDBlock, ReconHead, Decoder, CrossAttn, PromptGuidance")
```
with:
```python
    print(f"Encoders frozen. Training: FFBlockSCA(use_spatial={bool(args.use_spatial)}), FDBlock, ReconHead, Decoder, CrossAttn, PromptGuidance")
```

**Edit 5 — fine-tuning banner** (currently lines ~166-169):

Replace:
```python
    print(f"Fine-tuning Text_IF_Recon v2 from textif-me "
          f"(recon_weight={args.recon_weight}, lr={args.lr}, frozen_encoders=True, "
          f"max_ratio={args.max_ratio}, ssim_ratio={args.ssim_ratio}, "
          f"task_defaults=text_ratio{{3,2,3,2}}/max_ratio{{4,3,4,3}})")
```
with:
```python
    print(f"Fine-tuning Text_IF_Recon v5 from textif-me "
          f"(use_spatial={bool(args.use_spatial)}, recon_weight={args.recon_weight}, "
          f"lr={args.lr}, frozen_encoders=True, "
          f"max_ratio={args.max_ratio}, ssim_ratio={args.ssim_ratio}, "
          f"task_defaults=text_ratio{{3,2,3,2}}/max_ratio{{4,3,4,3}})")
```

**Edit 6 — default output dir** (currently line ~37):

Replace:
```python
        filefold_path = "./experiments/TextIF_full_recon_v2_ft_{}".format(file_name)
```
with:
```python
        filefold_path = "./experiments/TextIF_full_recon_v5_ft_{}_sp{}".format(file_name, int(args.use_spatial))
```

**Edit 7 — add `--use_spatial` CLI argument**. Insert immediately after the `--output_dir` argument (currently line ~280), before `opt = parser.parse_args()`:

```python
    parser.add_argument('--use_spatial', type=int, default=1, choices=[0, 1],
                        help='FFBlockSCA spatial attention: 1=on (default), 0=off (ablation control).')
```

- [ ] **Step 2: Verify `--help` shows the new flag**

Run:
```bash
$PY train_fusion_full_recon_v5_ft.py --help
```
Expected: help text prints without import errors; new `--use_spatial` argument listed with default 1 and choices {0, 1}.

- [ ] **Step 3: Sanity import check**

Run:
```bash
$PY -c "import train_fusion_full_recon_v5_ft; print('import OK')"
```
Expected: prints `import OK` with no errors.

- [ ] **Step 4: Commit**

```bash
git add train_fusion_full_recon_v5_ft.py
git commit -m "$(cat <<'EOF'
feat(train): add v5 fine-tune script for FFBlockSCA

Thin derivation of train_fusion_full_recon_v2_ft.py: switches the model
import, passes use_spatial, updates banners and default output dir, and
adds the --use_spatial CLI flag. Hyperparameters unchanged from v2-ft.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Create `evaluate_textif_full_recon_v5.py`

**Files:**
- Create: `evaluate_textif_full_recon_v5.py`

- [ ] **Step 1: Copy the v2 eval script and apply the deltas**

Run:
```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF
cp evaluate_textif_full_recon_v2.py evaluate_textif_full_recon_v5.py
```

Then apply exactly these edits to `evaluate_textif_full_recon_v5.py`:

**Edit 1 — module docstring** (top of file, lines 1-4):

Replace:
```
"""
Evaluate Text_IF_Recon v2 (FFBlock replaces feature_fusion) model on IVT_test.
Model forward returns 5 outputs: (fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis).
Only the fused output is used for quality metrics.
"""
```
with:
```
"""
Evaluate Text_IF_Recon v5 (FFBlockSCA) model on IVT_test.
Derived from evaluate_textif_full_recon_v2.py. Differences:
  - Uses Text_IF_recon_model_5
  - New --use_spatial CLI flag (must match the training run being evaluated)
Metric set and CSV schema are unchanged so existing aggregation scripts work.
"""
```

**Edit 2 — model import** (line ~23):

Replace:
```python
from model.Text_IF_recon_model_2 import Text_IF_Recon as create_model
```
with:
```python
from model.Text_IF_recon_model_5 import Text_IF_Recon_v5 as create_model
```

**Edit 3 — model construction**. Open the file and locate the line where the model is constructed (search for `create_model(model_clip`). Replace:

```python
    model = create_model(model_clip).to(device)
```
with:
```python
    model = create_model(model_clip, use_spatial=bool(args.use_spatial)).to(device)
```

**Edit 4 — add `--use_spatial` CLI argument**. Open the file, locate the argparse section, and insert the following argument in the same argument group as the other model-loading flags (next to `--weights` or `--resume`):

```python
    parser.add_argument('--use_spatial', type=int, default=1, choices=[0, 1],
                        help='FFBlockSCA spatial attention: must match the training run. 1=on, 0=off.')
```

- [ ] **Step 2: Verify `--help` shows the new flag**

Run:
```bash
$PY evaluate_textif_full_recon_v5.py --help
```
Expected: help text prints without import errors; `--use_spatial` argument listed with default 1.

- [ ] **Step 3: Sanity import check**

Run:
```bash
$PY -c "import evaluate_textif_full_recon_v5; print('import OK')"
```
Expected: prints `import OK` with no errors.

- [ ] **Step 4: Commit**

```bash
git add evaluate_textif_full_recon_v5.py
git commit -m "$(cat <<'EOF'
feat(eval): add v5 evaluation script for FFBlockSCA

Thin derivation of evaluate_textif_full_recon_v2.py: switches the model
import and adds the --use_spatial CLI flag. Metric set and CSV output
schema unchanged so existing aggregation scripts work unmodified.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Verify checkpoint transfer from v2-ft to v5 (V4)

**Files:**
- Create: `tests/test_v5_checkpoint_transfer.py`

**Prerequisite:** A v2-ft checkpoint at `experiments/TextIF_full_recon_v2_ft_20260508-100418/weights/checkpoint.pth` exists (it does at the time this plan was written). If unavailable, the test should be skipped, not failed — use `pytest.importorskip` style with a filesystem check.

- [ ] **Step 1: Write the test**

Create `tests/test_v5_checkpoint_transfer.py`:

```python
"""V4: Verify v2-ft checkpoint loads cleanly into v5 model.

Expected outcomes (per spec):
  - unexpected == []
  - missing keys contain exactly:
      * ffb_{1,2,3}.spatial_attn.conv.weight  (when use_spatial=True)
      * base.model_clip.*                       (CLIP, always reloaded separately)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import torch.nn as nn

from model.Text_IF_recon_model_5 import Text_IF_Recon_v5


CKPT_PATH = os.path.join(
    os.path.dirname(__file__), '..',
    'experiments', 'TextIF_full_recon_v2_ft_20260508-100418', 'weights', 'checkpoint.pth'
)


class _StubCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))


def test_v2ft_checkpoint_loads_into_v5_with_expected_missing_keys():
    if not os.path.exists(CKPT_PATH):
        pytest.skip(f"v2-ft checkpoint not found at {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    weights_dict = ckpt['model']

    # Build v5 with use_spatial=True (the typical deployment).
    model = Text_IF_Recon_v5(_StubCLIP(), use_spatial=True)

    missing, unexpected = model.load_state_dict(weights_dict, strict=False)

    # Hard requirement: no unexpected keys.
    assert unexpected == [], f"Unexpected keys (ckpt has them, v5 doesn't): {unexpected[:10]}"

    # The only non-CLIP missing keys must be the 3 spatial_attn conv weights.
    non_clip_missing = [k for k in missing if not k.startswith('base.model_clip')]
    expected_missing = {
        'ffb_1.spatial_attn.conv.weight',
        'ffb_2.spatial_attn.conv.weight',
        'ffb_3.spatial_attn.conv.weight',
    }
    assert set(non_clip_missing) == expected_missing, (
        f"Non-CLIP missing keys mismatch.\n"
        f"  Expected: {expected_missing}\n"
        f"  Got: {set(non_clip_missing)}"
    )


def test_v2ft_checkpoint_loads_into_v5_spatial_off_with_no_missing_non_clip():
    """When use_spatial=False, v5 is structurally identical to v2 — no non-CLIP missing keys."""
    if not os.path.exists(CKPT_PATH):
        pytest.skip(f"v2-ft checkpoint not found at {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    weights_dict = ckpt['model']

    model = Text_IF_Recon_v5(_StubCLIP(), use_spatial=False)
    missing, unexpected = model.load_state_dict(weights_dict, strict=False)

    assert unexpected == [], f"Unexpected keys: {unexpected[:10]}"
    non_clip_missing = [k for k in missing if not k.startswith('base.model_clip')]
    assert non_clip_missing == [], (
        f"Expected zero non-CLIP missing keys for use_spatial=False, got: {non_clip_missing}"
    )
```

- [ ] **Step 2: Run the test**

Run:
```bash
$PY -m pytest tests/test_v5_checkpoint_transfer.py -v
```
Expected: Both tests PASS (or SKIP if the checkpoint isn't present — in that case, ask the user for the correct path and update `CKPT_PATH`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_v5_checkpoint_transfer.py
git commit -m "$(cat <<'EOF'
test(checkpoint): verify v2-ft weights transfer cleanly to v5

V4: loading a v2-ft checkpoint into v5 (use_spatial=True) must yield
exactly three non-CLIP missing keys (the new spatial_attn convs) and
zero unexpected keys. With use_spatial=False, both must be zero.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Training smoke test (V5)

**Goal:** Confirm one full training epoch completes without error on a small data subset. This is a manual verification step, not an automated pytest (it requires the full data pipeline and CLIP load).

**Files:** none created — this task runs the script from Task 3 directly.

- [ ] **Step 1: Confirm data paths exist**

Run:
```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF
ls dataset/EMS_lite/Low_light/train 2>/dev/null | head -1
ls experiments/TextIF_train_20260408-185710/weights/checkpoint.pth
```
Expected: at least one file listed from each. If the EMS_lite path is different, note the correct path.

- [ ] **Step 2: Run a 1-epoch training smoke on a small subset**

This command runs the real script with `--epochs 1` and default args. If the dataset is large, you can interrupt with Ctrl-C once the first epoch's first few batches print loss values without error — the goal is to verify the training loop starts cleanly, not to converge.

Run:
```bash
cd d:/StudyFiles/MachineLearning/codes/Text-IF
$PY train_fusion_full_recon_v5_ft.py \
    --epochs 1 \
    --use_spatial 1 \
    --batch_size 4 \
    --val_every_epcho 1 \
    --output_dir ./experiments/_smoke_v5_sp1 2>&1 | tee /tmp/v5_smoke_sp1.log
```
Expected behaviors:
- "Loading IVF Fusion ..." banner prints 4 times
- "Encoders frozen. Training: FFBlockSCA(use_spatial=True), ..." prints
- "Fine-tuning Text_IF_Recon v5 from textif-me (use_spatial=True, ..." prints
- Training loop prints per-batch/per-epoch loss values that are finite
- Validation runs at the end of epoch 1 (since `val_every_epcho=1`)
- No Python tracebacks

If the full epoch would take too long, Ctrl-C after a few batches print loss values, and confirm the loss is finite and decreasing.

- [ ] **Step 3: Run the same smoke with `--use_spatial 0`**

Run:
```bash
$PY train_fusion_full_recon_v5_ft.py \
    --epochs 1 \
    --use_spatial 0 \
    --batch_size 4 \
    --val_every_epcho 1 \
    --output_dir ./experiments/_smoke_v5_sp0 2>&1 | tee /tmp/v5_smoke_sp0.log
```
Expected: same success indicators as Step 2 but with `use_spatial=False` in the banners.

- [ ] **Step 4: Clean up smoke artifacts**

Run:
```bash
rm -rf ./experiments/_smoke_v5_sp1 ./experiments/_smoke_v5_sp0
```

- [ ] **Step 5: Record results in EXPERIMENT_NOTES.md**

Append a new entry to `sweeps/EXPERIMENT_NOTES.md` under a new `## v5 smoke (2026-08-06)` heading:

```markdown
## v5 smoke (2026-08-06)

Verified FFBlockSCA training pipeline boots cleanly.
- `--use_spatial 1`: 1-epoch smoke run completes; loss finite; banners print correct flag.
- `--use_spatial 0`: 1-epoch smoke run completes; behavior matches v2-ft expectations.
- Both runs used batch_size=4, epochs=1; full A/B/C runs to follow.
```

Commit:
```bash
git add sweeps/EXPERIMENT_NOTES.md
git commit -m "$(cat <<'EOF'
docs(notes): record v5 FFBlockSCA training smoke results

V5: 1-epoch smoke runs complete cleanly with use_spatial={0,1}.
Pipeline ready for full A/B/C ablation runs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Final Acceptance Criteria

All of the following must be true before this plan is considered complete:

- [ ] Task 1: `tests/test_ffblock_sca.py` — all 4 tests pass.
- [ ] Task 2: `tests/test_text_if_recon_model_5.py` — all 3 tests pass.
- [ ] Task 3: `train_fusion_full_recon_v5_ft.py --help` runs without error and shows `--use_spatial`.
- [ ] Task 4: `evaluate_textif_full_recon_v5.py --help` runs without error and shows `--use_spatial`.
- [ ] Task 5: `tests/test_v5_checkpoint_transfer.py` — both tests pass (V4 satisfied).
- [ ] Task 6: Both 1-epoch smoke runs (`use_spatial=0` and `use_spatial=1`) complete without error (V5 satisfied).

After all six tasks pass, the implementation is ready for the A/B/C ablation experiments defined in spec §4.
