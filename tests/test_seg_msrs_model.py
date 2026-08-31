# tests/test_seg_msrs_model.py
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "references" / "SegFormer" / "mit_b1_20220624-02e5a6a1.pth"


def _model():
    from seg_msrs.model import SegFormerSeg
    return SegFormerSeg(num_classes=9)


def test_forward_shapes():
    model = _model().eval()
    with torch.no_grad():
        feats = model.backbone(torch.randn(1, 3, 480, 640))
        assert [f.shape[1] for f in feats] == [64, 128, 320, 512]
        assert [f.shape[2] for f in feats] == [120, 60, 30, 15]
        logits = model(torch.randn(1, 3, 480, 640))
    assert logits.shape == (1, 9, 480, 640)


def test_load_timm_checkpoint_strict():
    from seg_msrs.model import SegFormerSeg, remap_timm_mit_state
    model = SegFormerSeg(num_classes=9)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    remapped = remap_timm_mit_state(sd)
    model.backbone.load_state_dict(remapped, strict=True)  # 不抛异常即通过


def test_remap_splits_fused_qkv_correctly():
    """q must be in_proj rows [0:d]; kv must be k+v stacked rows [d:3d]."""
    from seg_msrs.model import remap_timm_mit_state
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    remapped = remap_timm_mit_state(sd)
    w = sd["layers.0.1.0.attn.attn.in_proj_weight"]
    b = sd["layers.0.1.0.attn.attn.in_proj_bias"]
    d = w.shape[0] // 3
    assert torch.equal(remapped["block1.0.attn.q.weight"], w[:d])
    assert torch.equal(remapped["block1.0.attn.q.bias"], b[:d])
    assert torch.equal(remapped["block1.0.attn.kv.weight"], w[d:3 * d])
    assert torch.equal(remapped["block1.0.attn.kv.bias"], b[d:3 * d])
    # ffn conv1x1 -> Linear flatten
    fc1 = sd["layers.0.1.0.ffn.layers.0.weight"]
    assert torch.equal(remapped["block1.0.mlp.fc1.weight"], fc1.flatten(1))
    # dwconv 3x3 stays 4D
    assert torch.equal(remapped["block1.0.mlp.dwconv.dwconv.weight"],
                       sd["layers.0.1.0.ffn.layers.1.weight"])
