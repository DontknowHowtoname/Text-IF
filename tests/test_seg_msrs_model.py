# tests/test_seg_msrs_model.py
import pytest
import torch

CKPT = "references/SegFormer/mit_b1_20220624-02e5a6a1.pth"


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
