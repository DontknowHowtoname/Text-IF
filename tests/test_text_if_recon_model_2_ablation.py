"""Ablation-flag smoke tests for Text_IF_Recon v2.

Verifies: flags default to False, any flag combination leaves state_dict
keys unchanged (checkpoint compatibility), and a shared ABLATION_MODEL_KWARGS
mapping exists for train/eval scripts.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

from model.Text_IF_recon_model_2 import Text_IF_Recon, ABLATION_MODEL_KWARGS


class _StubCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))


def _keys(**kw):
    m = Text_IF_Recon(_StubCLIP(), **kw)
    return set(m.state_dict().keys())


def test_flags_default_false():
    m = Text_IF_Recon(_StubCLIP())
    assert m.disable_fdblock is False
    assert m.disable_ffblock is False
    assert m.no_text is False


def test_state_dict_keys_identical_under_all_flags():
    base = _keys()
    for kw in ({"disable_fdblock": True}, {"disable_ffblock": True},
               {"no_text": True},
               {"disable_fdblock": True, "disable_ffblock": True, "no_text": True}):
        assert _keys(**kw) == base, f"state_dict keys changed with {kw}"


def test_ablation_model_kwargs_mapping():
    assert ABLATION_MODEL_KWARGS["full"] == {}
    assert ABLATION_MODEL_KWARGS["no_fdblock"] == {"disable_fdblock": True}
    assert ABLATION_MODEL_KWARGS["ff_feature_fusion"] == {"disable_ffblock": True}
    assert ABLATION_MODEL_KWARGS["no_text"] == {"no_text": True}
    # loss/training-only ablations map to the default model
    assert ABLATION_MODEL_KWARGS["no_dual_recon"] == {}
    assert ABLATION_MODEL_KWARGS["unfreeze_encoder"] == {}
    assert set(ABLATION_MODEL_KWARGS) == {
        "full", "no_fdblock", "no_dual_recon", "no_text",
        "ff_feature_fusion", "unfreeze_encoder"}


def test_forward_smoke_with_flags():
    """Forward pass under all ablation flags, without real CLIP.

    no_text=True skips base.get_text_feature and prompt guidance entirely,
    so a stub CLIP module is enough. disable_ffblock/disable_fdblock
    exercise the fallback fusion / degenerate decoupling branches.
    """
    model = Text_IF_Recon(_StubCLIP(), no_text=True,
                          disable_ffblock=True, disable_fdblock=True)
    model.eval()

    A = torch.randn(1, 3, 64, 64)
    B = torch.randn(1, 3, 64, 64)
    text = torch.zeros(1, 77, dtype=torch.long)  # content ignored under no_text

    with torch.no_grad():
        out = model(A, B, text)

    assert len(out) == 5
    names = ("fused", "recon_ir", "recon_vis", "recon_dec_ir", "recon_dec_vis")
    for name, o in zip(names, out):
        assert o.shape == (1, 3, 64, 64), f"{name} shape {tuple(o.shape)}"
        assert torch.isfinite(o).all(), f"{name} has non-finite values"
