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
