"""V4: Verify v2-ft checkpoint loads cleanly into v5 model.

Expected outcomes (per spec):
  - No non-CLIP unexpected keys (CLIP keys appear as unexpected because
    the test uses a stub CLIP, not the real one).
  - Non-CLIP missing keys contain exactly:
      * ffb_{1,2,3}.spatial_attn.conv.weight  (when use_spatial=True)
      * (nothing)                              (when use_spatial=False)

Note: CLIP keys (base.model_clip.*) are always mismatched in both
directions when using _StubCLIP -- the checkpoint has 302 real CLIP
params while the stub only has 'dummy'. We filter those out.
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

CLIP_PREFIX = 'base.model_clip'


def _filter_clip(keys):
    """Remove CLIP-related keys (expected mismatch due to stub CLIP)."""
    return [k for k in keys if not k.startswith(CLIP_PREFIX)]


class _StubCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))


def test_v2ft_checkpoint_loads_into_v5_with_expected_missing_keys():
    if not os.path.exists(CKPT_PATH):
        pytest.skip(f"v2-ft checkpoint not found at {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    weights_dict = ckpt['model']

    # Build v5 with use_spatial=True (the typical deployment).
    model = Text_IF_Recon_v5(_StubCLIP(), use_spatial=True)

    missing, unexpected = model.load_state_dict(weights_dict, strict=False)

    # Hard requirement: no non-CLIP unexpected keys.
    non_clip_unexpected = _filter_clip(unexpected)
    assert non_clip_unexpected == [], (
        f"Non-CLIP unexpected keys (ckpt has them, v5 doesn't): {non_clip_unexpected[:10]}"
    )

    # The only non-CLIP missing keys must be the 3 spatial_attn conv weights.
    non_clip_missing = _filter_clip(missing)
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
    """When use_spatial=False, v5 is structurally identical to v2 -- no non-CLIP missing keys."""
    if not os.path.exists(CKPT_PATH):
        pytest.skip(f"v2-ft checkpoint not found at {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    weights_dict = ckpt['model']

    model = Text_IF_Recon_v5(_StubCLIP(), use_spatial=False)
    missing, unexpected = model.load_state_dict(weights_dict, strict=False)

    non_clip_unexpected = _filter_clip(unexpected)
    assert non_clip_unexpected == [], (
        f"Non-CLIP unexpected keys: {non_clip_unexpected[:10]}"
    )
    non_clip_missing = _filter_clip(missing)
    assert non_clip_missing == [], (
        f"Expected zero non-CLIP missing keys for use_spatial=False, got: {non_clip_missing}"
    )
