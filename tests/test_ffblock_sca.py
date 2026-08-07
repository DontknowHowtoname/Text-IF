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
    """Spatial mask has shape [B, 1, H, W] -- broadcasts to all channels."""
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
