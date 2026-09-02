"""Unit tests for softlight core (pure numpy functions)."""
import numpy as np
import pytest

from probe_softlight import softlight, softlight_blend


def test_blend_midgray_returns_base():
    """blend=0.5 时 (2a-1)=0，柔光不改变 base。"""
    base = np.linspace(0.0, 1.0, 256).reshape(16, 16)
    blend = np.full_like(base, 0.5)
    np.testing.assert_allclose(softlight(base, blend), base, atol=1e-12)


def test_blend_white_brightens():
    """blend=1 → out = D(b) ≥ b，全图变亮。"""
    base = np.linspace(0.01, 0.99, 256).reshape(16, 16)
    out = softlight(base, np.ones_like(base))
    assert np.all(out >= base - 1e-12)
    assert np.any(out > base)


def test_blend_black_darkens():
    """blend=0 → out = 2b - D(b) ≤ b，全图变暗。"""
    base = np.linspace(0.01, 0.99, 256).reshape(16, 16)
    out = softlight(base, np.zeros_like(base))
    assert np.all(out <= base + 1e-12)
    assert np.any(out < base)


def test_opacity_zero_returns_base():
    base = np.random.rand(8, 8)
    blend = np.random.rand(8, 8)
    np.testing.assert_allclose(softlight_blend(base, blend, 0.0), base, atol=1e-12)


def test_opacity_one_equals_raw_softlight():
    base = np.random.rand(8, 8)
    blend = np.random.rand(8, 8)
    np.testing.assert_allclose(softlight_blend(base, blend, 1.0), softlight(base, blend), atol=1e-12)


def test_output_range():
    rng = np.random.default_rng(0)
    out = softlight(rng.random((16, 16)), rng.random((16, 16)))
    assert out.min() >= 0.0 and out.max() <= 1.0
