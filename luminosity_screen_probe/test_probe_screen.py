"""Unit tests for luminosity-masked screen blending core (pure numpy/cv2)."""
import numpy as np
import pytest

from probe_screen import highlight_mask, screen, gaussian_low_high


def test_mask_range_and_midpoint():
    """蒙版范围 [0,1]；I_ir=μ 处 M=0.5。"""
    ir = np.linspace(0.0, 1.0, 256).reshape(16, 16)
    m = highlight_mask(ir, alpha=8.0)
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert m.mean() == pytest.approx(0.5, abs=1e-6)  # μ=均值，对称分布


def test_mask_steeper_alpha_more_binary():
    """α 越大蒙版越陡（趋向二值 argmax）。"""
    ir = np.linspace(0.0, 1.0, 256).reshape(16, 16)
    m_lo = highlight_mask(ir, alpha=2.0)
    m_hi = highlight_mask(ir, alpha=64.0)
    assert np.abs(m_hi - 0.5).max() > np.abs(m_lo - 0.5).max()
    assert (m_hi[-1, -1] > 0.99) and (m_hi[0, 0] < 0.01)


def test_screen_zero_blend_returns_base():
    """H=0 时滤色恒等：结果=V。"""
    v = np.random.default_rng(0).random((8, 8))
    np.testing.assert_allclose(screen(v, np.zeros_like(v)), v, atol=1e-12)


def test_screen_full_blend_saturates():
    """H=1 时结果=1。"""
    v = np.random.default_rng(1).random((8, 8))
    np.testing.assert_allclose(screen(v, np.ones_like(v)), 1.0, atol=1e-12)


def test_screen_closed_form():
    v = np.array([0.2, 0.5, 0.8])
    h = np.array([0.3, 0.3, 0.3])
    np.testing.assert_allclose(screen(v, h), 1.0 - (1.0 - v) * (1.0 - h), atol=1e-12)


def test_low_high_reconstruct():
    """low + high 逐点还原原图。"""
    ir = np.random.default_rng(2).random((32, 32))
    low, high = gaussian_low_high(ir, sigma=3.0)
    np.testing.assert_allclose(low + high, ir, atol=1e-10)


def test_low_smooth_high_sharp():
    """低频比高频平滑（梯度能量更低）。"""
    ir = np.random.default_rng(3).random((64, 64))
    low, high = gaussian_low_high(ir, sigma=5.0)
    g = lambda x: np.abs(np.diff(x, axis=0)).sum() + np.abs(np.diff(x, axis=1)).sum()
    assert g(low) < g(high)
