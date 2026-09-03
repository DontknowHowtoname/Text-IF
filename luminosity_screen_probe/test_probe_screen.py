"""Unit tests for luminosity-masked screen blending core (pure numpy/cv2)."""
import numpy as np
import pytest

from probe_screen import highlight_mask, screen, gaussian_low_high


def test_mask_range_and_midpoint():
    """蒙版范围 [0,1]；μ=自适应均值时 sigmoid 输入关于 0 对称 → 蒙版均值必为 0.5。"""
    ir = np.linspace(0.3, 0.8, 256).reshape(16, 16)  # 均值 0.55 ≠ 0.5：若 μ 被硬编码 0.5 则均值≠0.5，可被抓到
    m = highlight_mask(ir, alpha=8.0)
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert m.mean() == pytest.approx(0.5, abs=1e-6)


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
    np.testing.assert_allclose(screen(v, h), np.array([0.44, 0.65, 0.86]), atol=1e-12)


def test_low_high_reconstruct():
    """low + high 逐点还原原图。"""
    ir = np.random.default_rng(2).random((32, 32))
    low, high = gaussian_low_high(ir, sigma=3.0)
    np.testing.assert_allclose(low + high, ir, atol=1e-10)


def test_low_smooth_high_sharp():
    """低频比高频平滑（梯度能量更低）。"""
    ir = np.random.default_rng(3).random((64, 64))
    low, high = gaussian_low_high(ir, sigma=5.0)

    def g(x):
        return np.abs(np.diff(x, axis=0)).sum() + np.abs(np.diff(x, axis=1)).sum()

    assert g(low) < g(high)


def test_fuse_screen_shape_and_range():
    from probe_screen import fuse_screen
    ir = np.random.default_rng(4).random((8, 8))
    vi = np.random.default_rng(5).random((8, 8, 3))
    out = fuse_screen(ir, vi, alpha=8.0, sigma=3.0, beta=1.0)
    assert out.shape == (8, 8, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_fuse_screen_zero_highfreq_and_dark_ir_returns_vi():
    """beta=0 且红外全暗（蒙版≈0，低频=0）时，滤色恒等，退化为 VI。"""
    from probe_screen import fuse_screen
    vi = np.random.default_rng(6).random((8, 8, 3))
    ir = np.zeros((8, 8))
    out = fuse_screen(ir, vi, alpha=8.0, sigma=3.0, beta=0.0)
    np.testing.assert_allclose(out, vi, atol=1e-6)


def test_fuse_screen_brightens():
    """正常参数下结果整体不暗于 VI（滤色只提亮，高频注入可正可负但整体提亮占优）。"""
    from probe_screen import fuse_screen
    ir = np.random.default_rng(7).random((16, 16)) * 0.9 + 0.1
    vi = np.full((16, 16, 3), 0.5)
    out = fuse_screen(ir, vi, alpha=8.0, sigma=3.0, beta=0.5)
    assert out.mean() > vi.mean()
