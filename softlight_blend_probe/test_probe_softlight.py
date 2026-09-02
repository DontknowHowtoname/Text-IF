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
    rng = np.random.default_rng(1)
    base = rng.random((8, 8))
    blend = rng.random((8, 8))
    np.testing.assert_allclose(softlight_blend(base, blend, 0.0), base, atol=1e-12)


def test_opacity_one_equals_raw_softlight():
    rng = np.random.default_rng(1)
    base = rng.random((8, 8))
    blend = rng.random((8, 8))
    np.testing.assert_allclose(softlight_blend(base, blend, 1.0), softlight(base, blend), atol=1e-12)


def test_output_range():
    rng = np.random.default_rng(0)
    out = softlight(rng.random((16, 16)), rng.random((16, 16)))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_softlight_closed_form_low_base():
    """b<=0.25 分支：blend=1 时 out=D(b)=16b^3-12b^2+4b。"""
    assert softlight(np.array(0.09), np.array(1.0)) == pytest.approx(16*0.09**3 - 12*0.09**2 + 4*0.09, abs=1e-12)


def test_softlight_closed_form_high_base():
    """b>0.25 分支：blend=1 时 out=D(b)=sqrt(b)。"""
    assert softlight(np.array(0.5), np.array(1.0)) == pytest.approx(np.sqrt(0.5), abs=1e-12)


def test_softlight_continuous_at_quarter():
    """两分支在 b=0.25 处均等于 0.5，公式连续。"""
    eps = 1e-9
    lo = softlight(np.array(0.25 - eps), np.array(0.7))
    hi = softlight(np.array(0.25 + eps), np.array(0.7))
    assert lo == pytest.approx(hi, abs=1e-6)


def test_fuse_ir_on_vi_shape_and_range():
    """主层序(base=vi, blend=ir)：彩色 vi 逐通道用同一 ir blend，输出形状/范围正确。"""
    from probe_softlight import fuse
    ir = np.random.rand(8, 8)
    vi = np.random.rand(8, 8, 3)
    out = fuse(ir, vi, order="ir_on_vi", alpha=0.8)
    assert out.shape == (8, 8, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_fuse_reverse_order():
    from probe_softlight import fuse
    ir = np.random.rand(8, 8)
    vi = np.random.rand(8, 8, 3)
    out = fuse(ir, vi, order="vi_on_ir", alpha=1.0)
    assert out.shape == (8, 8, 3)


def test_fuse_invalid_order_raises():
    from probe_softlight import fuse
    with pytest.raises(ValueError):
        fuse(np.random.rand(4, 4), np.random.rand(4, 4, 3), order="bad", alpha=0.8)


def test_fuse_alpha_zero_semantics():
    """α=0 时柔光退化为 base：ir_on_vi 返回 vi，vi_on_ir 返回 repeat(ir,3)。"""
    from probe_softlight import fuse
    ir = np.random.rand(8, 8)
    vi = np.random.rand(8, 8, 3)
    np.testing.assert_allclose(fuse(ir, vi, "ir_on_vi", 0.0), vi, atol=1e-12)
    np.testing.assert_allclose(fuse(ir, vi, "vi_on_ir", 0.0), np.repeat(ir[..., None], 3, axis=2), atol=1e-12)
