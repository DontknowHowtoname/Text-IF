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


def _write_pair(tmp_path, ir_h, ir_w, t5_h, t5_w):
    """合成 ir/vi/t5 三张 png：ir 灰度、vi 彩色、t5 为指定尺寸的融合样张。

    load_pair 的文件名约定是 ir_dir/vi_dir 下 f"{name}.png"，故 ir/vi 各占一个子目录。
    """
    import cv2
    import matplotlib.pyplot as plt
    ir_dir = tmp_path / "ir"
    vi_dir = tmp_path / "vi"
    ir_dir.mkdir(exist_ok=True)
    vi_dir.mkdir(exist_ok=True)
    ir = np.random.default_rng(2).integers(0, 256, (ir_h, ir_w), dtype=np.uint8)
    vi = np.random.default_rng(3).integers(0, 256, (ir_h, ir_w, 3), dtype=np.uint8)
    t5 = (np.random.default_rng(4).random((t5_h, t5_w, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(ir_dir / "x.png"), ir)
    cv2.imwrite(str(vi_dir / "x.png"), vi)
    # t5 用 cv2 写 3 通道 RGB png（plt.imsave 会写 RGBA，而真实 T5 输出为 RGB）
    cv2.imwrite(str(tmp_path / "t5_x.png"), cv2.cvtColor(t5, cv2.COLOR_RGB2BGR))
    return ir_dir, vi_dir


def test_load_pair_t5_identity_when_sizes_match(monkeypatch, tmp_path):
    """尺寸一致时（如 MSRS 480×640）不 resize，返回与磁盘内容一致。"""
    from probe_softlight import load_pair_t5, DATASET_DIRS
    ir_dir, vi_dir = _write_pair(tmp_path, 32, 32, 32, 32)
    monkeypatch.setitem(DATASET_DIRS, "FAKE", (ir_dir, vi_dir))
    # t5 图名约定：fused 目录下 f"{name}.png"；load_pair_t5 的 t5 参数直接传路径
    ir, vi, t5 = load_pair_t5("FAKE", "x", tmp_path / "t5_x.png")
    assert ir.shape == (32, 32) and vi.shape == (32, 32, 3) and t5.shape == (32, 32, 3)


def test_load_pair_t5_aligns_to_t5_size(monkeypatch, tmp_path):
    """T5 图为 16 倍数尺寸（如 TNO 368×448）时，ir/vi 被双线性对齐到该尺寸。"""
    from probe_softlight import load_pair_t5, DATASET_DIRS
    ir_dir, vi_dir = _write_pair(tmp_path, 40, 44, 32, 48)
    monkeypatch.setitem(DATASET_DIRS, "FAKE", (ir_dir, vi_dir))
    ir, vi, t5 = load_pair_t5("FAKE", "x", tmp_path / "t5_x.png")
    assert ir.shape == (32, 48) and vi.shape == (32, 48, 3) and t5.shape == (32, 48, 3)
