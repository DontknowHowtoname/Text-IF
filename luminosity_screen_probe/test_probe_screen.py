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
    """beta=0 且红外全暗 → low=high=0 → 滤色 blend=0 → 恒等退化为 VI。

    注：此时蒙版实为 0.5，本测试不覆盖蒙版抑制路径。
    """
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


def test_select_best_grid_prefers_higher_metrics():
    from probe_screen import select_best_grid
    # 两个配置，A 在所有 higher-better 指标上均值更高，应胜出
    metrics = {m: 1.0 for m in ["EN", "MI", "SF", "AG", "SD", "VIF", "SSIM", "Qabf", "SCD"]}
    metrics["Nabf"] = 0.0
    rec_a = {"alpha": 4.0, "sigma": 3.0, "beta": 0.5, **metrics}
    rec_b = {"alpha": 8.0, "sigma": 9.0, "beta": 1.0, **{k: v / 2 for k, v in metrics.items()}}
    best = select_best_grid([rec_a, rec_b], keys=("alpha", "sigma", "beta"))
    assert best == (4.0, 3.0, 0.5)


def test_select_best_grid_nabf_lower_better():
    from probe_screen import select_best_grid
    base = {m: 1.0 for m in ["EN", "MI", "SF", "AG", "SD", "VIF", "SSIM", "Qabf", "SCD"]}
    rec_a = {"alpha": 4.0, "sigma": 3.0, "beta": 0.5, **base, "Nabf": 0.9}
    rec_b = {"alpha": 8.0, "sigma": 9.0, "beta": 1.0, **base, "Nabf": 0.1}
    best = select_best_grid([rec_a, rec_b], keys=("alpha", "sigma", "beta"))
    assert best == (8.0, 9.0, 1.0)


def test_select_best_grid_tie_scores_equal():
    """平局感知排名：得分 = 严格劣于自己的配置数，并列双方均不得分。

    构造 3 配置：A=(1,1,1) 全指标严格最差；B=(2,2,2)、C=(3,3,3) 并列最优。
    场景 1（完全并列，固定文档化行为，新旧规则均通过）：B、C 在全部 10 个
    指标上取值相同 → 新规则各得 10 分（每指标仅胜 A），总分平局由 max 取
    先入序者（字典序最小）→ B 胜。
    场景 2（部分并列，区分性断言）：C 仅在 EN 上严格更优（1.5 vs 1.0），
    其余 9 指标仍与 B 并列。
      新规则：C = 2(EN 胜 B、A) + 9×1(并列指标各胜 A) = 11 > B = 1 + 9 = 10 → C 胜
      （胜者由严格优劣决定，与配置键/入序无关）。
      旧规则（sorted 稳定排序 + N-1-rank，即 probe_softlight.select_best_config
      的规则）：9 个并列指标上字典序小的 B 各得 2 分、C 各得 1 分 →
      B = 1 + 18 = 19 > C = 2 + 9 = 11 → B 胜。
    若回退旧规则，断言 best == (3.0, 3.0, 3.0) 必挂（已用等价模拟脚本验证）。
    """
    from probe_screen import select_best_grid
    metrics = ["EN", "MI", "SF", "AG", "SD", "VIF", "SSIM", "Qabf", "Nabf", "SCD"]
    tie_best = {m: 1.0 for m in metrics}
    tie_best["Nabf"] = 0.1
    worst = {m: (0.6 if m == "Nabf" else 0.5) for m in metrics}  # 全指标严格最差
    rec_a = {"alpha": 1.0, "sigma": 1.0, "beta": 1.0, **worst}
    rec_b = {"alpha": 2.0, "sigma": 2.0, "beta": 2.0, **dict(tie_best)}
    rec_c = {"alpha": 3.0, "sigma": 3.0, "beta": 3.0, **dict(tie_best)}

    # 场景 1：完全并列 → 总分相同 → 字典序最小的 B 胜
    best = select_best_grid([rec_a, rec_b, rec_c], keys=("alpha", "sigma", "beta"))
    assert best == (2.0, 2.0, 2.0)

    # 场景 2：C 仅 EN 严格更优 → 胜者只由严格优劣决定 → C 胜（旧规则下此处为 B）
    rec_c["EN"] = 1.5
    best = select_best_grid([rec_a, rec_b, rec_c], keys=("alpha", "sigma", "beta"))
    assert best == (3.0, 3.0, 3.0)


def test_select_best_grid_averages_across_records():
    """均值路径：每配置 2 条记录，配置内均值决定排名（B 每条都优于 A 每条）。"""
    from probe_screen import select_best_grid
    base = {m: 1.0 for m in ["EN", "MI", "SF", "AG", "SD", "VIF", "SSIM", "Qabf", "SCD"]}
    recs = [
        {"alpha": 4.0, "sigma": 3.0, "beta": 0.5, **base, "Nabf": 0.2},
        {"alpha": 4.0, "sigma": 3.0, "beta": 0.5, **base, "Nabf": 0.4},
        {"alpha": 8.0, "sigma": 9.0, "beta": 1.0, **{k: v - 0.2 for k, v in base.items()}, "Nabf": 0.1},
        {"alpha": 8.0, "sigma": 9.0, "beta": 1.0, **{k: v - 0.4 for k, v in base.items()}, "Nabf": 0.3},
    ]
    best = select_best_grid(recs, keys=("alpha", "sigma", "beta"))
    assert best == (4.0, 3.0, 0.5)
