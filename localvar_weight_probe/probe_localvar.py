"""Local-variance temperature-softmax weight probe (定性验证脚本)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def local_variance(gray: np.ndarray, win: int = 7, normalize: bool = True) -> np.ndarray:
    """局部方差 E[x^2] - E[x]^2，box filter 实现，reflect 边界填充。

    normalize=True 时除以最大值映射到 [0,1]（τ 的量纲约定基于归一化方差）。
    """
    gray = gray.astype(np.float64)
    k = np.ones((win, win), dtype=np.float64) / (win * win)
    mean = cv2.filter2D(gray, -1, k, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.filter2D(gray * gray, -1, k, borderType=cv2.BORDER_REFLECT)
    var = np.clip(mean_sq - mean * mean, 0.0, None)
    if normalize:
        var = var / (var.max() + 1e-12)
    return var


def temp_weight(v_ir: np.ndarray, v_vi: np.ndarray, tau: float) -> np.ndarray:
    """W_ir = exp(V_ir/tau) / (exp(V_ir/tau) + exp(V_vi/tau))，数值稳定的 Softmax。"""
    z = np.stack([v_ir, v_vi], axis=0) / tau
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e[0] / e.sum(axis=0)


def load_pair(data_root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """读取同名 ir/vi 灰度对，float64 [0,1]。"""
    ir = cv2.imread(str(data_root / "test" / "ir" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    vi = cv2.imread(str(data_root / "test" / "vi" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if ir is None or vi is None:
        raise FileNotFoundError(f"ir/vi pair not found for {name} under {data_root}/test")
    return ir.astype(np.float64) / 255.0, vi.astype(np.float64) / 255.0


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def fig1_texture(gray: np.ndarray, name: str, out: Path, win: int) -> None:
    """图1：灰度 | 局部方差 | Sobel | Canny 四联，验证方差提取纹理/边缘。"""
    var = local_variance(gray, win=win)
    mag = sobel_mag(gray)
    mag = mag / (mag.max() + 1e-12)
    canny = cv2.Canny((gray * 255).astype(np.uint8), 50, 150).astype(np.float64) / 255.0
    panels = [("gray", gray, "gray"), (f"local var (win={win})", var, "magma"),
              ("sobel magnitude", mag, "magma"), ("canny", canny, "gray")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"[1] texture/edge extraction - {name}")
    fig.tight_layout()
    fig.savefig(out / f"{name}_fig1_texture.png", dpi=150)
    plt.close(fig)


def fig2_weights(ir: np.ndarray, vi: np.ndarray, name: str, out: Path,
                 win: int, taus: list[float]) -> None:
    """图2：V_ir | V_vi | 各 τ 的 W_ir，验证极限行为与平滑过渡。"""
    v_ir, v_vi = local_variance(ir, win=win), local_variance(vi, win=win)
    panels = [("V_ir", v_ir, "magma"), ("V_vi", v_vi, "magma")]
    for t in taus:
        panels.append((f"W_ir (tau={t})", temp_weight(v_ir, v_vi, t), "coolwarm"))
    n = len(panels)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, img, cmap) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"[2] temperature weight W_ir - {name}")
    fig.tight_layout()
    fig.savefig(out / f"{name}_fig2_weights.png", dpi=150)
    plt.close(fig)


def fig3_fusion(ir: np.ndarray, vi: np.ndarray, name: str, out: Path,
                win: int, tau_soft: float = 0.5) -> None:
    """图3：梯度域融合，硬 max (tau=0) vs 软权重 (tau=tau_soft)。"""
    v_ir, v_vi = local_variance(ir, win=win), local_variance(vi, win=win)
    g_ir, g_vi = sobel_mag(ir), sobel_mag(vi)
    w_hard = (v_ir > v_vi).astype(np.float64)
    w_soft = temp_weight(v_ir, v_vi, tau_soft)
    f_hard = w_hard * g_ir + (1 - w_hard) * g_vi
    f_soft = w_soft * g_ir + (1 - w_soft) * g_vi
    panels = [("fused grad, tau=0 (hard max)", f_hard, "magma"),
              (f"fused grad, tau={tau_soft} (soft)", f_soft, "magma")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"[3] gradient-domain fusion - {name}")
    fig.tight_layout()
    fig.savefig(out / f"{name}_fig3_fusion.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", default=["00099D.png", "00016N.png"],
                    help="MSRS test 图像文件名（含 .png）")
    ap.add_argument("--data-root", type=Path, default=Path("dataset/MSRS-main"))
    ap.add_argument("--win", type=int, default=7)
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[0.05, 0.1, 0.5, 1.0, 10.0, 1000.0])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for fname in args.images:
        stem = Path(fname).stem
        ir, vi = load_pair(args.data_root, stem)
        fig1_texture(vi, stem, args.out, args.win)
        fig2_weights(ir, vi, stem, args.out, args.win, args.taus)
        fig3_fusion(ir, vi, stem, args.out, args.win)
        print(f"[done] {stem}: 3 figures -> {args.out}")


if __name__ == "__main__":
    main()
