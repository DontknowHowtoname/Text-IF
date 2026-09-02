"""Soft-light (W3C) blend probe: 柔光混合用于红外/可见光融合的初步验证。"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (须在 matplotlib.use 之后)

REPO_ROOT = Path(__file__).resolve().parent.parent


def softlight(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """W3C 柔光：out = b + (2a-1)·(D(b)-b)，D(b)=((16b-12)b+4)b (b<=0.25) 否则 sqrt(b)。

    base/blend 取值 [0,1]，逐像素计算，输出裁剪到 [0,1]。
    """
    base = np.clip(base.astype(np.float64), 0.0, 1.0)
    blend = np.clip(blend.astype(np.float64), 0.0, 1.0)
    d = np.where(base <= 0.25, ((16.0 * base - 12.0) * base + 4.0) * base, np.sqrt(base))
    return np.clip(base + (2.0 * blend - 1.0) * (d - base), 0.0, 1.0)


def softlight_blend(base: np.ndarray, blend: np.ndarray, alpha: float) -> np.ndarray:
    """带不透明度的柔光：out = (1-α)·base + α·softlight(base, blend)。"""
    return (1.0 - alpha) * base + alpha * softlight(base, blend)


DATASET_DIRS = {
    # dataset 名 -> (ir 目录, vi 目录)，相对 REPO_ROOT
    "MSRS": (REPO_ROOT / "dataset" / "MSRS-main" / "test" / "ir",
             REPO_ROOT / "dataset" / "MSRS-main" / "test" / "vi"),
    "TNO": (REPO_ROOT / "dataset" / "TNO" / "ir",
            REPO_ROOT / "dataset" / "TNO" / "vi"),
}


def load_pair(dataset: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """读取 ir(灰度 HxW) / vi(彩色 HxWx3)，float64 [0,1]。vi 若本身是灰度则转 3 通道。

    返回值为 RGB 色彩空间（cv2 读入的 BGR 已转换）。
    """
    ir_dir, vi_dir = DATASET_DIRS[dataset]
    ir = cv2.imread(str(ir_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    vi = cv2.imread(str(vi_dir / f"{name}.png"), cv2.IMREAD_COLOR)
    if ir is None or vi is None:
        raise FileNotFoundError(f"ir/vi not found for {dataset}/{name}")
    vi = cv2.cvtColor(vi, cv2.COLOR_BGR2RGB)
    return ir.astype(np.float64) / 255.0, vi.astype(np.float64) / 255.0


def fuse(ir: np.ndarray, vi: np.ndarray, order: str, alpha: float) -> np.ndarray:
    """柔光融合。order='ir_on_vi': base=vi(彩色), blend=ir；'vi_on_ir': 反向。

    输入/返回均为 RGB 色彩空间，输出 HxWx3 float [0,1]。
    """
    if order == "ir_on_vi":
        base, blend = vi, ir[..., None]          # blend 广播到 3 通道
    elif order == "vi_on_ir":
        base, blend = np.repeat(ir[..., None], 3, axis=2), vi
    else:
        raise ValueError(f"unknown order: {order}")
    return softlight_blend(base, blend, alpha)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def fig_compare(dataset: str, name: str, ir: np.ndarray, vi: np.ndarray,
                fused: np.ndarray, t5_path: Path, out: Path) -> None:
    """四联对比：IR | VI | 柔光(选定配置) | T5。"""
    t5 = plt.imread(str(t5_path))            # matplotlib 直接读 RGB
    if t5.ndim == 2:
        t5 = np.stack([t5] * 3, axis=2)
    panels = [("IR", ir, "gray"), ("VI", vi, None),
              ("SoftLight", fused, None), ("Text-IF T5", t5, None)]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"softlight vs T5 - {dataset}/{name}")
    fig.tight_layout()
    fig.savefig(out / f"{name}_compare.png", dpi=150)
    plt.close(fig)


def fig_grid(dataset: str, name: str, ir: np.ndarray, vi: np.ndarray,
             orders: list[str], alphas: list[float], out: Path) -> None:
    """配置网格：2 层序 × N α，便于挑参数。"""
    fig, axes = plt.subplots(len(orders), len(alphas), figsize=(4.5 * len(alphas), 4.5 * len(orders)))
    axes = np.atleast_2d(axes)
    for i, order in enumerate(orders):
        for j, alpha in enumerate(alphas):
            axes[i, j].imshow(fuse(ir, vi, order, alpha))
            axes[i, j].set_title(f"{order} α={alpha}"); axes[i, j].axis("off")
    fig.suptitle(f"softlight config grid - {dataset}/{name}")
    fig.tight_layout()
    fig.savefig(out / f"{name}_grid.png", dpi=150)
    plt.close(fig)
