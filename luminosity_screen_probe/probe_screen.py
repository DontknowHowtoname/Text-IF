"""Luminosity-masked screen blending probe: PS 高光选区 + 滤色 + 高低频分离。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # noqa: E402  (必须在 pyplot 之前；matplotlib 供后续出图任务使用)
import matplotlib.pyplot as plt  # noqa: F401  (后续任务使用)
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SOFTLIGHT_DIR = REPO_ROOT / "softlight_blend_probe"
if str(SOFTLIGHT_DIR) not in sys.path:
    sys.path.insert(0, str(SOFTLIGHT_DIR))

import probe_softlight as sl  # noqa: F401,E402  (复用指标/IO/柔光，不复制)


def highlight_mask(ir: np.ndarray, alpha: float, mu: float | None = None) -> np.ndarray:
    """高光蒙版 M = Sigmoid(alpha·(I_ir − mu))，对应 PS 高光选区 + 色阶中点。

    mu 缺省为该图红外均值（自适应中点）。返回 [0,1] float64。
    """
    ir = ir.astype(np.float64)
    if mu is None:
        mu = ir.mean()
    z = np.clip(alpha * (ir - mu), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def screen(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """滤色混合：out = 1 − (1−base)⊙(1−blend)。H=0 → base，H=1 → 1，天然无溢出。"""
    base = base.astype(np.float64)
    blend = blend.astype(np.float64)
    return 1.0 - (1.0 - base) * (1.0 - blend)


def gaussian_low_high(ir: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """高斯高低频分离：low = GaussianBlur(ir, sigma)，high = ir − low。"""
    ir = ir.astype(np.float64)
    low = cv2.GaussianBlur(ir, (0, 0), sigmaX=sigma)
    return low, ir - low


def fuse_screen(ir: np.ndarray, vi: np.ndarray, alpha: float, sigma: float,
                beta: float, mu: float | None = None) -> np.ndarray:
    """亮度蒙版滤色融合（RGB 返回）。

    F = clip( Screen(V, M⊙I_low) + beta·(M⊙I_high), 0, 1 )
    低频经滤色"打面光"（无溢出），高频经蒙版选区内加法注入细节。
    """
    m = highlight_mask(ir, alpha, mu)
    low, high = gaussian_low_high(ir, sigma)
    h_low = (m * low)[..., None]      # (H,W,1) 广播到 VI 三通道
    h_high = (m * high)[..., None]
    fused = screen(vi, h_low) + beta * h_high
    return np.clip(fused, 0.0, 1.0)
