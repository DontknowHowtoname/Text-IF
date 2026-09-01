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
