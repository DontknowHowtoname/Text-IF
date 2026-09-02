"""Soft-light (W3C) blend probe: 柔光混合用于红外/可见光融合的初步验证。"""
from __future__ import annotations

from pathlib import Path

import numpy as np


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
