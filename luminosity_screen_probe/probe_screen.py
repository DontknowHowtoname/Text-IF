"""Luminosity-masked screen blending probe: PS 高光选区 + 滤色 + 高低频分离。"""
from __future__ import annotations

import argparse
import csv
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

import probe_softlight as sl  # noqa: E402  (复用指标/IO/柔光，不复制)


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


LOWER_BETTER = {"Nabf"}


def select_best_grid(records: list[dict], keys: tuple[str, ...]) -> tuple:
    """rank 选优（泛化版）：每指标在配置间取均值后排名打分，总分最高胜出。

    配置键由 keys 指定（本探针为 ("alpha","sigma","beta")）。
    前置条件：records 须为同一 method 的记录，且包含全部 PROBE_METRICS 键。
    """
    configs = sorted({tuple(r[k] for k in keys) for r in records})
    scores = {c: 0 for c in configs}
    for m in sl.PROBE_METRICS:
        avg = {c: float(np.mean([r[m] for r in records
                                 if tuple(r[k] for k in keys) == c])) for c in configs}
        lower = m in LOWER_BETTER
        # 平局感知：得分 = 严格劣于自己的配置数（并列均不得分，无平局时等价于 N-1-rank）
        for c in configs:
            scores[c] += sum((avg[c] < avg[d]) if lower else (avg[c] > avg[d])
                             for d in configs if d != c)
    return max(scores, key=scores.get)


def fig_mask(dataset: str, name: str, ir: np.ndarray, alphas: list[float], out: Path) -> None:
    """IR 原图 + 各 α 的高光蒙版（coolwarm，红=1），验证高光选区抓取。"""
    panels = [("IR", ir, "gray")]
    panels += [(f"M alpha={a}", highlight_mask(ir, a), "coolwarm") for a in alphas]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
    for ax, (title, img, cmap) in zip(np.atleast_1d(axes).ravel(), panels):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title); ax.axis("off")
        if cmap == "coolwarm":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"highlight mask - {dataset}/{name}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / f"{name}_mask.png", dpi=150)
    plt.close(fig)


def fig_grid(dataset: str, name: str, ir: np.ndarray, vi: np.ndarray,
             alphas: list[float], sigmas: list[float], betas: list[float], out: Path) -> None:
    """12 配置网格：行 = alpha×sigma，列 = beta。"""
    rows = [(a, s) for a in alphas for s in sigmas]
    fig, axes = plt.subplots(len(rows), len(betas), squeeze=False,
                             figsize=(4.5 * len(betas), 4.5 * len(rows)))
    axes = np.asarray(axes)
    for i, (a, s) in enumerate(rows):
        for j, b in enumerate(betas):
            axes[i, j].imshow(fuse_screen(ir, vi, a, s, b))
            axes[i, j].set_title(f"α={a} σ={s} β={b}"); axes[i, j].axis("off")
    fig.suptitle(f"screen config grid - {dataset}/{name}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"{name}_grid.png", dpi=150)
    plt.close(fig)


def fig_compare(dataset: str, name: str, ir: np.ndarray, vi: np.ndarray,
                softlight: np.ndarray, screen_fused: np.ndarray,
                t5_path: Path, out: Path) -> None:
    """五联对比：IR | VI | 柔光(ir_on_vi/0.6) | 滤色(最优配置) | T5。"""
    t5 = plt.imread(str(t5_path))
    if t5.ndim == 2:
        t5 = np.stack([t5] * 3, axis=2)
    panels = [("IR", ir, "gray"), ("VI", vi, None),
              ("SoftLight", softlight, None), ("Screen", screen_fused, None),
              ("Text-IF T5", t5, None)]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5))
    for ax, (title, img, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"screen vs softlight vs T5 - {dataset}/{name}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / f"{name}_compare.png", dpi=150)
    plt.close(fig)


def save_metrics_csv(records: list[dict], path: Path) -> None:
    headers = ["dataset", "name", "method", "alpha", "sigma", "beta"] + sl.PROBE_METRICS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    ap = argparse.ArgumentParser(description="Luminosity-masked screen probe vs softlight vs T5")
    ap.add_argument("--datasets", nargs="+", default=["MSRS", "TNO"])
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--alphas", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[3.0, 9.0])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--softlight-alpha", type=float, default=0.6,
                    help="柔光参照配置的不透明度（层序固定 ir_on_vi）")
    ap.add_argument("--t5-root", type=Path, default=sl.REPO_ROOT / "sweeps" / "out" / "text_ratio_T5" / "gen")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for ds in args.datasets:
        fused_dir = args.t5_root / ds / "fused"
        names = sorted(f.stem for f in fused_dir.glob("*.png"))[: args.num]
        print(f"[{ds}] {len(names)} samples: {names}")
        for name in names:
            t5_path = fused_dir / f"{name}.png"
            ir, vi, t5_img = sl.load_pair_t5(ds, name, t5_path)
            cache[(ds, name)] = (ir, vi, t5_img)
            fig_mask(ds, name, ir, args.alphas, args.out)
            fig_grid(ds, name, ir, vi, args.alphas, args.sigmas, args.betas, args.out)

            records.append({"dataset": ds, "name": name, "method": "T5",
                            "alpha": float("nan"), "sigma": float("nan"),
                            "beta": float("nan"), **sl.compute_metrics(ir, vi, t5_img)})
            sl_fused = sl.fuse(ir, vi, "ir_on_vi", args.softlight_alpha)
            records.append({"dataset": ds, "name": name, "method": "softlight",
                            "alpha": args.softlight_alpha, "sigma": float("nan"),
                            "beta": float("nan"), **sl.compute_metrics(ir, vi, sl_fused)})
            for a in args.alphas:
                for s in args.sigmas:
                    for b in args.betas:
                        m = sl.compute_metrics(ir, vi, fuse_screen(ir, vi, a, s, b))
                        records.append({"dataset": ds, "name": name, "method": "screen",
                                        "alpha": a, "sigma": s, "beta": b, **m})

    if not records:
        sys.exit(f"no samples found under {args.t5_root} (check --t5-root / --datasets / --num)")

    best = select_best_grid([r for r in records if r["method"] == "screen"],
                            keys=("alpha", "sigma", "beta"))
    print(f"\nbest screen config: alpha={best[0]} sigma={best[1]} beta={best[2]}")

    for (ds, name), (ir, vi, _) in cache.items():
        sl_fused = sl.fuse(ir, vi, "ir_on_vi", args.softlight_alpha)
        sc_fused = fuse_screen(ir, vi, *best)
        fig_compare(ds, name, ir, vi, sl_fused, sc_fused,
                    args.t5_root / ds / "fused" / f"{name}.png", args.out)

    save_metrics_csv(records, args.out / "metrics.csv")

    print("\n=== metric means over sampled images ===")
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r["method"] == "T5":
            key = ("T5",)
        elif r["method"] == "softlight":
            key = ("softlight",)
        else:
            key = ("screen", r["alpha"], r["sigma"], r["beta"])
        groups.setdefault(key, []).append(r)
    print(f"{'method':<30}" + "".join(f"{m:>9}" for m in sl.PROBE_METRICS))
    for key in sorted(groups, key=str):
        rows = groups[key]
        label = "/".join(str(x) for x in key)
        print(f"{label:<30}" + "".join(f"{np.mean([r[m] for r in rows]):>9.3f}" for m in sl.PROBE_METRICS))
    print(f"\nsaved: {args.out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
