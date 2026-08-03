"""Generate trade-off visualizations: scatter pairs + normalized radar."""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT_ROOT = Path(__file__).resolve().parent / "out"
RUN_DIRS = [
    ("T0",  0),
    ("T1",  1),
    ("T2",  2),
    ("T3",  3),
    ("T5",  5),
    ("T8",  8),
    ("T10", 10),
    ("T15", 15),
]
# metrics where lower is better — will be inverted for radar
LOWER_BETTER = {"MSE", "Nabf", "CE"}

# trade-off scatter pairs: (x_metric, y_metric, x_label_hint, y_label_hint)
SCATTER_PAIRS = [
    ("Qabf", "Qy",   "edge detail transfer ↑", "structural fidelity ↑"),
    ("SF",   "CC",   "spatial frequency ↑",    "source correlation ↑"),
    ("VIF",  "CC",   "visual info fidelity ↑", "numerical consistency ↑"),
    ("EI",   "Qy",   "edge intensity ↑",       "structural fidelity ↑"),
]

# metrics used in the radar (representative subset, ~10)
RADAR_METRICS = [
    "SF", "AG", "EI", "Qabf", "Qcb",     # detail/edge side
    "SSIM", "MS_SSIM", "VIF", "CC", "Qy", # fidelity/structural side
]


def load_summary(run_dir: Path) -> dict[str, float]:
    summary = {}
    with (run_dir / "metrics" / "evaluation_summary.csv").open() as f:
        for row in csv.DictReader(f):
            summary[row["metric"]] = float(row["average"])
    return summary


def main() -> None:
    t_values = [t for _, t in RUN_DIRS]
    per_run = {}
    for name, t in RUN_DIRS:
        per_run[name] = load_summary(OUT_ROOT / f"text_ratio_T{t}")

    # ---------- 1. Scatter plots of trade-off pairs ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (xm, ym, xh, yh) in zip(axes, SCATTER_PAIRS):
        xs = [per_run[name][xm] for name, _ in RUN_DIRS]
        ys = [per_run[name][ym] for name, _ in RUN_DIRS]
        sc = ax.scatter(xs, ys, c=t_values, cmap="viridis", s=80, edgecolors="k")
        for x, y, t in zip(xs, ys, t_values):
            ax.annotate(f"T{t}", (x, y), fontsize=8,
                        textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel(f"{xm}  ({xh})")
        ax.set_ylabel(f"{ym}  ({yh})")
        ax.set_title(f"{ym} vs {xm}")
        ax.grid(True, linestyle=":", alpha=0.5)
        # trend line to show direction of trade-off
        if len(xs) > 1:
            z = np.polyfit(xs, ys, 1)
            x_line = np.linspace(min(xs), max(xs), 50)
            ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.4, linewidth=1)
        fig.colorbar(sc, ax=ax, label="text_ratio", shrink=0.8)
    fig.suptitle("Trade-off scatter: detail metrics vs fidelity metrics", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out1 = OUT_ROOT / "tradeoff_scatter.png"
    fig.savefig(out1, dpi=130)
    print(f"[tradeoff] wrote {out1}")

    # ---------- 2. Normalized radar chart ----------
    # normalize each metric across runs to [0,1]; invert LOWER_BETTER so outward = better
    norm = {}  # metric -> list of normalized values across runs
    for m in RADAR_METRICS:
        vals = np.array([per_run[name][m] for name, _ in RUN_DIRS])
        lo, hi = vals.min(), vals.max()
        rng = hi - lo if hi > lo else 1.0
        n = (vals - lo) / rng
        if m in LOWER_BETTER:
            n = 1.0 - n
        norm[m] = n

    angles = np.linspace(0, 2 * np.pi, len(RADAR_METRICS), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    cmap = plt.cm.viridis
    for idx, (name, t) in enumerate(RUN_DIRS):
        values = [norm[m][idx] for m in RADAR_METRICS]
        values += values[:1]
        color = cmap(t / max(t_values))
        ax.plot(angles, values, linewidth=1.8, label=name, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_METRICS, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Normalized radar (outward = better)\nLower-is-better metrics inverted: " +
                 ", ".join(sorted(LOWER_BETTER & set(RADAR_METRICS))) or "none",
                 fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), title="text_ratio")
    fig.tight_layout()
    out2 = OUT_ROOT / "tradeoff_radar.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"[tradeoff] wrote {out2}")


if __name__ == "__main__":
    main()
