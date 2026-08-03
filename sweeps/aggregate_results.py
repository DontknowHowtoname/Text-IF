"""Aggregate sweep results: build a summary CSV and a variation chart."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

OUT_ROOT = Path(__file__).resolve().parent / "out"
RUN_DIRS = [
    ("T0",  OUT_ROOT / "text_ratio_T0"),
    ("T1",  OUT_ROOT / "text_ratio_T1"),
    ("T2",  OUT_ROOT / "text_ratio_T2"),
    ("T3",  OUT_ROOT / "text_ratio_T3"),
    ("T5",  OUT_ROOT / "text_ratio_T5"),
    ("T8",  OUT_ROOT / "text_ratio_T8"),
    ("T10", OUT_ROOT / "text_ratio_T10"),
    ("T15", OUT_ROOT / "text_ratio_T15"),
]


def load_summary(run_dir: Path) -> dict[str, float]:
    summary = {}
    with (run_dir / "metrics" / "evaluation_summary.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary[row["metric"]] = float(row["average"])
    return summary


def main() -> None:
    labels = [name for name, _ in RUN_DIRS]
    t_values = [int(name[1:]) for name in labels]
    per_run = {name: load_summary(d) for name, d in RUN_DIRS}
    metrics = list(next(iter(per_run.values())).keys())

    # ---- Wide CSV: metric, T0, T1, ... ----
    wide_csv = OUT_ROOT / "sweep_summary.csv"
    with wide_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + labels)
        for m in metrics:
            w.writerow([m] + [f"{per_run[name][m]:.6f}" for name in labels])
    print(f"[aggregate] wrote {wide_csv}")

    # ---- Long CSV (tidy) ----
    long_csv = OUT_ROOT / "sweep_summary_long.csv"
    with long_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text_ratio", "metric", "value"])
        for name, tval in zip(labels, t_values):
            for m in metrics:
                w.writerow([tval, m, f"{per_run[name][m]:.6f}"])
    print(f"[aggregate] wrote {long_csv}")

    # ---- Chart: grid of subplots, one per metric ----
    cols = 4
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10), sharex=True)
    axes = axes.flatten()
    for ax, m in zip(axes, metrics):
        ys = [per_run[name][m] for name in labels]
        ax.plot(t_values, ys, marker="o", linewidth=1.5)
        ax.set_title(m, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        # annotate first/last
        ax.annotate(f"{ys[0]:.3f}", (t_values[0], ys[0]), fontsize=7,
                    textcoords="offset points", xytext=(4, -8))
        ax.annotate(f"{ys[-1]:.3f}", (t_values[-1], ys[-1]), fontsize=7,
                    textcoords="offset points", xytext=(4, -8))
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Sweep metrics vs text_ratio", fontsize=14)
    fig.supxlabel("text_ratio")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    chart_png = OUT_ROOT / "sweep_variation.png"
    fig.savefig(chart_png, dpi=130)
    print(f"[aggregate] wrote {chart_png}")


if __name__ == "__main__":
    main()
