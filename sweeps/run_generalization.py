"""Generalization eval: run T5/T8/T10 weights on 5 held-out datasets, 20 imgs each."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "sweeps" / "out"
EVAL_SCRIPT = REPO / "evaluate_textif_full_recon_v2.py"
PYTHON = r"D:/software/anaconda3/envs/xpu/python.exe"

# dataset short name -> data_path
DATASETS = {
    "MSRS":      REPO / "dataset" / "MSRS-main" / "test",
    "LLVIP":     REPO / "dataset" / "LLVIP",
    # NOTE: use RoadScene-aligned (cropinfrared + crop_LR_visible, 221 perfectly
    # aligned pairs). The raw RoadScene/{infrared,visible} folders are NOT
    # pixel-aligned (IR 640x512, VIS varied ~787x759 to 1819x1051) and would
    # produce stretched/misaligned fused outputs.
    "RoadScene": REPO / "dataset" / "RoadScene-aligned",
    "M3FD":      REPO / "dataset" / "M3FD_Detection",
    "FLIR":      REPO / "dataset" / "FLIR-align-3class" / "FLIR-align-3class",
}

TS = [5, 8, 10]


def run_one(T: int, name: str, data_path: Path, sample: int, seed: int, device: str) -> int:
    out_dir = OUT_ROOT / f"text_ratio_T{T}" / "gen" / name
    summary_csv = out_dir / "evaluation_summary.csv"
    if summary_csv.is_file():
        print(f"[gen] skip (exists): {summary_csv}")
        return 0

    cmd = [
        PYTHON, "-u", str(EVAL_SCRIPT),
        "--weights_path", str(OUT_ROOT / f"text_ratio_T{T}" / "train" / "weights" / "checkpoint.pth"),
        "--data_path",    str(data_path),
        "--output_dir",   str(out_dir),
        "--sample",       str(sample),
        "--seed",         str(seed),
        "--device",       device,
    ]
    log_path = out_dir / "eval.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen] T={T} dataset={name} -> {out_dir}")
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"[gen] FAILED T={T} {name} rc={proc.returncode}. See {log_path}", file=sys.stderr)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Generalization eval for T5/T8/T10 across datasets.")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--device", type=str, default="xpu")
    args = ap.parse_args()

    failures = []
    for T in TS:
        for name, path in DATASETS.items():
            if not path.is_dir():
                print(f"[gen] SKIP {name}: missing {path}", file=sys.stderr)
                failures.append((T, name, "missing dataset dir"))
                continue
            rc = run_one(T, name, path, args.sample, args.seed, args.device)
            if rc != 0:
                failures.append((T, name, f"rc={rc}"))

    if failures:
        print("\n[gen] FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)

    print("\n[gen] all done. Aggregating...")
    aggregate()


def aggregate() -> None:
    """Build a long CSV + wide CSV across T x dataset."""
    metrics_order = [
        "EN", "MI", "NMI", "SF", "AG", "SD", "CC", "SCD", "PSNR", "MSE",
        "VIF", "SSIM", "MS_SSIM", "Qabf", "Nabf", "CE", "QNCIE", "TE", "EI",
        "Qy", "Qcb",
    ]
    rows_long = []  # (T, dataset, metric, value)
    cells = {}      # (dataset, T) -> {metric: value}
    for T in TS:
        for name in DATASETS:
            summary_csv = OUT_ROOT / f"text_ratio_T{T}" / "gen" / name / "evaluation_summary.csv"
            if not summary_csv.is_file():
                continue
            with summary_csv.open() as f:
                for row in csv.DictReader(f):
                    val = float(row["average"])
                    cells.setdefault((name, T), {})[row["metric"]] = val
                    rows_long.append((T, name, row["metric"], val))

    datasets_present = sorted({n for (n, _) in cells})
    Ts_present = sorted({T for (_, T) in cells})

    # long CSV
    long_csv = OUT_ROOT / "generalization_long.csv"
    with long_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text_ratio", "dataset", "metric", "value"])
        for T, name, m, v in rows_long:
            w.writerow([T, name, m, f"{v:.6f}"])
    print(f"[gen] wrote {long_csv}")

    # wide CSV: metric, then one column per (dataset, T)
    wide_csv = OUT_ROOT / "generalization_wide.csv"
    cols = [f"{d}_T{T}" for d in datasets_present for T in Ts_present if (d, T) in cells]
    with wide_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + cols)
        for m in metrics_order:
            row = [m]
            for d in datasets_present:
                for T in Ts_present:
                    if (d, T) in cells:
                        v = cells[(d, T)].get(m)
                        row.append(f"{v:.6f}" if v is not None else "")
            w.writerow(row)
    print(f"[gen] wrote {wide_csv}")

    # chart: 5 subplots (one per dataset), 3 lines (T5/T8/T10) showing a
    # representative subset of metrics (since each dataset has 21 metrics,
    # we plot the per-T average rank across the 6 most diagnostic metrics).
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        diagnostic = ["SF", "AG", "EI", "Qabf", "SSIM", "VIF"]
        fig, axes = plt.subplots(1, len(datasets_present), figsize=(5 * len(datasets_present), 5),
                                 sharey=False)
        if len(datasets_present) == 1:
            axes = [axes]
        for ax, d in zip(axes, datasets_present):
            x = np.arange(len(diagnostic))
            width = 0.25
            for i, T in enumerate(Ts_present):
                if (d, T) not in cells:
                    continue
                ys = [cells[(d, T)].get(m, np.nan) for m in diagnostic]
                ax.bar(x + (i - 1) * width, ys, width, label=f"T{T}")
            ax.set_xticks(x)
            ax.set_xticklabels(diagnostic, rotation=30, ha="right", fontsize=9)
            ax.set_title(d)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            ax.legend(fontsize=8)
        fig.suptitle("Generalization: T5/T8/T10 on held-out datasets (20 imgs each)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        chart = OUT_ROOT / "generalization_bars.png"
        fig.savefig(chart, dpi=130)
        print(f"[gen] wrote {chart}")

        # Per-dataset T5/T8/T10 line chart for two key trade-off metrics: Qabf and SSIM
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for d in datasets_present:
            xs = [T for T in Ts_present if (d, T) in cells]
            ax1.plot(xs, [cells[(d, T)].get("Qabf", np.nan) for T in xs],
                     marker="o", label=d)
            ax2.plot(xs, [cells[(d, T)].get("SSIM", np.nan) for T in xs],
                     marker="s", label=d)
        ax1.set_title("Qabf across datasets")
        ax2.set_title("SSIM across datasets")
        for ax in (ax1, ax2):
            ax.set_xlabel("text_ratio")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(fontsize=8)
        fig.suptitle("Trade-off consistency across datasets", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        chart2 = OUT_ROOT / "generalization_lines.png"
        fig.savefig(chart2, dpi=130)
        print(f"[gen] wrote {chart2}")
    except Exception as e:
        print(f"[gen] chart skipped: {e}")


if __name__ == "__main__":
    main()
