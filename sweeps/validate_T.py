"""Validation driver across datasets, controlled entirely by CLI.

Runs evaluate_textif_full_recon_v2.py for each (T, dataset) pair, then
produces comparison CSVs, bar charts, and a winner print.

Idempotent: existing evaluation_summary.csv is reused (skipped) unless --force.

Usage examples:
    # Default: T5 and T10, all 5 datasets, 20 imgs each
    D:/software/anaconda3/envs/xpu/python.exe sweeps/validate_T.py

    # Compare three T values
    D:/software/anaconda3/envs/xpu/python.exe sweeps/validate_T.py --T 5 8 10

    # Subset of datasets, larger sample
    D:/software/anaconda3/envs/xpu/python.exe sweeps/validate_T.py --T 5 10 \\
        --datasets MSRS LLVIP --sample 50 --seed 42

    # Force re-eval everything
    D:/software/anaconda3/envs/xpu/python.exe sweeps/validate_T.py --force
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "sweeps" / "out"
EVAL_SCRIPT = REPO / "evaluate_textif_full_recon_v2.py"
PYTHON = r"D:/software/anaconda3/envs/xpu/python.exe"

# All available datasets (subset selectable via --datasets)
ALL_DATASETS = {
    "MSRS":      REPO / "dataset" / "MSRS-main" / "test",
    "LLVIP":     REPO / "dataset" / "LLVIP",
    # NOTE: aligned variant. Raw RoadScene folders are NOT pixel-aligned.
    "RoadScene": REPO / "dataset" / "RoadScene-aligned",
    "M3FD":      REPO / "dataset" / "M3FD_Detection",
    "FLIR":      REPO / "dataset" / "FLIR-align-3class" / "FLIR-align-3class",
}

HIGHER_BETTER = {
    "EN", "MI", "NMI", "SF", "AG", "SD", "CC", "SCD", "PSNR", "VIF",
    "SSIM", "MS_SSIM", "Qabf", "QNCIE", "TE", "EI", "Qy", "Qcb",
}
LOWER_BETTER = {"MSE", "Nabf", "CE"}
KEY_METRICS = ["SF", "AG", "EI", "Qabf", "Qcb", "VIF", "SSIM", "MS_SSIM", "CC", "PSNR", "EN", "MI"]


def run_eval(T: int, name: str, data_path: Path, sample: int, seed: int,
             device: str, force: bool) -> int:
    out_dir = OUT_ROOT / f"text_ratio_T{T}" / "validate" / name
    summary_csv = out_dir / "evaluation_summary.csv"
    if summary_csv.is_file() and not force:
        print(f"[validate] reuse existing: {summary_csv}")
        return 0

    weights = OUT_ROOT / f"text_ratio_T{T}" / "train" / "weights" / "checkpoint.pth"
    if not weights.is_file():
        print(f"[validate] SKIP T={T} {name}: weights missing {weights}", file=sys.stderr)
        return 1

    cmd = [
        PYTHON, "-u", str(EVAL_SCRIPT),
        "--weights_path", str(weights),
        "--data_path",    str(data_path),
        "--output_dir",   str(out_dir),
        "--sample",       str(sample),
        "--seed",         str(seed),
        "--device",       device,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval.log"
    print(f"[validate] T={T} dataset={name} -> {out_dir}")
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"[validate] FAILED T={T} {name} rc={proc.returncode}. See {log_path}", file=sys.stderr)
    return proc.returncode


def load_summary(run_dir: Path) -> dict[str, float]:
    summary = {}
    with (run_dir / "evaluation_summary.csv").open() as f:
        for row in csv.DictReader(f):
            summary[row["metric"]] = float(row["average"])
    return summary


def aggregate(Ts: list[int], datasets: list[str], tag: str) -> None:
    cells: dict[tuple[str, int], dict[str, float]] = {}
    for T in Ts:
        for name in datasets:
            csv_path = OUT_ROOT / f"text_ratio_T{T}" / "validate" / name / "evaluation_summary.csv"
            if csv_path.is_file():
                cells[(name, T)] = load_summary(csv_path.parent)

    if not cells:
        print("[validate] no results to aggregate", file=sys.stderr)
        return

    datasets_done = sorted({n for (n, _) in cells} & set(datasets))
    Ts_done = sorted({T for (_, T) in cells} & set(Ts))
    metrics = list(next(iter(cells.values())).keys())

    # ---- 1. Wide CSV: per dataset, one column per T ----
    wide_csv = OUT_ROOT / f"validate_{tag}_wide.csv"
    with wide_csv.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["metric"]
        for d in datasets_done:
            for T in Ts_done:
                header.append(f"{d}_T{T}")
        w.writerow(header)
        for m in metrics:
            row = [m]
            for d in datasets_done:
                for T in Ts_done:
                    v = cells.get((d, T), {}).get(m)
                    row.append(f"{v:.6f}" if v is not None else "")
            w.writerow(row)
    print(f"[validate] wrote {wide_csv}")

    # ---- 2. Long/tidy CSV ----
    long_csv = OUT_ROOT / f"validate_{tag}_long.csv"
    with long_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text_ratio", "dataset", "metric", "value"])
        for T in Ts_done:
            for d in datasets_done:
                for m in metrics:
                    v = cells.get((d, T), {}).get(m)
                    if v is not None:
                        w.writerow([T, d, m, f"{v:.6f}"])
    print(f"[validate] wrote {long_csv}")

    # ---- 3. Pairwise winner print (smallest vs largest T) ----
    if len(Ts_done) >= 2:
        t_lo, t_hi = Ts_done[0], Ts_done[-1]
        print("\n" + "=" * 80)
        print(f"{'dataset':<12}{'metric':<10}{f'T{t_lo}':>12}{f'T{t_hi}':>12}  winner")
        print("-" * 80)
        for d in datasets_done:
            for m in KEY_METRICS:
                v_lo = cells.get((d, t_lo), {}).get(m)
                v_hi = cells.get((d, t_hi), {}).get(m)
                if v_lo is None or v_hi is None:
                    continue
                if m in LOWER_BETTER:
                    winner = f"T{t_lo}" if v_lo < v_hi else (f"T{t_hi}" if v_hi < v_lo else "tie")
                else:
                    winner = f"T{t_lo}" if v_lo > v_hi else (f"T{t_hi}" if v_hi > v_lo else "tie")
                print(f"{d:<12}{m:<10}{v_lo:>12.4f}{v_hi:>12.4f}  {winner}")
            print("-" * 80)

    # ---- 4. Bar chart: per-dataset groups, one bar per T ----
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        n_metrics = len(KEY_METRICS)
        n_datasets = len(datasets_done)
        fig, axes = plt.subplots(n_datasets, 1, figsize=(11, 3 * n_datasets),
                                 sharex=True)
        if n_datasets == 1:
            axes = [axes]
        x = np.arange(n_metrics)
        width = 0.8 / len(Ts_done)
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        for ax, d in zip(axes, datasets_done):
            for i, T in enumerate(Ts_done):
                ys = [cells.get((d, T), {}).get(m, np.nan) for m in KEY_METRICS]
                offset = (i - (len(Ts_done) - 1) / 2) * width
                ax.bar(x + offset, ys, width, label=f"T{T}",
                       color=colors[i % len(colors)])
            ax.set_title(f"{d}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(KEY_METRICS, rotation=20, ha="right", fontsize=9)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            ax.legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Validation: T={Ts_done} on {n_datasets} datasets", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        chart = OUT_ROOT / f"validate_{tag}_bars.png"
        fig.savefig(chart, dpi=130)
        print(f"[validate] wrote {chart}")

        # ---- 5. Detail-improvement % chart (T_lo -> T_hi) ----
        if len(Ts_done) >= 2:
            detail_metrics = ["SF", "AG", "EI", "Qabf", "Qcb", "VIF"]
            fig, ax = plt.subplots(figsize=(11, 5))
            x = np.arange(len(detail_metrics))
            width = 0.8 / max(n_datasets, 1)
            for i, d in enumerate(datasets_done):
                deltas = []
                for m in detail_metrics:
                    v_lo = cells.get((d, t_lo), {}).get(m, np.nan)
                    v_hi = cells.get((d, t_hi), {}).get(m, np.nan)
                    if v_lo and not np.isnan(v_lo) and not np.isnan(v_hi):
                        deltas.append((v_hi - v_lo) / v_lo * 100)
                    else:
                        deltas.append(np.nan)
                offset = (i - (n_datasets - 1) / 2) * width
                ax.bar(x + offset, deltas, width, label=d,
                       color=colors[i % len(colors)])
            ax.axhline(0, color="k", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(detail_metrics)
            ax.set_ylabel(f"T{t_hi} vs T{t_lo} change (%)")
            ax.set_title(f"Detail-metric change T{t_lo} -> T{t_hi} across datasets")
            ax.legend(fontsize=8, ncol=3)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
            fig.tight_layout()
            chart2 = OUT_ROOT / f"validate_{tag}_detail_pct.png"
            fig.savefig(chart2, dpi=130)
            print(f"[validate] wrote {chart2}")
    except Exception as e:
        print(f"[validate] chart skipped: {e}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate one or more text_ratio values across datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--T", type=int, nargs="+", default=[5, 10],
                    help="One or more text_ratio values to compare")
    ap.add_argument("--datasets", type=str, nargs="+",
                    default=list(ALL_DATASETS.keys()),
                    choices=list(ALL_DATASETS.keys()),
                    help="Subset of datasets to evaluate")
    ap.add_argument("--sample", type=int, default=20,
                    help="Images per dataset (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="xpu")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if evaluation_summary.csv already exists")
    ap.add_argument("--tag", type=str, default=None,
                    help="Output filename tag (default: auto from T values, e.g. 'T5_T10')")
    args = ap.parse_args()

    Ts = sorted(set(args.T))
    tag = args.tag or "T" + "_".join(str(t) for t in Ts)
    selected = {n: ALL_DATASETS[n] for n in args.datasets}

    print(f"[validate] T values: {Ts}")
    print(f"[validate] datasets: {list(selected.keys())}")
    print(f"[validate] sample={args.sample}, seed={args.seed}, device={args.device}")
    print(f"[validate] force={args.force}, tag={tag}\n")

    failures = []
    for T in Ts:
        for name, path in selected.items():
            if not path.is_dir():
                print(f"[validate] SKIP {name}: missing {path}", file=sys.stderr)
                failures.append((T, name, "missing dataset dir"))
                continue
            rc = run_eval(T, name, path, args.sample, args.seed, args.device, args.force)
            if rc != 0:
                failures.append((T, name, f"rc={rc}"))

    if failures:
        print("\n[validate] FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)

    print("\n[validate] all evals done. Aggregating...")
    aggregate(Ts, list(selected.keys()), tag)


if __name__ == "__main__":
    main()
