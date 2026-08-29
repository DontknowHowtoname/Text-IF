"""Generalization eval: run T5/T8/T10 weights on held-out datasets, 20 imgs each.

HPC-compatible: uses ``sys.executable`` (no hardcoded interpreter path) and
resolves the repo / output root / dataset paths from CLI args and env vars, so
it runs unchanged under SLURM where datasets typically live on shared scratch.

Per-dataset paths can be overridden via env vars of the form
``TEXTIF_DS_<NAME>`` (e.g. ``TEXTIF_DS_TNO=/scratch/datasets/TNO``); otherwise
they default to ``<repo>/dataset/<...>`` as on the local dev box.
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL_SCRIPT_NAME = "evaluate_textif_full_recon_v2.py"

# dataset short name -> *default* data_path (relative to REPO). Override at
# runtime via env var TEXTIF_DS_<NAME> (uppercased short name).
DEFAULT_DATASETS = {
    "MSRS":      "dataset/MSRS-main/test",
    "LLVIP":     "dataset/LLVIP",
    # NOTE: use RoadScene-aligned (cropinfrared + crop_LR_visible, 221 perfectly
    # aligned pairs). The raw RoadScene/{infrared,visible} folders are NOT
    # pixel-aligned (IR 640x512, VIS varied ~787x759 to 1819x1051) and would
    # produce stretched/misaligned fused outputs.
    "RoadScene": "dataset/RoadScene-aligned",
    "M3FD":      "dataset/M3FD_Detection",
    "FLIR":      "dataset/FLIR-align-3class/FLIR-align-3class",
    # TNO: 153 paired grayscale PNGs (8-bit). ir/+vi/ layout, native
    # heterogeneous resolutions — handled online by the eval pipeline.
    "TNO":       "dataset/TNO/test",
}

TS = [5, 8, 10]


def _resolve_datasets(repo: Path) -> dict:
    """Build {name: Path}. Precedence: env TEXTIF_DS_<NAME> > default rel path."""
    out = {}
    for name, rel in DEFAULT_DATASETS.items():
        env_val = os.environ.get(f"TEXTIF_DS_{name.upper()}")
        out[name] = Path(env_val) if env_val else repo / rel
    return out


def run_one(T: int, name: str, data_path: Path, sample: int, seed: int, device: str,
            repo: Path, out_root: Path, out_tag: str = "gen") -> int:
    out_dir = out_root / f"text_ratio_T{T}" / out_tag / name
    summary_csv = out_dir / "evaluation_summary.csv"
    if summary_csv.is_file():
        print(f"[{out_tag}] skip (exists): {summary_csv}")
        return 0

    cmd = [
        sys.executable, "-u", str(repo / EVAL_SCRIPT_NAME),
        "--weights_path", str(out_root / f"text_ratio_T{T}" / "train" / "weights" / "checkpoint.pth"),
        "--data_path",    str(data_path),
        "--output_dir",   str(out_dir),
        "--sample",       str(sample),
        "--seed",         str(seed),
        "--device",       device,
    ]
    log_path = out_dir / "eval.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{out_tag}] T={T} dataset={name} -> {out_dir}")
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, cwd=str(repo), stdout=logf, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"[{out_tag}] FAILED T={T} {name} rc={proc.returncode}. See {log_path}", file=sys.stderr)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Generalization eval for T5/T8/T10 across datasets.")
    ap.add_argument("--repo_dir",    type=str, default=str(REPO),
                    help="Repository root (must contain evaluate_textif_full_recon_v2.py). "
                         "Default: derived from this file's location.")
    ap.add_argument("--output_root", type=str, default=None,
                    help="Parent dir for text_ratio_T<T>/ output. "
                         "Default: <repo_dir>/sweeps/out.")
    ap.add_argument("--sample",  type=int, default=20,
                    help="Images per dataset (0 = use the entire split).")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--device",  type=str, default="xpu",
                    help="Device passed to the eval script (e.g. 'xpu', 'cuda').")
    ap.add_argument("--datasets", type=str, default=None,
                    help="Comma-separated subset of dataset short names to run "
                         "(e.g. 'TNO'). Default: all datasets in DEFAULT_DATASETS.")
    ap.add_argument("--out_tag", type=str, default="gen",
                    help="Sub-folder under text_ratio_T{T}/. Use e.g. 'gen_full' to "
                         "keep an existing 20-sample 'gen' run intact.")
    ap.add_argument("--skip_aggregate", action="store_true",
                    help="Skip CSV/PNG aggregation at the end.")
    args = ap.parse_args()

    repo = Path(args.repo_dir)
    out_root = Path(args.output_root) if args.output_root else repo / "sweeps" / "out"
    datasets = _resolve_datasets(repo)
    if args.datasets:
        wanted = {s.strip() for s in args.datasets.split(",") if s.strip()}
        unknown = wanted - set(datasets)
        if unknown:
            print(f"ERROR: unknown --datasets: {sorted(unknown)}. "
                  f"Known: {sorted(datasets)}", file=sys.stderr)
            sys.exit(2)
        datasets = {k: v for k, v in datasets.items() if k in wanted}

    failures = []
    for T in TS:
        for name, path in datasets.items():
            if not path.is_dir():
                print(f"[{args.out_tag}] SKIP {name}: missing {path}", file=sys.stderr)
                failures.append((T, name, "missing dataset dir"))
                continue
            rc = run_one(T, name, path, args.sample, args.seed, args.device,
                         repo, out_root, args.out_tag)
            if rc != 0:
                failures.append((T, name, f"rc={rc}"))

    if failures:
        print(f"\n[{args.out_tag}] FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)

    if args.skip_aggregate:
        return
    print(f"\n[{args.out_tag}] all done. Aggregating...")
    aggregate(args.out_tag, out_root, datasets)


def aggregate(out_tag: str = "gen", out_root: Path = None,
              datasets: dict = None) -> None:
    """Build a long CSV + wide CSV across T x dataset.

    ``out_root`` and ``datasets`` default to the legacy module-level layout
    (``<repo>/sweeps/out`` and ``DEFAULT_DATASETS``) so the function remains
    usable as a library/import without explicit args.
    """
    if out_root is None:
        out_root = REPO / "sweeps" / "out"
    if datasets is None:
        datasets = _resolve_datasets(REPO)
    metrics_order = [
        "EN", "MI", "NMI", "SF", "AG", "SD", "CC", "SCD", "PSNR", "MSE",
        "VIF", "SSIM", "MS_SSIM", "Qabf", "Nabf", "CE", "QNCIE", "TE", "EI",
        "Qy", "Qcb",
    ]
    rows_long = []  # (T, dataset, metric, value)
    cells = {}      # (dataset, T) -> {metric: value}
    for T in TS:
        for name in datasets:
            summary_csv = out_root / f"text_ratio_T{T}" / out_tag / name / "evaluation_summary.csv"
            if not summary_csv.is_file():
                continue
            with summary_csv.open() as f:
                for row in csv.DictReader(f):
                    val = float(row["average"])
                    cells.setdefault((name, T), {})[row["metric"]] = val
                    rows_long.append((T, name, row["metric"], val))

    datasets_present = sorted({n for (n, _) in cells})
    Ts_present = sorted({T for (_, T) in cells})

    # suffix output file names by out_tag so different runs don't overwrite
    suffix = "" if out_tag == "gen" else f"_{out_tag}"

    # long CSV
    long_csv = out_root / f"generalization_long{suffix}.csv"
    with long_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text_ratio", "dataset", "metric", "value"])
        for T, name, m, v in rows_long:
            w.writerow([T, name, m, f"{v:.6f}"])
    print(f"[{out_tag}] wrote {long_csv}")

    # wide CSV: metric, then one column per (dataset, T)
    wide_csv = out_root / f"generalization_wide{suffix}.csv"
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
    print(f"[{out_tag}] wrote {wide_csv}")

    # chart: one subplot per dataset, 3 bars (T5/T8/T10) over the 6 most
    # diagnostic metrics.
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
        title_note = "full split" if out_tag != "gen" else "20 imgs each"
        fig.suptitle(f"Generalization [{out_tag}]: T5/T8/T10 ({title_note})", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        chart = out_root / f"generalization_bars{suffix}.png"
        fig.savefig(chart, dpi=130)
        print(f"[{out_tag}] wrote {chart}")

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
        fig.suptitle(f"Trade-off consistency across datasets [{out_tag}]", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        chart2 = out_root / f"generalization_lines{suffix}.png"
        fig.savefig(chart2, dpi=130)
        print(f"[{out_tag}] wrote {chart2}")
    except Exception as e:
        print(f"[{out_tag}] chart skipped: {e}")


if __name__ == "__main__":
    main()
