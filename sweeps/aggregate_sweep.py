"""Aggregate per-run evaluation_summary.csv files into one wide-format summary.

Each text_ratio sweep run produces a long-format CSV at
``sweeps/out/text_ratio_T<T>/metrics/evaluation_summary.csv`` with columns
``metric,average`` (written by ``evaluate_textif_full_recon_v2.py``). This
script pivots all such files into one wide-format CSV with one row per
text_ratio value and one column per metric, matching the schema of
``results/all_experiments_fusion_metrics.csv`` (plus a ``text_ratio`` column).

Usage:
    python sweeps/aggregate_sweep.py \\
        --out-root sweeps/out \\
        --output-csv sweeps/text_ratio_sweep_summary.csv \\
        [--expected-text-ratios 0,1,2,3,5,8,10,15]
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional


# Canonical metric column order (matches all_experiments_fusion_metrics.csv
# minus the "Experiment" column; text_ratio is prepended separately).
METRIC_COLUMNS = [
    "EN", "MI", "NMI", "SF", "AG", "SD", "CC", "SCD",
    "PSNR", "MSE", "VIF", "SSIM", "MS_SSIM",
    "Qabf", "Nabf", "CE", "QNCIE", "TE", "EI", "Qy", "Qcb",
]


def _read_long_format(path: Path) -> dict:
    """Read evaluation_summary.csv (metric,average) -> dict."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric")
            avg = row.get("average")
            if metric is None or avg is None:
                continue
            try:
                out[metric] = float(avg)
            except ValueError:
                # Non-numeric metric value; skip with warning
                print(f"WARN: non-numeric value for {metric}: {avg!r}", file=sys.stderr)
    return out


def _discover_run_dirs(out_root: Path) -> List[Path]:
    """Find all text_ratio_T*/metrics dirs with an evaluation_summary.csv."""
    if not out_root.is_dir():
        return []
    runs = []
    for child in sorted(out_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("text_ratio_T"):
            continue
        summary = child / "metrics" / "evaluation_summary.csv"
        if summary.is_file():
            runs.append(child)
    return runs


def _extract_text_ratio(run_dir: Path):
    """Parse 'text_ratio_T5' -> 5 (int) or 'text_ratio_T2.5' -> 2.5 (float).

    Returns the value as int when the suffix is integer-valued so that the
    string representation round-trips against the directory name (e.g. dir
    'text_ratio_T2' yields int 2, which serializes as '2' in the CSV, not
    '2.0'). Returns None on parse failure.
    """
    name = run_dir.name
    prefix = "text_ratio_T"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    try:
        # Prefer int when the suffix is integer-valued so str(value) matches
        # the directory suffix exactly (the test asserts endswith on this).
        if "." not in suffix:
            return int(suffix)
        return float(suffix)
    except ValueError:
        return None


def aggregate(out_root: str, output_csv: str,
              expected_text_ratios: Optional[List[float]] = None) -> None:
    """Aggregate per-run CSVs into one wide-format summary.

    Args:
        out_root: Directory containing text_ratio_T*/ subdirs.
        output_csv: Path to write the wide-format summary.
        expected_text_ratios: If provided, warn (stderr) about any values
            from this list that are missing in out_root.
    """
    out_root_path = Path(out_root)
    runs = _discover_run_dirs(out_root_path)

    found_ratios = set()
    rows = []
    for run_dir in runs:
        t = _extract_text_ratio(run_dir)
        if t is None:
            print(f"WARN: could not parse text_ratio from dir name: {run_dir.name}",
                  file=sys.stderr)
            continue
        metrics = _read_long_format(run_dir / "metrics" / "evaluation_summary.csv")
        row = {"text_ratio": t}
        for m in METRIC_COLUMNS:
            row[m] = metrics.get(m, "")
        # Use forward slashes for portability across platforms (tests and
        # downstream consumers assert on '/'-separated paths).
        row["experiment_dir"] = str(run_dir / "metrics").replace(os.sep, "/")
        rows.append(row)
        found_ratios.add(t)

    # Report missing
    if expected_text_ratios is not None:
        missing = [t for t in expected_text_ratios if t not in found_ratios]
        if missing:
            print(f"WARN: missing text_ratio runs: {sorted(missing)}", file=sys.stderr)

    # Sort by text_ratio ascending
    rows.sort(key=lambda r: r["text_ratio"])

    # Write
    fieldnames = ["text_ratio"] + METRIC_COLUMNS + ["experiment_dir"]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows -> {output_path}")


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", default="sweeps/out",
                        help="Root dir containing text_ratio_T*/ subdirs (default: sweeps/out)")
    parser.add_argument("--output-csv", default="sweeps/text_ratio_sweep_summary.csv",
                        help="Output wide-format CSV path")
    parser.add_argument("--expected-text-ratios", type=str, default=None,
                        help="Comma-separated expected text_ratio values, e.g. '0,1,2,3,5,8,10,15'")
    args = parser.parse_args()

    expected = _parse_float_list(args.expected_text_ratios) if args.expected_text_ratios else None
    aggregate(out_root=args.out_root, output_csv=args.output_csv,
              expected_text_ratios=expected)


if __name__ == "__main__":
    main()
