"""Tests for sweeps/aggregate_sweep.py.

Validates that the aggregator correctly pivots long-format evaluation_summary.csv
files (one per text_ratio run) into a single wide-format summary CSV.
"""
import csv
import os
import sys
import tempfile
from pathlib import Path

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sweeps"))


def _write_long_format(path: Path, metrics: dict):
    """Write a fake evaluation_summary.csv in the long format produced by
    evaluate_textif_full_recon_v2.py (columns: metric,average)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "average"])
        for k, v in metrics.items():
            w.writerow([k, v])


def test_aggregate_pivots_long_to_wide(tmp_path):
    """Two text_ratio runs should become two rows in the summary, with all
    21 metrics as columns plus text_ratio and experiment_dir."""
    from aggregate_sweep import aggregate

    # Synthetic metrics for T=2 and T=10
    out_root = tmp_path / "out"
    for t, en_val in [(2, 7.31), (10, 7.40)]:
        run_dir = out_root / f"text_ratio_T{t}" / "metrics"
        run_dir.mkdir(parents=True)
        metrics = {m: en_val + i * 0.01 for i, m in enumerate([
            "EN", "MI", "NMI", "SF", "AG", "SD", "CC", "SCD", "PSNR",
            "MSE", "VIF", "SSIM", "MS_SSIM", "Qabf", "Nabf", "CE",
            "QNCIE", "TE", "EI", "Qy", "Qcb"])}
        _write_long_format(run_dir / "evaluation_summary.csv", metrics)

    output_csv = tmp_path / "summary.csv"
    aggregate(out_root=str(out_root), output_csv=str(output_csv))

    assert output_csv.exists(), "summary CSV was not created"
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
    t_values = sorted(float(r["text_ratio"]) for r in rows)
    assert t_values == [2.0, 10.0], f"unexpected text_ratio values: {t_values}"
    # Check one metric column landed in wide format
    en_values = {float(r["text_ratio"]): float(r["EN"]) for r in rows}
    assert abs(en_values[2.0] - 7.31) < 1e-6
    assert abs(en_values[10.0] - 7.40) < 1e-6
    # experiment_dir column populated
    for r in rows:
        assert r["experiment_dir"].endswith(f"text_ratio_T{r['text_ratio']}/metrics")


def test_aggregate_skips_missing_runs(tmp_path, capsys):
    """If some text_ratio dirs are missing, aggregate what exists and report
    the missing ones on stderr."""
    from aggregate_sweep import aggregate

    out_root = tmp_path / "out"
    # Only create T=5; T=0,1,2,...,15 others missing
    run_dir = out_root / "text_ratio_T5" / "metrics"
    run_dir.mkdir(parents=True)
    _write_long_format(run_dir / "evaluation_summary.csv", {"EN": 7.5, "MI": 3.0})

    output_csv = tmp_path / "summary.csv"
    aggregate(out_root=str(out_root), output_csv=str(output_csv),
              expected_text_ratios=[0, 1, 2, 3, 5, 8, 10, 15])

    captured = capsys.readouterr()
    assert "missing" in captured.err.lower() or "skip" in captured.err.lower()
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]["text_ratio"]) == 5.0


def test_aggregate_handles_empty(tmp_path):
    """Empty out_root produces an empty CSV with headers (no rows)."""
    from aggregate_sweep import aggregate

    out_root = tmp_path / "out"
    out_root.mkdir()
    output_csv = tmp_path / "summary.csv"
    aggregate(out_root=str(out_root), output_csv=str(output_csv))

    assert output_csv.exists()
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert rows == [], "expected zero rows when no run dirs exist"
