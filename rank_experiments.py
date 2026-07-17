"""Rank experiments by comprehensive (z-score averaged) performance across fusion metrics."""
import csv
from pathlib import Path

CSV_PATH = Path(__file__).parent / "results" / "all_experiments_fusion_metrics.csv"

# Direction: True = higher better, False = lower better
DIRECTION = {
    "EN": True, "MI": True, "NMI": True, "SF": True, "AG": True, "SD": True,
    "CC": True, "SCD": True, "PSNR": True, "MSE": False, "VIF": True,
    "SSIM": True, "MS_SSIM": True, "Qabf": True, "Nabf": False, "CE": False,
    "QNCIE": True, "TE": True, "EI": True, "Qy": True, "Qcb": True,
}

# Weights reflecting "importance" for fusion quality (subjective but standard).
# Set to 1.0 for all to keep neutral; tweak if you want emphasis.
WEIGHTS = {
    "EN": 1.0, "MI": 1.0, "SF": 1.0, "AG": 1.0, "SD": 1.0, "SCD": 1.0,
    "PSNR": 1.0, "VIF": 1.0, "SSIM": 1.0, "MS_SSIM": 1.0, "Qabf": 1.0,
    "Qy": 1.0, "Qcb": 1.0, "CC": 1.0, "NMI": 1.0, "Nabf": 1.0, "CE": 1.0,
    "QNCIE": 1.0, "TE": 1.0, "EI": 1.0, "MSE": 1.0,
}


def main():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        metrics = header[1:]
        for row in reader:
            name = row[0]
            vals = {}
            for m, v in zip(metrics, row[1:]):
                try:
                    vals[m] = float(v)
                except ValueError:
                    vals[m] = None
            rows.append((name, vals))

    # Compute min/max per metric (only non-None)
    stats = {}
    for m in metrics:
        vals = [r[1][m] for r in rows if r[1][m] is not None]
        if vals:
            stats[m] = (min(vals), max(vals))

    # Normalize each metric to [0,1] by direction, then average weighted.
    print(f"{'Experiment':<32} | {'avg_score':>9} | {'rank':>4} | wins")
    print("-" * 90)
    results = []
    for name, vals in rows:
        scores = []
        per_metric_rank = {}
        for m in metrics:
            v = vals[m]
            if v is None or m not in stats:
                continue
            lo, hi = stats[m]
            if hi == lo:
                norm = 0.5
            else:
                norm = (v - lo) / (hi - lo)
                if not DIRECTION[m]:
                    norm = 1.0 - norm
            scores.append((WEIGHTS.get(m, 1.0), norm, m))
            per_metric_rank[m] = norm

        total_w = sum(w for w, _, _ in scores)
        avg = sum(w * n for w, n, _ in scores) / total_w if total_w else 0
        wins = sum(1 for _, _, m in scores if per_metric_rank[m] >= 0.95)
        results.append((name, avg, wins, per_metric_rank))

    # Sort by avg score desc
    results.sort(key=lambda x: -x[1])
    for i, (name, avg, wins, _) in enumerate(results, 1):
        print(f"{name:<32} | {avg:>9.4f} | {i:>4} | {wins} top metrics")

    # Top-3 per metric
    print("\n=== Best experiment per metric ===")
    for m in metrics:
        # find experiment with max value if direction up, else min
        candidates = [(r[0], r[3].get(m)) for r in results if m in r[3]]
        if not candidates:
            continue
        if DIRECTION[m]:
            best = max(candidates, key=lambda x: x[1] if x[1] is not None else -1)
        else:
            best = min(candidates, key=lambda x: x[1] if x[1] is not None else 1e9)
        print(f"  {m:<10} -> {best[0]:<30} (norm={best[1]:.3f})")

    # Write a rankings CSV
    out = Path(__file__).parent / "results" / "all_experiments_ranking.csv"
    metric_cols = list(stats.keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "experiment", "comprehensive_score", "top_metric_count"] + metric_cols)
        for i, (name, avg, wins, ranks) in enumerate(results, 1):
            row = [i, name, f"{avg:.4f}", wins]
            for m in metric_cols:
                v = ranks.get(m)
                row.append(f"{v:.3f}" if v is not None else "")
            w.writerow(row)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
