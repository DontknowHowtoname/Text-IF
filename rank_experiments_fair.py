"""Rank experiments by comprehensive performance - FAIR comparison (same test set only)."""
import csv
from pathlib import Path

CSV_PATH = Path(__file__).parent / "results" / "all_experiments_fusion_metrics.csv"

DIRECTION = {
    "EN": True, "MI": True, "NMI": True, "SF": True, "AG": True, "SD": True,
    "CC": True, "SCD": True, "PSNR": True, "MSE": False, "VIF": True,
    "SSIM": True, "MS_SSIM": True, "Qabf": True, "Nabf": False, "CE": False,
    "QNCIE": True, "TE": True, "EI": True, "Qy": True, "Qcb": True,
}

# Exclude cross-dataset evals (different test set, unfair comparison)
EXCLUDE = {
    "full_recon_v2 @ LLVIP",
    "full_recon_v2 @ M3FD",
    "full_recon_v2 @ MSRS",
    "full_recon_v2 @ RoadScene",
    "textif-me (120ep)",  # duplicate of textif-me (mine, simple)
}


def main():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        metrics = header[1:]
        for row in reader:
            name = row[0]
            if name in EXCLUDE:
                continue
            vals = {}
            for m, v in zip(metrics, row[1:]):
                try:
                    vals[m] = float(v)
                except ValueError:
                    vals[m] = None
            rows.append((name, vals))

    stats = {}
    for m in metrics:
        vals = [r[1][m] for r in rows if r[1][m] is not None]
        if vals:
            stats[m] = (min(vals), max(vals))

    results = []
    for name, vals in rows:
        norms = {}
        for m in metrics:
            v = vals[m]
            if v is None or m not in stats:
                continue
            lo, hi = stats[m]
            if hi == lo:
                norms[m] = 0.5
            else:
                n = (v - lo) / (hi - lo)
                if not DIRECTION[m]:
                    n = 1.0 - n
                norms[m] = n
        avg = sum(norms.values()) / len(norms) if norms else 0
        wins = sum(1 for n in norms.values() if n >= 0.95)
        results.append((name, avg, wins, norms))

    results.sort(key=lambda x: -x[1])

    print(f"{'Rank':<5}{'Experiment':<32}{'Score':>8}{'#Top':>6}")
    print("-" * 60)
    for i, (name, avg, wins, _) in enumerate(results, 1):
        print(f"{i:<5}{name:<32}{avg:>8.4f}{wins:>6}")

    print("\n=== Per-metric winner ===")
    for m in metrics:
        if m not in stats:
            continue
        candidates = [(r[0], r[3].get(m)) for r in results if m in r[3]]
        best = max(candidates, key=lambda x: x[1] if x[1] is not None else -1)
        print(f"  {m:<10} -> {best[0]:<30} (norm={best[1]:.3f})")

    out = Path(__file__).parent / "results" / "all_experiments_ranking_fair.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "experiment", "comprehensive_score", "top_metric_count"] + list(stats.keys()))
        for i, (name, avg, wins, norms) in enumerate(results, 1):
            row = [i, name, f"{avg:.4f}", wins]
            for m in stats:
                v = norms.get(m)
                row.append(f"{v:.3f}" if v is not None else "")
            w.writerow(row)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
