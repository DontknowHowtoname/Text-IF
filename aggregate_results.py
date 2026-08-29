"""Aggregate all experimental fusion/detection metrics into CSV + Markdown tables."""
import csv
import os
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

# Mapping: result folder -> friendly experiment name (None means skip / non-metric folder)
FUSION_DIRS = {
    "textif_simple_eval_0":               "textif-pre (pretrained)",
    "textif_simple_eval":                 "textif-me (mine, simple)",
    "textif_simple_eval_1":               "textif-me (120ep)",
    "textif_simple_eval_sample2_0":       "textif-simple-sample2",
    "textif_full_recon_eval_0":           "text_rcon (recon v1)",
    "textif_full_recon_v2_eval_ft_1":     "text_recon2_ft",
    "textif_full_recon_v2_eval_ft_myweight": "text_recon2_ft_myweight",
    "textif_full_recon_v2_eval_ft_proweight": "text_recon2_ft_proweight",
    "textif_obj_enhance_eval":            "textif_obj_enhance",
    "textif_full_recon_v2_eval_LLVIP":    "full_recon_v2 @ LLVIP",
    "textif_full_recon_v2_eval_M3FD":     "full_recon_v2 @ M3FD",
    "textif_full_recon_v2_eval_MSRS":     "full_recon_v2 @ MSRS",
    "textif_full_recon_v2_eval_RoadScene":"full_recon_v2 @ RoadScene",
    "textif_full_recon_v2_eval_TNO":      "full_recon_v2 @ TNO",
}

DETECTION_DIRS = {
    "LLVIP_visible_baseline_detection":          "LLVIP visible (baseline)",
    "textif_full_recon_v2_eval_LLVIP_detection": "LLVIP fused (full_recon_v2)",
}


def read_fusion_summary(folder: Path):
    csv_path = folder / "evaluation_summary.csv"
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) >= 2 and row[0]:
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    pass
    return out


def read_detection_summary(folder: Path):
    csv_path = folder / "detection_summary.csv"
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2 and row[0]:
                key, val = row[0], row[1]
                if key in ("mAP@0.5", "mAP@0.75", "mAP@0.5:0.95",
                           "num_images", "total_gt_boxes", "total_detections"):
                    try:
                        out[key] = float(val)
                    except ValueError:
                        out[key] = val
    return out


def main():
    # Collect fusion metrics
    fusion_data = {}
    all_fusion_metrics = []
    for folder, name in FUSION_DIRS.items():
        path = RESULTS / folder
        metrics = read_fusion_summary(path)
        if not metrics:
            print(f"[warn] no fusion metrics in {folder}")
            continue
        fusion_data[name] = metrics
        for m in metrics:
            if m not in all_fusion_metrics:
                all_fusion_metrics.append(m)

    # Collect detection metrics
    detection_data = {}
    all_det_metrics = []
    for folder, name in DETECTION_DIRS.items():
        path = RESULTS / folder
        metrics = read_detection_summary(path)
        if not metrics:
            print(f"[warn] no detection metrics in {folder}")
            continue
        detection_data[name] = metrics
        for m in metrics:
            if m not in all_det_metrics:
                all_det_metrics.append(m)

    # ---- Write fusion CSV ----
    fusion_csv = RESULTS / "all_experiments_fusion_metrics.csv"
    with open(fusion_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Experiment"] + all_fusion_metrics)
        for name, metrics in fusion_data.items():
            w.writerow([name] + [f"{metrics.get(m, ''):.6f}" if isinstance(metrics.get(m), float) else ""
                                  for m in all_fusion_metrics])

    # ---- Write detection CSV ----
    det_csv = RESULTS / "all_experiments_detection_metrics.csv"
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Experiment"] + all_det_metrics)
        for name, metrics in detection_data.items():
            row = [name]
            for m in all_det_metrics:
                v = metrics.get(m, "")
                if isinstance(v, float):
                    row.append(f"{v:.6f}")
                else:
                    row.append(v)
            w.writerow(row)

    # ---- Markdown fusion ----
    md = ["# Fusion Metrics Summary\n"]
    md.append("所有融合实验的客观指标汇总（来源：各 `evaluation_summary.csv`）。\n")
    md.append("| " + " | ".join(["Experiment"] + all_fusion_metrics) + " |")
    md.append("|" + "|".join(["---"] * (len(all_fusion_metrics) + 1)) + "|")
    for name, metrics in fusion_data.items():
        cells = [name]
        for m in all_fusion_metrics:
            v = metrics.get(m)
            cells.append(f"{v:.4f}" if isinstance(v, float) else "-")
        md.append("| " + " | ".join(cells) + " |")

    md.append("\n# Detection Metrics Summary\n")
    md.append("LLVIP 数据集上 YOLOv5 检测结果汇总（来源：各 `detection_summary.csv`，target=person）。\n")
    md.append("| " + " | ".join(["Experiment"] + all_det_metrics) + " |")
    md.append("|" + "|".join(["---"] * (len(all_det_metrics) + 1)) + "|")
    for name, metrics in detection_data.items():
        cells = [name]
        for m in all_det_metrics:
            v = metrics.get(m)
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v) if v != "" else "-")
        md.append("| " + " | ".join(cells) + " |")

    md_path = RESULTS / "all_experiments_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Wrote: {fusion_csv}")
    print(f"Wrote: {det_csv}")
    print(f"Wrote: {md_path}")
    print(f"Fusion experiments: {len(fusion_data)} | metrics: {len(all_fusion_metrics)}")
    print(f"Detection experiments: {len(detection_data)} | metrics: {len(all_det_metrics)}")


if __name__ == "__main__":
    main()
