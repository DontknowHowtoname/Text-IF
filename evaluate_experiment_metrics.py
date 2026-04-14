import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


METRIC_DIR = os.path.join(os.path.dirname(__file__), "metric")
if METRIC_DIR not in sys.path:
    sys.path.insert(0, METRIC_DIR)

from Metric_torch import (  # noqa: E402
    AG_function,
    CC_function,
    CE_function,
    EI_function,
    EN_function,
    MI_function,
    MSE_function,
    MS_SSIM_function,
    NMI_function,
    Nabf_function,
    PSNR_function,
    QNCIE_function,
    Qabf_function,
    Qcb_function,
    Qy_function,
    SCD_function,
    SD_function,
    SF_function,
    SSIM_function,
    TE_function,
    VIF_function,
)


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRIC_NAMES = [
    "CE",
    "NMI",
    "QNCIE",
    "TE",
    "EI",
    "Qy",
    "Qcb",
    "EN",
    "MI",
    "SF",
    "AG",
    "SD",
    "CC",
    "SCD",
    "VIF",
    "MSE",
    "PSNR",
    "Qabf",
    "Nabf",
    "SSIM",
    "MS_SSIM",
]


@dataclass
class Triplet:
    fused: str
    vis: str
    ir: str
    name: str


def resolve_device(device_name: str):
    if device_name == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(device_name)


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img)


def safe_float(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.item()
    try:
        return float(value)
    except Exception:
        return float("nan")


def is_finite_number(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def compute_all_metrics(ir_path: str, vis_path: str, fused_path: str, device: torch.device) -> Dict[str, float]:
    fused = load_gray(fused_path)
    ir = load_gray(ir_path)
    vis = load_gray(vis_path)

    fused_tensor = torch.tensor(fused).float().to(device)
    ir_tensor = torch.tensor(ir).float().to(device)
    vis_tensor = torch.tensor(vis).float().to(device)

    fused_int = fused.astype(np.int32)
    ir_int = ir.astype(np.int32)
    vis_int = vis.astype(np.int32)

    fused_float = fused.astype(np.float32)
    ir_float = ir.astype(np.float32)
    vis_float = vis.astype(np.float32)

    results = {
        "CE": CE_function(ir_tensor, vis_tensor, fused_tensor),
        "NMI": NMI_function(ir_int, vis_int, fused_int, gray_level=256),
        "QNCIE": QNCIE_function(ir_tensor, vis_tensor, fused_tensor),
        "TE": TE_function(ir_tensor, vis_tensor, fused_tensor),
        "EI": EI_function(fused_tensor),
        "Qy": Qy_function(ir_tensor, vis_tensor, fused_tensor),
        "Qcb": Qcb_function(ir_tensor, vis_tensor, fused_tensor),
        "EN": EN_function(fused_tensor),
        "MI": MI_function(ir_int, vis_int, fused_int, gray_level=256),
        "SF": SF_function(fused_tensor),
        "AG": AG_function(fused_tensor),
        "SD": SD_function(fused_tensor),
        "CC": CC_function(ir_tensor, vis_tensor, fused_tensor),
        "SCD": SCD_function(ir_tensor, vis_tensor, fused_tensor),
        "VIF": VIF_function(ir_tensor, vis_tensor, fused_tensor),
        "MSE": MSE_function(ir_tensor, vis_tensor, fused_tensor),
        "PSNR": PSNR_function(ir_tensor, vis_tensor, fused_tensor),
        "Qabf": Qabf_function(ir_float, vis_float, fused_float),
        "Nabf": Nabf_function(ir_tensor, vis_tensor, fused_tensor),
        "SSIM": SSIM_function(ir_float, vis_float, fused_float),
        "MS_SSIM": MS_SSIM_function(ir_float, vis_float, fused_float),
    }

    return {k: safe_float(v) for k, v in results.items()}


def find_triplets(epoch_dir: str) -> List[Triplet]:
    files = sorted(os.listdir(epoch_dir), key=natural_key)
    file_set = set(files)
    triplets: List[Triplet] = []

    for name in files:
        stem, ext = os.path.splitext(name)
        if ext.lower() not in SUPPORTED_EXTS:
            continue
        if stem.endswith("vis") or stem.endswith("ir"):
            continue

        vis_name = f"{stem}vis{ext}"
        ir_name = f"{stem}ir{ext}"
        if vis_name in file_set and ir_name in file_set:
            triplets.append(
                Triplet(
                    fused=os.path.join(epoch_dir, name),
                    vis=os.path.join(epoch_dir, vis_name),
                    ir=os.path.join(epoch_dir, ir_name),
                    name=stem,
                )
            )

    return triplets


def collect_epoch_dirs(img_root: str, requested_epochs: Optional[List[str]]) -> List[Tuple[str, str]]:
    epoch_dirs = []
    if requested_epochs:
        for epoch in requested_epochs:
            epoch_path = os.path.join(img_root, epoch)
            if os.path.isdir(epoch_path):
                epoch_dirs.append((epoch, epoch_path))
            else:
                print(f"[WARN] Epoch folder not found: {epoch_path}")
    else:
        for folder in sorted(os.listdir(img_root), key=natural_key):
            path = os.path.join(img_root, folder)
            if os.path.isdir(path):
                epoch_dirs.append((folder, path))
    return epoch_dirs


def mean_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for metric in METRIC_NAMES:
        values = [r[metric] for r in records if metric in r and is_finite_number(r[metric])]
        summary[metric] = float(np.mean(values)) if values else float("nan")
    return summary


def save_csv(path: str, rows: List[Dict[str, float]]):
    headers = ["epoch", "num_samples"] + METRIC_NAMES
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: str, payload: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fused results in an experiment folder using all metrics in metric/."
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment folder, e.g. experiments/TextIF_train_20260408-185710",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="",
        help="Optional image root. Default: <experiment-dir>/img",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="",
        help="Optional comma-separated epoch folders, e.g. 2,20,100. Default: all folders under img/",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="metrics",
        help="Output filename prefix saved in experiment folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, xpu, cuda, or cpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_dir = os.path.abspath(args.experiment_dir)
    img_root = os.path.abspath(args.img_dir) if args.img_dir else os.path.join(experiment_dir, "img")

    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f"Experiment folder not found: {experiment_dir}")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"Image folder not found: {img_root}")

    requested_epochs = [e.strip() for e in args.epochs.split(",") if e.strip()] if args.epochs else None
    epoch_dirs = collect_epoch_dirs(img_root, requested_epochs)
    if not epoch_dirs:
        raise RuntimeError(f"No epoch folders found under: {img_root}")

    device = resolve_device(args.device)
    epoch_rows: List[Dict[str, float]] = []
    json_payload = {
        "experiment_dir": experiment_dir,
        "img_root": img_root,
        "device": str(device),
        "epochs": {},
    }

    print(f"[INFO] Evaluating experiment: {experiment_dir}")
    print(f"[INFO] Using image folder: {img_root}")
    print(f"[INFO] Using device: {device}")

    for epoch_name, epoch_path in epoch_dirs:
        triplets = find_triplets(epoch_path)
        if not triplets:
            print(f"[WARN] No valid triplets found in: {epoch_path}")
            continue

        metric_records: List[Dict[str, float]] = []
        pbar = tqdm(triplets, desc=f"epoch {epoch_name}", unit="img")
        for triplet in pbar:
            try:
                result = compute_all_metrics(triplet.ir, triplet.vis, triplet.fused, device)
                metric_records.append(result)
            except Exception as exc:
                print(f"[WARN] Skip {triplet.name} in epoch {epoch_name}: {exc}")

        if not metric_records:
            print(f"[WARN] No metrics computed in epoch {epoch_name}.")
            continue

        epoch_mean = mean_metrics(metric_records)
        row = {"epoch": epoch_name, "num_samples": len(metric_records), **epoch_mean}
        epoch_rows.append(row)
        json_payload["epochs"][epoch_name] = {
            "num_samples": len(metric_records),
            "mean": epoch_mean,
        }

    if not epoch_rows:
        raise RuntimeError("No metric result generated. Please check image naming and folders.")

    overall_metrics = mean_metrics(epoch_rows)
    overall_row = {"epoch": "ALL_MEAN", "num_samples": int(sum(r["num_samples"] for r in epoch_rows)), **overall_metrics}
    epoch_rows.append(overall_row)

    json_payload["overall_mean"] = overall_metrics

    csv_path = os.path.join(experiment_dir, f"{args.output_prefix}_summary.csv")
    json_path = os.path.join(experiment_dir, f"{args.output_prefix}_summary.json")
    save_csv(csv_path, epoch_rows)
    save_json(json_path, json_payload)

    print(f"[DONE] CSV saved to: {csv_path}")
    print(f"[DONE] JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
