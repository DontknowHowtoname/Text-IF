"""
Batch-evaluate all hyperparameter search trials under a single experiment directory.

Scans each trial sub-directory (e.g. lr0.0001_rw0.03/), loads its best checkpoint,
runs full metric evaluation on the given dataset, and produces a comparison table.

Usage:
    python eval_hyperparam_trials.py \
        --exp_dir experiments/hyperparam_search/20260519-220812 \
        --data_path data/IVT_test \
        --sample 20
"""
import os
import csv
import argparse
import gc
import json
import warnings
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import clip

from model.Text_IF_recon_model_2 import Text_IF_Recon as create_model

METRIC_DIR = os.path.join(os.path.dirname(__file__), "metric")
if METRIC_DIR not in sys.path:
    sys.path.insert(0, METRIC_DIR)

from metric.Metric_torch import (
    EN_function, CE_function, NMI_function, QNCIE_function,
    TE_function, EI_function, Qy_function, Qcb_function,
    MI_function, SF_function, SD_function, AG_function,
    PSNR_function, MSE_function, VIF_function, CC_function,
    SCD_function, Qabf_function, Nabf_function,
    SSIM_function, MS_SSIM_function,
)

try:
    from natsort import natsorted
except Exception:
    def natsorted(items):
        return sorted(items)

warnings.filterwarnings("ignore")

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
METRIC_NAMES = [
    "EN", "MI", "NMI", "SF", "AG", "SD", "CC", "SCD",
    "PSNR", "MSE", "VIF", "SSIM", "MS_SSIM", "Qabf",
    "Nabf", "CE", "QNCIE", "TE", "EI", "Qy", "Qcb",
]


# ── helpers (shared with evaluate_textif_full_recon_v2) ──────────────────

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


def clear_device_cache(device: torch.device):
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    gc.collect()


def load_model(weights_path: str, device: torch.device):
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model(model_clip).to(device)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v

    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model


def resize_to_multiple_of_16(img: Image.Image) -> Image.Image:
    w, h = img.size
    new_w = max(16, (w // 16) * 16)
    new_h = max(16, (h // 16) * 16)
    if new_w == w and new_h == h:
        return img
    return img.resize((new_w, new_h), Image.BILINEAR)


def to_tensor_rgb(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = resize_to_multiple_of_16(img)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def evaluate_metrics(ir_path: str, vis_path: str, fused: torch.Tensor, device: torch.device) -> Dict[str, float]:
    ir_img = Image.open(ir_path).convert("L")
    vi_img = Image.open(vis_path).convert("L")

    fused_gray = fused.mean(dim=1, keepdim=True)
    f_np = (fused_gray.squeeze(0).squeeze(0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    f_img = Image.fromarray(f_np).convert("L")

    if ir_img.size != vi_img.size:
        vi_img = vi_img.resize(ir_img.size, Image.BILINEAR)
    if f_img.size != ir_img.size:
        f_img = f_img.resize(ir_img.size, Image.BILINEAR)

    f_tensor = torch.tensor(np.array(f_img)).float().to(device)
    ir_tensor = torch.tensor(np.array(ir_img)).float().to(device)
    vi_tensor = torch.tensor(np.array(vi_img)).float().to(device)

    f_int = np.array(f_img).astype(np.int32)
    ir_int = np.array(ir_img).astype(np.int32)
    vi_int = np.array(vi_img).astype(np.int32)

    f_float = np.array(f_img).astype(np.float32)
    ir_float = np.array(ir_img).astype(np.float32)
    vi_float = np.array(vi_img).astype(np.float32)

    try:
        metrics = {
            "EN": EN_function(f_tensor),
            "MI": MI_function(ir_int, vi_int, f_int, gray_level=256),
            "NMI": NMI_function(ir_int, vi_int, f_int, gray_level=256),
            "SF": SF_function(f_tensor),
            "AG": AG_function(f_tensor),
            "SD": SD_function(f_tensor),
            "CC": CC_function(ir_tensor, vi_tensor, f_tensor),
            "SCD": SCD_function(ir_tensor, vi_tensor, f_tensor),
            "PSNR": PSNR_function(ir_tensor, vi_tensor, f_tensor),
            "MSE": MSE_function(ir_tensor, vi_tensor, f_tensor),
            "VIF": VIF_function(ir_tensor, vi_tensor, f_tensor),
            "SSIM": SSIM_function(ir_float, vi_float, f_float),
            "MS_SSIM": MS_SSIM_function(ir_float, vi_float, f_float),
            "Qabf": Qabf_function(ir_float, vi_float, f_float),
            "Nabf": Nabf_function(ir_tensor, vi_tensor, f_tensor),
            "CE": CE_function(ir_tensor, vi_tensor, f_tensor),
            "QNCIE": QNCIE_function(ir_tensor, vi_tensor, f_tensor),
            "TE": TE_function(ir_tensor, vi_tensor, f_tensor),
            "EI": EI_function(f_tensor),
            "Qy": Qy_function(ir_tensor, vi_tensor, f_tensor),
            "Qcb": Qcb_function(ir_tensor, vi_tensor, f_tensor),
        }
    finally:
        del f_tensor, ir_tensor, vi_tensor

    out = {}
    for k, v in metrics.items():
        out[k] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
    return out


def _resolve_ir_vis_dirs(data_path: str) -> Tuple[str, str]:
    for ir_name, vis_name in [("ir", "vis"), ("infrared", "visible")]:
        ir_dir = os.path.join(data_path, ir_name)
        vis_dir = os.path.join(data_path, vis_name)
        if os.path.isdir(ir_dir) and os.path.isdir(vis_dir):
            return ir_dir, vis_dir
    raise FileNotFoundError(
        f"data_path must contain ir/+vis/ or infrared/+visible/. "
        f"Got: {os.listdir(data_path)}"
    )


def prepare_image_list(data_path: str, sample: int, seed: int) -> Tuple[List[str], str, str]:
    import random
    ir_dir, vis_dir = _resolve_ir_vis_dirs(data_path)

    def _maybe_enter_split(d):
        entries = [e for e in os.listdir(d) if os.path.isdir(os.path.join(d, e))]
        image_files = [f for f in os.listdir(d) if f.lower().endswith(SUPPORTED_EXTS)]
        if not image_files and "test" in entries:
            return os.path.join(d, "test")
        return d

    ir_dir = _maybe_enter_split(ir_dir)
    vis_dir = _maybe_enter_split(vis_dir)

    ir_images = natsorted([x for x in os.listdir(ir_dir) if x.lower().endswith(SUPPORTED_EXTS)])
    vis_images = natsorted([x for x in os.listdir(vis_dir) if x.lower().endswith(SUPPORTED_EXTS)])
    common_names = set(ir_images) & set(vis_images)
    image_list = natsorted(list(common_names))

    if not image_list:
        raise RuntimeError(f"No matching pairs in {ir_dir} / {vis_dir}")

    if sample > 0 and sample < len(image_list):
        random.seed(seed)
        image_list = natsorted(random.sample(image_list, sample))

    return image_list, ir_dir, vis_dir


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── discover trials ───────────────────────────────────────────────────────

def discover_trials(exp_dir: str) -> List[Dict]:
    """Scan exp_dir for trial sub-directories with weights/checkpoint.pth."""
    trials = []
    for name in natsorted(os.listdir(exp_dir)):
        trial_dir = os.path.join(exp_dir, name)
        if not os.path.isdir(trial_dir):
            continue
        ckpt = os.path.join(trial_dir, "weights", "checkpoint.pth")
        if not os.path.isfile(ckpt):
            continue

        # parse lr & recon_weight from directory name like "lr0.0001_rw0.03"
        lr_str, rw_str = None, None
        parts = name.split("_")
        for p in parts:
            if p.startswith("lr"):
                lr_str = p[2:]
            elif p.startswith("rw"):
                rw_str = p[2:]
        trials.append({
            "name": name,
            "dir": trial_dir,
            "weights": ckpt,
            "lr": float(lr_str) if lr_str else None,
            "recon_weight": float(rw_str) if rw_str else None,
        })
    return trials


# ── evaluate a single trial ───────────────────────────────────────────────

def evaluate_trial(
    trial: Dict,
    image_list: List[str],
    ir_dir: str,
    vis_dir: str,
    text: torch.Tensor,
    device: torch.device,
    output_dir: str,
) -> Dict[str, float]:
    """Load checkpoint, run fusion, compute metrics. Returns averaged metrics."""
    trial_out = os.path.join(output_dir, trial["name"])
    fused_dir = os.path.join(trial_out, "fused")
    os.makedirs(fused_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Trial: {trial['name']}  (lr={trial['lr']}, rw={trial['recon_weight']})")
    print(f"Checkpoint: {trial['weights']}")
    print(f"{'='*70}")

    model = load_model(trial["weights"], device)

    metric_sum = {m: 0.0 for m in METRIC_NAMES}
    n_ok = 0

    for img_name in tqdm(image_list, desc=f"  {trial['name']}"):
        clear_device_cache(device)

        ir_tensor = vis_tensor = fused = None
        recon_ir = recon_vis = recon_dec_ir = recon_dec_vis = None
        metrics = None

        try:
            ir_tensor = to_tensor_rgb(os.path.join(ir_dir, img_name)).to(device)
            vis_tensor = to_tensor_rgb(os.path.join(vis_dir, img_name)).to(device)

            if ir_tensor.shape[-2:] != vis_tensor.shape[-2:]:
                vis_tensor = F.interpolate(vis_tensor, size=ir_tensor.shape[-2:],
                                           mode="bilinear", align_corners=True)

            with torch.no_grad():
                fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = \
                    model(vis_tensor, ir_tensor, text)

            # save fused image
            fused_name = os.path.splitext(img_name)[0] + ".png"
            Image.fromarray(tensor_to_image(fused)).save(os.path.join(fused_dir, fused_name))

            metrics = evaluate_metrics(
                os.path.join(ir_dir, img_name),
                os.path.join(vis_dir, img_name),
                fused, device,
            )

            for m in METRIC_NAMES:
                metric_sum[m] += metrics[m]
            n_ok += 1

        except Exception as e:
            print(f"\n  [Error] {img_name}: {e}")
        finally:
            del ir_tensor, vis_tensor, fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, metrics
            clear_device_cache(device)

    if n_ok == 0:
        print(f"  [WARN] No images successfully evaluated for trial {trial['name']}")
        return {m: float("nan") for m in METRIC_NAMES}

    avg = {m: metric_sum[m] / n_ok for m in METRIC_NAMES}

    # save per-trial summary
    write_csv(
        os.path.join(trial_out, "summary.csv"),
        ["metric", "average"],
        [{"metric": k, "average": v} for k, v in avg.items()],
    )

    print(f"  Evaluated {n_ok}/{len(image_list)} images")
    for m in ["EN", "MI", "AG", "SD", "PSNR", "SSIM", "VIF", "SCD", "Qabf"]:
        print(f"    {m:>8s}: {avg[m]:.4f}")

    del model
    clear_device_cache(device)

    return avg


# ── main ──────────────────────────────────────────────────────────────────

def main(args):
    device = resolve_device(args.device)

    # discover trials
    trials = discover_trials(args.exp_dir)
    if not trials:
        raise RuntimeError(f"No trials with weights/checkpoint.pth found in {args.exp_dir}")
    print(f"Found {len(trials)} trials in {args.exp_dir}")

    # prepare dataset (shared across all trials)
    image_list, ir_dir, vis_dir = prepare_image_list(args.data_path, args.sample, args.seed)
    print(f"Dataset: {args.data_path}")
    print(f"  IR dir : {ir_dir}")
    print(f"  VIS dir: {vis_dir}")
    print(f"  Images : {len(image_list)}")

    text = clip.tokenize([args.input_text]).to(device)

    output_dir = os.path.join(args.exp_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # run evaluation for each trial
    # --- RESUME: skip already-evaluated trials (first 5) ---
    start_idx = 5  # resume from trial 6 (0-indexed)
    # load previously completed results
    prev_json = os.path.join(output_dir, "comparison.json")
    all_results = []
    if os.path.isfile(prev_json) and start_idx > 0:
        with open(prev_json, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} previous results from {prev_json}")
    else:
        print(f"[WARN] No previous results found, starting fresh from trial {start_idx+1}")

    for i in range(start_idx, len(trials)):
        trial = trials[i]
        print(f"\n[{i+1}/{len(trials)}]", end="")
        avg = evaluate_trial(trial, image_list, ir_dir, vis_dir, text, device, output_dir)
        row = {"trial": trial["name"], "lr": trial["lr"], "recon_weight": trial["recon_weight"]}
        row.update(avg)
        all_results.append(row)

    # ── write comparison table ────────────────────────────────────────────
    comparison_path = os.path.join(output_dir, "comparison.csv")
    fieldnames = ["trial", "lr", "recon_weight"] + METRIC_NAMES
    write_csv(comparison_path, fieldnames, all_results)

    # ── rank by key metrics & print summary ───────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (all trials)")
    print("=" * 80)

    header = f"{'Trial':<22s} {'LR':<10s} {'RW':<6s}"
    for m in ["EN", "MI", "AG", "SD", "PSNR", "SSIM", "VIF", "SCD", "Qabf"]:
        header += f" {m:>8s}"
    print(header)
    print("-" * len(header))

    for row in all_results:
        line = f"{row['trial']:<22s} {str(row['lr']):<10s} {row['recon_weight']:<6.2f}"
        for m in ["EN", "MI", "AG", "SD", "PSNR", "SSIM", "VIF", "SCD", "Qabf"]:
            line += f" {row[m]:>8.4f}"
        print(line)

    # highlight best for key metrics
    print()
    for m in ["EN", "PSNR", "SSIM", "VIF", "SCD"]:
        best = max(all_results, key=lambda r: r[m])
        print(f"  Best {m:>8s}: {best[m]:.4f}  ({best['trial']})")

    # save comparison JSON for easy downstream use
    json_path = os.path.join(output_dir, "comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir}")
    print(f"  CSV: {comparison_path}")
    print(f"  JSON: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-evaluate all hyperparameter search trials"
    )
    parser.add_argument("--exp_dir", type=str,
                        default="experiments/hyperparam_search/20260519-220812",
                        help="Experiment directory containing trial sub-dirs")
    parser.add_argument("--data_path", type=str, default="data/IVT_test",
                        help="Dataset path (ir/+vis/ or infrared/+visible/)")
    parser.add_argument("--input_text", type=str,
                        default="This is the infrared and visible light image fusion task.",
                        help="Text prompt")
    parser.add_argument("--sample", type=int, default=20,
                        help="Number of sampled images (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    main(args)
