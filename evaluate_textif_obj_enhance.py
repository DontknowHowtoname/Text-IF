"""
Evaluate Text_IF_Recon v3 (object-level enhancement) model on test datasets.

Based on evaluate_textif_full_recon_v2.py, adds:
  - Loads Text_IF_Recon_v3 with MaskGuidedAffine
  - Loads pre-computed masks from mask directory
  - Supports mask-weighted object brightness evaluation
"""
import os
import csv
import argparse
import gc
import random
import warnings
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import clip

from model.Text_IF_recon_model_3 import Text_IF_Recon_v3 as create_model
from model.Text_IF_recon_model_4 import Text_IF_Recon_v4 as create_model_v4
from model.sam_iterative_filter import IterativeSAMFilter

METRIC_DIR = os.path.join(os.path.dirname(__file__), "metric")
if METRIC_DIR not in sys.path:
    sys.path.insert(0, METRIC_DIR)

from metric.Metric_torch import (
    EN_function,
    CE_function,
    NMI_function,
    QNCIE_function,
    TE_function,
    EI_function,
    Qy_function,
    Qcb_function,
    MI_function,
    SF_function,
    SD_function,
    AG_function,
    PSNR_function,
    MSE_function,
    VIF_function,
    CC_function,
    SCD_function,
    Qabf_function,
    Nabf_function,
    SSIM_function,
    MS_SSIM_function,
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
    "Nabf", "CE", "QNCIE", "TE", "EI", "Qy", "Qcb"
]


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_device_cache(device: torch.device):
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    gc.collect()


def load_model(weights_path: str, device: torch.device):
    """Load Text_IF_Recon_v3 with key remapping for backward compatibility."""
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model(model_clip).to(device)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # Clean DataParallel prefix
    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v

    # Remap prompt_guidance_X.MLP -> prompt_guidance_X.global_affine.MLP
    remapped = {}
    for k, v in clean_state.items():
        for level in ['2', '3', '4']:
            old_prefix = f'base.prompt_guidance_{level}.'
            new_prefix = f'base.prompt_guidance_{level}.global_affine.'
            if k.startswith(old_prefix) and 'global_affine' not in k:
                k = k.replace(old_prefix, new_prefix)
                break
        remapped[k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    loaded_count = len(remapped) - len(unexpected)
    print(f"Loaded weights: {loaded_count}/{len(remapped)} keys")
    new_keys = [k for k in missing if 'mask_encode' in k or 'spatial_refine' in k]
    if new_keys:
        print(f"  MaskGuidedAffine keys (random init): {len(new_keys)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def load_model_v4(weights_path: str, device: torch.device, iterations=2):
    """Load Text_IF_Recon_v4 with key remapping and return (model, model_clip)."""
    model_clip, _ = clip.load("ViT-B/32", device=device)
    model = create_model_v4(model_clip, iterations=iterations).to(device)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace("module.", "")] = v

    remapped = {}
    for k, v in clean_state.items():
        for level in ['2', '3', '4']:
            old_prefix = f'base.prompt_guidance_{level}.'
            new_prefix = f'base.prompt_guidance_{level}.global_affine.'
            if k.startswith(old_prefix) and 'global_affine' not in k:
                k = k.replace(old_prefix, new_prefix)
                break
        remapped[k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"Loaded weights: {len(remapped) - len(unexpected)}/{len(remapped)} keys")

    model.eval()
    return model, model_clip


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
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def load_mask_tensor(mask_path: str, target_h: int, target_w: int) -> torch.Tensor:
    """Load mask and resize to target dimensions. Returns [1, 1, H, W] tensor."""
    if not os.path.exists(mask_path):
        return torch.zeros(1, 1, target_h, target_w)

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((target_w, target_h), Image.NEAREST)
    arr = np.array(mask).astype(np.float32) / 255.0
    # Binarize
    arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def save_fused_image(fused: torch.Tensor, out_path: str):
    img = Image.fromarray(tensor_to_image(fused))
    img.save(out_path)


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
        if isinstance(v, torch.Tensor):
            out[k] = float(v.item())
        else:
            out[k] = float(v)
    return out


def _resolve_ir_vis_dirs(data_path: str):
    """Auto-detect IR and VIS directories for different dataset formats."""
    candidates_ir = ["ir", "infrared"]
    candidates_vis = ["vis", "visible", "vi"]

    for ir_name in candidates_ir:
        for vis_name in candidates_vis:
            ir_dir = os.path.join(data_path, ir_name)
            vis_dir = os.path.join(data_path, vis_name)
            if os.path.isdir(ir_dir) and os.path.isdir(vis_dir):
                return ir_dir, vis_dir

    for split_name in ["test", "val"]:
        split_dir = os.path.join(data_path, split_name)
        if os.path.isdir(split_dir):
            for ir_name in candidates_ir:
                for vis_name in candidates_vis:
                    ir_dir = os.path.join(split_dir, ir_name)
                    vis_dir = os.path.join(split_dir, vis_name)
                    if os.path.isdir(ir_dir) and os.path.isdir(vis_dir):
                        return ir_dir, vis_dir

    raise FileNotFoundError(
        f"data_path must contain ir/+vis/(vi/) or infrared/+visible/ folders. "
        f"Got: {os.listdir(data_path)}"
    )


def prepare_image_list(data_path: str, sample: int, seed: int) -> List[str]:
    ir_dir, vis_dir = _resolve_ir_vis_dirs(data_path)

    def _maybe_enter_split(d):
        entries = [e for e in os.listdir(d) if os.path.isdir(os.path.join(d, e))]
        image_files = [f for f in os.listdir(d)
                       if f.lower().endswith(SUPPORTED_EXTS)]
        if not image_files and "test" in entries:
            return os.path.join(d, "test")
        return d

    ir_dir = _maybe_enter_split(ir_dir)
    vis_dir = _maybe_enter_split(vis_dir)

    ir_images = natsorted([x for x in os.listdir(ir_dir) if x.lower().endswith(SUPPORTED_EXTS)])
    vis_images = natsorted([x for x in os.listdir(vis_dir) if x.lower().endswith(SUPPORTED_EXTS)])

    ir_stems = {os.path.splitext(f)[0]: f for f in ir_images}
    vis_stems = {os.path.splitext(f)[0]: f for f in vis_images}
    common_stems = set(ir_stems.keys()) & set(vis_stems.keys())

    image_list = natsorted(list(common_stems))

    if not image_list:
        raise RuntimeError(
            f"No matching image pairs found (matched by stem).\n"
            f"  IR dir : {ir_dir}  ({len(ir_images)} images)\n"
            f"  VIS dir: {vis_dir}  ({len(vis_images)} images)"
        )

    if sample > 0 and sample < len(image_list):
        random.seed(seed)
        image_list = random.sample(image_list, sample)
        image_list = natsorted(image_list)
        print(f"[Sample Mode] Randomly sampled {sample} images with seed={seed}")

    return image_list, ir_dir, vis_dir, ir_stems, vis_stems


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    set_seed(args.seed)
    device = resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    fused_dir = os.path.join(args.output_dir, "fused")
    os.makedirs(fused_dir, exist_ok=True)

    image_list, ir_dir, vis_dir, ir_stems, vis_stems = prepare_image_list(args.data_path, args.sample, args.seed)

    print(f"Using device: {device}")
    print(f"IR dir : {ir_dir}")
    print(f"VIS dir: {vis_dir}")
    print(f"Mask dir: {args.mask_dir}")
    print(f"Image pairs to evaluate: {len(image_list)}")

    if args.iterative:
        print("Using iterative v4 model with online SAM mask generation")
        model, model_clip = load_model_v4(args.weights_path, device, iterations=args.iterations)
        _, clip_preprocess = clip.load("ViT-B/32", device=device)
        sam_filter = IterativeSAMFilter(
            sam_ckpt=args.sam_ckpt_iter,
            obj_text=args.obj_text,
            clip_model=model.base.model_clip,
            clip_preprocess=clip_preprocess,
            device=device,
            clip_threshold=args.clip_threshold
        )
    else:
        model = load_model(args.weights_path, device)
        sam_filter = None

    text = clip.tokenize([args.input_text]).to(device)

    use_mask = args.mask_dir != "" and os.path.isdir(args.mask_dir)
    if not use_mask and args.mask_dir != "":
        print(f"WARNING: mask_dir '{args.mask_dir}' not found, evaluating without masks")

    detail_rows = []
    metric_sum = {m: 0.0 for m in METRIC_NAMES}

    clear_device_cache(device)

    for img_name in tqdm(image_list, desc="Evaluating"):
        clear_device_cache(device)

        ir_path = os.path.join(ir_dir, ir_stems[img_name])
        vis_path = os.path.join(vis_dir, vis_stems[img_name])

        ir_tensor = None
        vis_tensor = None
        mask_tensor = None
        fused = None
        fused_1 = None
        recon_ir = None
        recon_vis = None
        recon_dec_ir = None
        recon_dec_vis = None
        metrics = None

        try:
            ir_tensor = to_tensor_rgb(ir_path).to(device)
            vis_tensor = to_tensor_rgb(vis_path).to(device)

            if ir_tensor.shape[-2:] != vis_tensor.shape[-2:]:
                vis_tensor = F.interpolate(vis_tensor, size=ir_tensor.shape[-2:], mode="bilinear", align_corners=True)

            # Load mask
            h, w = ir_tensor.shape[-2], ir_tensor.shape[-1]
            if use_mask:
                stem = os.path.splitext(img_name)[0]
                # Try exact name, then stem + .png
                mask_path = os.path.join(args.mask_dir, img_name)
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(args.mask_dir, stem + ".png")
                mask_tensor = load_mask_tensor(mask_path, h, w).to(device)
            else:
                mask_tensor = None

            if args.iterative:
                with torch.no_grad():
                    fused, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
                        vis_tensor, ir_tensor, text, sam_filter=sam_filter)
            else:
                with torch.no_grad():
                    fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = model(
                        vis_tensor, ir_tensor, text, mask=mask_tensor)

            fused_name = os.path.splitext(img_name)[0] + ".png"
            save_fused_image(fused, os.path.join(fused_dir, fused_name))

            metrics = evaluate_metrics(ir_path, vis_path, fused, device)
            row = {"filename": img_name}
            row.update(metrics)
            detail_rows.append(row)

            for m in METRIC_NAMES:
                metric_sum[m] += metrics[m]
        except Exception as e:
            print(f"\n[Error] Failed on {img_name}: {e}")
            continue
        finally:
            if args.iterative:
                del ir_tensor, vis_tensor, mask_tensor, fused, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, metrics
            else:
                del ir_tensor, vis_tensor, mask_tensor, fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, metrics
            clear_device_cache(device)

    details_path = os.path.join(args.output_dir, "evaluation_details.csv")
    summary_path = os.path.join(args.output_dir, "evaluation_summary.csv")
    sampled_list_path = os.path.join(args.output_dir, "sampled_filenames.txt")

    write_csv(details_path, ["filename"] + METRIC_NAMES, detail_rows)

    avg_row = {m: metric_sum[m] / len(detail_rows) for m in METRIC_NAMES}
    write_csv(summary_path, ["metric", "average"], [{"metric": k, "average": v} for k, v in avg_row.items()])

    with open(sampled_list_path, "w", encoding="utf-8") as f:
        for name in image_list:
            f.write(name + "\n")

    print("=" * 80)
    print(f"Done. Results saved to: {args.output_dir}")
    print(f"Details: {details_path}")
    print(f"Summary: {summary_path}")
    mask_info = f"Iterative SAM (obj='{args.obj_text}')" if args.iterative else ('Yes' if use_mask else 'No (evaluating without masks)')
    print(f"Mask usage: {mask_info}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Text_IF_Recon v3/v4 (object-level enhancement)")
    parser.add_argument("--data_path", type=str, default="data/IVT_test",
                        help="Path containing ir/+vis/ or infrared/+visible/")
    parser.add_argument("--weights_path", type=str,
                        default="experiments/TextIF_obj_enhance_20260601-210510/weights/checkpoint.pth",
                        help="Text_IF_Recon v3 model weight path")
    parser.add_argument("--mask_dir", type=str, default="data/IVT_test/masks",
                        help="Directory containing pre-computed masks (leave empty to skip mask)")
    parser.add_argument("--output_dir", type=str, default="results/textif_obj_enhance_eval",
                        help="Directory to save outputs")
    parser.add_argument("--input_text", type=str,
                        default="This is the infrared and visible light image fusion task.",
                        help="Text prompt")
    parser.add_argument("--sample", type=int, default=20, help="Number of sampled images (0 means all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/xpu/cuda/cpu")
    parser.add_argument("--iterative", action="store_true",
                        help="Use iterative v4 model with online SAM mask generation")
    parser.add_argument("--iterations", type=int, default=2,
                        help="Number of fusion iterations for v4 (default: 2)")
    parser.add_argument("--obj_text", type=str, default="person",
                        help="Object category for CLIP filtering (v4 iterative mode)")
    parser.add_argument("--sam_ckpt_iter", type=str,
                        default="references/segment-anything/checkpoints/sam_vit_b_01ec64.pth",
                        help="SAM ViT-B checkpoint for iterative mask generation")
    parser.add_argument("--clip_threshold", type=float, default=0.22,
                        help="CLIP similarity threshold for mask filtering")

    args = parser.parse_args()
    main(args)
