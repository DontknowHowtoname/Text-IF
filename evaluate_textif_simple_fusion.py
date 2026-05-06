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

from model.Text_IF_model import Text_IF as create_model

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
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


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
        # 显式释放指标计算中的 GPU 张量
        del f_tensor, ir_tensor, vi_tensor

    out = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            out[k] = float(v.item())
        else:
            out[k] = float(v)
    return out


def prepare_image_list(data_path: str, sample: int, seed: int) -> List[str]:
    ir_dir = os.path.join(data_path, "ir")
    vis_dir = os.path.join(data_path, "vis")
    if not os.path.isdir(ir_dir) or not os.path.isdir(vis_dir):
        raise FileNotFoundError("data_path must contain ir/ and vis/ folders")

    ir_images = natsorted([x for x in os.listdir(ir_dir) if x.lower().endswith(SUPPORTED_EXTS)])
    vis_images = natsorted([x for x in os.listdir(vis_dir) if x.lower().endswith(SUPPORTED_EXTS)])

    common_names = set(ir_images) & set(vis_images)
    image_list = natsorted(list(common_names))

    if sample > 0 and sample < len(image_list):
        random.seed(seed)
        image_list = random.sample(image_list, sample)
        image_list = natsorted(image_list)
        print(f"[Sample Mode] Randomly sampled {sample} images with seed={seed}")

    return image_list


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

    image_list = prepare_image_list(args.data_path, args.sample, args.seed)
    if not image_list:
        raise RuntimeError("No matching image pairs found in ir/ and vis/")

    print(f"Using device: {device}")
    print(f"Image pairs to evaluate: {len(image_list)}")

    model = load_model(args.weights_path, device)
    text = clip.tokenize([args.input_text]).to(device)

    ir_dir = os.path.join(args.data_path, "ir")
    vis_dir = os.path.join(args.data_path, "vis")

    detail_rows = []
    metric_sum = {m: 0.0 for m in METRIC_NAMES}

    # 在推理循环开始前，先释放模型加载阶段可能残留的临时显存
    clear_device_cache(device)

    for img_name in tqdm(image_list, desc="Evaluating"):
        ir_path = os.path.join(ir_dir, img_name)
        vis_path = os.path.join(vis_dir, img_name)

        ir_tensor = None
        vis_tensor = None
        fused = None
        metrics = None

        try:
            ir_tensor = to_tensor_rgb(ir_path).to(device)
            vis_tensor = to_tensor_rgb(vis_path).to(device)

            if ir_tensor.shape[-2:] != vis_tensor.shape[-2:]:
                vis_tensor = F.interpolate(vis_tensor, size=ir_tensor.shape[-2:], mode="bilinear", align_corners=True)

            with torch.no_grad():
                fused = model(vis_tensor, ir_tensor, text)

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
            # 显式释放本轮所有中间张量，再清理显存缓存
            del ir_tensor, vis_tensor, fused, metrics
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
    print(f"Sample list: {sampled_list_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Text-IF simple_fusion on IVT_test with reproducible sampling")
    parser.add_argument("--data_path", type=str, default="data/IVT_test", help="Path containing ir/ and vis/")
    parser.add_argument("--weights_path", type=str, default="experiments/TextIF_train_20260408-185710/weights/checkpoint.pth", help="Text-IF model weight path")
    parser.add_argument("--output_dir", type=str, default="results/textif_simple_eval", help="Directory to save outputs")
    parser.add_argument("--input_text", type=str, default="This is the infrared and visible light image fusion task.", help="Text prompt")
    parser.add_argument("--sample", type=int, default=20, help="Number of sampled images (0 means all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/xpu/cuda/cpu")

    args = parser.parse_args()
    main(args)
