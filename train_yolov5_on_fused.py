"""
Generate fused images from LLVIP train set using Text_IF_Recon v2 model,
convert VOC annotations to YOLO format, then fine-tune YOLOv5.

Usage:
    # Full pipeline: fusion generation + YOLOv5 fine-tuning
    python train_yolov5_on_fused.py \
        --fusion_weights experiments/TextIF_full_recon_2_20260506-213402/weights/checkpoint.pth \
        --llvip_root D:/StudyFiles/MachineLearning/datasets/LLVIP

    # Skip fusion (already generated), only fine-tune
    python train_yolov5_on_fused.py --skip_fusion \
        --output_dir results/yolov5_finetune_llvip

    # Custom settings
    python train_yolov5_on_fused.py \
        --fusion_weights experiments/.../checkpoint.pth \
        --llvip_root D:/StudyFiles/MachineLearning/datasets/LLVIP \
        --yolo_epochs 100 --yolo_batch_size 8 --val_split 0.15
"""

import os
import sys
import gc
import argparse
import random
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import clip

from model.Text_IF_recon_model_2 import Text_IF_Recon as create_model


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def resolve_device(device_name: str) -> torch.device:
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


# ---------------------------------------------------------------------------
# Step 1: Load fusion model
# ---------------------------------------------------------------------------

def load_fusion_model(weights_path: str, device: torch.device):
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


# ---------------------------------------------------------------------------
# Step 2: Generate fused images from LLVIP train set
# ---------------------------------------------------------------------------

def generate_fused_images(args):
    """Run fusion model on LLVIP train images and save results."""
    device = resolve_device(args.device)

    ir_dir = os.path.join(args.llvip_root, "infrared", "train")
    vis_dir = os.path.join(args.llvip_root, "visible", "train")

    if not os.path.isdir(ir_dir) or not os.path.isdir(vis_dir):
        raise FileNotFoundError(
            f"Cannot find infrared/train or visible/train under {args.llvip_root}"
        )

    # Discover image pairs (match by stem)
    ir_files = {os.path.splitext(f)[0]: f for f in os.listdir(ir_dir)
                if f.lower().endswith(SUPPORTED_EXTS)}
    vis_files = {os.path.splitext(f)[0]: f for f in os.listdir(vis_dir)
                 if f.lower().endswith(SUPPORTED_EXTS)}
    common_stems = sorted(set(ir_files.keys()) & set(vis_files.keys()))

    if not common_stems:
        raise RuntimeError("No matching IR/VIS image pairs found in train set")

    print(f"[Fusion] Found {len(common_stems)} image pairs in LLVIP train set")

    # Train/val split
    random.seed(args.seed)
    random.shuffle(common_stems)
    n_val = max(1, int(len(common_stems) * args.val_split))
    val_stems = set(common_stems[:n_val])
    train_stems = [s for s in common_stems if s not in val_stems]

    print(f"[Fusion] Train: {len(train_stems)}, Val: {len(val_stems)}")

    # Create output dirs
    fused_train_dir = os.path.join(args.output_dir, "images", "train")
    fused_val_dir = os.path.join(args.output_dir, "images", "val")
    os.makedirs(fused_train_dir, exist_ok=True)
    os.makedirs(fused_val_dir, exist_ok=True)

    # Load model
    print(f"[Fusion] Loading model: {args.fusion_weights}")
    model = load_fusion_model(args.fusion_weights, device)
    text = clip.tokenize([args.text_prompt]).to(device)

    clear_device_cache(device)

    # Save split info
    split_info_path = os.path.join(args.output_dir, "split_stems.txt")
    with open(split_info_path, "w", encoding="utf-8") as f:
        f.write("=== TRAIN ===\n")
        for s in sorted(train_stems):
            f.write(s + "\n")
        f.write("=== VAL ===\n")
        for s in sorted(val_stems):
            f.write(s + "\n")

    # Inference loop
    n_success = 0
    n_fail = 0
    for stem in tqdm(common_stems, desc="[Fusion] Generating"):
        out_dir = fused_val_dir if stem in val_stems else fused_train_dir
        out_path = os.path.join(out_dir, stem + ".png")

        # Skip already generated
        if os.path.exists(out_path):
            n_success += 1
            continue

        ir_tensor = None
        vis_tensor = None
        try:
            ir_tensor = to_tensor_rgb(os.path.join(ir_dir, ir_files[stem])).to(device)
            vis_tensor = to_tensor_rgb(os.path.join(vis_dir, vis_files[stem])).to(device)

            if ir_tensor.shape[-2:] != vis_tensor.shape[-2:]:
                vis_tensor = F.interpolate(vis_tensor, size=ir_tensor.shape[-2:],
                                           mode="bilinear", align_corners=True)

            with torch.no_grad():
                fused, _, _, _, _ = model(vis_tensor, ir_tensor, text)

            save_fused_image(fused, out_path)
            n_success += 1
        except Exception as e:
            print(f"\n[Error] Failed on {stem}: {e}")
            n_fail += 1
        finally:
            del ir_tensor
            clear_device_cache(device)

    print(f"[Fusion] Done: {n_success} succeeded, {n_fail} failed")
    return train_stems, val_stems


# ---------------------------------------------------------------------------
# Step 3: VOC XML -> YOLO TXT label conversion
# ---------------------------------------------------------------------------

def parse_voc_annotation(xml_path):
    """Parse a VOC XML annotation. Returns (img_width, img_height, list of objects)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        objects.append({"name": name, "bbox": [xmin, ymin, xmax, ymax]})
    return img_w, img_h, objects


def voc_to_yolo(box, img_w, img_h):
    """Convert VOC [xmin, ymin, xmax, ymax] to YOLO [x_center, y_center, w, h] (normalized)."""
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return [x_center, y_center, w, h]


def convert_annotations(args, train_stems, val_stems):
    """Convert VOC XML annotations to YOLO TXT format."""
    ann_dir = os.path.join(args.llvip_root, "Annotations")

    # Class mapping (LLVIP only has 'person')
    class_to_id = {"person": 0}

    # Read the fused image dimensions for each stem
    fused_train_dir = os.path.join(args.output_dir, "images", "train")
    fused_val_dir = os.path.join(args.output_dir, "images", "val")

    label_train_dir = os.path.join(args.output_dir, "labels", "train")
    label_val_dir = os.path.join(args.output_dir, "labels", "val")
    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_val_dir, exist_ok=True)

    all_stems = {"train": train_stems, "val": list(val_stems)}
    img_dirs = {"train": fused_train_dir, "val": fused_val_dir}
    lbl_dirs = {"train": label_train_dir, "val": label_val_dir}

    total_labels = 0
    total_empty = 0

    for split, stems in all_stems.items():
        n_converted = 0
        for stem in tqdm(stems, desc=f"[Labels] Converting {split}"):
            xml_path = os.path.join(ann_dir, stem + ".xml")
            label_path = os.path.join(lbl_dirs[split], stem + ".txt")

            if not os.path.exists(xml_path):
                # No annotation -> write empty label file
                with open(label_path, "w") as f:
                    pass
                total_empty += 1
                continue

            # Get actual fused image dimensions (may differ from original after resize)
            img_dir = img_dirs[split]
            # Find the actual image file
            img_path = None
            for ext in SUPPORTED_EXTS:
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                # Image not generated, skip
                continue

            with Image.open(img_path) as img:
                fused_w, fused_h = img.size

            # Parse original annotation
            orig_w, orig_h, objects = parse_voc_annotation(xml_path)

            # Scale bboxes from original image size to fused image size
            scale_x = fused_w / orig_w
            scale_y = fused_h / orig_h

            lines = []
            for obj in objects:
                cls_name = obj["name"]
                if cls_name not in class_to_id:
                    continue
                cls_id = class_to_id[cls_name]
                xmin, ymin, xmax, ymax = obj["bbox"]
                # Scale to fused image coordinates
                scaled_box = [xmin * scale_x, ymin * scale_y,
                              xmax * scale_x, ymax * scale_y]
                yolo_box = voc_to_yolo(scaled_box, fused_w, fused_h)
                # Clamp to [0, 1]
                yolo_box = [max(0.0, min(1.0, v)) for v in yolo_box]
                lines.append(f"{cls_id} {' '.join(f'{v:.6f}' for v in yolo_box)}")

            with open(label_path, "w") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

            total_labels += len(lines)
            n_converted += 1

        print(f"[Labels] {split}: converted {n_converted} annotations")

    print(f"[Labels] Total: {total_labels} bounding boxes, {total_empty} empty annotations")


# ---------------------------------------------------------------------------
# Step 4: Create YOLOv5 dataset YAML and launch training
# ---------------------------------------------------------------------------

def create_dataset_yaml(args):
    """Create YOLOv5 dataset configuration YAML."""
    yaml_path = os.path.join(args.output_dir, "dataset.yaml")
    # Use absolute paths to avoid path resolution issues
    images_dir = os.path.abspath(os.path.join(args.output_dir, "images"))

    content = f"""# LLVIP fused image dataset for YOLOv5 fine-tuning
path: {images_dir}  # dataset root dir
train: train  # train images (relative to 'path')
val: val      # val images (relative to 'path')

# Classes
names:
  0: person
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[Dataset] YAML saved to: {yaml_path}")
    return yaml_path


def run_yolov5_training(args, yaml_path):
    """Launch YOLOv5 training via subprocess."""
    yolo_train = os.path.join(
        os.path.dirname(__file__), "references", "yolov5-master", "train.py"
    )

    if not os.path.exists(yolo_train):
        raise FileNotFoundError(f"YOLOv5 train.py not found: {yolo_train}")

    # Resolve device for YOLOv5
    device = resolve_device(args.device)
    yolo_device = ""
    if device.type == "cuda":
        yolo_device = "0"
    elif device.type == "xpu":
        yolo_device = "xpu"
    else:
        yolo_device = "cpu"

    yolo_project = os.path.join(args.output_dir, "yolov5_runs")
    os.makedirs(yolo_project, exist_ok=True)

    cmd = [
        sys.executable, yolo_train,
        "--data", yaml_path,
        "--weights", args.yolo_weights,
        "--epochs", str(args.yolo_epochs),
        "--batch-size", str(args.yolo_batch_size),
        "--imgsz", str(args.yolo_img_size),
        "--device", yolo_device,
        "--project", yolo_project,
        "--name", args.yolo_exp_name,
        "--exist-ok",
        "--seed", str(args.seed),
        "--cache", "ram",
    ]

    # Add optional freeze layers (freeze backbone by default for fine-tuning)
    if args.yolo_freeze:
        cmd.extend(["--freeze"] + [str(x) for x in args.yolo_freeze])

    print(f"\n[YOLOv5] Starting fine-tuning ...")
    print(f"[YOLOv5] Command: {' '.join(cmd)}")
    print("=" * 80)

    result = subprocess.run(cmd, cwd=os.path.dirname(yolo_train))

    if result.returncode != 0:
        print(f"\n[YOLOv5] Training exited with code {result.returncode}")
    else:
        print(f"\n[YOLOv5] Training completed successfully!")
        print(f"[YOLOv5] Results saved to: {yolo_project}/{args.yolo_exp_name}")

    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate fused images from LLVIP and fine-tune YOLOv5"
    )

    # Fusion model settings
    parser.add_argument("--fusion_weights", type=str,
                        default="experiments/TextIF_full_recon_2_20260506-213402/weights/checkpoint.pth",
                        help="Path to Text_IF_Recon v2 model weights")
    parser.add_argument("--text_prompt", type=str,
                        default="This is the infrared and visible light image fusion task.",
                        help="Text prompt for fusion model")

    # Dataset settings
    parser.add_argument("--llvip_root", type=str,
                        default="D:/StudyFiles/MachineLearning/datasets/LLVIP",
                        help="LLVIP dataset root directory")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")

    # YOLOv5 settings
    parser.add_argument("--yolo_weights", type=str,
                        default="references/yolov5-master/models/yolov5m.pt",
                        help="YOLOv5 initial weights for fine-tuning")
    parser.add_argument("--yolo_epochs", type=int, default=50,
                        help="YOLOv5 training epochs")
    parser.add_argument("--yolo_batch_size", type=int, default=16,
                        help="YOLOv5 batch size")
    parser.add_argument("--yolo_img_size", type=int, default=640,
                        help="YOLOv5 training image size")
    parser.add_argument("--yolo_exp_name", type=str, default="llvip_fused",
                        help="YOLOv5 experiment name")
    parser.add_argument("--yolo_freeze", type=int, nargs="*", default=[0],
                        help="Freeze YOLOv5 layers (0=backbone, default: [0])")

    # General settings
    parser.add_argument("--output_dir", type=str,
                        default="results/yolov5_finetune_llvip",
                        help="Output root directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto/cuda/xpu/cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip_fusion", action="store_true",
                        help="Skip fusion generation (use existing fused images)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip YOLOv5 training (only generate fusion + labels)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("LLVIP Fusion + YOLOv5 Fine-tuning Pipeline")
    print("=" * 80)
    print(f"  LLVIP root:       {args.llvip_root}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Device:           {args.device}")
    print(f"  Fusion weights:   {args.fusion_weights}")
    print(f"  YOLOv5 weights:   {args.yolo_weights}")
    print(f"  Val split:        {args.val_split}")
    print("=" * 80)

    train_stems = None
    val_stems = None

    # Step 1: Generate fused images
    if not args.skip_fusion:
        print("\n>>> Step 1/4: Generating fused images ...")
        train_stems, val_stems = generate_fused_images(args)
    else:
        print("\n>>> Step 1/4: Skipping fusion (using existing images)")
        # Reconstruct split from saved file
        split_path = os.path.join(args.output_dir, "split_stems.txt")
        if os.path.exists(split_path):
            train_stems, val_stems = [], []
            current = None
            with open(split_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line == "=== TRAIN ===":
                        current = train_stems
                    elif line == "=== VAL ===":
                        current = val_stems
                    elif line and current is not None:
                        current.append(line)
            print(f"  Loaded split: Train={len(train_stems)}, Val={len(val_stems)}")
        else:
            # Fallback: scan directories
            fused_train_dir = os.path.join(args.output_dir, "images", "train")
            fused_val_dir = os.path.join(args.output_dir, "images", "val")
            train_stems = [os.path.splitext(f)[0] for f in os.listdir(fused_train_dir)
                           if f.lower().endswith(SUPPORTED_EXTS)] if os.path.isdir(fused_train_dir) else []
            val_stems = [os.path.splitext(f)[0] for f in os.listdir(fused_val_dir)
                         if f.lower().endswith(SUPPORTED_EXTS)] if os.path.isdir(fused_val_dir) else []
            print(f"  Scanned dirs: Train={len(train_stems)}, Val={len(val_stems)}")

    # Step 2: Convert annotations
    print("\n>>> Step 2/4: Converting VOC annotations to YOLO format ...")
    convert_annotations(args, train_stems, val_stems)

    # Step 3: Create dataset YAML
    print("\n>>> Step 3/4: Creating YOLOv5 dataset config ...")
    yaml_path = create_dataset_yaml(args)

    # Step 4: Launch YOLOv5 training
    if not args.skip_training:
        print("\n>>> Step 4/4: Launching YOLOv5 fine-tuning ...")
        ret = run_yolov5_training(args, yaml_path)
        if ret != 0:
            print(f"\n[Warning] YOLOv5 training exited with code {ret}")
    else:
        print("\n>>> Step 4/4: Skipping YOLOv5 training (--skip_training)")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print(f"  Fused images:  {os.path.join(args.output_dir, 'images')}")
    print(f"  Labels:        {os.path.join(args.output_dir, 'labels')}")
    print(f"  Dataset YAML:  {yaml_path}")
    if not args.skip_training:
        print(f"  YOLOv5 runs:   {os.path.join(args.output_dir, 'yolov5_runs', args.yolo_exp_name)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
