"""Offline mask generation using SAM (ViT-H) + CLIP filtering.

For each image in the dataset:
1. SAM AutomaticMaskGenerator produces all candidate masks
2. Each mask region is cropped and CLIP-encoded
3. Cosine similarity with the object text filters relevant masks
4. Filtered masks are merged into a single binary mask and saved as PNG

Supports two directory layouts:
  1) EMS_lite (multi-task, multi-split):
        {data_root}/{task}/{split}/Visible/
        masks -> {data_root}/{task}/{split}/masks/
     e.g. dataset/EMS_lite/Low_light/train/Visible/
  2) Flat test set (vis/ or ir/ at root):
        {data_root}/vis/   (or visible/)
        masks -> {data_root}/masks/
     e.g. data/IVT_test/vis/

Usage:
    # EMS_lite (auto-detected)
    python scripts/generate_masks.py --data_root dataset/EMS_lite --obj_text "person"

    # Flat test set (auto-detected)
    python scripts/generate_masks.py --data_root data/IVT_test --obj_text "person"

Output:
    masks/0001.png  (255=object, 0=background)
"""
import os
import sys
import argparse
import random
import numpy as np
from PIL import Image
import torch
import clip

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Add SAM to path
sam_path = os.path.join(os.path.dirname(__file__), '..', 'references', 'segment-anything')
sys.path.insert(0, os.path.abspath(sam_path))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def load_sam_generator(ckpt_path, device):
    """Load SAM ViT-H model and create automatic mask generator."""
    print(f"Loading SAM from: {ckpt_path}")
    sam = sam_model_registry["vit_h"](checkpoint=ckpt_path)
    sam.to(device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    return generator


def load_clip_model(device):
    """Load CLIP ViT-B/32 model."""
    print("Loading CLIP ViT-B/32...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


def get_mask_crop(image_np, mask):
    """Crop the masked region from the image for CLIP encoding."""
    # mask is a dict from SAM: {'segmentation': np.ndarray (H,W), 'bbox': [x,y,w,h]}
    bbox = mask['bbox']  # [x, y, w, h]
    x, y, w, h = bbox
    # Add padding
    pad = max(w, h) // 4
    x1 = max(0, int(x - pad))
    y1 = max(0, int(y - pad))
    x2 = min(image_np.shape[1], int(x + w + pad))
    y2 = min(image_np.shape[0], int(y + h + pad))

    crop = image_np[y1:y2, x1:x2]
    seg = mask['segmentation'][y1:y2, x1:x2]

    # Mask out background
    crop_masked = crop * seg[:, :, np.newaxis]
    return crop_masked


def filter_masks_by_clip(masks, image_np, obj_text, clip_model, clip_preprocess, device,
                         threshold=0.22):
    """Filter SAM masks by CLIP similarity with object text."""
    if len(masks) == 0:
        return []

    # Encode object text
    text_tokens = clip.tokenize([obj_text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    filtered = []
    for mask in masks:
        # Skip very small masks
        if mask['area'] < 500:
            continue

        # Crop masked region
        crop = get_mask_crop(image_np, mask)
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        # CLIP encode the crop
        crop_pil = Image.fromarray(crop)
        crop_input = clip_preprocess(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(crop_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (image_features @ text_features.T).squeeze().item()

        if similarity >= threshold:
            filtered.append(mask)

    return filtered


def merge_masks(masks, height, width):
    """Merge multiple SAM mask dicts into a single binary mask."""
    combined = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, mask['segmentation'].astype(np.uint8) * 255)
    return combined


def process_directory(generator, clip_model, clip_preprocess, image_dir, mask_out_dir,
                      obj_text, device, threshold, sample=0, seed=42):
    """Process images in a directory and save masks."""
    os.makedirs(mask_out_dir, exist_ok=True)

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", ".tif", ".TIF"]
    images = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f)[-1] in supported])

    if sample > 0 and sample < len(images):
        random.seed(seed)
        images = sorted(random.sample(images, sample))
        print(f"  [Sample Mode] Randomly sampled {sample} images with seed={seed}")

    print(f"  Processing {len(images)} images from {image_dir}")
    no_mask_count = 0

    for i, fname in enumerate(images):
        img_path = os.path.join(image_dir, fname)
        image = np.array(Image.open(img_path).convert('RGB'))

        # Generate all masks with SAM
        sam_masks = generator.generate(image)

        # Filter by CLIP
        filtered = filter_masks_by_clip(
            sam_masks, image, obj_text, clip_model, clip_preprocess, device, threshold
        )

        # Merge and save
        if len(filtered) > 0:
            merged = merge_masks(filtered, image.shape[0], image.shape[1])
        else:
            merged = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            no_mask_count += 1

        out_path = os.path.join(mask_out_dir, fname)
        Image.fromarray(merged).save(out_path)

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(images)}] {len(filtered)} object masks found"
                  f" (no_mask: {no_mask_count})")

    print(f"  Done. {no_mask_count}/{len(images)} images have no object mask.")


def detect_dirs(data_root):
    """Auto-detect directory layout and return list of (image_dir, mask_dir) pairs.

    Layout 1 - EMS_lite (multi-task, multi-split):
        data_root/{task}/{split}/Visible/
        -> data_root/{task}/{split}/masks/

    Layout 2 - Flat test set (vis/ or visible/ at root):
        data_root/vis/  (or visible/)
        -> data_root/masks/
    """
    jobs = []

    # --- Try Layout 1: EMS_lite ---
    tasks = ['Low_light', 'Over_exposure', 'IR_Low_contrast', 'IR_Noise']
    splits = [('train', 'Visible'), ('eval', 'Visible')]
    found_ems = False

    for task in tasks:
        for split, subdir in splits:
            image_dir = os.path.join(data_root, task, split, subdir)
            if os.path.isdir(image_dir):
                mask_dir = os.path.join(data_root, task, split, 'masks')
                jobs.append((image_dir, mask_dir, f"Task: {task}, Split: {split}"))
                found_ems = True

    if found_ems:
        return jobs, "EMS_lite (multi-task)"

    # --- Try Layout 2: Flat test set ---
    vis_candidates = ['vis', 'visible', 'Vis', 'Visible', 'vi']
    ir_candidates = ['ir', 'infrared', 'Ir', 'Infrared']

    for vis_name in vis_candidates:
        vis_dir = os.path.join(data_root, vis_name)
        if os.path.isdir(vis_dir):
            mask_dir = os.path.join(data_root, 'masks')
            jobs.append((vis_dir, mask_dir, f"Flat (vis={vis_name})"))
            break

    if not jobs:
        # Also try IR dir as fallback (infrared images often contain objects)
        for ir_name in ir_candidates:
            ir_dir = os.path.join(data_root, ir_name)
            if os.path.isdir(ir_dir):
                mask_dir = os.path.join(data_root, 'masks')
                jobs.append((ir_dir, mask_dir, f"Flat (ir={ir_name})"))
                break

    if jobs:
        return jobs, "Flat test set"

    raise FileNotFoundError(
        f"Cannot detect directory layout in: {data_root}\n"
        f"Expected either:\n"
        f"  EMS_lite: {{data_root}}/{{task}}/{{split}}/Visible/\n"
        f"  Flat:     {{data_root}}/vis/ or {{data_root}}/visible/"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate object masks using SAM + CLIP")
    parser.add_argument('--sam_ckpt', type=str,
                        default='references/segment-anything/checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--data_root', type=str, default='dataset/EMS_lite',
                        help='Dataset root directory (EMS_lite or flat test set)')
    parser.add_argument('--obj_text', type=str, default='person',
                        help='Object category text for CLIP filtering')
    parser.add_argument('--clip_threshold', type=float, default=0.22,
                        help='CLIP similarity threshold for mask filtering')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sample', type=int, default=0,
                        help='Number of images to sample per directory (0 = all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Detect directory layout
    jobs, layout_type = detect_dirs(args.data_root)
    print(f"Detected layout: {layout_type}")
    print(f"Found {len(jobs)} directory pair(s) to process")

    # Load models
    generator = load_sam_generator(args.sam_ckpt, device)
    clip_model, clip_preprocess = load_clip_model(device)

    for image_dir, mask_dir, desc in jobs:
        print(f"\n=== {desc} ===")
        print(f"  Images: {image_dir}")
        print(f"  Masks:  {mask_dir}")
        process_directory(
            generator, clip_model, clip_preprocess,
            image_dir, mask_dir, args.obj_text, device, args.clip_threshold,
            sample=args.sample, seed=args.seed
        )

    print("\nMask generation complete!")


if __name__ == '__main__':
    main()
