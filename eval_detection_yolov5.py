"""
Object detection evaluation on fused images using YOLOv5.
Computes mAP@0.5, mAP@0.75, and mAP@0.5:0.95 against LLVIP ground truth annotations.

Usage:
    python eval_detection_yolov5.py
    python eval_detection_yolov5.py --fused_dir results/textif_full_recon_v2_eval_LLVIP/fused
    python eval_detection_yolov5.py --weights references/yolov5-master/models/yolov5m.pt
"""

import os
import csv
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Ground-truth parsing (VOC XML -> per-image bbox list)
# ---------------------------------------------------------------------------

def parse_voc_annotation(xml_path):
    """Parse a single VOC XML annotation file. Returns list of (xmin, ymin, xmax, ymax)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        difficult = obj.find("difficult")
        is_difficult = int(difficult.text) if difficult is not None else 0
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        boxes.append({
            "name": name,
            "bbox": [xmin, ymin, xmax, ymax],
            "difficult": is_difficult,
        })
    return boxes


def load_ground_truths(ann_dir, image_stems):
    """Load all ground truth boxes for the given image stems.

    Returns:
        gt_all: dict  stem -> list of [xmin, ymin, xmax, ymax]
        gt_difficult: dict  stem -> list of bool  (parallel to gt_all)
        class_names: set of class names found
    """
    gt_all = {}
    gt_difficult = {}
    class_names = set()
    for stem in image_stems:
        xml_path = os.path.join(ann_dir, stem + ".xml")
        if not os.path.exists(xml_path):
            gt_all[stem] = []
            gt_difficult[stem] = []
            continue
        objs = parse_voc_annotation(xml_path)
        boxes, diffs = [], []
        for o in objs:
            boxes.append(o["bbox"])
            diffs.append(o["difficult"])
            class_names.add(o["name"])
        gt_all[stem] = boxes
        gt_difficult[stem] = diffs
    return gt_all, gt_difficult, class_names


# ---------------------------------------------------------------------------
# IoU & mAP computation
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """IoU between two [xmin, ymin, xmax, ymax] boxes."""
    ixmin = max(box_a[0], box_b[0])
    iymin = max(box_a[1], box_b[1])
    ixmax = min(box_a[2], box_b[2])
    iymax = min(box_a[3], box_b[3])
    iw = max(0.0, ixmax - ixmin)
    ih = max(0.0, iymax - iymin)
    inter = iw * ih
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / max(area_a + area_b - inter, 1e-7)


def compute_ap(recall, precision):
    """Compute AP using all-point interpolation (COCO-style)."""
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing (right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Area under curve
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def evaluate_single_class(preds_by_image, gt_by_image, gt_difficult_by_image, iou_threshold):
    """Evaluate predictions for a single class at a given IoU threshold.

    Args:
        preds_by_image: dict  stem -> list of {"bbox": [...], "score": float}
        gt_by_image:    dict  stem -> list of [xmin, ymin, xmax, ymax]
        gt_difficult_by_image: dict  stem -> list of bool
        iou_threshold:  float

    Returns:
        ap: float (average precision)
        precision_arr, recall_arr: numpy arrays for the PR curve
    """
    # Collect all predictions with image id, sorted by confidence descending
    all_preds = []
    n_gt = 0  # total non-difficult ground truths
    for stem in gt_by_image:
        gts = gt_by_image[stem]
        diffs = gt_difficult_by_image.get(stem, [False] * len(gts))
        n_gt += sum(1 for d in diffs if not d)

        for pred in preds_by_image.get(stem, []):
            all_preds.append((pred["score"], stem, pred["bbox"]))

    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    all_preds.sort(key=lambda x: -x[0])

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Track which GT boxes have been matched per image
    matched = defaultdict(set)

    for idx, (score, stem, pred_box) in enumerate(all_preds):
        gts = gt_by_image.get(stem, [])
        diffs = gt_difficult_by_image.get(stem, [False] * len(gts))
        best_iou = 0.0
        best_gt_idx = -1

        for gi, gt_box in enumerate(gts):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            if best_gt_idx not in matched[stem]:
                if not diffs[best_gt_idx]:  # non-difficult
                    tp[idx] = 1
                    matched[stem].add(best_gt_idx)
                # difficult GT matched -> ignore (neither TP nor FP)
            else:
                fp[idx] = 1  # already matched -> duplicate detection
        else:
            fp[idx] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp)

    ap = compute_ap(recall, precision)
    return ap, precision, recall


def compute_map(preds_by_image, gt_by_image, gt_difficult_by_image, iou_thresholds):
    """Compute mAP over the given IoU thresholds (single-class)."""
    aps = []
    for iou_t in iou_thresholds:
        ap, _, _ = evaluate_single_class(
            preds_by_image, gt_by_image, gt_difficult_by_image, iou_t
        )
        aps.append(ap)
    return aps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 object detection evaluation on fused images")
    parser.add_argument("--fused_dir", type=str,
                        default="results/textif_full_recon_v2_eval_LLVIP/fused",
                        help="Directory of fused images")
    parser.add_argument("--ann_dir", type=str,
                        default="D:/StudyFiles/MachineLearning/datasets/LLVIP/Annotations",
                        help="LLVIP VOC annotation directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: fused_dir parent + _detection)")
    parser.add_argument("--weights", type=str,
                        default="results/yolov5_finetune_llvip_2/yolov5_runs/llvip_fused/weights/best.pt",
                        help="Path to YOLOv5 weights (.pt file)")
    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--iou_thres", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Inference image size")
    parser.add_argument("--device", type=str, default="",
                        help="Device: '' (auto) / cuda / cpu")
    parser.add_argument("--visualize", type=int, default=-1,
                        help="Number of visualization images to save (-1 = all, 0 = disable)")
    parser.add_argument("--vis_seed", type=int, default=42,
                        help="Random seed for selecting visualization images")
    args = parser.parse_args()

    # Output dir
    if args.output_dir is None:
        parent = os.path.dirname(args.fused_dir.rstrip("/\\"))
        args.output_dir = parent + "_detection"
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Discover fused images
    # ------------------------------------------------------------------
    SUPPORTED = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    fused_files = sorted([
        f for f in os.listdir(args.fused_dir)
        if f.lower().endswith(SUPPORTED)
    ])
    if not fused_files:
        print(f"No images found in {args.fused_dir}")
        return

    image_stems = [os.path.splitext(f)[0] for f in fused_files]
    print(f"Found {len(fused_files)} fused images")

    # ------------------------------------------------------------------
    # 2. Load ground truth
    # ------------------------------------------------------------------
    print("Loading ground truth annotations ...")
    gt_all, gt_difficult, class_names = load_ground_truths(args.ann_dir, image_stems)
    n_missing = sum(1 for s in image_stems if len(gt_all[s]) == 0 and os.path.exists(os.path.join(args.ann_dir, s + ".xml")) is False)
    n_with_gt = sum(1 for s in image_stems if len(gt_all[s]) > 0)
    print(f"  Classes: {class_names}")
    print(f"  Images with GT boxes: {n_with_gt} / {len(image_stems)}")
    print(f"  Images missing annotation file: {n_missing}")

    total_gt = sum(len(v) for v in gt_all.values())
    total_gt_nondiff = sum(
        sum(1 for d in gt_difficult.get(s, []) if not d)
        for s in gt_all
    )
    print(f"  Total GT boxes: {total_gt} (non-difficult: {total_gt_nondiff})")

    # ------------------------------------------------------------------
    # 3. Load YOLOv5 model
    # ------------------------------------------------------------------
    print(f"Loading YOLOv5 model: {args.weights} ...")

    # Fix torch.hub Authorization header bug in newer PyTorch versions
    import torch.hub as _hub
    _orig_validate = _hub._validate_not_a_forked_repo
    def _patched_validate(*a, **k):
        try:
            _orig_validate(*a, **k)
        except KeyError:
            pass
    _hub._validate_not_a_forked_repo = _patched_validate

    # Resolve device (YOLOv5 hubconf only accepts cpu/cuda, so we load on cpu
    # then move to XPU if needed)
    device = args.device
    hub_device = "cpu"
    if device == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif torch.cuda.is_available():
            hub_device = "cuda"
            device = "cuda"
        else:
            device = "cpu"

    if device == "xpu":
        # YOLOv5 hubconf doesn't know about XPU; load on CPU then transfer
        model = torch.hub.load("ultralytics/yolov5", "custom",
                               path=args.weights, source="github",
                               device="cpu")
        # Move underlying model to XPU (keep the AutoShape wrapper)
        model.xpu()
        print(f"Model moved to XPU")
    else:
        model = torch.hub.load("ultralytics/yolov5", "custom",
                               path=args.weights, source="github",
                               device=hub_device or None)
    model.conf = args.conf_thres
    model.iou = args.iou_thres
    print("Model loaded.")

    # ------------------------------------------------------------------
    # 4. Run inference
    # ------------------------------------------------------------------
    print("Running inference on fused images ...")
    TARGET_CLASS = "person"
    preds_by_image = {}

    for fname in tqdm(fused_files, desc="Detecting"):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(args.fused_dir, fname)
        results = model(img_path, size=args.img_size)

        preds = results.pandas().xyxy[0]  # DataFrame: xmin, ymin, xmax, ymax, confidence, class, name
        person_preds = preds[preds["name"] == TARGET_CLASS]

        det_list = []
        for _, row in person_preds.iterrows():
            det_list.append({
                "bbox": [float(row["xmin"]), float(row["ymin"]),
                         float(row["xmax"]), float(row["ymax"])],
                "score": float(row["confidence"]),
            })
        preds_by_image[stem] = det_list

    total_det = sum(len(v) for v in preds_by_image.values())
    n_with_det = sum(1 for v in preds_by_image.values() if len(v) > 0)
    print(f"  Total detections ({TARGET_CLASS}): {total_det}")
    print(f"  Images with detections: {n_with_det} / {len(image_stems)}")

    # ------------------------------------------------------------------
    # 5. Compute mAP
    # ------------------------------------------------------------------
    print("Computing mAP metrics ...")

    # mAP@0.5
    iou_thresholds_50 = [0.5]
    aps_50 = compute_map(preds_by_image, gt_all, gt_difficult, iou_thresholds_50)
    map_50 = aps_50[0]

    # mAP@0.75
    iou_thresholds_75 = [0.75]
    aps_75 = compute_map(preds_by_image, gt_all, gt_difficult, iou_thresholds_75)
    map_75 = aps_75[0]

    # mAP@0.5:0.95
    iou_thresholds_range = np.arange(0.5, 1.0, 0.05).tolist()
    aps_range = compute_map(preds_by_image, gt_all, gt_difficult, iou_thresholds_range)
    map_50_95 = float(np.mean(aps_range))

    # Detailed per-threshold results
    detail_rows = []
    for iou_t, ap_val in zip(iou_thresholds_range, aps_range):
        detail_rows.append({
            "IoU_threshold": f"{iou_t:.2f}",
            "AP": f"{ap_val:.6f}",
        })

    # ------------------------------------------------------------------
    # 6. Visualization
    # ------------------------------------------------------------------
    if args.visualize != 0:
        # Pick images that have both GT and predictions for richer visuals
        candidate_stems = [
            s for s in image_stems
            if len(gt_all.get(s, [])) > 0 and len(preds_by_image.get(s, [])) > 0
        ]
        n_vis = len(candidate_stems) if args.visualize < 0 else min(args.visualize, len(candidate_stems))
        print(f"Generating {n_vis} visualization images ...")

        rng = np.random.RandomState(args.vis_seed)
        vis_stems = sorted(rng.choice(candidate_stems, n_vis, replace=False).tolist())

        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Color constants
        GT_COLOR = (0, 255, 0)       # green for GT boxes
        DET_COLOR = (255, 50, 50)    # red for detection boxes
        GT_WIDTH = 3
        DET_WIDTH = 2

        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        for stem in tqdm(vis_stems, desc="Drawing boxes"):
            fname = stem + ".png"
            img_path = os.path.join(args.fused_dir, fname)
            if not os.path.exists(img_path):
                # try .jpg
                fname = stem + ".jpg"
                img_path = os.path.join(args.fused_dir, fname)
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Draw GT boxes (green, dashed-style via double line)
            for gi, gt_box in enumerate(gt_all.get(stem, [])):
                x1, y1, x2, y2 = gt_box
                draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=GT_WIDTH)
                draw.text((x1, y1 - 20), f"GT#{gi}", fill=GT_COLOR, font=font)

            # Draw detection boxes (red)
            for det in preds_by_image.get(stem, []):
                x1, y1, x2, y2 = det["bbox"]
                score = det["score"]
                draw.rectangle([x1, y1, x2, y2], outline=DET_COLOR, width=DET_WIDTH)
                label = f"{TARGET_CLASS} {score:.2f}"
                draw.text((x1, y2 + 2), label, fill=DET_COLOR, font=font)

            vis_path = os.path.join(vis_dir, f"{stem}_det.png")
            img.save(vis_path)

        print(f"  Visualization images saved to: {vis_dir}")

    # ------------------------------------------------------------------
    # 7. Save CSV results
    # ------------------------------------------------------------------
    summary_path = os.path.join(args.output_dir, "detection_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["model", os.path.basename(args.weights)])
        writer.writerow(["target_class", TARGET_CLASS])
        writer.writerow(["conf_thres", args.conf_thres])
        writer.writerow(["iou_nms", args.iou_thres])
        writer.writerow(["img_size", args.img_size])
        writer.writerow(["num_images", len(fused_files)])
        writer.writerow(["total_gt_boxes", total_gt])
        writer.writerow(["total_gt_nondifficult", total_gt_nondiff])
        writer.writerow(["total_detections", total_det])
        writer.writerow(["images_with_gt", n_with_gt])
        writer.writerow(["images_with_det", n_with_det])
        writer.writerow(["mAP@0.5", f"{map_50:.6f}"])
        writer.writerow(["mAP@0.75", f"{map_75:.6f}"])
        writer.writerow(["mAP@0.5:0.95", f"{map_50_95:.6f}"])

    details_path = os.path.join(args.output_dir, "detection_details.csv")
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["IoU_threshold", "AP"])
        writer.writeheader()
        writer.writerows(detail_rows)

    # Per-image detection counts
    per_image_path = os.path.join(args.output_dir, "per_image_detection.csv")
    with open(per_image_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "num_gt", "num_det"])
        for stem in image_stems:
            writer.writerow([stem, len(gt_all.get(stem, [])),
                             len(preds_by_image.get(stem, []))])

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Object Detection Evaluation Results")
    print("=" * 60)
    print(f"  Model:           {os.path.basename(args.weights)}")
    print(f"  Target class:    {TARGET_CLASS}")
    print(f"  Confidence:      {args.conf_thres}")
    print(f"  Images:          {len(fused_files)}")
    print(f"  GT boxes:        {total_gt} (non-difficult: {total_gt_nondiff})")
    print(f"  Detections:      {total_det}")
    print("-" * 60)
    print(f"  mAP@0.5:         {map_50:.4f}")
    print(f"  mAP@0.75:        {map_75:.4f}")
    print(f"  mAP@0.5:0.95:    {map_50_95:.4f}")
    print("=" * 60)
    print(f"Summary saved to:    {summary_path}")
    print(f"Per-threshold saved: {details_path}")
    print(f"Per-image saved:     {per_image_path}")
    if args.visualize > 0:
        print(f"Visualizations:      {os.path.join(args.output_dir, 'visualizations')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
