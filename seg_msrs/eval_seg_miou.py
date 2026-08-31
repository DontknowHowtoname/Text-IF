# seg_msrs/eval_seg_miou.py
"""mIoU evaluation: MSRS test set or arbitrary fused-image directory.

Usage:
  python seg_msrs/eval_seg_miou.py --mode msrs --checkpoint seg_msrs/runs/segformer_b1/best_miou.pth
  python seg_msrs/eval_seg_miou.py --mode dir --images results/fused_msrs --checkpoint ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seg_msrs.dataset import MSRSSegDataset, ImageDirDataset
from seg_msrs.model import SegFormerSeg

CLASSES = ("background", "car", "person", "bike", "curve",
           "car_stop", "guardrail", "color_cone", "bump")


class ConfusionMeter:
    def __init__(self, num_classes, ignore_index=255):
        self.k, self.ignore = num_classes, ignore_index
        self.conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, label):
        """pred, label: LongTensor (N,H,W) or (H,W), values in [0,k) or ignore."""
        pred, label = pred.flatten(), label.flatten()
        mask = (pred != self.ignore) & (label != self.ignore)
        pred, label = pred[mask].numpy(), label[mask].numpy()
        idx = pred * self.k + label  # conf[pred, label]
        counts = np.bincount(idx, minlength=self.k * self.k)
        self.conf += counts.reshape(self.k, self.k)


def confusion_to_iou(conf):
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(1) + conf.sum(0) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return iou


@torch.no_grad()
def evaluate(model, loader, device, num_classes=9):
    model.eval()
    meter = ConfusionMeter(num_classes)
    for imgs, labels in loader:
        logits = model(imgs.to(device))
        pred = logits.argmax(1).cpu()
        meter.update(pred, labels)
    iou = confusion_to_iou(meter.conf)
    miou = float(np.nanmean(iou))
    per_class = {c: (None if np.isnan(v) else round(float(v), 4))
                 for c, v in zip(CLASSES, iou)}
    return miou, per_class


def write_results(out_dir, miou, per_class, name="eval"):
    with open(os.path.join(out_dir, f"{name}_miou.json"), "w") as f:
        json.dump({"mIoU": round(miou, 4), "per_class": per_class}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("msrs", "dir"), required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", help="fused image dir (mode=dir)")
    ap.add_argument("--data-root", default="dataset/MSRS-main")
    ap.add_argument("--modality", default="vi", choices=("vi", "ir"))
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--out", default="seg_msrs/eval_out")
    ap.add_argument("--save-pred", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = SegFormerSeg(num_classes=9).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                     weights_only=False), strict=True)

    os.makedirs(args.out, exist_ok=True)
    if args.mode == "msrs":
        ds = MSRSSegDataset(args.data_root, "test", args.modality, train=False)
        loader = DataLoader(ds, batch_size=4)
        miou, per_class = evaluate(model, loader, device)
    else:
        assert args.images, "--images required for mode=dir"
        ds = ImageDirDataset(args.images)
        loader = DataLoader(ds, batch_size=4)
        lbl_dir = os.path.join(args.data_root, "test", "Segmentation_labels")
        meter = ConfusionMeter(9)
        missing = []
        model.eval()
        with torch.no_grad():
            for imgs, names in loader:
                pred = model(imgs.to(device)).argmax(1).cpu()
                for i, name in enumerate(names):
                    lbl_path = os.path.join(lbl_dir, name)
                    if not os.path.exists(lbl_path):
                        missing.append(name)
                        continue
                    label = torch.from_numpy(
                        np.array(Image.open(lbl_path))).long()
                    assert pred[i].shape == label.shape, (
                        f"{name}: pred {tuple(pred[i].shape)} vs GT {tuple(label.shape)} — "
                        f"fused images must be 480x640")
                    meter.update(pred[i], label)
                    if args.save_pred:
                        os.makedirs(args.save_pred, exist_ok=True)
                        Image.fromarray(pred[i].numpy().astype(np.uint8)).save(
                            os.path.join(args.save_pred, name))
        if missing:
            print(f"WARNING: {len(missing)} images without GT skipped: {missing[:10]}")
        iou = confusion_to_iou(meter.conf)
        miou = float(np.nanmean(iou))
        per_class = {c: (None if np.isnan(v) else round(float(v), 4))
                     for c, v in zip(CLASSES, iou)}

    write_results(args.out, miou, per_class)
    print(f"mIoU: {miou:.4f}")
    for c, v in per_class.items():
        print(f"  {c:10s} {v}")


if __name__ == "__main__":
    main()
