# seg_msrs/train_segformer_msrs.py
"""Train SegFormer-B1 on MSRS (9 classes), device-agnostic (xpu/cuda/cpu).

Usage:
  python seg_msrs/train_segformer_msrs.py --device xpu
  python seg_msrs/train_segformer_msrs.py --device xpu --epochs 1 --limit 20  # smoke
"""
import argparse
import csv
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seg_msrs.dataset import MSRSSegDataset
from seg_msrs.eval_seg_miou import evaluate, write_results
from seg_msrs.model import SegFormerSeg

DEFAULT_CKPT = os.path.join("references", "SegFormer", "mit_b1_20220624-02e5a6a1.pth")


def build_optimizer(model, lr, wd):
    """mmseg paramwise cfg: head lr x10, norms & patch embeds no weight decay."""
    decay, no_decay, head = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("decode_head"):
            head.append(p)
        elif p.ndim <= 1 or "proj" in name.split("."):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0},
         {"params": head, "weight_decay": wd, "lr": lr * 10}], lr=lr)


def lr_at(step, total, base_lr, warmup=1500, power=1.0):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    return base_lr * (1 - step / total) ** power


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="dataset/MSRS-main")
    ap.add_argument("--pretrained", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--crop-size", type=int, default=480)
    ap.add_argument("--modality", default="vi", choices=("vi", "ir"))
    ap.add_argument("--eval-interval", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train images")
    ap.add_argument("--out", default="seg_msrs/runs/segformer_b1")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    train_ds = MSRSSegDataset(args.data_root, "train", args.modality,
                              train=True, crop_size=args.crop_size)
    val_ds = MSRSSegDataset(args.data_root, "test", args.modality, train=False)
    if args.limit:
        train_ds.items, val_ds.items = train_ds.items[:args.limit], val_ds.items[:args.limit]
    assert not train_ds.missing and not val_ds.missing, \
        f"unaligned files: train={train_ds.missing} val={val_ds.missing}"

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True,
                          pin_memory=False)
    val_ld = DataLoader(val_ds, batch_size=4, num_workers=args.num_workers)

    model = SegFormerSeg(num_classes=9, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = build_optimizer(model, args.lr, args.wd)
    total_iters = args.epochs * len(train_ld)

    log_path = os.path.join(args.out, "train_log.csv")
    with open(log_path, "w", newline=""):
        pass
    best = -1.0
    step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, labels in train_ld:
            lr = lr_at(step, total_iters, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr * (10.0 if g is optimizer.param_groups[-1] else 1.0)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            step += 1
        print(f"epoch {epoch + 1}/{args.epochs} loss {running / len(train_ld):.4f} lr {lr:.2e}", flush=True)

        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs:
            miou, per_class = evaluate(model, val_ld, device, num_classes=9)
            print(f"epoch {epoch + 1} val mIoU {miou:.4f}", flush=True)
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, running / len(train_ld), miou,
                                        json.dumps(per_class)])
            torch.save(model.state_dict(), os.path.join(args.out, "last.pth"))
            if miou > best:
                best = miou
                torch.save(model.state_dict(), os.path.join(args.out, "best_miou.pth"))
                write_results(args.out, miou, per_class, name="best")
    print(f"done. best mIoU {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
