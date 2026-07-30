"""Train TextIFSpatial on FLIR-align-3class.

Usage:
    python train_fusion_flir.py --epochs 120 --batch-size 8

Device: prefers Intel XPU, falls back to CUDA, then CPU.
"""
import os
# Mitigate CUDA fragmentation: the OOM error reported reserved >> allocated.
# Must be set before torch is imported. Harmless on XPU/CPU.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

import argparse
import datetime
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import clip

# Make project root importable when running from anywhere
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model.text_if_spatial import TextIFSpatial
from data.flir_dataset import FLIRPromptDataSet
from scripts.flir_utils import train_one_epoch, evaluate, save_fused_images, save_attention_samples


def parse_args():
    p = argparse.ArgumentParser(
        description='Train TextIFSpatial on FLIR-align-3class (XPU)')
    p.add_argument(
        '--data_root',
        default='D:/StudyFiles/MachineLearning/datasets/FLIR-align-3class/FLIR-align-3class')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--grad_clip', type=float, default=1.0,
                   help='max grad norm for clipping; 0 to disable')
    p.add_argument('--save_attn_samples', type=int, default=20,
                   help='attention map visualization: 0=off, -1=all test, N>0=sample N (default 20)')
    p.add_argument('--gate_scale', type=float, default=0.3,
                   help='bounds TextSpatialAffine spatial gate to [1-s,1+s]. '
                        'Higher gives the spatial/attention path more influence. '
                        'Default 0.3 (was 0.1 originally).')
    p.add_argument('--attn_loss_weight', type=float, default=0.01,
                   help='weight for thermal-saliency attention supervision MSE. '
                        'Set to 0 to disable. Default 0.01.')
    p.add_argument('--saliency_top_k', type=float, default=0.15,
                   help='fraction of brightest IR pixels kept in saliency IR branch.')
    p.add_argument('--saliency_sigma', type=float, default=0.3,
                   help='Gaussian sigma as fraction of per-axis bbox dims in saliency bbox branch.')
    p.add_argument('--val_every_epoch', type=int, default=5)
    p.add_argument('--input_h', type=int, default=512)
    p.add_argument('--input_w', type=int, default=640)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_root', default='experiments')
    p.add_argument('--device', default='auto',
                   help='auto | xpu | cuda | cpu')
    p.add_argument('--use_dp', action='store_true', help='use DataParallel (CUDA only)')
    p.add_argument('--amp', action='store_true', default=True,
                   help='use CUDA mixed precision (fp16). Default on for CUDA.')
    p.add_argument('--no_amp', dest='amp', action='store_false',
                   help='disable mixed precision')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    if name == 'xpu':
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu')
        print('XPU requested but unavailable; falling back to CPU.')
        return torch.device('cpu')
    if name == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def build_lr_scheduler(optimizer, steps_per_epoch, warmup_epochs, total_epochs):
    """Cosine schedule with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f'Device: {device}')

    # Output dir
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    filefold_path = os.path.join(args.output_root, f'TextIF_flir_{timestamp}')
    os.makedirs(os.path.join(filefold_path, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(filefold_path, 'img'), exist_ok=True)
    os.makedirs(os.path.join(filefold_path, 'log'), exist_ok=True)
    print(f'Output: {filefold_path}')

    # Transforms
    tf = transforms.Compose([
        transforms.Resize((args.input_h, args.input_w)),
        transforms.ToTensor(),
    ])

    # Datasets
    train_ds = FLIRPromptDataSet(
        ir_dir=os.path.join(args.data_root, 'infrared', 'train'),
        vis_dir=os.path.join(args.data_root, 'visible', 'train'),
        label_dir=os.path.join(args.data_root, 'labels', 'train'),
        attrs_cache=os.path.join(args.data_root, 'labels', 'train', 'attrs.json'),
        transform=tf, phase='train', seed=args.seed,
    )
    test_ds = FLIRPromptDataSet(
        ir_dir=os.path.join(args.data_root, 'infrared', 'test'),
        vis_dir=os.path.join(args.data_root, 'visible', 'test'),
        label_dir=os.path.join(args.data_root, 'labels', 'test'),
        attrs_cache=os.path.join(args.data_root, 'labels', 'test', 'attrs.json'),
        transform=tf, phase='test', seed=args.seed,
    )
    print(f'Train: {len(train_ds)} samples | Test: {len(test_ds)} samples')

    # pin_memory only helps CUDA; safe-but-no-op on XPU in modern torch
    pin_memory = device.type in ('cuda', 'xpu')
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=pin_memory,
        collate_fn=train_ds.collate_fn)
    val_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        collate_fn=test_ds.collate_fn)

    # Model
    model_clip, _ = clip.load('ViT-B/32', device=str(device))
    model = TextIFSpatial(model_clip, dim=16, gate_scale=args.gate_scale).to(device)
    print(f'gate_scale={args.gate_scale}  attn_loss_weight={args.attn_loss_weight}')
    # IMPORTANT: Explicitly freeze CLIP. The base Text_IF only calls .eval(),
    # which does NOT set requires_grad=False. Training without this line
    # would update CLIP weights via the model's internal CLIP submodule.
    for p in model.model_clip.parameters():
        p.requires_grad = False

    if args.use_dp and device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
    elif args.use_dp:
        print('DataParallel only enabled for CUDA. Continuing on a single device.')

    # Optimizer: only train non-CLIP params
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {n_trainable:,} / {n_total:,}')
    assert n_trainable < n_total, \
        'CLIP not frozen! Check requires_grad logic.'

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = build_lr_scheduler(
        optimizer, len(train_loader), args.warmup_epochs, args.epochs)

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(filefold_path, 'log'))

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        (train_loss, t_ssim, t_max, t_color, t_text, t_attn, lr) = train_one_epoch(
            model=model, model_clip=model_clip, optimizer=optimizer,
            lr_scheduler=lr_scheduler, data_loader=train_loader,
            device=device, epoch=epoch, use_amp=(args.amp and device.type == 'cuda'),
            grad_clip=args.grad_clip,
            attn_loss_weight=args.attn_loss_weight,
            saliency_top_k=args.saliency_top_k,
            saliency_sigma=args.saliency_sigma,
        )

        tb_writer.add_scalar('train/total', train_loss, epoch)
        tb_writer.add_scalar('train/ssim', t_ssim, epoch)
        tb_writer.add_scalar('train/max', t_max, epoch)
        tb_writer.add_scalar('train/color', t_color, epoch)
        tb_writer.add_scalar('train/text', t_text, epoch)
        tb_writer.add_scalar('train/attn', t_attn, epoch)
        tb_writer.add_scalar('train/lr', lr, epoch)

        do_val = ((epoch + 1) % args.val_every_epoch == 0) or (epoch == args.epochs - 1)
        if do_val:
            val_loss, v_ssim, v_max, v_color, v_text = evaluate(
                model=model, model_clip=model_clip,
                data_loader=val_loader, device=device,
                epoch=epoch, lr=lr,
            )
            tb_writer.add_scalar('val/total', val_loss, epoch)
            tb_writer.add_scalar('val/ssim', v_ssim, epoch)
            tb_writer.add_scalar('val/max', v_max, epoch)
            tb_writer.add_scalar('val/color', v_color, epoch)
            tb_writer.add_scalar('val/text', v_text, epoch)
            print(f'[epoch {epoch}] val_loss={val_loss:.4f} val_text={v_text:.4f}')

            # Attention map visualization for the current epoch
            if args.save_attn_samples != 0:
                attn_dir = os.path.join(filefold_path, 'attn_vis')
                model_for_vis = model.module if isinstance(model, nn.DataParallel) else model
                save_attention_samples(
                    model=model_for_vis,
                    data_loader=val_loader,
                    device=device,
                    output_dir=attn_dir,
                    n_samples=args.save_attn_samples,
                    seed=args.seed,
                    epoch=epoch,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                save_file = os.path.join(filefold_path, 'weights', 'best.pth')
                torch.save({'model': model_to_save.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss}, save_file)
                print(f'  Saved best -> {save_file}')

        # Always save latest
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({'model': model_to_save.state_dict(), 'epoch': epoch},
                   os.path.join(filefold_path, 'weights', 'latest.pth'))

    # Save final fused images on test set
    final_fused_dir = os.path.join(filefold_path, 'fused_test')
    save_fused_images(model, val_loader, device, final_fused_dir)
    print(f'Fused test images -> {final_fused_dir}')

    tb_writer.close()
    print('Done.')


if __name__ == '__main__':
    main()
