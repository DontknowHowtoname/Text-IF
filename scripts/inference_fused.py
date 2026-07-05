"""Run inference with a trained TextIFSpatial checkpoint."""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import clip
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))

from model.text_if_spatial import TextIFSpatial
from data.flir_dataset import FLIRPromptDataSet
from scripts.flir_utils import save_fused_images


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root',
                   default='D:/StudyFiles/MachineLearning/datasets/FLIR-align-3class/FLIR-align-3class')
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--input_h', type=int, default=512)
    p.add_argument('--input_w', type=int, default=640)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()

    # Device: prefer Intel XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    model_clip, _ = clip.load('ViT-B/32', device=str(device))
    model = TextIFSpatial(model_clip, dim=16).to(device)
    ckpt = torch.load(args.checkpoint, map_location=str(device))
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint epoch={ckpt.get('epoch','?')} val_loss={ckpt.get('val_loss','?')}")

    tf = transforms.Compose([
        transforms.Resize((args.input_h, args.input_w)),
        transforms.ToTensor(),
    ])
    ds = FLIRPromptDataSet(
        ir_dir=os.path.join(args.data_root, 'infrared', args.split),
        vis_dir=os.path.join(args.data_root, 'visible', args.split),
        label_dir=os.path.join(args.data_root, 'labels', args.split),
        attrs_cache=os.path.join(args.data_root, 'labels', args.split, 'attrs.json'),
        transform=tf, phase=args.split, seed=0,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    save_fused_images(model, loader, device, args.output_dir)
    print(f'Saved {len(ds)} fused images -> {args.output_dir}')


if __name__ == '__main__':
    main()
