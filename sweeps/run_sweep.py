"""Full-pipeline sweep driver: train + eval + aggregate for each text_ratio.

Single Python invocation runs N text_ratio values serially (one GPU).
Fail-fast: any subprocess failure exits the whole sweep.

Usage:
    python sweeps/run_sweep.py \\
        --text_ratios 0,1,2,3,5,8,10,15 \\
        --repo_dir /path/to/Text-IF \\
        --pretrained_weights /path/to/ckpt.pth \\
        --dataset_ll ... --dataset_oe ... --dataset_ic ... --dataset_in ... \\
        --eval_data_path ...
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args. Pass argv=None to read sys.argv[1:]."""
    p = argparse.ArgumentParser(
        description="Serial full-pipeline sweep over text_ratio values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--text_ratios", type=str, required=True,
                   help="Comma-separated text_ratio values, e.g. '0,1,2,3,5,8,10,15'")
    p.add_argument("--repo_dir", type=str, required=True,
                   help="Repository root (must contain train_fusion_full_recon_v2_ft.py "
                        "and evaluate_textif_full_recon_v2.py)")
    p.add_argument("--pretrained_weights", type=str, required=True,
                   help="Path to textif-me pretrained checkpoint (.pth)")
    p.add_argument("--dataset_ll", type=str, required=True,
                   help="EMS_lite/Low_light directory")
    p.add_argument("--dataset_oe", type=str, required=True,
                   help="EMS_lite/Over_exposure directory")
    p.add_argument("--dataset_ic", type=str, required=True,
                   help="EMS_lite/IR_Low_contrast directory")
    p.add_argument("--dataset_in", type=str, required=True,
                   help="EMS_lite/IR_Noise directory")
    p.add_argument("--eval_data_path", type=str, required=True,
                   help="Evaluation dataset path (e.g. data/IVT_test)")
    p.add_argument("--output_root", type=str, default="sweeps/out",
                   help="Parent dir for text_ratio_T<T>/ output subdirs")
    p.add_argument("--val_every_epcho", type=int, default=1,
                   help="Forwarded to training script. Default 1 ensures checkpoint "
                        "is written at epoch 1 (see parent spec C1).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Forwarded to training script. None = use script default (50).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns process exit code."""
    args = parse_args(argv)
    print(f"[sweep] parsed args: text_ratios={args.text_ratios!r}, "
          f"repo_dir={args.repo_dir!r}, output_root={args.output_root!r}")
    print("[sweep] (skeleton — sweep loop not implemented yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
