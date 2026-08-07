"""
Fine-tuning script: Text_IF_Recon v2 / v5 from a v2-ft pretrained checkpoint,
on a single standard IVIF benchmark (TNO/MSRS/M3FD/RoadScene/LLVIP).

Defends against catastrophic forgetting via mixed EMS_lite replay:
each target step has replay_ratio probability of an additional EMS forward+backward.

Examples:
    # Fine-tune v2 on TNO with EMS replay (default)
    python train_finetune_v2ft.py --dataset_name TNO --model_version v2 \
        --weights experiments/TextIF_full_recon_v2_ft_.../weights/checkpoint.pth

    # Disable replay (single-dataset fine-tune)
    python train_finetune_v2ft.py --dataset_name LLVIP --model_version v2 --ems_root ""
"""
import os
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import clip

from data.prompt_dataset import PromptDataSet, SingleTaskDataSet
from scripts.utils import (
    read_data, read_data_for_finetune,
    train_one_epoch_replay, evaluate_replay, create_lr_scheduler,
)
import transforms as T


DATASET_CONFIGS = {
    "TNO":       "./dataset/TNO",
    "MSRS":      "./dataset/MSRS",
    "M3FD":      "./dataset/M3FD",
    "RoadScene": "./dataset/RoadScene",
    "LLVIP":     "./dataset/LLVIP",
}


def _build_target_datasets(dataset_root, data_transform):
    """Build train/val SingleTaskDataSet from the target dataset root."""
    train_vis, train_ir, val_vis, val_ir = read_data_for_finetune(dataset_root)
    print(f"[target] {len(train_vis)} train pairs, {len(val_vis)} val pairs from {dataset_root}")
    train_ds = SingleTaskDataSet(train_vis, train_ir, phase="train",
                                  transform=data_transform["train"])
    val_ds = SingleTaskDataSet(val_vis, val_ir, phase="val",
                                transform=data_transform["val"])
    return train_ds, val_ds


def _build_ems_dataset(ems_root, data_transform):
    """Build the EMS replay PromptDataSet (all 4 tasks). Returns None if root missing."""
    if not ems_root or not os.path.isdir(ems_root):
        print(f"[replay] EMS root '{ems_root}' missing — replay disabled.")
        return None

    tasks = ["Low_light", "Over_exposure", "IR_Low_contrast", "IR_Noise"]
    train_path_lists = []
    val_path_lists = []
    for task_dir in tasks:
        root = os.path.join(ems_root, task_dir)
        if not os.path.isdir(root):
            print(f"[replay] WARNING: EMS task missing: {root}")
            return None
        train_paths, val_paths = read_data(root)
        train_path_lists.append(train_paths)
        val_path_lists.append(val_paths)

    # PromptDataSet takes 4 task path_lists. Combine into one dataset.
    (train_low_light, train_over, train_contrast, train_noise) = train_path_lists
    train_ds = PromptDataSet(
        train_low_light_path_list=train_low_light,
        val_low_light_path_list=val_path_lists[0],
        train_over_exposure_path_list=train_over,
        val_over_exposure_path_list=val_path_lists[1],
        train_ir_low_contrast_path_list=train_contrast,
        val_ir_low_contrast_path_list=val_path_lists[2],
        train_ir_noise_path_list=train_noise,
        val_ir_noise_path_list=val_path_lists[3],
        phase="train",
        transform=data_transform["train"],
    )
    print(f"[replay] EMS dataset loaded from {ems_root}")
    return train_ds


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir is not None:
        filefold_path = args.output_dir
    else:
        filefold_path = "./experiments/finetune_{}_{}_{}".format(
            args.model_version, args.dataset_name, file_name)
    os.makedirs(filefold_path, exist_ok=True)
    for sub in ["img", "weights", "log"]:
        os.makedirs(os.path.join(filefold_path, sub), exist_ok=True)
    tb_writer = SummaryWriter(log_dir=os.path.join(filefold_path, "log"))

    # ---- Datasets ----
    data_transform = {
        "train": T.Compose([T.RandomCrop(96), T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5), T.ToTensor()]),
        "val":   T.Compose([T.Resize_16(), T.ToTensor()]),
    }

    target_root = DATASET_CONFIGS[args.dataset_name]
    train_target_ds, val_target_ds = _build_target_datasets(target_root, data_transform)
    train_ems_ds = _build_ems_dataset(args.ems_root, data_transform)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    target_loader = torch.utils.data.DataLoader(
        train_target_ds, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=nw, collate_fn=train_target_ds.collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_target_ds, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=nw, collate_fn=val_target_ds.collate_fn)

    ems_loader = None
    if train_ems_ds is not None and args.replay_ratio > 0.0:
        ems_loader = torch.utils.data.DataLoader(
            train_ems_ds, batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=nw, collate_fn=train_ems_ds.collate_fn)

    # ---- Model ----
    model_clip, _ = clip.load("ViT-B/32", device=device)
    if args.model_version == "v2":
        from model.Text_IF_recon_model_2 import Text_IF_Recon as create_model
    elif args.model_version == "v5":
        from model.Text_IF_recon_model_5 import Text_IF_Recon_v5 as create_model
    else:
        raise ValueError(f"Unknown --model_version: {args.model_version}")

    use_spatial = bool(args.use_spatial) if args.model_version == "v5" else False
    if args.model_version == "v5":
        model = create_model(model_clip, use_spatial=use_spatial).to(device)
    else:
        model = create_model(model_clip).to(device)

    # Freeze CLIP + encoders
    for param in model.base.model_clip.parameters():
        param.requires_grad = False
    for param in model.base.encoder_A.parameters():
        param.requires_grad = False
    for param in model.base.encoder_B.parameters():
        param.requires_grad = False

    # Load v2-ft weights (reuse v5-ft's key-remap logic)
    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device, weights_only=False)["model"]
        has_base_prefix = any(k.startswith("base.") for k in weights_dict)
        if not has_base_prefix:
            weights_dict = {f'base.{k}': v for k, v in weights_dict.items()}
        missing, unexpected = model.load_state_dict(weights_dict, strict=False)
        loaded_base = len(weights_dict) - len(unexpected)
        print(f"Loaded pretrained weights from: {args.weights}")
        print(f"  Keys loaded: {loaded_base}/{len(weights_dict)}")
        non_clip_missing = [k for k in missing if not k.startswith("base.model_clip")]
        if non_clip_missing:
            print(f"  Missing keys (random init, non-CLIP): {non_clip_missing[:5]}... ({len(non_clip_missing)} total)")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:5]}... ({len(unexpected)} total)")

    if args.use_dp:
        model = torch.nn.DataParallel(model).cuda()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5e-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(target_loader), args.epochs, warmup=True)

    replay_status = f"ON (ratio={args.replay_ratio})" if ems_loader is not None else "OFF"
    print(f"Fine-tuning {args.model_version} on {args.dataset_name} | replay={replay_status} | "
          f"lr={args.lr} epochs={args.epochs} recon_weight={args.recon_weight}")

    best_val_loss = 1e5
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        (train_loss, train_ssim, train_max, train_color,
         train_text, train_recon, lr) = train_one_epoch_replay(
            model=model, model_clip=model_clip, optimizer=optimizer,
            lr_scheduler=lr_scheduler, target_loader=target_loader,
            ems_loader=ems_loader, replay_ratio=args.replay_ratio,
            device=device, epoch=epoch, recon_weight=args.recon_weight,
            max_ratio=args.max_ratio, ssim_ratio=args.ssim_ratio, text_ratio=args.text_ratio)

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_ssim_loss", train_ssim, epoch)
        tb_writer.add_scalar("train_max_loss", train_max, epoch)
        tb_writer.add_scalar("train_color_loss", train_color, epoch)
        tb_writer.add_scalar("train_text_loss", train_text, epoch)
        tb_writer.add_scalar("train_recon_loss", train_recon, epoch)

        if epoch % args.val_every_epcho == 0 and epoch != 0:
            # NOTE: filefold_path=None because evaluate_replay does not save images.
            (val_loss, val_ssim, val_max, val_color,
             val_text, val_recon) = evaluate_replay(
                model=model, data_loader=val_loader, device=device,
                epoch=epoch, lr=lr,
                filefold_path=None,
                max_ratio=args.max_ratio, ssim_ratio=args.ssim_ratio, text_ratio=args.text_ratio)

            tb_writer.add_scalar("val_total_loss", val_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss", val_ssim, epoch)
            tb_writer.add_scalar("val_max_loss", val_max, epoch)
            tb_writer.add_scalar("val_color_loss", val_color, epoch)
            tb_writer.add_scalar("val_text_loss", val_text, epoch)
            tb_writer.add_scalar("val_recon_loss", val_recon, epoch)

            if val_loss < best_val_loss:
                save_file = {
                    "model": (model.module if args.use_dp else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch, "args": args,
                }
                torch.save(save_file, os.path.join(filefold_path, "weights", "checkpoint.pth"))
                best_val_loss = val_loss

            save_file = {
                "model": (model.module if args.use_dp else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch, "args": args,
            }
            torch.save(save_file, os.path.join(filefold_path, "weights", "checkpoint_lastest.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune v2-ft on standard IVIF benchmarks with optional EMS replay")
    parser.add_argument("--dataset_name", type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--model_version", type=str, default="v2", choices=["v2", "v5"])
    parser.add_argument("--weights", type=str, default="experiments/TextIF_train_20260408-185710/weights/checkpoint.pth",
                        help="Path to v2-ft pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--val_every_epcho", type=int, default=2)
    parser.add_argument("--recon_weight", type=float, default=0.3)
    parser.add_argument("--max_ratio", type=float, default=None)
    parser.add_argument("--ssim_ratio", type=float, default=None)
    parser.add_argument("--text_ratio", type=float, default=None)
    parser.add_argument("--use_spatial", type=int, default=1, choices=[0, 1], help="v5 only")
    parser.add_argument("--replay_ratio", type=float, default=0.2,
                        help="Probability of an EMS replay step per target step")
    parser.add_argument("--ems_root", type=str, default="./dataset/EMS_lite",
                        help="EMS_lite root for replay; set to '' to disable")
    parser.add_argument("--use_dp", default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--output_dir", default=None,
                        help="Override default ./experiments/finetune_<v>_<dataset>_<timestamp>")
    args = parser.parse_args()
    main(args)
