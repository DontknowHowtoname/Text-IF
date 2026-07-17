"""
Training script: Text_IF_Recon v4 with iterative SAM-fusion feedback.

Pass 1: Global fusion without mask (v2 baseline).
Pass 2: SAM ViT-B generates mask from fused_1, fusion runs again with mask.

No pre-computed masks needed. SAM+CLIP runs online (frozen).
Based on train_fusion_obj_enhance.py but uses PromptDataSet (no masks).
"""
import os
import argparse

# Must set CUDA_VISIBLE_DEVICES before importing torch,
# otherwise the CUDA runtime may already be initialized and ignore this env var.
# Default gpu_id=0; override via --gpu_id before any torch import takes effect.
_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
# Allow early override from command line: --gpu_id X parsed manually
for _i, _arg in enumerate(os.sys.argv):
    if _arg == "--gpu_id" and _i + 1 < len(os.sys.argv):
        _gpu_id = os.sys.argv[_i + 1]
        break
os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_id

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import clip
from data.prompt_dataset import PromptDataSetWithMask

from model.Text_IF_recon_model_4 import Text_IF_Recon_v4 as create_model
from model.sam_iterative_filter import IterativeSAMFilter
from scripts.utils import (read_data, train_one_epoch_iterative, evaluate_iterative,
                            create_lr_scheduler)
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import transforms as T


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Device info ----
    print("=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  cuDNN version   : {torch.backends.cudnn.version()}")
        print(f"  GPU count       : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU[{i}]         : {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU[{i}] memory  : {mem_total:.1f} GB")
    print(f"  Device          : {device}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/TextIF_iterative_{}".format(file_name)
    os.makedirs(filefold_path)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_val_loss = 1e5
    start_epoch = 0

    print("Loading IVF Fusion and Low-Light Task!")
    if args.low_light_path is not None:
        train_low_light_path_list, val_low_light_path_list = read_data(args.low_light_path)
    else:
        train_low_light_path_list = val_low_light_path_list = None

    print("Loading IVF Fusion and Over-Exposure Task!")
    if args.over_exposure_path is not None:
        train_over_exposure_path_list, val_over_exposure_path_list = read_data(args.over_exposure_path)
    else:
        train_over_exposure_path_list = val_over_exposure_path_list = None

    print("Loading IVF Fusion and ir_low_contrast Task!")
    if args.ir_low_contrast_path is not None:
        train_ir_low_contrast_path_list, val_ir_low_contrast_path_list = read_data(args.ir_low_contrast_path)
    else:
        train_ir_low_contrast_path_list = val_ir_low_contrast_path_list = None

    print("Loading IVF Fusion and ir_noise_path Task!")
    if args.ir_noise_path is not None:
        train_ir_noise_path_list, val_ir_noise_path_list = read_data(args.ir_noise_path)
    else:
        train_ir_noise_path_list = val_ir_noise_path_list = None

    data_transform = {
        "train": T.Compose([T.RandomCrop(96),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    # Use PromptDataSetWithMask for collate_fn compatibility (masks will be zeros)
    train_dataset = PromptDataSetWithMask(
        train_low_light_path_list=train_low_light_path_list,
        val_low_light_path_list=val_low_light_path_list,
        train_over_exposure_path_list=train_over_exposure_path_list,
        val_over_exposure_path_list=val_over_exposure_path_list,
        train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
        val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
        train_ir_noise_path_list=train_ir_noise_path_list,
        val_ir_noise_path_list=val_ir_noise_path_list,
        phase="train",
        transform=data_transform["train"])

    val_dataset = PromptDataSetWithMask(
        train_low_light_path_list=train_low_light_path_list,
        val_low_light_path_list=val_low_light_path_list,
        train_over_exposure_path_list=train_over_exposure_path_list,
        val_over_exposure_path_list=val_over_exposure_path_list,
        train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
        val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
        train_ir_noise_path_list=train_ir_noise_path_list,
        val_ir_noise_path_list=val_ir_noise_path_list,
        phase="val",
        transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # Load CLIP (shared between model and SAM filter)
    model_clip, clip_preprocess = clip.load("ViT-B/32", device=device)

    # Create v4 model
    model = create_model(model_clip, iterations=args.iterations).to(device)

    # Freeze CLIP
    for param in model.base.model_clip.parameters():
        param.requires_grad = False

    # Load pretrained weights with key remapping
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]

        # Clean DataParallel prefix
        weights_dict = {k.replace("module.", ""): v for k, v in weights_dict.items()}

        has_base_prefix = any(k.startswith('base.') for k in weights_dict)
        if not has_base_prefix:
            weights_dict = {f'base.{k}': v for k, v in weights_dict.items()}

        # Remap prompt_guidance_X.MLP -> prompt_guidance_X.global_affine.MLP
        new_weights = {}
        for k, v in weights_dict.items():
            for level in ['2', '3', '4']:
                old_prefix = f'base.prompt_guidance_{level}.'
                new_prefix = f'base.prompt_guidance_{level}.global_affine.'
                if k.startswith(old_prefix) and 'global_affine' not in k:
                    k = k.replace(old_prefix, new_prefix)
                    break
            new_weights[k] = v

        missing, unexpected = model.load_state_dict(new_weights, strict=False)
        loaded_count = len(new_weights) - len(unexpected)
        print(f"Loaded pretrained weights from: {args.weights}")
        print(f"  Keys loaded: {loaded_count}/{len(new_weights)}")

    # Freeze encoders
    for param in model.base.encoder_A.parameters():
        param.requires_grad = False
    for param in model.base.encoder_B.parameters():
        param.requires_grad = False
    print("Encoders frozen. Training: MaskGuidedAffine, FFBlock, FDBlock, ReconHead, Decoder")

    # Create IterativeSAMFilter (frozen SAM ViT-B + shared CLIP)
    sam_filter = IterativeSAMFilter(
        sam_ckpt=args.sam_ckpt_iter,
        obj_text=args.obj_text,
        clip_model=model.base.model_clip,
        clip_preprocess=clip_preprocess,
        device=device,
        clip_threshold=args.clip_threshold
    )

    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    print(f"Training Text_IF_Recon_v4 iterative fusion "
          f"(iterations={args.iterations}, obj_text='{args.obj_text}', "
          f"pass1_weight={args.pass1_weight}, lr={args.lr})")

    for epoch in range(start_epoch, args.epochs):
        # Train
        (train_loss, train_ssim, train_max, train_color, train_text,
         train_recon, train_mask, lr) = train_one_epoch_iterative(
            model=model,
            model_clip=model_clip,
            sam_filter=sam_filter,
            optimizer=optimizer,
            data_loader=train_loader,
            lr_scheduler=lr_scheduler,
            device=device,
            epoch=epoch,
            recon_weight=args.recon_weight,
            enhance_factor=args.enhance_factor,
            bg_factor=args.bg_factor,
            mask_loss_weight=args.mask_loss_weight,
            pass1_weight=args.pass1_weight)

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_ssim_loss", train_ssim, epoch)
        tb_writer.add_scalar("train_max_loss", train_max, epoch)
        tb_writer.add_scalar("train_color_loss", train_color, epoch)
        tb_writer.add_scalar("train_text_loss", train_text, epoch)
        tb_writer.add_scalar("train_recon_loss", train_recon, epoch)
        tb_writer.add_scalar("train_mask_loss", train_mask, epoch)

        if epoch % args.val_every_epcho == 0 and epoch != 0:
            (val_loss, val_ssim, val_max, val_color, val_text,
             val_recon, val_mask) = evaluate_iterative(
                model=model,
                sam_filter=sam_filter,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
                lr=lr,
                filefold_path=file_img_path,
                recon_weight=args.recon_weight,
                enhance_factor=args.enhance_factor,
                bg_factor=args.bg_factor,
                mask_loss_weight=args.mask_loss_weight,
                pass1_weight=args.pass1_weight)

            tb_writer.add_scalar("val_total_loss", val_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss", val_ssim, epoch)
            tb_writer.add_scalar("val_max_loss", val_max, epoch)
            tb_writer.add_scalar("val_color_loss", val_color, epoch)
            tb_writer.add_scalar("val_text_loss", val_text, epoch)
            tb_writer.add_scalar("val_recon_loss", val_recon, epoch)
            tb_writer.add_scalar("val_mask_loss", val_mask, epoch)

            if val_loss < best_val_loss:
                if args.use_dp == True:
                    save_file = {"model": model.module.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
                else:
                    save_file = {"model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
                torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")
                best_val_loss = val_loss

            if args.use_dp == True:
                    save_file = {"model": model.module.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
            else:
                    save_file = {"model": model.state_dict(),
                                 "optimizer": optimizer.state_dict(),
                                 "lr_scheduler": lr_scheduler.state_dict(),
                                 "epoch": epoch,
                                 "args": args}
            torch.save(save_file, file_weights_path + "/" + "checkpoint_lastest.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--low_light_path', type=str, default="./dataset/EMS_lite/Low_light")
    parser.add_argument('--over_exposure_path', type=str, default="./dataset/EMS_lite/Over_exposure")
    parser.add_argument('--ir_low_contrast_path', type=str, default="./dataset/EMS_lite/IR_Low_contrast")
    parser.add_argument('--ir_noise_path', type=str, default="./dataset/EMS_lite/IR_Noise")

    parser.add_argument('--weights', type=str,
                        default='experiments/TextIF_train_20260408-185710/weights/checkpoint.pth',
                        help='textif-me pretrained weights path')
    parser.add_argument('--val_every_epcho', type=int, default=2, help='val every epcho')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default=False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')

    # Reconstruction loss parameters
    parser.add_argument('--recon_weight', type=float, default=0.05)
    parser.add_argument('--pass1_weight', type=float, default=0.3,
                        help='Weight for Pass 1 loss (default: 0.3)')

    # Mask enhancement parameters
    parser.add_argument('--enhance_factor', type=float, default=1.5)
    parser.add_argument('--bg_factor', type=float, default=0.5)
    parser.add_argument('--mask_loss_weight', type=float, default=1.0)

    # Iterative SAM parameters
    parser.add_argument('--iterations', type=int, default=2,
                        help='Number of fusion iterations (default: 2)')
    parser.add_argument('--obj_text', type=str, default='person',
                        help='Object category for CLIP filtering (default: person)')
    parser.add_argument('--sam_ckpt_iter', type=str,
                        default='references/segment-anything/checkpoints/sam_vit_b_01ec64.pth',
                        help='SAM ViT-B checkpoint for iterative mask generation')
    parser.add_argument('--clip_threshold', type=float, default=0.22,
                        help='CLIP similarity threshold for mask filtering')

    opt = parser.parse_args()
    main(opt)
