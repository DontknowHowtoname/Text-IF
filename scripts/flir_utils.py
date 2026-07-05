"""Train/eval loops for FLIRPromptDataSet + TextIFSpatial.

Adapted from scripts/utils.py but uses dict batches (ir, vis, text, ...)
instead of positional tuples. Uses scripts.losses_xpu for XPU compatibility
(the original scripts.losses hardcodes .cuda() calls inside L_Grad_position
and L_SSIM that fail on Intel XPU).

fusion_prompt_loss signature (from scripts/losses_xpu.py):
    forward(image_A, image_B, image_fused, task)
    -> (total_loss, ssim_loss, max_loss, color_loss, grad_loss)
where `task` is an iterable of strings in {"low_light", "over_exposure",
"ir_low_contrast", "ir_noise"}. FLIR has no degradation type, so we use
"ir_low_contrast" as the default (balanced max_ratio=8, ssim_ratio=1,
text_ratio=10).
"""
import os
import sys

import torch
from tqdm import tqdm

from scripts.losses_xpu import fusion_prompt_loss


# FLIR has no named degradation; use the most general balanced task config.
DEFAULT_TASK = "ir_low_contrast"


def _move_loss_to_device(loss_fn, device):
    """Move loss module parameters (sobel kernels, ssim window) to device."""
    return loss_fn.to(device)


def _make_task_list(batch_size, device):
    """Build the per-sample task list expected by fusion_prompt_loss."""
    return [DEFAULT_TASK] * batch_size


def train_one_epoch(model, model_clip, optimizer, lr_scheduler, data_loader,
                    device, epoch, use_obj_intensity=False, obj_intensity_weight=0.05):
    """One training epoch.

    Args:
        model: TextIFSpatial instance
        model_clip: CLIP model (frozen, used by fusion_prompt_loss indirectly)
        optimizer, lr_scheduler: standard
        data_loader: yields dict batches with keys ir/vis/text/stem
        device: torch device
        epoch: int (for tqdm description)
        use_obj_intensity: reserved (currently unused; FLIR has no GT)
        obj_intensity_weight: reserved

    Returns:
        avg_total_loss, avg_ssim, avg_max, avg_color, avg_text, lr
        (per-batch averages, matching scripts/utils.py convention)
    """
    model.train()
    model_clip.eval()
    loss_function = fusion_prompt_loss()
    loss_function = _move_loss_to_device(loss_function, device)

    accu_total = torch.zeros(1).to(device)
    accu_ssim = torch.zeros(1).to(device)
    accu_max = torch.zeros(1).to(device)
    accu_color = torch.zeros(1).to(device)
    accu_text = torch.zeros(1).to(device)

    optimizer.zero_grad()

    tbar = tqdm(data_loader, file=sys.stdout)
    for step, batch in enumerate(tbar):
        ir = batch['ir'].to(device)
        vis = batch['vis'].to(device)
        text = batch['text'].to(device)

        # FLIR has no clean GT reference; use the visible and IR images
        # themselves as structural targets (consistent with prior Text-IF
        # training where the loss compares fused against the two sources).
        ir_gt = ir
        vis_gt = vis

        task = _make_task_list(ir.size(0), device)

        I_fused = model(vis, ir, text)
        loss, loss_ssim, loss_max, loss_color, loss_text = \
            loss_function(vis_gt, ir_gt, I_fused, task)

        loss.backward()

        accu_total += loss.detach()
        accu_ssim += loss_ssim.detach()
        accu_max += loss_max.detach()
        accu_color += loss_color.detach()
        accu_text += loss_text.detach()

        lr = optimizer.param_groups[0]["lr"]

        tbar.desc = ("[train epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                     "color: {:.3f}  text: {:.3f}  lr: {:.6f}").format(
            epoch,
            accu_total.item() / (step + 1),
            accu_ssim.item() / (step + 1),
            accu_max.item() / (step + 1),
            accu_color.item() / (step + 1),
            accu_text.item() / (step + 1),
            lr,
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()

    n_steps = max(step + 1, 1)
    return (accu_total.item() / n_steps,
            accu_ssim.item() / n_steps,
            accu_max.item() / n_steps,
            accu_color.item() / n_steps,
            accu_text.item() / n_steps,
            lr)


@torch.no_grad()
def evaluate(model, model_clip, data_loader, device,
             epoch=0, lr=0.0, filefold_path=None):
    """Validation loop. Returns same tuple as train_one_epoch.

    Args:
        model, model_clip, data_loader, device: standard
        epoch, lr, filefold_path: kept for API parity with scripts/utils.evaluate;
            used only for optional sample-image dumping when filefold_path is set.

    Returns:
        avg_total_loss, avg_ssim, avg_max, avg_color, avg_text
    """
    loss_function = fusion_prompt_loss()
    loss_function = _move_loss_to_device(loss_function, device)
    model.eval()

    accu_total = torch.zeros(1).to(device)
    accu_ssim = torch.zeros(1).to(device)
    accu_max = torch.zeros(1).to(device)
    accu_color = torch.zeros(1).to(device)
    accu_text = torch.zeros(1).to(device)

    tbar = tqdm(data_loader, file=sys.stdout)
    for step, batch in enumerate(tbar):
        ir = batch['ir'].to(device)
        vis = batch['vis'].to(device)
        text = batch['text'].to(device)

        ir_gt = ir
        vis_gt = vis
        task = _make_task_list(ir.size(0), device)

        I_fused = model(vis, ir, text)
        loss, loss_ssim, loss_max, loss_color, loss_text = \
            loss_function(vis_gt, ir_gt, I_fused, task)

        accu_total += loss
        accu_ssim += loss_ssim.detach()
        accu_max += loss_max.detach()
        accu_color += loss_color.detach()
        accu_text += loss_text

        tbar.desc = ("[val epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                     "color: {:.3f}  text: {:.3f}  lr: {:.6f}").format(
            epoch,
            accu_total.item() / (step + 1),
            accu_ssim.item() / (step + 1),
            accu_max.item() / (step + 1),
            accu_color.item() / (step + 1),
            accu_text.item() / (step + 1),
            lr,
        )

    n_steps = max(step + 1, 1)
    return (accu_total.item() / n_steps,
            accu_ssim.item() / n_steps,
            accu_max.item() / n_steps,
            accu_color.item() / n_steps,
            accu_text.item() / n_steps)


@torch.no_grad()
def save_fused_images(model, data_loader, device, output_dir):
    """Run inference and dump fused images to output_dir.

    Args:
        model: TextIFSpatial
        data_loader: yields dict batches (must include 'stem')
        device: torch device
        output_dir: target directory (created if missing)
    """
    from PIL import Image
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    for batch in tqdm(data_loader, desc='[inference]'):
        ir = batch['ir'].to(device)
        vis = batch['vis'].to(device)
        text = batch['text'].to(device)
        fused = model(vis, ir, text)
        fused = fused.clamp(0, 1).cpu().numpy()
        stems = batch['stem']
        if isinstance(stems, torch.Tensor):
            stems = stems.tolist()
        for i in range(fused.shape[0]):
            stem = stems[i]
            arr = (fused[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(output_dir, f'{stem}.png'))
