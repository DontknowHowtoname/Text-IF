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
                    device, epoch, use_obj_intensity=False, obj_intensity_weight=0.05,
                    use_amp=False, grad_clip=1.0):
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
        use_amp: enable mixed precision
        grad_clip: max grad norm for clip_grad_norm_. Set to 0 or None to disable.

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

    optimizer.zero_grad(set_to_none=True)

    # AMP GradScaler must be created outside autocast. Only used when use_amp.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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

        with torch.cuda.amp.autocast(enabled=use_amp):
            I_fused = model(vis, ir, text)
            loss, loss_ssim, loss_max, loss_color, loss_text = \
                loss_function(vis_gt, ir_gt, I_fused, task)

        scaler.scale(loss).backward()

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
            # Skip the step instead of aborting the whole run. Critical for
            # stability: cross-attention can occasionally produce inf/NaN on
            # a single batch without meaning the whole epoch is lost.
            print(f'[warn] non-finite loss at step {step}: {loss.item():.4f}; skipping batch')
            optimizer.zero_grad(set_to_none=True)
            continue

        # Gradient clipping BEFORE scaler.step to prevent explosion.
        # Must unscale first when using AMP so clip reads true grad values.
        if use_amp:
            scaler.unscale_(optimizer)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

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


@torch.no_grad()
def save_attention_samples(model, data_loader, device, output_dir,
                           n_samples=20, seed=42, epoch=None):
    """Save attention visualizations for a subset of samples.

    Outputs per sample (in `{output_dir}/epoch_{N}/` or `{output_dir}/`):
      - `{stem}.png`            composite grid: IR | VIS | Fused + L1-L4 attn
      - `{stem}_fused.png`      fused image only
      - `{stem}_attn_L{1-4}.png` per-level mean-over-heads attention heatmap

    Args:
        model: TextIFSpatial (must implement forward_with_attn)
        data_loader: dict-batch loader with keys ir/vis/text/stem
        device: torch device
        output_dir: base directory
        n_samples: 0 = no-op; -1 = all; N>0 = randomly sample N (seeded)
        seed: RNG seed for deterministic sampling
        epoch: if not None, samples go to {output_dir}/epoch_{epoch}/ else {output_dir}/
    """
    if n_samples == 0:
        return

    import random
    import numpy as np
    from PIL import Image
    try:
        import matplotlib.pyplot as plt
        _HAS_MPL = True
    except ImportError:
        _HAS_MPL = False

    subdir = output_dir if epoch is None else os.path.join(output_dir, f'epoch_{epoch}')
    os.makedirs(subdir, exist_ok=True)

    n_total = len(data_loader.dataset)
    if n_samples == -1 or n_samples >= n_total:
        keep_indices = set(range(n_total))
    else:
        rng = random.Random(seed)
        keep_indices = set(rng.sample(range(n_total), n_samples))

    model.eval()
    visited = 0
    target_count = len(keep_indices)
    saved = 0
    for batch in tqdm(data_loader, desc=f'[attn-vis target={target_count}]'):
        ir = batch['ir'].to(device)
        vis = batch['vis'].to(device)
        text = batch['text'].to(device)
        stems = batch['stem']
        if isinstance(stems, torch.Tensor):
            stems = stems.tolist()

        fused, attn_dict = model.forward_with_attn(vis, ir, text)
        fused = fused.clamp(0, 1).cpu().numpy()

        B = ir.size(0)
        for i in range(B):
            idx = visited
            visited += 1
            if idx not in keep_indices:
                continue
            saved += 1

            stem = stems[i]
            ir_np = (ir[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            vis_np = (vis[i].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            fused_np = (fused[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(fused_np).save(os.path.join(subdir, f'{stem}_fused.png'))

            # Per-level mean-over-heads attention, min-max normalized
            attn_grids = []
            for level in ('L1', 'L2', 'L3', 'L4'):
                a = attn_dict[level][i].mean(dim=0).cpu().numpy()
                a = (a - a.min()) / max(a.max() - a.min(), 1e-12)
                attn_grids.append(a)
                a_uint = (a * 255).astype(np.uint8)
                Image.fromarray(a_uint, mode='L').save(
                    os.path.join(subdir, f'{stem}_attn_{level}.png'))

            # Composite figure (matplotlib preferred; PIL fallback)
            if _HAS_MPL:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                axes[0, 0].imshow(ir_np);  axes[0, 0].set_title('IR');    axes[0, 0].axis('off')
                axes[0, 1].imshow(vis_np); axes[0, 1].set_title('VIS');   axes[0, 1].axis('off')
                axes[0, 2].imshow(fused_np); axes[0, 2].set_title('Fused'); axes[0, 2].axis('off')
                axes[0, 3].axis('off')
                im = None
                for j, level in enumerate(('L1', 'L2', 'L3', 'L4')):
                    im = axes[1, j].imshow(attn_grids[j], cmap='jet', vmin=0, vmax=1)
                    axes[1, j].set_title(f'Attn {level}'); axes[1, j].axis('off')
                if im is not None:
                    fig.colorbar(im, ax=axes[1, :], fraction=0.02, pad=0.02)
                plt.tight_layout()
                fig.savefig(os.path.join(subdir, f'{stem}.png'), dpi=80, bbox_inches='tight')
                plt.close(fig)
            else:
                strip = np.concatenate([ir_np, vis_np, fused_np], axis=1)
                Image.fromarray(strip).save(os.path.join(subdir, f'{stem}.png'))

            if saved >= target_count:
                return
