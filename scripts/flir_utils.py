"""Train/eval loops for FLIRPromptDataSet + TextIFSpatial.

Adapted from scripts/utils.py but uses dict batches (ir, vis, text, ...)
instead of positional tuples. Uses scripts.losses_xpu for XPU compatibility
(the original scripts.losses hardcodes .cuda() calls inside L_Grad_position
and L_SSIM that fail on Intel XPU).

Loss strategy: bypass the task-based fusion_prompt_loss wrapper and call the
inner fusion_loss directly with FLIR-tuned weights. The original task
"ir_low_contrast" (max=8, ssim=1, text=10, color=12 default) drove training
to divergence on FLIR because:
  - L_color with ratio=12 forces fused image's chroma (Cb/Cr in YCbCr) to
    match VIS, but FLIR IR is grayscale (no chroma info to contribute) so
    the loss degenerates into "copy VIS color" — fights with intensity and
    gradient terms that involve IR.
  - color loss dominated the total (168-264 weighted vs 30-40 for others)
    and tracked total loss divergence exactly.

FLIR tuned weights (see docs/superpowers/specs/2026-07-03-flir-text-fusion-design.md):
  - max_ratio=8 (kept from ir_low_contrast; preserves intensity of both sources)
  - ssim_ratio=1 (kept)
  - color_ratio=4 (was 12 — lowered because IR has no chroma; raised from 2
    after L_Intensity was fixed to keep VIS chroma everywhere, since the
    gray/color boundary that originally required a low color_ratio is gone)
  - text_ratio=10 (kept)
"""
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.losses_xpu import fusion_loss
from scripts.thermal_saliency import build_thermal_saliency, resize_saliency_to_attn


# FLIR-tuned loss weights.
# - color_ratio=12 (default) caused divergence: L_color dominated total loss
#   and fought intensity/gradient terms because IR has no chroma.
# - color_ratio=2 stabilized training but too weak to constrain edge chroma,
#   producing purple fringes at object boundaries (see L_Intensity note).
# - color_ratio=4 is the compromise: enough to constrain edge chroma locally
#   now that L_Intensity no longer introduces a gray/color boundary.
FLIR_LOSS_WEIGHTS = dict(max_ratio=8, ssim_ratio=1, color_ratio=4, text_ratio=10)


def _move_loss_to_device(loss_fn, device):
    """Move loss module parameters (sobel kernels, ssim window) to device."""
    return loss_fn.to(device)


def train_one_epoch(model, model_clip, optimizer, lr_scheduler, data_loader,
                    device, epoch, use_obj_intensity=False, obj_intensity_weight=0.05,
                    use_amp=False, grad_clip=1.0,
                    attn_loss_weight=0.01, saliency_top_k=0.15, saliency_sigma=0.3):
    """One training epoch.

    Args:
        model: TextIFSpatial instance
        model_clip: CLIP model (frozen, used by fusion_prompt_loss indirectly)
        optimizer, lr_scheduler: standard
        data_loader: yields dict batches with keys ir/vis/text/stem/bboxes
        device: torch device
        epoch: int (for tqdm description)
        use_obj_intensity: reserved (currently unused; FLIR has no GT)
        obj_intensity_weight: reserved
        use_amp: enable mixed precision
        grad_clip: max grad norm for clip_grad_norm_. Set to 0 or None to disable.
        attn_loss_weight: weight for attention-supervision MSE loss. Set to 0
            to disable attention supervision and recover original behavior.
        saliency_top_k: fraction of brightest IR pixels kept in the IR branch
            of thermal saliency (only used when attn_loss_weight > 0).
        saliency_sigma: Gaussian sigma as fraction of per-axis bbox dims in the
            bbox branch of thermal saliency.

    Returns:
        avg_total_loss, avg_ssim, avg_max, avg_color, avg_text, avg_attn, lr
        (per-batch averages; avg_attn is the unweighted attention MSE for
        monitoring, even though attn_loss_weight scales its contribution)
    """
    model.train()
    model_clip.eval()
    loss_function = fusion_loss()
    loss_function = _move_loss_to_device(loss_function, device)

    accu_total = torch.zeros(1).to(device)
    accu_ssim = torch.zeros(1).to(device)
    accu_max = torch.zeros(1).to(device)
    accu_color = torch.zeros(1).to(device)
    accu_text = torch.zeros(1).to(device)
    accu_attn = torch.zeros(1).to(device)

    optimizer.zero_grad(set_to_none=True)

    # AMP GradScaler must be created outside autocast. Only used when use_amp.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    supervise_attn = attn_loss_weight > 0
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

        with torch.cuda.amp.autocast(enabled=use_amp):
            # Use forward_with_attn to get per-level attention maps for
            # supervision. Graph is built (forward_with_attn no longer carries
            # @torch.no_grad) so attn loss backprops into q_proj/k_proj/gate_conv.
            I_fused, attn_dict = model.forward_with_attn(vis, ir, text)
            loss_rec, loss_ssim, loss_max, loss_color, loss_text = \
                loss_function(vis_gt, ir_gt, I_fused, **FLIR_LOSS_WEIGHTS)

            loss_attn = torch.zeros(1, device=device)
            if supervise_attn:
                bboxes = [b.to(device) if isinstance(b, torch.Tensor) else b
                          for b in batch['bboxes']]
                saliency = build_thermal_saliency(
                    ir, bboxes,
                    top_k_pct=saliency_top_k,
                    sigma_factor=saliency_sigma,
                )
                for level, attn in attn_dict.items():
                    # attn: [B, num_heads, H_l, W_l]
                    attn_mean = attn.mean(dim=1)  # [B, H_l, W_l]
                    B_, Hl, Wl = attn_mean.shape
                    flat = attn_mean.view(B_, -1)
                    mn = flat.min(dim=-1, keepdim=True).values
                    mx = flat.max(dim=-1, keepdim=True).values
                    attn_norm = (flat - mn) / (mx - mn + 1e-8)
                    attn_norm = attn_norm.view(B_, Hl, Wl)
                    sal_resized = resize_saliency_to_attn(saliency, Hl, Wl)
                    loss_attn = loss_attn + F.mse_loss(attn_norm, sal_resized)
                loss_attn = loss_attn / len(attn_dict)  # avg over L1-L4

            loss = loss_rec + attn_loss_weight * loss_attn

        scaler.scale(loss).backward()

        accu_total += loss.detach()
        accu_ssim += loss_ssim.detach()
        accu_max += loss_max.detach()
        accu_color += loss_color.detach()
        accu_text += loss_text.detach()
        accu_attn += loss_attn.detach()

        lr = optimizer.param_groups[0]["lr"]

        tbar.desc = ("[train epoch {}] loss: {:.3f}  ssim: {:.3f}  max: {:.3f}  "
                     "color: {:.3f}  text: {:.3f}  attn: {:.4f}  lr: {:.6f}").format(
            epoch,
            accu_total.item() / (step + 1),
            accu_ssim.item() / (step + 1),
            accu_max.item() / (step + 1),
            accu_color.item() / (step + 1),
            accu_text.item() / (step + 1),
            accu_attn.item() / (step + 1),
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
            accu_attn.item() / n_steps,
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
    loss_function = fusion_loss()
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

        I_fused = model(vis, ir, text)
        loss, loss_ssim, loss_max, loss_color, loss_text = \
            loss_function(vis_gt, ir_gt, I_fused, **FLIR_LOSS_WEIGHTS)

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
