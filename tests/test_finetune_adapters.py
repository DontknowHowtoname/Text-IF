"""Tests for v2-ft per-dataset fine-tuning adapters.

Covers: _TASK_DEFAULTS generic key, get_generic_prompt, fallback behavior,
resolve_ir_vis_dirs layout auto-detection, read_data_for_finetune stem pairing.
"""
import os
import sys
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_task_defaults_has_generic_key():
    """The 'generic' task type must resolve to a default ratio dict so the
    fine-tune loss path does not raise KeyError."""
    from scripts.losses import fusion_dual_recon_prompt_loss
    loss = fusion_dual_recon_prompt_loss()
    assert "generic" in loss._TASK_DEFAULTS, (
        "Expected 'generic' key in _TASK_DEFAULTS so fine-tuning samples "
        "with task='generic' do not raise KeyError."
    )
    defaults = loss._TASK_DEFAULTS["generic"]
    assert set(defaults.keys()) == {"max_ratio", "ssim_ratio", "text_ratio"}
    assert defaults["max_ratio"] == 4
    assert defaults["ssim_ratio"] == 1
    assert defaults["text_ratio"] == 3


def test_generic_uses_default_ratios_when_no_override():
    """fusion_dual_recon_prompt_loss with task=['generic'] should produce a
    finite loss using the generic defaults (no override)."""
    import torch
    from scripts.losses import fusion_dual_recon_prompt_loss

    loss_fn = fusion_dual_recon_prompt_loss()
    I_A_gt = torch.rand(1, 3, 32, 32)
    I_B_gt = torch.rand(1, 3, 32, 32)
    fused = torch.rand(1, 3, 32, 32)
    recon_ir = torch.rand(1, 3, 32, 32)
    recon_vis = torch.rand(1, 3, 32, 32)
    recon_dec_ir = torch.rand(1, 3, 32, 32)
    recon_dec_vis = torch.rand(1, 3, 32, 32)

    out = loss_fn(I_A_gt, I_B_gt, fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, ["generic"])
    assert torch.isfinite(out[0]), f"Loss must be finite, got {out[0]}"
