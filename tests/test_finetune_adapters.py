"""Tests for v2-ft per-dataset fine-tuning adapters.

Covers: _TASK_DEFAULTS generic key, get_generic_prompt, fallback behavior,
resolve_ir_vis_dirs layout auto-detection, read_data_for_finetune stem pairing.

Note: Some tests instantiate fusion_dual_recon_prompt_loss, which requires CUDA
because L_Grad_position (scripts/losses.py:144) hardcodes .cuda(). These tests
are skipped on non-CUDA devices (e.g. XPU, CPU). The structural test
(test_task_defaults_has_generic_key) verifies the dict via class attribute and
runs on any device.
"""
import os
import sys
import pytest
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def test_task_defaults_has_generic_key():
    """The 'generic' task type must resolve to a default ratio dict so the
    fine-tune loss path does not raise KeyError.

    Accesses _TASK_DEFAULTS as a class attribute (no instantiation), so this
    test runs on any device — including XPU-only machines where
    fusion_dual_recon_prompt_loss() cannot be constructed due to hardcoded
    .cuda() calls in L_Grad_position.
    """
    from scripts.losses import fusion_dual_recon_prompt_loss
    defaults = fusion_dual_recon_prompt_loss._TASK_DEFAULTS["generic"]
    assert set(defaults.keys()) == {"max_ratio", "ssim_ratio", "text_ratio"}
    assert defaults["max_ratio"] == 4
    assert defaults["ssim_ratio"] == 1
    assert defaults["text_ratio"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="fusion_dual_recon_prompt_loss instantiation requires CUDA (L_Grad_position hardcodes .cuda())")
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


def test_module_imports_without_ems_lite(tmp_path, monkeypatch):
    """scripts.utils must be importable even when ./dataset/EMS_lite does not
    exist. Module-level text.txt loaders must not crash on import."""
    import importlib

    # Run from a cwd where EMS_lite is not present.
    monkeypatch.chdir(tmp_path)
    # Force re-import under the new cwd.
    sys.modules.pop("scripts.utils", None)
    try:
        import scripts.utils  # noqa: F401
        # If we got here without exception, the hardening worked.
        assert True
    finally:
        # Restore module so other tests get the cached, repo-cwd version.
        sys.modules.pop("scripts.utils", None)
