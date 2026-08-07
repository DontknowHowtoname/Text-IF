"""Smoke tests for train_one_epoch_replay.

These tests use a tiny in-memory model and fake dataloaders to validate the
plumbing (no CUDA required). Loss semantics are not checked here — only that
the function runs end-to-end, accumulates losses, and respects replay_ratio.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import pytest


def _stub_loss(*args, **kwargs):
    """Stub loss that returns 6 finite scalars regardless of input.

    The total loss has requires_grad=True so .backward() works in the
    training loop. This avoids needing the real fusion_dual_recon_prompt_loss
    which hardcodes .cuda() internally (L_Grad_position).
    """
    def _forward(I_A_gt, I_B_gt, fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis, task):
        return (torch.tensor(1.0, requires_grad=True),
                torch.tensor(0.5), torch.tensor(0.5),
                torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5))
    class _Stub:
        def __call__(self, *a, **k): return _forward(*a, **k)
        def to(self, device): return self
    return _Stub()


@pytest.fixture(autouse=True)
def _patch_loss(monkeypatch):
    """Replace fusion_dual_recon_prompt_loss with a stub for all tests in this module."""
    import scripts.utils as su
    monkeypatch.setattr(su, "fusion_dual_recon_prompt_loss", _stub_loss)


class _TinyModel(nn.Module):
    """Returns 5 tensors like Text_IF_Recon (fused + 4 recon heads)."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Conv2d(3, 3, 1)
    def forward(self, I_A, I_B, text):
        # text arg is accepted but ignored in this stub.
        f = self.linear(I_A)
        return f, f.clone(), f.clone(), f.clone(), f.clone()


def _make_loader(n_batches, batch_size=2, H=32, W=32):
    """Fake loader yielding (I_A, I_B, I_A_gt, I_B_gt, I_full, task_tuple, name_tuple)."""
    batches = []
    for _ in range(n_batches):
        I_A = torch.rand(batch_size, 3, H, W)
        I_B = torch.rand(batch_size, 3, H, W)
        I_A_gt = I_A.clone()
        I_B_gt = I_B.clone()
        I_full = I_A.clone()
        task = ("generic",) * batch_size
        name = ("x",) * batch_size
        batches.append((I_A, I_B, I_A_gt, I_B_gt, I_full, task, name))
    return batches


def test_train_one_epoch_replay_replay_off_runs():
    """With replay_ratio=0.0 and ems_loader=None, must run without error and
    return a 7-tuple of finite floats."""
    from scripts.utils import train_one_epoch_replay

    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target_loader = _make_loader(n_batches=3, batch_size=2, H=32, W=32)
    class _FakeSched:
        def step(self): pass
    scheduler = _FakeSched()

    out = train_one_epoch_replay(
        model=model,
        model_clip=None,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        target_loader=target_loader,
        ems_loader=None,
        replay_ratio=0.0,
        device=torch.device("cpu"),
        epoch=0,
        recon_weight=0.3,
        max_ratio=None,
        ssim_ratio=None,
        text_ratio=None,
    )
    assert len(out) == 7, f"Expected 7-tuple, got {len(out)}"
    for v in out:
        assert isinstance(v, (int, float)), f"Expected scalar, got {type(v)}"
        assert torch.isfinite(torch.tensor(float(v))), f"Non-finite value: {v}"


def test_train_one_epoch_replay_with_ems_runs():
    """With an EMS loader provided and replay_ratio=1.0, every step must also
    consume an EMS batch (no crash). The function must still return finite
    scalars."""
    from scripts.utils import train_one_epoch_replay

    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target_loader = _make_loader(n_batches=2, batch_size=2, H=32, W=32)
    ems_batches = []
    for _ in range(5):
        I_A = torch.rand(2, 3, 32, 32)
        I_B = torch.rand(2, 3, 32, 32)
        I_A_gt = I_A.clone(); I_B_gt = I_B.clone(); I_full = I_A.clone()
        task = ("low_light", "low_light")
        name = ("e", "e")
        ems_batches.append((I_A, I_B, I_A_gt, I_B_gt, I_full, task, name))

    class _FakeSched:
        def step(self): pass
    scheduler = _FakeSched()

    out = train_one_epoch_replay(
        model=model, model_clip=None, optimizer=optimizer, lr_scheduler=scheduler,
        target_loader=target_loader, ems_loader=ems_batches, replay_ratio=1.0,
        device=torch.device("cpu"), epoch=0,
        recon_weight=0.3, max_ratio=None, ssim_ratio=None, text_ratio=None,
    )
    assert len(out) == 7
    for v in out:
        assert torch.isfinite(torch.tensor(float(v))), f"Non-finite: {v}"


def test_evaluate_replay_runs_on_target_only():
    """evaluate_replay must run on a target val loader, using generic prompt,
    and return a 6-tuple of finite scalars."""
    from scripts.utils import evaluate_replay

    model = _TinyModel()
    val_loader = _make_loader(n_batches=2, batch_size=1, H=32, W=32)

    out = evaluate_replay(
        model=model,
        data_loader=val_loader,
        device=torch.device("cpu"),
        epoch=0, lr=1e-4,
        filefold_path=None,  # pass None to skip image saving
        max_ratio=None, ssim_ratio=None, text_ratio=None,
    )
    assert len(out) == 6, f"Expected 6-tuple, got {len(out)}"
    for v in out:
        assert torch.isfinite(torch.tensor(float(v))), f"Non-finite: {v}"
