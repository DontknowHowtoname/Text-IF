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
    # Run from a cwd where EMS_lite is not present.
    monkeypatch.chdir(tmp_path)
    # Force re-import under the new cwd.
    sys.modules.pop("scripts.utils", None)
    try:
        import scripts.utils  # noqa: F401
        # If we got here without exception, the hardening worked.
        assert hasattr(scripts.utils, "low_light_lines"), \
            "module must expose low_light_lines after import"
    finally:
        # Restore module so other tests get the cached, repo-cwd version.
        sys.modules.pop("scripts.utils", None)


def test_get_generic_prompt_returns_constant_string():
    """get_generic_prompt() must return the fixed generic prompt string."""
    from scripts.utils import get_generic_prompt
    p = get_generic_prompt()
    assert isinstance(p, str)
    assert p == "This is the infrared-visible light fusion task."


def test_task_prompt_falls_back_to_generic_when_lines_empty(monkeypatch):
    """If low_light_lines is empty (EMS_lite missing), get_low_light_prompt()
    must fall back to the generic prompt instead of raising IndexError."""
    import scripts.utils as su
    monkeypatch.setattr(su, "low_light_lines", [])
    p = su.get_low_light_prompt()
    assert p == su.get_generic_prompt()


def test_task_prompt_returns_random_line_when_lines_present(monkeypatch):
    """When low_light_lines is populated, get_low_light_prompt returns a
    stripped random choice from that list."""
    import scripts.utils as su
    monkeypatch.setattr(su, "low_light_lines", ["  hello world  \n", "  another  \n"])
    # Force random.choice to deterministically return the first entry.
    monkeypatch.setattr(su.random, "choice", lambda seq: seq[0])
    p = su.get_low_light_prompt()
    assert p == "hello world"


def _make_image(path):
    """Write a 16x16 black PNG so file enumeration finds it."""
    from PIL import Image
    img = Image.new("RGB", (16, 16), (0, 0, 0))
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


@pytest.fixture
def tno_layout(tmp_path):
    """Layout 1: <root>/ir/ + <root>/vis/ with same-stem PNGs."""
    root = tmp_path / "TNO"
    _make_image(root / "ir" / "0001.png")
    _make_image(root / "vis" / "0001.png")
    _make_image(root / "ir" / "0002.png")
    _make_image(root / "vis" / "0002.png")
    return root


@pytest.fixture
def msrs_layout(tmp_path):
    """Layout 3: <root>/train/{ir,vi}/ + <root>/test/{ir,vi}/."""
    root = tmp_path / "MSRS"
    for split in ["train", "test"]:
        _make_image(root / split / "ir" / f"{split}_0.png")
        _make_image(root / split / "vi" / f"{split}_0.png")
    return root


@pytest.fixture
def llvip_layout(tmp_path):
    """Layout 2: <root>/infrared/ + <root>/visible/, each with train/test subdir."""
    root = tmp_path / "LLVIP"
    for split in ["train", "test"]:
        _make_image(root / "infrared" / split / f"img_{split}.png")
        _make_image(root / "visible" / split / f"img_{split}.png")
    return root


def test_resolve_ir_vis_dirs_tno_layout(tno_layout):
    from scripts.utils import resolve_ir_vis_dirs
    ir, vis = resolve_ir_vis_dirs(str(tno_layout), "train")
    assert os.path.basename(ir) == "ir"
    assert os.path.basename(vis) == "vis"


def test_resolve_ir_vis_dirs_msrs_train_split(msrs_layout):
    from scripts.utils import resolve_ir_vis_dirs
    ir, vis = resolve_ir_vis_dirs(str(msrs_layout), "train")
    # Should pick the train split.
    assert os.path.basename(os.path.dirname(ir)) == "train"
    assert os.path.basename(ir) == "ir"
    assert os.path.basename(vis) == "vi"


def test_resolve_ir_vis_dirs_llvip_train_split(llvip_layout):
    from scripts.utils import resolve_ir_vis_dirs
    ir, vis = resolve_ir_vis_dirs(str(llvip_layout), "train")
    assert ir.endswith(os.path.join("infrared", "train"))
    assert vis.endswith(os.path.join("visible", "train"))


def test_resolve_ir_vis_dirs_raises_when_unresolvable(tmp_path):
    from scripts.utils import resolve_ir_vis_dirs
    (tmp_path / "junk").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_ir_vis_dirs(str(tmp_path / "junk"), "train")


def test_read_data_for_finetune_pairs_by_stem(tno_layout):
    """read_data_for_finetune must pair ir/vis images by filename stem and
    return sorted lists of equal length."""
    from scripts.utils import read_data_for_finetune
    train_vis, train_ir, val_vis, val_ir = read_data_for_finetune(str(tno_layout))
    assert len(train_vis) == len(train_ir), "vis/ir must be paired 1:1"
    # With only 2 images and an 80/20 split, expect 2 train + 0 val OR
    # 1 train + 1 val depending on rounding; both are acceptable.
    assert len(train_vis) >= 1
    # Train and val should be disjoint
    train_stems = {os.path.splitext(os.path.basename(p))[0] for p in train_vis}
    val_stems = {os.path.splitext(os.path.basename(p))[0] for p in val_vis}
    assert train_stems.isdisjoint(val_stems), "train/val must not share stems"


def test_read_data_for_finetune_handles_mixed_extensions(tmp_path):
    """RoadScene-style: ir/*.png + vis/*.jpg, same stems."""
    from PIL import Image
    from scripts.utils import read_data_for_finetune

    root = tmp_path / "RoadScene"
    (root / "ir").mkdir(parents=True, exist_ok=True)
    (root / "vis").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16)).save(root / "ir" / "0001.png")
    Image.new("RGB", (16, 16)).save(root / "vis" / "0001.jpg")
    Image.new("RGB", (16, 16)).save(root / "ir" / "0002.png")
    Image.new("RGB", (16, 16)).save(root / "vis" / "0002.jpg")

    train_vis, train_ir, val_vis, val_ir = read_data_for_finetune(str(root))
    # Each pair should share a stem despite different extensions.
    for v, ir in zip(train_vis, train_ir):
        assert os.path.splitext(os.path.basename(v))[0] == \
               os.path.splitext(os.path.basename(ir))[0]


def test_read_data_for_finetune_uses_test_split_when_available(msrs_layout):
    """When <root>/train and <root>/test both exist, train->train, test->val."""
    from scripts.utils import read_data_for_finetune
    train_vis, train_ir, val_vis, val_ir = read_data_for_finetune(str(msrs_layout))
    # Train samples should come from train/ split
    assert all("train" in p for p in train_vis)
    # Val samples should come from test/ split (the eval alias)
    assert all("test" in p for p in val_vis) if val_vis else True
