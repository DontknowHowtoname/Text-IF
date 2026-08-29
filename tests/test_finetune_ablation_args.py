"""Tests for resolve_ablation() pure helper in train_finetune_v2ft.py."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from types import SimpleNamespace
from train_finetune_v2ft import resolve_ablation


def _args(ablation, recon_weight=0.3):
    return SimpleNamespace(ablation=ablation, recon_weight=recon_weight)


def test_full_is_noop():
    r = resolve_ablation(_args("full"))
    assert r["model_kwargs"] == {}
    assert r["freeze_encoders"] is True
    assert r["recon_weight"] == 0.3


def test_no_dual_recon_forces_zero_recon_weight():
    r = resolve_ablation(_args("no_dual_recon"))
    assert r["recon_weight"] == 0.0
    assert r["model_kwargs"] == {}
    assert r["freeze_encoders"] is True


def test_unfreeze_encoder_skips_freeze():
    r = resolve_ablation(_args("unfreeze_encoder"))
    assert r["freeze_encoders"] is False


def test_model_ablations_map_kwargs():
    from model.Text_IF_recon_model_2 import ABLATION_MODEL_KWARGS
    for name in ("no_fdblock", "no_text", "ff_feature_fusion"):
        r = resolve_ablation(_args(name))
        assert r["model_kwargs"] == ABLATION_MODEL_KWARGS[name]
