"""Tests for evaluate_textif_full_recon_v2.validate_checkpoint_arm (pure helper)."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluate_textif_full_recon_v2 import validate_checkpoint_arm


def test_namespace_args_match_passes():
    ckpt = {"model": {}, "args": SimpleNamespace(ablation="no_text")}
    assert validate_checkpoint_arm(ckpt, "no_text") == "no_text"


def test_dict_args_match_passes():
    ckpt = {"model": {}, "args": {"ablation": "full"}}
    assert validate_checkpoint_arm(ckpt, "full") == "full"


def test_legacy_checkpoint_without_args_is_skipped():
    # legacy checkpoints saved before the arm was recorded must not raise
    assert validate_checkpoint_arm({"model": {}}, "full") is None
    assert validate_checkpoint_arm({"model": {}, "args": None}, "full") is None
    # plain state_dict (not a dict-with-args checkpoint) is also skipped
    assert validate_checkpoint_arm({"layer.weight": []}, "full") is None


def test_args_without_ablation_field_is_skipped():
    ckpt = {"model": {}, "args": SimpleNamespace(epochs=30)}
    assert validate_checkpoint_arm(ckpt, "full") is None
    ckpt = {"model": {}, "args": {"epochs": 30}}
    assert validate_checkpoint_arm(ckpt, "full") is None


def test_mismatch_raises():
    ckpt = {"model": {}, "args": SimpleNamespace(ablation="no_text")}
    with pytest.raises(ValueError, match="no_text"):
        validate_checkpoint_arm(ckpt, "full")
    ckpt = {"model": {}, "args": {"ablation": "full"}}
    with pytest.raises(ValueError, match="full"):
        validate_checkpoint_arm(ckpt, "unfreeze_encoder")
