"""Unit tests for sweeps/run_sweep.py.

Tests cover pure-logic pieces (text_ratios parsing, path validation).
Subprocess and sweep-loop behavior are NOT tested here — they require
GPU + datasets and are verified via HPC smoke tests documented in
sweeps/README.md.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sweeps"))


def test_parse_text_ratios_basic():
    from run_sweep import _parse_text_ratios
    assert _parse_text_ratios("0,1,2,3,5,8,10,15") == [0, 1, 2, 3, 5, 8, 10, 15]


def test_parse_text_ratios_single():
    from run_sweep import _parse_text_ratios
    assert _parse_text_ratios("5") == [5]


def test_parse_text_ratios_strips_whitespace():
    from run_sweep import _parse_text_ratios
    assert _parse_text_ratios(" 0 , 1 , 2 ") == [0, 1, 2]


def test_parse_text_ratios_rejects_garbage():
    from run_sweep import _parse_text_ratios
    with pytest.raises(ValueError):
        _parse_text_ratios("0,abc,2")


def test_parse_text_ratios_rejects_empty():
    from run_sweep import _parse_text_ratios
    with pytest.raises(ValueError):
        _parse_text_ratios("")


def test_validate_paths_all_exist(tmp_path):
    """When all required paths exist, _validate_paths returns without raising."""
    from run_sweep import _validate_paths

    # Build fake repo_dir with the two scripts
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train_fusion_full_recon_v2_ft.py").touch()
    (repo / "evaluate_textif_full_recon_v2.py").touch()

    # Build fake required files/dirs
    weights = tmp_path / "weights.pth"
    weights.touch()
    for name in ["ll", "oe", "ic", "in", "eval"]:
        (tmp_path / name).mkdir()

    # Should not raise
    _validate_paths(
        repo_dir=str(repo),
        pretrained_weights=str(weights),
        dataset_ll=str(tmp_path / "ll"),
        dataset_oe=str(tmp_path / "oe"),
        dataset_ic=str(tmp_path / "ic"),
        dataset_in=str(tmp_path / "in"),
        eval_data_path=str(tmp_path / "eval"),
    )


def test_validate_paths_missing_weights_exits(tmp_path, capsys):
    """Missing pretrained_weights file → SystemExit with code 2."""
    from run_sweep import _validate_paths

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train_fusion_full_recon_v2_ft.py").touch()
    (repo / "evaluate_textif_full_recon_v2.py").touch()
    for name in ["ll", "oe", "ic", "in", "eval"]:
        (tmp_path / name).mkdir()

    with pytest.raises(SystemExit) as exc:
        _validate_paths(
            repo_dir=str(repo),
            pretrained_weights=str(tmp_path / "nonexistent.pth"),
            dataset_ll=str(tmp_path / "ll"),
            dataset_oe=str(tmp_path / "oe"),
            dataset_ic=str(tmp_path / "ic"),
            dataset_in=str(tmp_path / "in"),
            eval_data_path=str(tmp_path / "eval"),
        )
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "pretrained_weights" in captured.err
    assert "nonexistent.pth" in captured.err


def test_validate_paths_missing_repo_script_exits(tmp_path, capsys):
    """repo_dir without the training script → SystemExit with code 2."""
    from run_sweep import _validate_paths

    repo = tmp_path / "repo"
    repo.mkdir()
    # Missing: train_fusion_full_recon_v2_ft.py
    (repo / "evaluate_textif_full_recon_v2.py").touch()

    weights = tmp_path / "weights.pth"
    weights.touch()
    for name in ["ll", "oe", "ic", "in", "eval"]:
        (tmp_path / name).mkdir()

    with pytest.raises(SystemExit) as exc:
        _validate_paths(
            repo_dir=str(repo),
            pretrained_weights=str(weights),
            dataset_ll=str(tmp_path / "ll"),
            dataset_oe=str(tmp_path / "oe"),
            dataset_ic=str(tmp_path / "ic"),
            dataset_in=str(tmp_path / "in"),
            eval_data_path=str(tmp_path / "eval"),
        )
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "train_fusion_full_recon_v2_ft.py" in captured.err
