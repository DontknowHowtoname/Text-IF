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


def test_run_subprocess_with_log_captures_output(tmp_path):
    """Run a stub Python script via _run_subprocess_with_log; verify log file
    captures stdout+stderr and returncode is propagated."""
    from run_sweep import _run_subprocess_with_log

    # Stub script that prints to stdout, stderr, and exits 0
    stub = tmp_path / "stub.py"
    stub.write_text(
        'import sys\n'
        'print("stdout line", flush=True)\n'
        'print("stderr line", file=sys.stderr, flush=True)\n'
        'sys.exit(0)\n',
        encoding="utf-8",
    )
    log_file = tmp_path / "stub.log"

    rc = _run_subprocess_with_log(
        cmd=[sys.executable, str(stub)],
        log_path=log_file,
        cwd=str(tmp_path),
        label="stub",
    )
    assert rc == 0
    assert log_file.is_file()
    contents = log_file.read_text(encoding="utf-8")
    assert "stdout line" in contents
    assert "stderr line" in contents


def test_run_subprocess_with_log_propagates_failure(tmp_path):
    """Stub script that exits 7 — returncode is propagated."""
    from run_sweep import _run_subprocess_with_log

    stub = tmp_path / "fail.py"
    stub.write_text(
        'import sys\n'
        'print("about to fail", flush=True)\n'
        'sys.exit(7)\n',
        encoding="utf-8",
    )
    log_file = tmp_path / "fail.log"

    rc = _run_subprocess_with_log(
        cmd=[sys.executable, str(stub)],
        log_path=log_file,
        cwd=str(tmp_path),
        label="fail",
    )
    assert rc == 7


def test_args_default_output_root():
    """--output_root defaults to 'sweeps/out' when not specified."""
    from run_sweep import parse_args

    # Minimal required argv
    argv = [
        "--text_ratios", "5",
        "--repo_dir", "/fake/repo",
        "--pretrained_weights", "/fake/weights.pth",
        "--dataset_ll", "/fake/ll",
        "--dataset_oe", "/fake/oe",
        "--dataset_ic", "/fake/ic",
        "--dataset_in", "/fake/in",
        "--eval_data_path", "/fake/eval",
    ]
    args = parse_args(argv)
    assert args.output_root == "sweeps/out"


def test_args_default_val_every_epcho():
    """--val_every_epcho defaults to 1.

    This pins down the C1 mitigation from the parent spec: default 2 would
    skip epoch 1 entirely, so checkpoint writing wouldn't start until epoch 2.
    If a future refactor changes this default, this test will catch it.
    """
    from run_sweep import parse_args

    argv = [
        "--text_ratios", "5",
        "--repo_dir", "/fake/repo",
        "--pretrained_weights", "/fake/weights.pth",
        "--dataset_ll", "/fake/ll",
        "--dataset_oe", "/fake/oe",
        "--dataset_ic", "/fake/ic",
        "--dataset_in", "/fake/in",
        "--eval_data_path", "/fake/eval",
    ]
    args = parse_args(argv)
    assert args.val_every_epcho == 1
