"""Tests for sweeps/run_ablation.py: command construction + run_arm gating + aggregation."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.Text_IF_recon_model_2 import ABLATION_MODEL_KWARGS
from sweeps import run_ablation
from sweeps.run_ablation import (
    ARMS,
    ARM_LABELS,
    aggregate,
    build_detect_cmd,
    build_eval_cmd,
    build_train_cmd,
    run_arm,
)


def _make_args(stage="all"):
    return SimpleNamespace(
        stage=stage,
        base_weights="w.pth",
        dataset_root="dataset/LLVIP",
        epochs=30,
        sample=0,
        seed=42,
        device="xpu",
        ann_dir="ann",
        yolo_weights="y.pt",
        detect_device="auto",
    )


def test_arms_order_and_labels():
    assert ARMS == ["full", "no_fdblock", "no_dual_recon", "no_text",
                    "ff_feature_fusion", "unfreeze_encoder"]
    assert len(ARM_LABELS) == len(ARMS)


def test_arms_match_model_kwargs_vocabulary():
    # adding a 7th arm in one place only must fail loudly
    assert set(ARMS) == set(ABLATION_MODEL_KWARGS.keys())


def test_build_train_cmd():
    cmd = build_train_cmd(arm="no_text", repo=Path("/repo"), base_weights="w.pth",
                          epochs=30, out_root=Path("/out"))
    assert cmd[1:3] == ["-u", "/repo/train_finetune_v2ft.py"]
    joined = " ".join(cmd)
    assert "--ablation no_text" in joined
    assert "--weights w.pth" in joined
    assert "--output_dir /out/no_text/train" in joined


def test_build_eval_cmd():
    cmd = build_eval_cmd(arm="no_fdblock", repo=Path("/repo"),
                         data_path="dataset/LLVIP", out_root=Path("/out"),
                         sample=0, seed=42, device="xpu")
    joined = " ".join(cmd)
    assert "--ablation no_fdblock" in joined
    assert "--weights_path /out/no_fdblock/train/weights/checkpoint.pth" in joined
    assert "--output_dir /out/no_fdblock/eval" in joined


def test_build_detect_cmd():
    cmd = build_detect_cmd(arm="full", repo=Path("/repo"), out_root=Path("/out"),
                           ann_dir="ann", yolo_weights="y.pt", device="auto")
    joined = " ".join(cmd)
    assert "--fused_dir /out/full/eval/fused" in joined
    assert "--output_dir /out/full/detection" in joined


def test_run_arm_skips_when_done_sentinel_exists(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_ablation, "_run",
                        lambda cmd, log: calls.append(cmd) or 0)
    # fully-completed arm: train sentinel + eval/detect summary markers
    done = tmp_path / "full" / "train" / ".done"
    done.parent.mkdir(parents=True)
    done.write_text("2026-08-30T00:00:00\n")
    (tmp_path / "full" / "eval").mkdir(parents=True)
    (tmp_path / "full" / "eval" / "evaluation_summary.csv").write_text("x\n")
    (tmp_path / "full" / "detection").mkdir(parents=True)
    (tmp_path / "full" / "detection" / "detection_summary.csv").write_text("x\n")
    rc = run_arm("full", _make_args(stage="all"), Path("/repo"), tmp_path)
    assert rc == 0
    assert calls == []  # train, eval, detect all skip


def test_run_arm_train_failure_shortcircuits(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, log):
        calls.append(cmd)
        # fail only the train command
        return 1 if "train_finetune_v2ft.py" in " ".join(cmd) else 0

    monkeypatch.setattr(run_ablation, "_run", fake_run)
    rc = run_arm("no_text", _make_args(stage="all"), Path("/repo"), tmp_path)
    assert rc != 0
    assert len(calls) == 1  # eval/detect never launched
    assert (tmp_path / "no_text" / "train" / ".done").exists() is False


def test_run_arm_interrupted_train_reruns(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_ablation, "_run",
                        lambda cmd, log: calls.append(cmd) or 0)
    # interrupted run left a checkpoint but no .done sentinel
    ckpt = tmp_path / "full" / "train" / "weights" / "checkpoint.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"partial")
    rc = run_arm("full", _make_args(stage="all"), Path("/repo"), tmp_path)
    assert rc == 0
    assert any("train_finetune_v2ft.py" in " ".join(c) for c in calls)
    # successful train wrote the sentinel
    assert (tmp_path / "full" / "train" / ".done").is_file()


def test_aggregate_writes_table(tmp_path):
    # fabricate one arm's eval + detection outputs
    arm_dir = tmp_path / "full"
    (arm_dir / "eval").mkdir(parents=True)
    (arm_dir / "eval" / "evaluation_summary.csv").write_text(
        "metric,average\nEN,7.30\nVIF,0.90\nSSIM,0.81\nSF,16.7\nQabf,0.62\n")
    det = arm_dir / "detection"
    det.mkdir(parents=True)
    (det / "detection_summary.csv").write_text("Metric,Value\nmAP@0.5,0.7556\n")

    aggregate(tmp_path)

    md = (tmp_path / "ablation_table.md").read_text(encoding="utf-8")
    assert "| 配置 | EN | VIF | SSIM | SF | Qabf | mAP@0.5 |" in md
    assert "完整方法" in md and "0.7556" in md
    # missing arms appear as failed rows
    assert "no_text" in md
    csv = (tmp_path / "ablation_table.csv").read_text(encoding="utf-8")
    assert "full" in csv
