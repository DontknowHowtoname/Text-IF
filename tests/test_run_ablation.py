"""Tests for sweeps/run_ablation.py: command construction + table aggregation."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

from sweeps.run_ablation import (
    ARMS, ARM_LABELS, build_train_cmd, build_eval_cmd, build_detect_cmd, aggregate,
)


def test_arms_order_and_labels():
    assert ARMS == ["full", "no_fdblock", "no_dual_recon", "no_text",
                    "ff_feature_fusion", "unfreeze_encoder"]
    assert len(ARM_LABELS) == len(ARMS)


def test_build_train_cmd():
    cmd = build_train_cmd(arm="no_text", repo=Path("/repo"), base_weights="w.pth",
                          dataset_root="dataset/LLVIP", epochs=30, out_root=Path("/out"))
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
                           ann_dir="ann", yolo_weights="yolo.pt", device="auto")
    joined = " ".join(cmd)
    assert "--fused_dir /out/full/eval/fused" in joined
    assert "--output_dir /out/full/detection" in joined


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
