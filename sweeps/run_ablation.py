"""§4.3 ablation pipeline: train -> fusion metrics -> YOLO mAP, per arm.

Idempotent: each step is skipped when its completion marker already exists
(train: .done sentinel written only after a successful train subprocess;
eval/detect: their summary CSVs), so the script is safe to re-run after
partial failures.

Example (CUDA training box):
    python sweeps/run_ablation.py \
        --base_weights experiments/TextIF_train_20260408-185710/weights/checkpoint.pth \
        --yolo_weights <yolov5m.pt> --device cuda --epochs 30

Note: training requires CUDA (scripts/losses.py calls .cuda()); XPU boxes
should only run --stage eval/detect/aggregate on already-trained arms.
"""
import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# §4.3 table row order
ARMS = ["full", "no_fdblock", "no_dual_recon", "no_text",
        "ff_feature_fusion", "unfreeze_encoder"]
ARM_LABELS = {
    "full":             "完整方法",
    "no_fdblock":       "- 去除 FDBlock（无双路径解耦）",
    "no_dual_recon":    "- 去除 DualReconLoss（仅融合损失）",
    "no_text":          "- 去除文本 prompt（task=None）",
    "ff_feature_fusion": "- FFBlock 换回 feature_fusion",
    "unfreeze_encoder": "- 不冻结编码器（全量微调）",
}
TABLE_METRICS = ["EN", "VIF", "SSIM", "SF", "Qabf"]
TABLE_HEADER = ["配置"] + TABLE_METRICS + ["mAP@0.5"]


def _p(path) -> str:
    """Posix-style string for a Path (forward slashes work on Windows too,
    and keep commands deterministic across platforms)."""
    return Path(path).as_posix()


def build_train_cmd(arm, repo, base_weights, epochs, out_root):
    return [sys.executable, "-u", _p(repo / "train_finetune_v2ft.py"),
            "--dataset_name", "LLVIP", "--model_version", "v2",
            "--ablation", arm,
            "--weights", str(base_weights),
            "--epochs", str(epochs),
            "--output_dir", _p(out_root / arm / "train"),
            "--device", "cuda", "--gpu_id", "0"]


def build_eval_cmd(arm, repo, data_path, out_root, sample, seed, device):
    return [sys.executable, "-u", _p(repo / "evaluate_textif_full_recon_v2.py"),
            "--ablation", arm,
            "--weights_path", _p(out_root / arm / "train" / "weights" / "checkpoint.pth"),
            "--data_path", str(data_path),
            "--output_dir", _p(out_root / arm / "eval"),
            "--sample", str(sample), "--seed", str(seed),
            "--device", device]


def build_detect_cmd(arm, repo, out_root, ann_dir, yolo_weights, device):
    return [sys.executable, "-u", _p(repo / "eval_detection_yolov5.py"),
            "--fused_dir", _p(out_root / arm / "eval" / "fused"),
            "--ann_dir", str(ann_dir),
            "--weights", str(yolo_weights),
            "--output_dir", _p(out_root / arm / "detection"),
            "--device", device]


def _run(cmd, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("  $ " + " ".join(cmd))
    # append: a re-run of a failed arm must not destroy the failure evidence
    with log_path.open("a") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=logf,
                              stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"  FAILED rc={proc.returncode}, see {log_path}", file=sys.stderr)
    return proc.returncode


def run_arm(arm, args, repo, out_root) -> int:
    # Train completion is gated on a sentinel written ONLY after a successful
    # train subprocess. checkpoint.pth appears mid-training (saved on every val
    # improvement from ~epoch 2), so gating on it would silently skip
    # re-training after an interrupted run and evaluate a partial model.
    done = out_root / arm / "train" / ".done"
    summary = out_root / arm / "eval" / "evaluation_summary.csv"
    det = out_root / arm / "detection" / "detection_summary.csv"

    rc = 0
    if args.stage in ("all", "train"):
        if done.is_file():
            print(f"[{arm}] skip train (done): {done}")
        else:
            print(f"[{arm}] train ...")
            rc |= _run(build_train_cmd(arm, repo, args.base_weights,
                                       args.epochs, out_root),
                       out_root / arm / "train.log")
            if rc == 0:
                done.parent.mkdir(parents=True, exist_ok=True)
                done.write_text(
                    datetime.now().isoformat(timespec="seconds") + "\n",
                    encoding="utf-8")
    if rc:
        return rc

    if args.stage in ("all", "eval"):
        if summary.is_file():
            print(f"[{arm}] skip eval (exists): {summary}")
        else:
            print(f"[{arm}] eval ...")
            rc |= _run(build_eval_cmd(arm, repo, args.dataset_root, out_root,
                                      args.sample, args.seed, args.device),
                       out_root / arm / "eval.log")

    if args.stage in ("all", "detect") and not rc:
        if det.is_file():
            print(f"[{arm}] skip detect (exists): {det}")
        else:
            print(f"[{arm}] detect ...")
            rc |= _run(build_detect_cmd(arm, repo, out_root, args.ann_dir,
                                        args.yolo_weights, args.detect_device),
                       out_root / arm / "detect.log")
    return rc


def aggregate(out_root: Path) -> Path:
    """Read per-arm summary CSVs -> ablation_table.{csv,md} (§4.3 format)."""
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ARMS:
        row = {"配置": ARM_LABELS[arm]}
        summary = out_root / arm / "eval" / "evaluation_summary.csv"
        if summary.is_file():
            with summary.open(encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if r["metric"] in TABLE_METRICS:
                        row[r["metric"]] = float(r["average"])
        det = out_root / arm / "detection" / "detection_summary.csv"
        if det.is_file():
            with det.open(encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if r["Metric"] == "mAP@0.5":
                        row["mAP@0.5"] = float(r["Value"])
        rows.append(row)

    def _fmt(v, nd=3):
        if v is None or v == "":
            return "—"
        return f"{v:.{nd}f}"

    def _cells(row):
        return [_fmt(row.get(k)) for k in TABLE_METRICS] + \
               [_fmt(row.get("mAP@0.5"), nd=4)]

    csv_path = out_root / "ablation_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arm"] + TABLE_HEADER)
        for arm, row in zip(ARMS, rows):
            w.writerow([arm] + _cells(row))
    print(f"wrote {csv_path}")

    md_path = out_root / "ablation_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(TABLE_HEADER) + " |\n")
        f.write("|" + "---|" * len(TABLE_HEADER) + "\n")
        for arm, row in zip(ARMS, rows):
            # include the arm id so missing/failed arms are identifiable in the table
            cells = [f"{row['配置']} ({arm})"] + _cells(row)
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"wrote {md_path}")
    return md_path


def main() -> None:
    ap = argparse.ArgumentParser(description="§4.3 ablation pipeline (train+eval+detect)")
    ap.add_argument("--base_weights", type=str, required=True,
                    help="Shared v2-ft checkpoint all arms fine-tune from")
    ap.add_argument("--dataset_root", type=str, default="dataset/LLVIP",
                    help="LLVIP root ({infrared,visible}/{train,test}); used by "
                         "the EVAL stage only (training resolves LLVIP via its "
                         "own DATASET_CONFIGS)")
    ap.add_argument("--ann_dir", type=str,
                    default="D:/StudyFiles/MachineLearning/datasets/LLVIP/Annotations",
                    help="LLVIP VOC annotations for detection eval")
    ap.add_argument("--yolo_weights", type=str, required=True,
                    help="Zero-shot YOLOv5m weights (.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--sample", type=int, default=0, help="0 = full test split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="xpu",
                    help="Device for the fusion eval script")
    ap.add_argument("--detect_device", type=str, default="auto",
                    help="Device for eval_detection_yolov5.py")
    ap.add_argument("--out_root", type=str, default=None,
                    help="Default: <repo>/sweeps/out/ablation")
    ap.add_argument("--arms", type=str, default=None,
                    help="Comma-separated subset of arms to run")
    ap.add_argument("--stage", type=str, default="all",
                    choices=["all", "train", "eval", "detect", "aggregate"],
                    help="Run a single pipeline stage (aggregate always runs at end)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without running")
    args = ap.parse_args()

    repo = REPO
    out_root = Path(args.out_root) if args.out_root else repo / "sweeps" / "out" / "ablation"
    arms = ([s.strip() for s in args.arms.split(",")] if args.arms else ARMS)
    # fail fast: validate all arm names before launching any subprocess,
    # so a typo cannot waste hours of training before exiting
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        print(f"ERROR: unknown arms {unknown} (valid: {ARMS})", file=sys.stderr)
        sys.exit(2)

    failures = []
    for arm in arms:
        if args.dry_run:
            if args.stage in ("all", "train"):
                print(" ".join(build_train_cmd(arm, repo, args.base_weights,
                                               args.epochs, out_root)))
            if args.stage in ("all", "eval"):
                print(" ".join(build_eval_cmd(arm, repo, args.dataset_root, out_root,
                                              args.sample, args.seed, args.device)))
            if args.stage in ("all", "detect"):
                print(" ".join(build_detect_cmd(arm, repo, out_root, args.ann_dir,
                                                args.yolo_weights, args.detect_device)))
            continue
        if run_arm(arm, args, repo, out_root) != 0:
            failures.append(arm)

    if args.dry_run:
        return  # no side effects in dry-run mode
    # aggregate always runs at the end of a real run, refreshing the table
    # from whatever summaries already exist (missing arms become "—" rows)
    md = aggregate(out_root)
    print(f"\nDone. Table: {md}")
    if failures:
        print(f"FAILED arms: {failures}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
