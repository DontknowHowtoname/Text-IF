"""Full-pipeline sweep driver: train + eval + aggregate for each text_ratio.

Single Python invocation runs N text_ratio values serially (one GPU).
Fail-fast: any subprocess failure exits the whole sweep.

Usage:
    python sweeps/run_sweep.py \\
        --text_ratios 0,1,2,3,5,8,10,15 \\
        --repo_dir /path/to/Text-IF \\
        --pretrained_weights /path/to/ckpt.pth \\
        --dataset_ll ... --dataset_oe ... --dataset_ic ... --dataset_in ... \\
        --eval_data_path ...
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args. Pass argv=None to read sys.argv[1:]."""
    p = argparse.ArgumentParser(
        description="Serial full-pipeline sweep over text_ratio values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--text_ratios", type=str, required=True,
                   help="Comma-separated text_ratio values, e.g. '0,1,2,3,5,8,10,15'")
    p.add_argument("--repo_dir", type=str, required=True,
                   help="Repository root (must contain train_fusion_full_recon_v2_ft.py "
                        "and evaluate_textif_full_recon_v2.py)")
    p.add_argument("--pretrained_weights", type=str, required=True,
                   help="Path to textif-me pretrained checkpoint (.pth)")
    p.add_argument("--dataset_ll", type=str, required=True,
                   help="EMS_lite/Low_light directory")
    p.add_argument("--dataset_oe", type=str, required=True,
                   help="EMS_lite/Over_exposure directory")
    p.add_argument("--dataset_ic", type=str, required=True,
                   help="EMS_lite/IR_Low_contrast directory")
    p.add_argument("--dataset_in", type=str, required=True,
                   help="EMS_lite/IR_Noise directory")
    p.add_argument("--eval_data_path", type=str, required=True,
                   help="Evaluation dataset path (e.g. data/IVT_test)")
    p.add_argument("--output_root", type=str, default="sweeps/out",
                   help="Parent dir for text_ratio_T<T>/ output subdirs")
    p.add_argument("--val_every_epcho", type=int, default=1,
                   help="Forwarded to training script. Default 1 ensures checkpoint "
                        "is written at epoch 1 (see parent spec C1).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Forwarded to training script. None = use script default (50).")
    return p.parse_args(argv)


def _parse_text_ratios(s: str) -> List[int]:
    """Parse '0,1,2,3,5,8,10,15' -> [0, 1, 2, 3, 5, 8, 10, 15].

    Raises ValueError on empty string or non-integer tokens.
    """
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    if not tokens:
        raise ValueError(f"empty text_ratios list: {s!r}")
    out = []
    for t in tokens:
        try:
            out.append(int(t))
        except ValueError:
            raise ValueError(f"non-integer text_ratio token: {t!r}") from None
    return out


def _validate_paths(repo_dir: str, pretrained_weights: str,
                    dataset_ll: str, dataset_oe: str,
                    dataset_ic: str, dataset_in: str,
                    eval_data_path: str) -> None:
    """Verify all required paths exist. Prints offending path to stderr and
    raises SystemExit(2) on any miss."""
    missing = []

    repo_path = Path(repo_dir)
    if not repo_path.is_dir():
        missing.append(f"repo_dir (not a directory): {repo_dir}")
    else:
        for script in ("train_fusion_full_recon_v2_ft.py",
                       "evaluate_textif_full_recon_v2.py"):
            sp = repo_path / script
            if not sp.is_file():
                missing.append(f"repo_dir missing script: {sp}")

    if not Path(pretrained_weights).is_file():
        missing.append(f"pretrained_weights (not a file): {pretrained_weights}")

    for label, p in [("dataset_ll", dataset_ll), ("dataset_oe", dataset_oe),
                     ("dataset_ic", dataset_ic), ("dataset_in", dataset_in),
                     ("eval_data_path", eval_data_path)]:
        if not Path(p).is_dir():
            missing.append(f"{label} (not a directory): {p}")

    if missing:
        for m in missing:
            print(f"ERROR: {m}", file=sys.stderr)
        raise SystemExit(2)


def _run_subprocess_with_log(cmd: List[str], log_path: Path,
                             cwd: str, label: str) -> int:
    """Run cmd as subprocess, tee stdout+stderr to log_path AND to our own
    stdout. Returns the subprocess return code.

    Uses Python -u equivalent (env PYTHONUNBUFFERED=1) so output flushes
    promptly — important when running under SLURM where buffered output
    would hide progress for minutes.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[{label}] cmd: {' '.join(cmd)}")
    print(f"[{label}] log: {log_path}")

    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        proc.wait()
        return proc.returncode


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns process exit code."""
    args = parse_args(argv)
    text_ratios = _parse_text_ratios(args.text_ratios)
    _validate_paths(
        repo_dir=args.repo_dir,
        pretrained_weights=args.pretrained_weights,
        dataset_ll=args.dataset_ll,
        dataset_oe=args.dataset_oe,
        dataset_ic=args.dataset_ic,
        dataset_in=args.dataset_in,
        eval_data_path=args.eval_data_path,
    )

    repo = Path(args.repo_dir)
    output_root = Path(args.output_root)
    train_script = repo / "train_fusion_full_recon_v2_ft.py"
    eval_script = repo / "evaluate_textif_full_recon_v2.py"

    print(f"[sweep] text_ratios={text_ratios}")
    print(f"[sweep] output_root={output_root}")
    print(f"[sweep] starting serial sweep ({len(text_ratios)} values)")

    for T in text_ratios:
        run_dir = output_root / f"text_ratio_T{T}"
        train_dir = run_dir / "train"
        metrics_dir = run_dir / "metrics"
        train_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        weights_path = train_dir / "weights" / "checkpoint.pth"

        # Skip if already done (idempotent re-run)
        summary_csv = metrics_dir / "evaluation_summary.csv"
        if summary_csv.is_file():
            print(f"[sweep] T={T} already evaluated at {summary_csv}, skipping")
            continue

        # --- Train ---
        train_cmd = [
            sys.executable, "-u", str(train_script),
            "--text_ratio", str(T),
            "--weights", args.pretrained_weights,
            "--low_light_path", args.dataset_ll,
            "--over_exposure_path", args.dataset_oe,
            "--ir_low_contrast_path", args.dataset_ic,
            "--ir_noise_path", args.dataset_in,
            "--val_every_epcho", str(args.val_every_epcho),
            "--output_dir", str(train_dir),
        ]
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])

        print(f"\n[sweep] === T={T} | training ===")
        rc = _run_subprocess_with_log(
            cmd=train_cmd,
            log_path=run_dir / "train.log",
            cwd=str(repo),
            label=f"T{T}/train",
        )
        if rc != 0:
            print(f"[sweep] T={T} TRAIN FAILED (rc={rc}). "
                  f"Log: {run_dir / 'train.log'}", file=sys.stderr)
            print("[sweep] Aborting sweep (fail-fast).", file=sys.stderr)
            return 1

        if not weights_path.is_file():
            print(f"[sweep] T={T} train OK but weights file missing: "
                  f"{weights_path}", file=sys.stderr)
            return 1

        # --- Evaluate ---
        eval_cmd = [
            sys.executable, "-u", str(eval_script),
            "--weights_path", str(weights_path),
            "--data_path", args.eval_data_path,
            "--output_dir", str(metrics_dir),
        ]

        print(f"\n[sweep] === T={T} | evaluating ===")
        rc = _run_subprocess_with_log(
            cmd=eval_cmd,
            log_path=run_dir / "eval.log",
            cwd=str(repo),
            label=f"T{T}/eval",
        )
        if rc != 0:
            print(f"[sweep] T={T} EVAL FAILED (rc={rc}). "
                  f"Log: {run_dir / 'eval.log'}", file=sys.stderr)
            print("[sweep] Aborting sweep (fail-fast).", file=sys.stderr)
            return 1

        print(f"[sweep] T={T} done")

    # --- Aggregate ---
    # Import shared aggregator (same folder). sys.path already has sweeps/.
    try:
        from aggregate_sweep import aggregate
    except ImportError:
        # When invoked as `python sweeps/run_sweep.py`, sweeps/ may not be on path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from aggregate_sweep import aggregate

    summary_csv = repo / "sweeps" / "text_ratio_sweep_summary.csv"
    print(f"\n[sweep] aggregating -> {summary_csv}")
    aggregate(
        out_root=str(output_root),
        output_csv=str(summary_csv),
        expected_text_ratios=text_ratios,
    )
    print(f"[sweep] summary written: {summary_csv}")
    print("[sweep] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
