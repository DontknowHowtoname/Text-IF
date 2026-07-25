# Python Sweep Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sweeps/run_sweep.py` — a CLI-driven Python script that runs the full text_ratio sweep (train + eval + aggregate) serially in one invocation, plus unit tests for its pure-logic parts.

**Architecture:** Single-file Python orchestrator using `argparse` + `subprocess`. Each `text_ratio` value triggers two `subprocess.run` calls (train script, then eval script) with stdout/stderr tee'd to per-run log files. Final aggregation reuses the existing `sweeps/aggregate_sweep.py:aggregate` function via direct import. Fail-fast: any subprocess returning non-zero exits the whole sweep with exit 1.

**Tech Stack:** Python 3 stdlib (`argparse`, `subprocess`, `sys`, `os`, `pathlib`, `shutil`), pytest for unit tests. No third-party deps.

**Spec:** [docs/superpowers/specs/2026-07-25-python-sweep-runner-design.md](../specs/2026-07-25-python-sweep-runner-design.md)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `sweeps/run_sweep.py` (new) | CLI parsing, path validation, sweep loop (train+eval subprocess), aggregation |
| `tests/test_run_sweep.py` (new) | Unit tests for CLI parsing & path validation (no subprocess, no real training) |
| `sweeps/README.md` (modify) | Document Python runner as an alternative to the bash+SLURM-array approach |

---

### Task 1: Scaffolding — argument parsing + skeleton

Set up the CLI surface and module structure. No sweep loop yet, just argparse + a `main()` that prints parsed args.

**Files:**
- Create: `sweeps/run_sweep.py`

- [ ] **Step 1: Write the file skeleton**

Create `sweeps/run_sweep.py` with this content:

```python
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


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns process exit code."""
    args = parse_args(argv)
    print(f"[sweep] parsed args: text_ratios={args.text_ratios!r}, "
          f"repo_dir={args.repo_dir!r}, output_root={args.output_root!r}")
    print("[sweep] (skeleton — sweep loop not implemented yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify `--help` works**

Run:
```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe sweeps/run_sweep.py --help
```

Expected: help text listing all 11 args (`--text_ratios`, `--repo_dir`, `--pretrained_weights`, `--dataset_ll`, `--dataset_oe`, `--dataset_ic`, `--dataset_in`, `--eval_data_path`, `--output_root`, `--val_every_epcho`, `--epochs`).

- [ ] **Step 3: Verify missing-required-arg fails with exit 2**

Run:
```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe sweeps/run_sweep.py 2>&1; echo "EXIT=$?"
```

Expected: argparse error about required args, `EXIT=2`.

- [ ] **Step 4: Commit**

```bash
git add sweeps/run_sweep.py
git commit -m "feat(sweep): scaffold run_sweep.py CLI (argparse + main skeleton)

11 args: 8 required (text_ratios, repo_dir, weights, 4 datasets,
eval_data_path) + 3 optional (output_root, val_every_epcho=1, epochs=None).
No sweep logic yet; main() just echoes parsed args."
```

---

### Task 2: TDD — text_ratios parsing + path validation

Write the failing tests for the two pure functions we need next: `_parse_text_ratios` and `_validate_paths`. Both live in `sweeps/run_sweep.py`.

**Files:**
- Create: `tests/test_run_sweep.py`
- Modify: `sweeps/run_sweep.py` (add the two functions, wire into `main()`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_sweep.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe -m pytest tests/test_run_sweep.py -v
```

Expected: `ImportError: cannot import name '_parse_text_ratios' from 'run_sweep'` (or similar). 0 passed, 8 errors.

- [ ] **Step 3: Implement `_parse_text_ratios` and `_validate_paths`**

Edit `sweeps/run_sweep.py`. **Add these two functions after `parse_args`** (before `main`):

```python
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
```

**Then edit `main`** to call both before printing the success message:

Replace:
```python
def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns process exit code."""
    args = parse_args(argv)
    print(f"[sweep] parsed args: text_ratios={args.text_ratios!r}, "
          f"repo_dir={args.repo_dir!r}, output_root={args.output_root!r}")
    print("[sweep] (skeleton — sweep loop not implemented yet)")
    return 0
```

with:

```python
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
    print(f"[sweep] text_ratios={text_ratios}, output_root={args.output_root!r}")
    print("[sweep] (sweep loop not implemented yet)")
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe -m pytest tests/test_run_sweep.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_run_sweep.py sweeps/run_sweep.py
git commit -m "feat(sweep): add text_ratios parsing and path validation (TDD)

_parse_text_ratios: comma-separated string -> List[int], rejects
empty/garbage with ValueError. _validate_paths: checks repo_dir contains
both training scripts, weights file exists, all 5 dataset/eval paths are
directories; on miss prints to stderr and exits with code 2.

8 unit tests covering happy paths, whitespace, garbage, missing weights,
missing repo script. No subprocess, no GPU."
```

---

### Task 3: Subprocess runner for a single (script, args) call

Encapsulate the subprocess + log-tee pattern in a helper that we'll call twice per text_ratio (train, then eval).

**Files:**
- Modify: `sweeps/run_sweep.py` (add `_run_subprocess_with_log`)
- Modify: `tests/test_run_sweep.py` (add one test using a stub Python script)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_sweep.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe -m pytest tests/test_run_sweep.py -v -k "run_subprocess"
```

Expected: ImportError on `_run_subprocess_with_log`, 2 errors.

- [ ] **Step 3: Implement `_run_subprocess_with_log`**

Edit `sweeps/run_sweep.py`. **Add this function after `_validate_paths`** (before `main`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe -m pytest tests/test_run_sweep.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add sweeps/run_sweep.py tests/test_run_sweep.py
git commit -m "feat(sweep): add _run_subprocess_with_log helper

Runs subprocess with stdout+stderr merged, teed line-by-line to both
our stdout and a per-run log file. Sets PYTHONUNBUFFERED=1 for prompt
flush under SLURM. Two tests: success path verifies log content;
failure path verifies returncode propagation."
```

---

### Task 4: Sweep loop + aggregation

Wire everything together: per-T train → eval → aggregate. Fail-fast on any subprocess failure.

**Files:**
- Modify: `sweeps/run_sweep.py` (rewrite `main` body)

- [ ] **Step 1: Read current `main`**

Verify current `main` shape (should still print "sweep loop not implemented yet"). Then replace it wholesale.

- [ ] **Step 2: Replace `main` with the full sweep loop**

Edit `sweeps/run_sweep.py`. Replace the existing `main` function with:

```python
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
```

- [ ] **Step 3: Verify `--help` still works**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe sweeps/run_sweep.py --help 2>&1 | head -5
```

Expected: usage line shown, no errors.

- [ ] **Step 4: Verify path validation still triggers**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe sweeps/run_sweep.py \
    --text_ratios 5 \
    --repo_dir /nonexistent \
    --pretrained_weights /nonexistent.pth \
    --dataset_ll /nonexistent --dataset_oe /nonexistent \
    --dataset_ic /nonexistent --dataset_in /nonexistent \
    --eval_data_path /nonexistent 2>&1; echo "EXIT=$?"
```

Expected: multiple `ERROR:` lines listing the missing paths, `EXIT=2`. No sweep starts.

- [ ] **Step 5: Re-run full unit test suite**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe -m pytest tests/test_run_sweep.py tests/test_aggregate_sweep.py -v
```

Expected: 10 + 3 = 13 passed. (No regressions in the existing aggregator tests.)

- [ ] **Step 6: Commit**

```bash
git add sweeps/run_sweep.py
git commit -m "feat(sweep): wire full sweep loop with fail-fast + aggregation

main() now:
  1. parse + validate
  2. for each T in text_ratios:
     a. skip if evaluation_summary.csv already exists (idempotent re-run)
     b. subprocess: train_fusion_full_recon_v2_ft.py --text_ratio T
     c. assert weights/checkpoint.pth exists
     d. subprocess: evaluate_textif_full_recon_v2.py --weights_path ...
  3. import and call aggregate_sweep.aggregate()
Fail-fast: any subprocess rc!=0 exits with rc=1, prints log path.

Subprocess uses sys.executable -u with PYTHONUNBUFFERED=1 for prompt
output flush under SLURM."
```

---

### Task 5: Update `sweeps/README.md` to document the Python runner

Add a section showing how to use `run_sweep.py` as an alternative to the bash+SLURM-array harness.

**Files:**
- Modify: `sweeps/README.md`

- [ ] **Step 1: Read the current README**

Read `sweeps/README.md` to locate the right insertion point (after the existing bash/SLURM documentation).

- [ ] **Step 2: Insert the Python runner section**

Add the following section immediately **after** the "Local sanity check (no SLURM)" section at the end of `sweeps/README.md`:

````markdown

## Alternative: full-pipeline Python runner (`run_sweep.py`)

If you prefer a single Python invocation that runs the whole sweep serially
(train + eval + aggregate in one call) instead of a SLURM array job, use
`sweeps/run_sweep.py`. This is useful when:
- Your sbatch is a single-task job (not an array)
- You want one combined log per text_ratio
- You want idempotent re-runs (already-completed T values are skipped)

### Submit (single-task sbatch)

In your sbatch:

```bash
# activate conda env first (omitted)
python sweeps/run_sweep.py \
    --text_ratios 0,1,2,3,5,8,10,15 \
    --repo_dir /path/to/Text-IF \
    --pretrained_weights /path/to/textif-me/checkpoint.pth \
    --dataset_ll /path/to/EMS_lite/Low_light \
    --dataset_oe /path/to/EMS_lite/Over_exposure \
    --dataset_ic /path/to/EMS_lite/IR_Low_contrast \
    --dataset_in /path/to/EMS_lite/IR_Noise \
    --eval_data_path /path/to/IVT_test
```

### Outputs

Same directory layout as the SLURM-array approach:

```
sweeps/out/
├── text_ratio_T0/
│   ├── train/              # weights/, img/, log/
│   ├── metrics/            # evaluation_summary.csv, fused/
│   ├── train.log           # captured train stdout+stderr
│   └── eval.log            # captured eval stdout+stderr
├── text_ratio_T1/...
└── text_ratio_T15/...
sweeps/text_ratio_sweep_summary.csv
```

### Behavior

- **Serial**: one T at a time (one GPU assumed)
- **Fail-fast**: if any T's training or eval returns non-zero, abort the whole sweep
- **Idempotent**: if `text_ratio_T<T>/metrics/evaluation_summary.csv` already exists, that T is skipped
- **Auto-aggregate**: after all T finish, runs `aggregate_sweep.aggregate()` and writes `sweeps/text_ratio_sweep_summary.csv`

### Local unit tests (no GPU needed)

```bash
python -m pytest tests/test_run_sweep.py tests/test_aggregate_sweep.py -v
```
````

- [ ] **Step 3: Commit**

```bash
git add sweeps/README.md
git commit -m "docs(sweep): document Python full-pipeline runner (run_sweep.py)

Adds 'Alternative: full-pipeline Python runner' section to sweeps/README
showing usage, outputs, fail-fast/idempotent behavior, and pointing to
the unit tests for local verification."
```

---

### Task 6: End-to-end verification

Verify the full harness hangs together without running real training (which needs HPC + GPU).

- [ ] **Step 1: Confirm all files in place**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && ls sweeps/ tests/test_run_sweep.py tests/test_aggregate_sweep.py
```

Expected: `sweep_text_ratio.sbatch`, `run_single.sh`, `aggregate_sweep.py`, `run_sweep.py`, `README.md` under `sweeps/`; both test files under `tests/`.

- [ ] **Step 2: Full unit test suite passes**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe -m pytest tests/test_run_sweep.py tests/test_aggregate_sweep.py -v
```

Expected: 13 passed.

- [ ] **Step 3: `--help` shows all 11 args**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe sweeps/run_sweep.py --help 2>&1 | grep -E "^\s+--"
```

Expected: 11 lines, one per arg, with defaults shown.

- [ ] **Step 4: Path validation triggers correctly**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe sweeps/run_sweep.py \
    --text_ratios 5 \
    --repo_dir /tmp \
    --pretrained_weights /nope.pth \
    --dataset_ll /nope --dataset_oe /nope \
    --dataset_ic /nope --dataset_in /nope \
    --eval_data_path /nope 2>&1 | head -10; echo "EXIT=$?"
```

Expected: multiple `ERROR:` lines, `EXIT=2`, no sweep starts.

- [ ] **Step 5: Garbage text_ratios is rejected**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && D:/software/anaconda3/envs/xpu/python.exe sweeps/run_sweep.py \
    --text_ratios "0,abc,2" \
    --repo_dir /tmp \
    --pretrained_weights /nope.pth \
    --dataset_ll /nope --dataset_oe /nope \
    --dataset_ic /nope --dataset_in /nope \
    --eval_data_path /nope 2>&1 | tail -5; echo "EXIT=$?"
```

Expected: a `ValueError: non-integer text_ratio token: 'abc'` traceback (uncaught exception), `EXIT=1`.

(Note: this is a Python traceback, not a clean error message. We accept that — the user gets a clear hint to fix the CLI. If we wanted polished UX we'd catch and re-format, but that's out of scope per spec YAGNI.)

- [ ] **Step 6: Spec drift check**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && git diff main docs/superpowers/specs/2026-07-25-python-sweep-runner-design.md
```

Expected: empty (spec was committed at design stage; no further edits).

If any check fails, fix the code to match spec.

- [ ] **Step 7: Final commit if any fixups**

```bash
cd "d:/StudyFiles/MachineLearning/codes/Text-IF" && git status
# If fixups:
git add -A && git commit -m "chore: post-verification fixups"
```

---

## Self-Review Notes

**Spec coverage**: every section of the spec maps to a task
- §2.1 CLI → Task 1
- §2.3 必需参数校验 → Task 2 (paths) + Task 2 (text_ratios parse)
- §3 执行流程 → Task 3 (subprocess helper) + Task 4 (sweep loop, train+eval+aggregate, fail-fast)
- §3.1 子进程调用约定 → Task 3 (sys.executable, PYTHONUNBUFFERED=1, tee)
- §3.2 聚合阶段 → Task 4 (`from aggregate_sweep import aggregate`)
- §4 输出目录结构 → Task 4 (mkdir train_dir/metrics_dir; same layout as bash harness)
- §5.1 单元测试 → Task 2 (5 parse/validate tests) + Task 3 (2 subprocess tests) = 7 originally; expanded to 8 in Task 2 (`test_validate_paths_missing_repo_script_exits` added). Total = 10 unit tests
- §5.2 HPC smoke → Task 6 Step 5 (CLI smoke only; real HPC training is user's job)
- §6 验收标准 1-4 verifiable in Task 6; 5-6 require HPC

**Placeholder scan**: no TBD/TODO; every code step shows the actual code.

**Type consistency**:
- `_parse_text_ratios(s: str) -> List[int]` — used as int list throughout (matches spec §3.2 `expected_text_ratios=text_ratios` which aggregate expects as `List[float]` — int is assignable to float param in Python, no type error)
- `_validate_paths(repo_dir, pretrained_weights, dataset_ll, dataset_oe, dataset_ic, dataset_in, eval_data_path)` — same 7 names everywhere
- `_run_subprocess_with_log(cmd, log_path, cwd, label) -> int` — consistent across Task 3 def and Task 4 callers
- `main(argv=None) -> int` — consistent
- `text_ratio_T{T}` directory name matches existing bash harness

**Out of scope (per spec §1.3)**: no resume beyond skip-existing, no parallelism, no multi-dim grid, no conda activation.
