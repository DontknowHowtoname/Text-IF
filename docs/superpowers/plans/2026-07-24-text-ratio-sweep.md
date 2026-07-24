# Text Ratio Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `text_ratio` override plumbing to the dual-recon training pipeline, plus a SLURM array-job sweep harness that produces a wide-format metrics CSV over 8 `text_ratio` values for Pareto-frontier analysis.

**Architecture:** Three-file plumbing change mirrors the existing `max_ratio`/`ssim_ratio` override pattern (CLI → train/eval functions → loss class). The sweep harness is four new files under `sweeps/`: a SLURM array `sbatch`, a per-run wrapper script, a Python aggregator (TDD with synthetic CSVs), and a README.

**Tech Stack:** PyTorch, argparse, SLURM (`sbatch --array`), Python 3, CSV I/O. Training script = `train_fusion_full_recon_v2_ft.py` (dual-path reconstruction variant). Eval script = `evaluate_textif_full_recon_v2.py` (loads weights, runs forward, writes long-format `evaluation_summary.csv`).

**Spec:** [docs/superpowers/specs/2026-07-24-text-ratio-sweep-design.md](../specs/2026-07-24-text-ratio-sweep-design.md)

---

### Task 1: Add `text_ratio` to `fusion_dual_recon_prompt_loss`

The innermost layer of the plumbing. Mirror the existing `max_ratio`/`ssim_ratio` override pattern exactly.

**Files:**
- Modify: `scripts/losses.py:460-501` (`fusion_dual_recon_prompt_loss.__init__` and `.forward`)

- [ ] **Step 1: Inspect current `__init__` signature and `_TASK_DEFAULTS`**

Run: `python -c "from scripts.losses import fusion_dual_recon_prompt_loss; help(fusion_dual_recon_prompt_loss.__init__)"`
Expected: shows signature `(self, upper_weight=1.3, recon_weight=1.0, max_ratio=None, ssim_ratio=None)` — no `text_ratio` yet.

- [ ] **Step 2: Add `text_ratio=None` to `__init__`**

Edit [scripts/losses.py:460-466](../../../scripts/losses.py#L460-L466). Replace:

```python
class fusion_dual_recon_prompt_loss(nn.Module):
    """Full loss: original task-specific fusion loss + dual-path reconstruction loss."""
    def __init__(self, upper_weight=1.3, recon_weight=1.0, max_ratio=None, ssim_ratio=None):
        super(fusion_dual_recon_prompt_loss, self).__init__()
        self.fusion_loss = fusion_loss()
        self.dual_recon_loss = DualReconLoss(upper_weight=upper_weight)
        self.recon_weight = recon_weight
        self.max_ratio = max_ratio
        self.ssim_ratio = ssim_ratio
```

with:

```python
class fusion_dual_recon_prompt_loss(nn.Module):
    """Full loss: original task-specific fusion loss + dual-path reconstruction loss."""
    def __init__(self, upper_weight=1.3, recon_weight=1.0,
                 max_ratio=None, ssim_ratio=None, text_ratio=None):
        super(fusion_dual_recon_prompt_loss, self).__init__()
        self.fusion_loss = fusion_loss()
        self.dual_recon_loss = DualReconLoss(upper_weight=upper_weight)
        self.recon_weight = recon_weight
        self.max_ratio = max_ratio
        self.ssim_ratio = ssim_ratio
        self.text_ratio = text_ratio
```

- [ ] **Step 3: Use `self.text_ratio` in `forward`**

Edit [scripts/losses.py:497-501](../../../scripts/losses.py#L497-L501). Replace:

```python
            defaults = self._TASK_DEFAULTS[task_type]
            mr = self.max_ratio if self.max_ratio is not None else defaults["max_ratio"]
            sr = self.ssim_ratio if self.ssim_ratio is not None else defaults["ssim_ratio"]
            loss, ssim_l, max_l, color_l, text_l = self.fusion_loss(
                img_A, img_B, img_f, max_ratio=mr, ssim_ratio=sr, text_ratio=defaults["text_ratio"])
```

with:

```python
            defaults = self._TASK_DEFAULTS[task_type]
            mr = self.max_ratio if self.max_ratio is not None else defaults["max_ratio"]
            sr = self.ssim_ratio if self.ssim_ratio is not None else defaults["ssim_ratio"]
            tr = self.text_ratio if self.text_ratio is not None else defaults["text_ratio"]
            loss, ssim_l, max_l, color_l, text_l = self.fusion_loss(
                img_A, img_B, img_f, max_ratio=mr, ssim_ratio=sr, text_ratio=tr)
```

- [ ] **Step 4: Smoke-test constructor compat**

Run: `python -c "from scripts.losses import fusion_dual_recon_prompt_loss; m = fusion_dual_recon_prompt_loss(text_ratio=5); print('text_ratio=', m.text_ratio)"`
Expected: prints `text_ratio= 5.0` with no error.

Run: `python -c "from scripts.losses import fusion_dual_recon_prompt_loss; m = fusion_dual_recon_prompt_loss(); print('text_ratio=', m.text_ratio)`
Expected: prints `text_ratio= None` (default behavior preserved).

- [ ] **Step 5: Commit**

```bash
git add scripts/losses.py
git commit -m "feat(loss): add text_ratio override to fusion_dual_recon_prompt_loss

Mirrors the existing max_ratio/ssim_ratio pattern. When text_ratio is None,
per-task defaults from _TASK_DEFAULTS are used (preserving current behavior).
This is the innermost layer of three for the text_ratio CLI plumbing."
```

---

### Task 2: Plumb `text_ratio` through `train_one_epoch_recon_dual` and `evaluate_recon_dual`

Middle layer. Both functions live in `scripts/utils.py` and instantiate the loss class.

**Files:**
- Modify: `scripts/utils.py:515-520` (`train_one_epoch_recon_dual` signature + loss construction)
- Modify: `scripts/utils.py:596-598` (`evaluate_recon_dual` signature + loss construction)

- [ ] **Step 1: Read current signatures**

Run: `python -c "import scripts.utils as u; help(u.train_one_epoch_recon_dual)" 2>&1 | head -10`
Expected: shows signature without `text_ratio`. If import fails, run from project root: `cd <repo-root> && python -c "..."`.

- [ ] **Step 2: Add `text_ratio=None` to `train_one_epoch_recon_dual` and pass to loss**

Edit [scripts/utils.py:515-520](../../../scripts/utils.py#L515-L520). Replace:

```python
def train_one_epoch_recon_dual(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch,
                                recon_weight=1.0, max_ratio=None, ssim_ratio=None):
    model.train()
    model_clip.eval()
    loss_function = fusion_dual_recon_prompt_loss(recon_weight=recon_weight,
                                                  max_ratio=max_ratio, ssim_ratio=ssim_ratio)
```

with:

```python
def train_one_epoch_recon_dual(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch,
                                recon_weight=1.0, max_ratio=None, ssim_ratio=None, text_ratio=None):
    model.train()
    model_clip.eval()
    loss_function = fusion_dual_recon_prompt_loss(recon_weight=recon_weight,
                                                  max_ratio=max_ratio, ssim_ratio=ssim_ratio,
                                                  text_ratio=text_ratio)
```

- [ ] **Step 3: Add `text_ratio=None` to `evaluate_recon_dual` and pass to loss**

Read [scripts/utils.py:596-598](../../../scripts/utils.py#L596-L598) first to confirm current form (may differ slightly from `train_one_epoch_recon_dual`). Then edit:

Replace:

```python
def evaluate_recon_dual(model, data_loader, device, epoch, lr, filefold_path,
                        max_ratio=None, ssim_ratio=None):
    ...
    loss_function = fusion_dual_recon_prompt_loss(max_ratio=max_ratio, ssim_ratio=ssim_ratio)
```

with:

```python
def evaluate_recon_dual(model, data_loader, device, epoch, lr, filefold_path,
                        max_ratio=None, ssim_ratio=None, text_ratio=None):
    ...
    loss_function = fusion_dual_recon_prompt_loss(max_ratio=max_ratio, ssim_ratio=ssim_ratio,
                                                  text_ratio=text_ratio)
```

**Note:** If `evaluate_recon_dual`'s actual signature differs (e.g., extra `recon_weight` arg), preserve all existing params and only add `text_ratio=None`. Re-read the file to confirm before editing.

- [ ] **Step 4: Smoke-test both functions accept `text_ratio`**

Run:
```bash
python -c "
import inspect
from scripts.utils import train_one_epoch_recon_dual, evaluate_recon_dual
print('train:', 'text_ratio' in inspect.signature(train_one_epoch_recon_dual).parameters)
print('eval:', 'text_ratio' in inspect.signature(evaluate_recon_dual).parameters)
"
```
Expected: both print `True`.

- [ ] **Step 5: Commit**

```bash
git add scripts/utils.py
git commit -m "feat(utils): plumb text_ratio through train/eval dual-recon functions

Middle layer of the three-layer text_ratio plumbing. Both functions now
accept text_ratio=None (default preserves current behavior) and forward
it to fusion_dual_recon_prompt_loss."
```

---

### Task 3: Add `--text_ratio` and `--output_dir` CLI to training script

Outermost layer. Also adds `--output_dir` so the sweep harness can predict output paths (spec section 3.3, option A).

**Files:**
- Modify: `train_fusion_full_recon_v2_ft.py:33-34` (output dir construction)
- Modify: `train_fusion_full_recon_v2_ft.py:171-181` (train_one_epoch_recon_dual call)
- Modify: `train_fusion_full_recon_v2_ft.py:~192` (evaluate_recon_dual call — re-read to find exact line)
- Modify: `train_fusion_full_recon_v2_ft.py:269-272` (CLI arg addition)

- [ ] **Step 1: Add `--text_ratio` and `--output_dir` CLI args**

Edit [train_fusion_full_recon_v2_ft.py:269-272](../../../train_fusion_full_recon_v2_ft.py#L269-L272). Replace:

```python
    parser.add_argument('--ssim_ratio', type=float, default=None,
                        help='Override ssim_ratio for all tasks (default: None = per-task defaults)')

    opt = parser.parse_args()
```

with:

```python
    parser.add_argument('--ssim_ratio', type=float, default=None,
                        help='Override ssim_ratio for all tasks (default: None = per-task defaults)')
    parser.add_argument('--text_ratio', type=float, default=None,
                        help='Override text_ratio (L_Grad_position weight) for all tasks '
                             '(default: None = per-task defaults {3,2,3,2})')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory override. Default: ./experiments/TextIF_full_recon_v2_ft_<timestamp>')

    opt = parser.parse_args()
```

- [ ] **Step 2: Use `args.output_dir` in path construction**

Edit [train_fusion_full_recon_v2_ft.py:33-34](../../../train_fusion_full_recon_v2_ft.py#L33-L34). Replace:

```python
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/TextIF_full_recon_v2_ft_{}".format(file_name)
```

with:

```python
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir is not None:
        filefold_path = args.output_dir
    else:
        filefold_path = "./experiments/TextIF_full_recon_v2_ft_{}".format(file_name)
```

- [ ] **Step 3: Pass `text_ratio=args.text_ratio` to `train_one_epoch_recon_dual`**

Edit [train_fusion_full_recon_v2_ft.py:179-181](../../../train_fusion_full_recon_v2_ft.py#L179-L181). Replace:

```python
            recon_weight=args.recon_weight,
            max_ratio=args.max_ratio,
            ssim_ratio=args.ssim_ratio)
```

with:

```python
            recon_weight=args.recon_weight,
            max_ratio=args.max_ratio,
            ssim_ratio=args.ssim_ratio,
            text_ratio=args.text_ratio)
```

- [ ] **Step 4: Pass `text_ratio=args.text_ratio` to `evaluate_recon_dual`**

Re-read [train_fusion_full_recon_v2_ft.py:190-200](../../../train_fusion_full_recon_v2_ft.py#L190-L200) to find the exact call site of `evaluate_recon_dual`. Then add `text_ratio=args.text_ratio` to its argument list (after `ssim_ratio=...` if present, otherwise after the last existing arg).

Expected form after edit:

```python
        if epoch % args.val_every_epcho == 0 and epoch != 0:
            (val_loss, val_ssim_loss, val_max_loss, val_color_loss,
             val_text_loss, val_recon_loss) = evaluate_recon_dual(
                model=model,
                data_loader=val_loader,
                ...,
                text_ratio=args.text_ratio)
```

- [ ] **Step 5: Verify `--help` shows both new args**

Run: `python train_fusion_full_recon_v2_ft.py --help 2>&1 | grep -E "text_ratio|output_dir"`
Expected: two lines containing `--text_ratio` and `--output_dir`.

- [ ] **Step 6: Verify defaults preserve current behavior**

Run: `python -c "import sys; sys.argv = ['x']; exec(open('train_fusion_full_recon_v2_ft.py').read().replace('if __name__ == ', 'if False and __name__ == ')); print('parsed')" 2>&1 | tail -3`

**Easier alternative** — just confirm via `--help` that defaults are `None`:
Run: `python train_fusion_full_recon_v2_ft.py --help 2>&1 | grep -A1 "text_ratio\|output_dir"`
Expected: both show `default: None`.

- [ ] **Step 7: Commit**

```bash
git add train_fusion_full_recon_v2_ft.py
git commit -m "feat(train): expose --text_ratio and --output_dir CLI in v2 ft script

Completes the three-layer text_ratio plumbing (CLI -> utils -> loss).
--output_dir lets the sweep harness predict output paths without parsing
timestamps. Both default to None, preserving current behavior."
```

---

### Task 4: `sweeps/run_single.sh` — per-run wrapper

Single SLURM task = one `text_ratio` value = one call to this script.

**Files:**
- Create: `sweeps/run_single.sh`

- [ ] **Step 1: Create directory and write script**

```bash
mkdir -p sweeps
```

Write `sweeps/run_single.sh`:

```bash
#!/usr/bin/env bash
# Wrapper for a single text_ratio sweep run: train + evaluate.
# Invoked by sweep_text_ratio.sbatch once per array task.
#
# Usage: run_single.sh <text_ratio_value>
#
# Reads the following environment variables (set in sweep_text_ratio.sbatch):
#   REPO_DIR, PRETRAINED_WEIGHTS,
#   DATASET_LL, DATASET_OE, DATASET_IC, DATASET_IN,
#   EVAL_DATA_PATH, CONDA_ENV

set -euo pipefail

T="$1"
if [[ -z "${T:-}" ]]; then
    echo "ERROR: text_ratio value required as \$1" >&2
    exit 1
fi

# Required env vars
for v in REPO_DIR PRETRAINED_WEIGHTS DATASET_LL DATASET_OE DATASET_IC DATASET_IN EVAL_DATA_PATH CONDA_ENV; do
    if [[ -z "${!v:-}" ]]; then
        echo "ERROR: env var $v is not set" >&2
        exit 1
    fi
done

OUT_DIR="${REPO_DIR}/sweeps/out/text_ratio_T${T}"
TRAIN_DIR="${OUT_DIR}/train"
METRICS_DIR="${OUT_DIR}/metrics"

mkdir -p "${TRAIN_DIR}" "${METRICS_DIR}"

cd "${REPO_DIR}"

# Activate environment (adjust if your HPC uses modules or a different activate path)
source "${CONDA_ENV}/bin/activate"

echo "[$(date)] text_ratio=${T} | training -> ${TRAIN_DIR}"
python train_fusion_full_recon_v2_ft.py \
    --text_ratio "${T}" \
    --weights "${PRETRAINED_WEIGHTS}" \
    --low_light_path "${DATASET_LL}" \
    --over_exposure_path "${DATASET_OE}" \
    --ir_low_contrast_path "${DATASET_IC}" \
    --ir_noise_path "${DATASET_IN}" \
    --output_dir "${TRAIN_DIR}"

WEIGHTS="${TRAIN_DIR}/weights/checkpoint.pth"
if [[ ! -f "${WEIGHTS}" ]]; then
    echo "ERROR: expected weights file not found: ${WEIGHTS}" >&2
    exit 1
fi

echo "[$(date)] text_ratio=${T} | evaluating -> ${METRICS_DIR}"
python evaluate_textif_full_recon_v2.py \
    --weights_path "${WEIGHTS}" \
    --data_path "${EVAL_DATA_PATH}" \
    --output_dir "${METRICS_DIR}"

echo "[$(date)] text_ratio=${T} | done"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x sweeps/run_single.sh`

- [ ] **Step 3: Smoke-test argument validation (no training actually runs)**

Run: `bash sweeps/run_single.sh` (no args)
Expected: exits with `ERROR: text_ratio value required as \$1`, exit code 1.

Run (from repo root): `REPO_DIR= CONDA_ENV= bash sweeps/run_single.sh 5`
Expected: exits with `ERROR: env var REPO_DIR is not set` (or similar).

- [ ] **Step 4: Commit**

```bash
git add sweeps/run_single.sh
git commit -m "feat(sweep): add run_single.sh per-run wrapper (train + eval)

One text_ratio value = one invocation. Validates args and env vars
before doing any work so misconfigured SLURM tasks fail fast."
```

---

### Task 5: `sweeps/sweep_text_ratio.sbatch` — SLURM array job

Driver script. Submit once with `sbatch sweeps/sweep_text_ratio.sbatch`, fires off 8 parallel tasks.

**Files:**
- Create: `sweeps/sweep_text_ratio.sbatch`

- [ ] **Step 1: Write the sbatch script**

Write `sweeps/sweep_text_ratio.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=text_ratio_sweep
#SBATCH --array=0-7
#SBATCH --output=sweeps/logs/sweep_%A_%a.out
#SBATCH --error=sweeps/logs/sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# HPC-specific: uncomment and adjust for your cluster
# #SBATCH --partition=<your-partition>
# #SBATCH --account=<your-account>

# ============================================================
# USER CONFIGURATION: fill in these variables for your HPC.
# ============================================================
REPO_DIR="/path/to/Text-IF"                                    # CHANGE ME
PRETRAINED_WEIGHTS="${REPO_DIR}/experiments/TextIF_train_20260408-185710/weights/checkpoint.pth"
DATASET_LL="${REPO_DIR}/dataset/EMS_lite/Low_light"            # CHANGE ME
DATASET_OE="${REPO_DIR}/dataset/EMS_lite/Over_exposure"        # CHANGE ME
DATASET_IC="${REPO_DIR}/dataset/EMS_lite/IR_Low_contrast"      # CHANGE ME
DATASET_IN="${REPO_DIR}/dataset/EMS_lite/IR_Noise"             # CHANGE ME
EVAL_DATA_PATH="${REPO_DIR}/data/IVT_test"                     # CHANGE ME
CONDA_ENV="/path/to/conda/envs/xpu"                            # CHANGE ME

# ============================================================
# Sweep grid: 8 text_ratio values, indexed by SLURM_ARRAY_TASK_ID
# ============================================================
TEXT_RATIOS=(0 1 2 3 5 8 10 15)

mkdir -p "${REPO_DIR}/sweeps/logs" "${REPO_DIR}/sweeps/out"

# Export for run_single.sh
export REPO_DIR PRETRAINED_WEIGHTS
export DATASET_LL DATASET_OE DATASET_IC DATASET_IN EVAL_DATA_PATH CONDA_ENV

T="${TEXT_RATIOS[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> text_ratio=${T}"

bash "${REPO_DIR}/sweeps/run_single.sh" "${T}"
```

- [ ] **Step 2: Verify syntax**

Run: `bash -n sweeps/sweep_text_ratio.sbatch`
Expected: no output (syntax OK).

- [ ] **Step 3: Verify array indexing logic (dry run)**

Run:
```bash
TEXT_RATIOS=(0 1 2 3 5 8 10 15)
for i in 0 1 2 3 4 5 6 7; do
    echo "task $i -> text_ratio ${TEXT_RATIOS[$i]}"
done
```
Expected:
```
task 0 -> text_ratio 0
task 1 -> text_ratio 1
task 2 -> text_ratio 2
task 3 -> text_ratio 3
task 4 -> text_ratio 5
task 5 -> text_ratio 8
task 6 -> text_ratio 10
task 7 -> text_ratio 15
```

- [ ] **Step 4: Commit**

```bash
git add sweeps/sweep_text_ratio.sbatch
git commit -m "feat(sweep): add sweep_text_ratio.sbatch SLURM array driver

8-array job over text_ratio in {0,1,2,3,5,8,10,15}. User fills in HPC
paths (REPO_DIR, datasets, conda env) at the top. Per-task logs in
sweeps/logs/sweep_<jobid>_<taskid>.{out,err}."
```

---

### Task 6: `sweeps/aggregate_sweep.py` — TDD

Aggregate per-run `evaluation_summary.csv` files into one wide-format summary. This is pure Python with file I/O — suitable for real TDD.

**Files:**
- Create: `tests/test_aggregate_sweep.py`
- Create: `sweeps/aggregate_sweep.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregate_sweep.py`:

```python
"""Tests for sweeps/aggregate_sweep.py.

Validates that the aggregator correctly pivots long-format evaluation_summary.csv
files (one per text_ratio run) into a single wide-format summary CSV.
"""
import csv
import os
import sys
import tempfile
from pathlib import Path

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sweeps"))


def _write_long_format(path: Path, metrics: dict):
    """Write a fake evaluation_summary.csv in the long format produced by
    evaluate_textif_full_recon_v2.py (columns: metric,average)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "average"])
        for k, v in metrics.items():
            w.writerow([k, v])


def test_aggregate_pivots_long_to_wide(tmp_path):
    """Two text_ratio runs should become two rows in the summary, with all
    21 metrics as columns plus text_ratio and experiment_dir."""
    from aggregate_sweep import aggregate

    # Synthetic metrics for T=2 and T=10
    out_root = tmp_path / "out"
    for t, en_val in [(2, 7.31), (10, 7.40)]:
        run_dir = out_root / f"text_ratio_T{t}" / "metrics"
        run_dir.mkdir(parents=True)
        metrics = {m: en_val + i * 0.01 for i, m in enumerate([
            "EN", "MI", "NMI", "SF", "AG", "SD", "CC", "SCD", "PSNR",
            "MSE", "VIF", "SSIM", "MS_SSIM", "Qabf", "Nabf", "CE",
            "QNCIE", "TE", "EI", "Qy", "Qcb"])}
        _write_long_format(run_dir / "evaluation_summary.csv", metrics)

    output_csv = tmp_path / "summary.csv"
    aggregate(out_root=str(out_root), output_csv=str(output_csv))

    assert output_csv.exists(), "summary CSV was not created"
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
    t_values = sorted(float(r["text_ratio"]) for r in rows)
    assert t_values == [2.0, 10.0], f"unexpected text_ratio values: {t_values}"
    # Check one metric column landed in wide format
    en_values = {float(r["text_ratio"]): float(r["EN"]) for r in rows}
    assert abs(en_values[2.0] - 7.31) < 1e-6
    assert abs(en_values[10.0] - 7.40) < 1e-6
    # experiment_dir column populated
    for r in rows:
        assert r["experiment_dir"].endswith(f"text_ratio_T{r['text_ratio']}/metrics")


def test_aggregate_skips_missing_runs(tmp_path, capsys):
    """If some text_ratio dirs are missing, aggregate what exists and report
    the missing ones on stderr."""
    from aggregate_sweep import aggregate

    out_root = tmp_path / "out"
    # Only create T=5; T=0,1,2,...,15 others missing
    run_dir = out_root / "text_ratio_T5" / "metrics"
    run_dir.mkdir(parents=True)
    _write_long_format(run_dir / "evaluation_summary.csv", {"EN": 7.5, "MI": 3.0})

    output_csv = tmp_path / "summary.csv"
    aggregate(out_root=str(out_root), output_csv=str(output_csv),
              expected_text_ratios=[0, 1, 2, 3, 5, 8, 10, 15])

    captured = capsys.readouterr()
    assert "missing" in captured.err.lower() or "skip" in captured.err.lower()
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]["text_ratio"]) == 5.0


def test_aggregate_handles_empty(tmp_path):
    """Empty out_root produces an empty CSV with headers (no rows)."""
    from aggregate_sweep import aggregate

    out_root = tmp_path / "out"
    out_root.mkdir()
    output_csv = tmp_path / "summary.csv"
    aggregate(out_root=str(out_root), output_csv=str(output_csv))

    assert output_csv.exists()
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert rows == [], "expected zero rows when no run dirs exist"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aggregate_sweep.py -v`
Expected: `ModuleNotFoundError: No module named 'aggregate_sweep'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation**

Create `sweeps/aggregate_sweep.py`:

```python
"""Aggregate per-run evaluation_summary.csv files into one wide-format summary.

Each text_ratio sweep run produces a long-format CSV at
``sweeps/out/text_ratio_T<T>/metrics/evaluation_summary.csv`` with columns
``metric,average`` (written by ``evaluate_textif_full_recon_v2.py``). This
script pivots all such files into one wide-format CSV with one row per
text_ratio value and one column per metric, matching the schema of
``results/all_experiments_fusion_metrics.csv`` (plus a ``text_ratio`` column).

Usage:
    python sweeps/aggregate_sweep.py \\
        --out-root sweeps/out \\
        --output-csv sweeps/text_ratio_sweep_summary.csv \\
        [--expected-text-ratios 0,1,2,3,5,8,10,15]
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional


# Canonical metric column order (matches all_experiments_fusion_metrics.csv
# minus the "Experiment" column; text_ratio is prepended separately).
METRIC_COLUMNS = [
    "EN", "MI", "NMI", "SF", "AG", "SD", "CC", "SCD",
    "PSNR", "MSE", "VIF", "SSIM", "MS_SSIM",
    "Qabf", "Nabf", "CE", "QNCIE", "TE", "EI", "Qy", "Qcb",
]


def _read_long_format(path: Path) -> dict:
    """Read evaluation_summary.csv (metric,average) -> dict."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric")
            avg = row.get("average")
            if metric is None or avg is None:
                continue
            try:
                out[metric] = float(avg)
            except ValueError:
                # Non-numeric metric value; skip with warning
                print(f"WARN: non-numeric value for {metric}: {avg!r}", file=sys.stderr)
    return out


def _discover_run_dirs(out_root: Path) -> List[Path]:
    """Find all text_ratio_T*/metrics dirs with an evaluation_summary.csv."""
    if not out_root.is_dir():
        return []
    runs = []
    for child in sorted(out_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("text_ratio_T"):
            continue
        summary = child / "metrics" / "evaluation_summary.csv"
        if summary.is_file():
            runs.append(child)
    return runs


def _extract_text_ratio(run_dir: Path) -> Optional[float]:
    """Parse 'text_ratio_T5' -> 5.0; returns None on parse failure."""
    name = run_dir.name
    prefix = "text_ratio_T"
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix):])
    except ValueError:
        return None


def aggregate(out_root: str, output_csv: str,
              expected_text_ratios: Optional[List[float]] = None) -> None:
    """Aggregate per-run CSVs into one wide-format summary.

    Args:
        out_root: Directory containing text_ratio_T*/ subdirs.
        output_csv: Path to write the wide-format summary.
        expected_text_ratios: If provided, warn (stderr) about any values
            from this list that are missing in out_root.
    """
    out_root_path = Path(out_root)
    runs = _discover_run_dirs(out_root_path)

    found_ratios = set()
    rows = []
    for run_dir in runs:
        t = _extract_text_ratio(run_dir)
        if t is None:
            print(f"WARN: could not parse text_ratio from dir name: {run_dir.name}",
                  file=sys.stderr)
            continue
        metrics = _read_long_format(run_dir / "metrics" / "evaluation_summary.csv")
        row = {"text_ratio": t}
        for m in METRIC_COLUMNS:
            row[m] = metrics.get(m, "")
        row["experiment_dir"] = str(run_dir / "metrics")
        rows.append(row)
        found_ratios.add(t)

    # Report missing
    if expected_text_ratios is not None:
        missing = [t for t in expected_text_ratios if t not in found_ratios]
        if missing:
            print(f"WARN: missing text_ratio runs: {sorted(missing)}", file=sys.stderr)

    # Sort by text_ratio ascending
    rows.sort(key=lambda r: r["text_ratio"])

    # Write
    fieldnames = ["text_ratio"] + METRIC_COLUMNS + ["experiment_dir"]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows -> {output_path}")


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", default="sweeps/out",
                        help="Root dir containing text_ratio_T*/ subdirs (default: sweeps/out)")
    parser.add_argument("--output-csv", default="sweeps/text_ratio_sweep_summary.csv",
                        help="Output wide-format CSV path")
    parser.add_argument("--expected-text-ratios", type=str, default=None,
                        help="Comma-separated expected text_ratio values, e.g. '0,1,2,3,5,8,10,15'")
    args = parser.parse_args()

    expected = _parse_float_list(args.expected_text_ratios) if args.expected_text_ratios else None
    aggregate(out_root=args.out_root, output_csv=args.output_csv,
              expected_text_ratios=expected)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_aggregate_sweep.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_aggregate_sweep.py sweeps/aggregate_sweep.py
git commit -m "feat(sweep): add aggregate_sweep.py with TDD tests

Pivots long-format evaluation_summary.csv files into one wide-format
summary CSV matching all_experiments_fusion_metrics.csv schema.
Handles missing runs gracefully (warns on stderr, still writes what
exists). Tests cover: basic pivot, missing runs, empty out_root."
```

---

### Task 7: `sweeps/README.md` — usage docs

**Files:**
- Create: `sweeps/README.md`

- [ ] **Step 1: Write README**

Write `sweeps/README.md`:

````markdown
# Text Ratio Sweep

Controlled single-variable sweep of `text_ratio` (L_Grad_position loss weight)
for the paper's metric trade-off analysis. See
[design spec](../docs/superpowers/specs/2026-07-24-text-ratio-sweep-design.md)
for motivation.

## Files

| File | Purpose |
|------|---------|
| `sweep_text_ratio.sbatch` | SLURM array job driver (8 parallel tasks) |
| `run_single.sh` | Per-run wrapper: train + eval for one `text_ratio` |
| `aggregate_sweep.py` | Pivot per-run `evaluation_summary.csv` -> one wide-format CSV |
| `out/text_ratio_T<T>/` | Per-run output (train/, metrics/, fused/) |

## Setup (HPC)

1. **Fill in `sweep_text_ratio.sbatch`** — edit the USER CONFIGURATION block at
   the top: `REPO_DIR`, `PRETRAINED_WEIGHTS`, `DATASET_*`, `EVAL_DATA_PATH`,
   `CONDA_ENV`. Also uncomment/adjust the `#SBATCH --partition` line for your
   cluster.
2. **Verify the env activation line in `run_single.sh`** — it currently uses
   `source "${CONDA_ENV}/bin/activate"`. If your HPC uses `module load` or a
   different activate path, adjust accordingly.
3. **Verify pretrained weights exist** at `$PRETRAINED_WEIGHTS` on the HPC.

## Submit

```bash
# from repo root on the HPC login node
sbatch sweeps/sweep_text_ratio.sbatch
```

This submits an 8-task array job. Each task runs one `text_ratio` value from
`{0, 1, 2, 3, 5, 8, 10, 15}`.

## Monitor

```bash
squeue -u $USER              # pending / running
ls sweeps/out/               # completed runs appear as text_ratio_T*/
tail -f sweeps/logs/sweep_<jobid>_<taskid>.out
```

## Rerun a failed task

```bash
# rerun only task index 3 (text_ratio=3)
sbatch --array=3-3 sweeps/sweep_text_ratio.sbatch
```

## Aggregate

After all 8 tasks finish:

```bash
python sweeps/aggregate_sweep.py \
    --out-root sweeps/out \
    --output-csv sweeps/text_ratio_sweep_summary.csv \
    --expected-text-ratios 0,1,2,3,5,8,10,15
```

Output CSV schema matches `results/all_experiments_fusion_metrics.csv` (plus a
leading `text_ratio` column), so you can feed it into `rank_experiments.py` or
plot Pareto curves directly.

## Local sanity check (no SLURM)

Verify the aggregator works on synthetic data:

```bash
python -m pytest tests/test_aggregate_sweep.py -v
```
````

- [ ] **Step 2: Commit**

```bash
git add sweeps/README.md
git commit -m "docs(sweep): add sweeps/README with HPC usage instructions

Covers setup (env vars to fill in), submission, monitoring, rerunning
failed tasks, aggregation, and local sanity check."
```

---

### Task 8: Final end-to-end verification

Verify the whole pipeline hangs together without running real training.

- [ ] **Step 1: Confirm all pieces are in place**

Run:
```bash
ls sweeps/
```
Expected: `README.md  aggregate_sweep.py  run_single.sh  sweep_text_ratio.sbatch`

- [ ] **Step 2: Re-run unit tests**

Run: `python -m pytest tests/test_aggregate_sweep.py -v`
Expected: 3 passed.

- [ ] **Step 3: Verify CLI plumbing (training script)**

Run: `python train_fusion_full_recon_v2_ft.py --help 2>&1 | grep -E "text_ratio|output_dir"`
Expected: both `--text_ratio` and `--output_dir` are listed.

- [ ] **Step 4: Verify loss class plumbing**

Run:
```bash
python -c "
from scripts.losses import fusion_dual_recon_prompt_loss
# Override path
m = fusion_dual_recon_prompt_loss(text_ratio=5)
assert m.text_ratio == 5.0
# Default path (per-task defaults still apply)
m = fusion_dual_recon_prompt_loss()
assert m.text_ratio is None
print('OK')
"
```
Expected: prints `OK`.

- [ ] **Step 5: Verify aggregate_sweep CLI on empty dir**

Run:
```bash
mkdir -p /tmp/empty_out
python sweeps/aggregate_sweep.py --out-root /tmp/empty_out --output-csv /tmp/empty_summary.csv
cat /tmp/empty_summary.csv
```
Expected: CSV with header row only (`text_ratio,EN,MI,...,Qcb,experiment_dir`), zero data rows.

- [ ] **Step 6: Inspect final spec for any drift**

Run: `git diff main docs/superpowers/specs/2026-07-24-text-ratio-sweep-design.md`
Expected: empty (spec was committed at design stage; no further edits).

If anything is inconsistent between spec and code, fix the code to match spec.

- [ ] **Step 7: Final commit if any fixups were needed**

```bash
git status
# If fixups:
git add -A
git commit -m "chore: post-verification fixups"
```

---

## Self-Review Notes

- **Spec coverage**: every section of the spec maps to a task
  - §3.1 改动 1 (losses.py) -> Task 1
  - §3.1 改动 2 (utils.py) -> Task 2
  - §3.1 改动 3 (training script) + §3.3 (--output_dir) -> Task 3
  - §3.2 文件 1 (run_single.sh) -> Task 4
  - §3.2 文件 2 (sbatch) -> Task 5
  - §3.2 文件 3 (aggregate_sweep.py) -> Task 6 (TDD)
  - §3.2 文件 4 (README) -> Task 7
  - §5 验收标准 -> Task 8 (criteria 1-2-5 verifiable locally; 3-4 require HPC)
- **Placeholder scan**: no TBD/TODO; every code step shows the actual edit
- **Type consistency**: `text_ratio=None` param name matches across all three files; METRIC_COLUMNS list in aggregate matches the spec's 21-metric schema
- **Out of scope (per spec §1.3)**: no Pareto plotting, no `_TASK_DEFAULTS` changes, no new loss terms, no local training
