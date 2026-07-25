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
