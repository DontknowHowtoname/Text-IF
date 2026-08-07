# Text Ratio Sweep — Experiment Notes

**Date:** 2026-08-03
**Sweep variable:** `text_ratio` (weight on the L_Grad_position / text-injection loss)
**Goal:** Characterize the metric trade-off introduced by text-injection strength, and find a Pareto-optimal value that generalizes across datasets.

## Setup

- **Training data (degraded pairs):** `dataset/EMS_lite/{Low_light, Over_exposure, IR_Low_contrast, IR_Noise}`
- **Eval data (main sweep):** `dataset/LLVIP/{infrared,visible}/test` (in-distribution test set)
- **Pretrained base:** Text_IF full-recon v2 (FFBlock variant) — fine-tuned per T value
- **Hardware/env:** Intel XPU, `D:/software/anaconda3/envs/xpu/python.exe` (torch 2.10.0+xpu)
- **Pipeline:** `sweeps/run_sweep.py` (train + eval + aggregate, fail-fast, idempotent)
- **Eval script:** `evaluate_textif_full_recon_v2.py` (21 fusion metrics per image)

## Phase 1 — Main sweep (T ∈ {0, 1, 2, 3, 5, 8, 10, 15})

Full results in [`out/sweep_summary.csv`](out/sweep_summary.csv) (wide) and [`out/sweep_summary_long.csv`](out/sweep_summary_long.csv) (tidy).
Visualizations: [`out/sweep_variation.png`](out/sweep_variation.png) (per-metric curves), [`out/tradeoff_scatter.png`](out/tradeoff_scatter.png), [`out/tradeoff_radar.png`](out/tradeoff_radar.png).

### Trends vs text_ratio (T0 → T15)

| Trend | Metrics | Magnitude |
|---|---|---|
| Strong ↑ | SF, AG, EI, Qabf, Qcb, VIF | SF 11.4→19.6, EI 29.3→41.2, Qabf 0.465→0.670 |
| Mild ↑ | SSIM, MS_SSIM, SD, Nabf | MS_SSIM 0.914→0.945 |
| Flat | EN, NMI, MI, CE, QNCIE, TE, PSNR, MSE | EN ≈ 7.25 throughout, PSNR ≈ 14.7 |
| Mild ↓ | CC, SCD, Qy | CC 0.655→0.620, Qy 0.932→0.906 |

### Key trade-off

Detail/edge metrics improve monotonically with `text_ratio`, while source-correlation metrics (CC, Qy, SCD) degrade. Information-theoretic metrics (EN, MI, NMI, CE) are essentially insensitive.

### Phase-1 conclusion

- **T0** is Pareto-dominated by T1/T2 (no text → lowest detail, no fidelity advantage).
- **T5–T10** is the Pareto front: structural metrics near peak, detail metrics substantially improved, fidelity loss still acceptable.
- **T15** shows diminishing returns — detail barely improves over T10, but Qy/CC continue to drop.

## Phase 2 — Generalization study (T ∈ {5, 8, 10} × 5 datasets, 20 imgs each)

Sampling: 20 images per dataset, `seed=42` (same sample across T values for fair comparison).
Full results in [`out/generalization_wide.csv`](out/generalization_wide.csv) and [`out/generalization_long.csv`](out/generalization_long.csv).
Visualizations: [`out/generalization_bars.png`](out/generalization_bars.png), [`out/generalization_lines.png`](out/generalization_lines.png).

### Datasets

| Short | Path | Resolution | Role |
|---|---|---|---|
| MSRS  | `dataset/MSRS-main/test`                          | 640×480  | held-out |
| LLVIP | `dataset/LLVIP`                                   | 1280×1024 | in-distribution (matches Phase-1 test set) |
| RoadScene | `dataset/RoadScene-aligned`                  | ~500×330 (per-pair aligned) | held-out, distribution-shifted |
| M3FD  | `dataset/M3FD_Detection`                          | 1024×768 | held-out |
| FLIR  | `dataset/FLIR-align-3class/FLIR-align-3class`     | 640×512  | held-out, high sensor noise |

> **RoadScene alignment caveat (corrected 2026-08-03).** The raw
> `dataset/RoadScene/{infrared,visible}/` folders are NOT pixel-aligned
> (IR is uniformly 640×512; VIS varies from 787×759 to 1819×1051, 0/221 pairs
> match in size). Running the eval on the raw folder produced visibly stretched
> fused images — VIS got bilinarly squashed to IR's 640×512 aspect ratio AND
> the scene content was not actually aligned to begin with. The aligned
> variant `dataset/RoadScene-aligned/` was built from
> `RoadScene-master/{cropinfrared,crop_LR_visible}/` (221/221 perfectly
> size-matched pairs) and is what the numbers below use. The first run on
> unaligned data inflated SF/AG/Qabf/MS_SSIM (stretching artifacts) and
> deflated MI; the corrected numbers are more honest.

### Consistency of the detail-improvement trend (T5 → T10)

| Metric | FLIR | LLVIP | M3FD | MSRS | RoadScene |
|---|---|---|---|---|---|
| SF  | 26→30 | 15→18 | 15→16 | 17→21 | 29→31 |
| AG  | 6.3→7.1 | 3.5→3.8 | 4.5→4.8 | 4.2→4.7 | 5.8→6.5 |
| EI  | 68→75 | 38→41 | 49→51 | 47→51 | 63→69 |
| Qabf | .540→.582 | .624→.663 | .635→.662 | .645→.660 | .526→.577 |
| Qcb  | .441→.467 | .444→.484 | .483→.511 | .532→.529 | .407→.436 |

→ **Detail/edge improvement generalizes to every dataset tested.** This is the most robust finding.

### Dataset-specific fidelity response

- **SSIM** degrades on FLIR / MSRS / M3FD but **improves on LLVIP** (0.819 → 0.848). Not a universal trade-off.
- **CC** drops on 4/5 datasets; M3FD is essentially flat.
- **PSNR** is most sensitive on FLIR (12.7 → 12.4); others are stable.

### Per-dataset characteristics (corrected, RoadScene-aligned)

- **RoadScene** (aligned): distribution-shifted. SF 20.8→22.4, AG 5.4→5.9, Qabf 0.489→0.519 (T5→T10) — detail trend holds but absolute values are lower than other datasets because crops are ~500×330 (less high-frequency content per pixel). CC ≈ 0.40, MI ≈ 3.04 (highest mutual information, suggesting strong scene overlap).
- **MSRS** is closest to the training distribution (highest NMI ≈ 0.56); most stable behavior.
- **FLIR** has the highest sensor noise — PSNR baseline low (12.7); most sensitive to text_ratio increases.

## Overall conclusions

1. **Detail/edge improvement from `text_ratio` is robust** — generalizes across all 5 tested datasets.
2. **Fidelity trade-off is dataset-dependent** — present on most datasets but absent or even reversed on LLVIP (SSIM).
3. **T5 is the best-generalizing sweet spot** — improves detail everywhere with minimal fidelity loss, including on distribution-shifted datasets (FLIR, RoadScene).
4. **T10 is acceptable when the target distribution is close to training** (MSRS/LLVIP/M3FD); avoid T10 on high-noise data (FLIR).

## File index

| File | Description |
|---|---|
| [`out/sweep_summary.csv`](out/sweep_summary.csv) | Phase-1 wide CSV (metric × T) |
| [`out/sweep_summary_long.csv`](out/sweep_summary_long.csv) | Phase-1 long/tidy CSV |
| [`out/sweep_variation.png`](out/sweep_variation.png) | 21 metrics vs text_ratio |
| [`out/tradeoff_scatter.png`](out/tradeoff_scatter.png) | Detail-vs-fidelity scatter pairs |
| [`out/tradeoff_radar.png`](out/tradeoff_radar.png) | Normalized radar per T |
| [`out/generalization_wide.csv`](out/generalization_wide.csv) | Phase-2 wide CSV |
| [`out/generalization_long.csv`](out/generalization_long.csv) | Phase-2 long/tidy CSV |
| [`out/generalization_bars.png`](out/generalization_bars.png) | Per-dataset bar groups |
| [`out/generalization_lines.png`](out/generalization_lines.png) | Qabf / SSIM per dataset |
| [`aggregate_results.py`](aggregate_results.py) | Phase-1 aggregation + charts |
| [`tradeoff_plots.py`](tradeoff_plots.py) | Phase-1 trade-off scatter + radar |
| [`run_generalization.py`](run_generalization.py) | Phase-2 cross-dataset eval driver |

## Reproduce

```bash
# Phase 1 (already run; idempotent — skips completed T values)
D:/software/anaconda3/envs/xpu/python.exe sweeps/run_sweep.py \
    --text_ratios 0,1,2,3,5,8,10,15 \
    --pretrained_weights <path/to/textif-me/checkpoint.pth> \
    --dataset_ll dataset/EMS_lite/Low_light \
    --dataset_oe dataset/EMS_lite/Over_exposure \
    --dataset_ic dataset/EMS_lite/IR_Low_contrast \
    --dataset_in dataset/EMS_lite/IR_Noise \
    --eval_data_path dataset/LLVIP \
    --output_root sweeps/out

# Phase 1 aggregation + charts
D:/software/anaconda3/envs/xpu/python.exe sweeps/aggregate_results.py
D:/software/anaconda3/envs/xpu/python.exe sweeps/tradeoff_plots.py

# Phase 2 generalization (T5/T8/T10 × 5 datasets × 20 imgs)
D:/software/anaconda3/envs/xpu/python.exe sweeps/run_generalization.py --sample 20 --seed 42 --device xpu
```

## v5 smoke (2026-08-07)

Verified v5 pipeline boots correctly up to the training-iteration boundary. Deferred to CUDA for full smoke (see "Smoke scope" below).

**Verified on XPU (boot path, not training):**
- Data loading: 4× "Loading IVF Fusion ..." banners, all 4 task datasets enumerated with correct counts.
- Checkpoint transfer: 659/659 v2-ft pretrained keys loaded; only `ffb_{1,2,3}.spatial_attn.conv.weight` (+ CLIP) random-init as expected (matches V4 spec).
- Model construction: `FFBlockSCA(use_spatial=True)` instantiates cleanly.
- Status banners: "Encoders frozen. Training: FFBlockSCA(use_spatial=True), ..." and "Fine-tuning Text_IF_Recon v5 from textif-me (use_spatial=True, ..." print correctly.

**Smoke scope (deferred to CUDA):**
The first training iteration requires `scripts/losses.py`, which (like all v2-ft training) uses hardcoded `.cuda()` calls (e.g. `L_Grad_position.__init__` at line 144). The Intel XPU env (torch 2.10.0+xpu) has no CUDA. **This is identical to v2-ft behavior** — v2-ft also imports `scripts.utils` → `scripts.losses` and would crash the same way on XPU. The repo has `scripts/losses_xpu.py` / `scripts/utils_xpu.py` but only `fusion_prompt_loss` is ported (the v2-ft recon losses are not), and there is no `train_fusion_full_recon_v2_ft_xpu.py`. v2-ft was originally trained on CUDA; v5-ft will follow the same workflow.

**torch.load fix (commit 3e98a62):** PyTorch 2.6+ `weights_only=True` default blocked loading the v2-ft checkpoint's `argparse.Namespace`. Added `weights_only=False` to `torch.load` at lines 134 and 162 of `train_fusion_full_recon_v5_ft.py`. The same fix was needed in `tests/test_v5_checkpoint_transfer.py`.

**Status:** v5 code complete and unit-tested (V1–V4 pass). Pipeline boots correctly on XPU. Full-iteration smoke + A/B/C ablation runs deferred to CUDA machine.
