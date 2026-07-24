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

T="${1:-}"
if [[ -z "${T}" ]]; then
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
