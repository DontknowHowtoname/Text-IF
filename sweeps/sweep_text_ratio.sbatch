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
