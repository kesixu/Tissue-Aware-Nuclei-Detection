#!/bin/bash
# =============================================================================
# TAND Training Example — SLURM Job Script
# =============================================================================
# Best configuration (V7a): Spatial-FiLM + augmentation + rare-class oversampling
#
# Reproduces the results from the paper:
#   - Detection F1: 0.864 +/- 0.008
#   - Classification Macro-F1: 0.398
#
# Adjust --partition, --gres, --batch-size for your GPU hardware:
#   - NVIDIA L4 (24 GB):  batch_size=2, pretrain_batch_size=6,  ~24h per fold
#   - NVIDIA A100 (80 GB): batch_size=8, pretrain_batch_size=10, ~10h per fold
# =============================================================================
set -e

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=python                           # or path to your conda env python
DATA_ROOT="${PROJECT_DIR}/data/puma_coco_folds"   # PUMA dataset (5-fold split)
FOLD=1                                  # fold to train (1-5)
BATCH_SIZE=2                            # adjust for your GPU VRAM
PRETRAIN_BS=6                           # Virchow seg head pretrain batch size
GPU_TYPE="l4"                           # l4, a100, v100, etc.
PARTITION="gpu"                         # your SLURM partition name
TIME_LIMIT="28:00:00"                   # max wall time
# ──────────────────────────────────────────────────────────────────────────────

TISSUE_ROOT="${DATA_ROOT}/fold_${FOLD}/tissue_masks"
RUN_TAG="tand_v7a_${GPU_TYPE}_fold${FOLD}_$(date +%Y%m%d)"
SCRIPT="${PROJECT_DIR}/scripts/train.py"

# Sqrt-inverse-frequency class weights for PUMA 10-class setting
# Computed from training set class distribution
SQRT_WEIGHTS="1.03,1.77,0.69,0.18,0.50,1.01,0.91,1.59,2.03,0.28"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tand_f${FOLD}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:${GPU_TYPE}:1
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${PROJECT_DIR}/logs/tand_f${FOLD}_%j.out
#SBATCH --error=${PROJECT_DIR}/logs/tand_f${FOLD}_%j.err

echo "=== TAND Training ==="
echo "Job: \$SLURM_JOB_ID | Fold: ${FOLD} | GPU: ${GPU_TYPE} | \$(hostname) | \$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd ${PROJECT_DIR}
export PYTHONPATH=\${PWD}:\${PYTHONPATH}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set TORCH_HOME to avoid downloading models to home directory
export TORCH_HOME=${PROJECT_DIR}/.cache/torch
export HF_HOME=${PROJECT_DIR}/.cache/huggingface

mkdir -p logs .cache/torch .cache/huggingface

${PYTHON} ${SCRIPT} \\
    --model virchow_fused \\
    --trainer virchow_fused \\
    --fusion-model efficient \\
    --data-root ${DATA_ROOT} \\
    --data-mode full \\
    --full-resize 1024 \\
    --no-confirm \\
    --mode film_only \\
    --film-limit 0.5 \\
    --film-start-epoch 20 \\
    --det-thr 0.35 \\
    --nms-radius 3 \\
    --match-radius 15 \\
    --num-tissue 6 \\
    --heat-focal-alpha 0.25 \\
    --heat-focal-gamma 2.0 \\
    --seg-loss-weight 0.3 \\
    --heat-loss-weight 1.0 \\
    --cls-loss-weight 1.0 \\
    --lam-bias 0.0 \\
    --conf-thr 0.6 \\
    --tau 1.0 \\
    --pretrain-earlystop-patience -1 \\
    --pretrain-lr-schedule cosine \\
    --pretrain-amp off \\
    --train-amp off \\
    --save-every 20 \\
    --epochs 200 \\
    --cls-focal-gamma 2.0 \\
    --cls-class-weights ${SQRT_WEIGHTS} \\
    --lr-schedule cosine \\
    --lr-warmup-epochs 10 \\
    --augment \\
    --oversample-rare 3.0 \\
    --batch-size ${BATCH_SIZE} \\
    --pretrain-batch-size ${PRETRAIN_BS} \\
    --pretrain-seg-epochs 50 \\
    --pretrain-lr 0.0002 \\
    --only-fold ${FOLD} \\
    --tissue-mask-root ${TISSUE_ROOT} \\
    --run-tag ${RUN_TAG}

echo "Done: TAND fold=${FOLD} at \$(date)"
EOF

echo "Submitted TAND training: fold=${FOLD}, gpu=${GPU_TYPE}, bs=${BATCH_SIZE}"
echo "Monitor: squeue -u \$(whoami)"
