#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Why this file exists:
#   SLURM batch script for the Stage 1 connector-alignment smoke test of the
#   VLM knowledge-distillation pipeline. Runs `train_stage1.py --smoke_test`
#   (50 samples, 2 optimizer steps) on a single GPU.
#
#   Edit the SBATCH header (account/partition/path placeholders below) and the
#   PROJECT_DIR / DATA_DIR variables to match your cluster, then submit with:
#       sbatch examples/scripts/distill_vlm/slurm/stage1_smoke.sbatch
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=trl_distill_vlm_stage1_smoke
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ── Edit these to match your environment ─────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-$HOME/trl}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data/stage1}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/checkpoints/stage1_smoke}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
# Activation depends on how you installed the fork. Pick one:
#   - conda:  source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate trl
#   - venv:   source $PROJECT_DIR/.venv/bin/activate
#   - module: module load python/3.11 cuda/12.4
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$(dirname "${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out")"

cd "$PROJECT_DIR"
export HF_HOME

echo "[stage1_smoke] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python examples/scripts/distill_vlm/train_stage1.py \
  --dataset_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --smoke_test
