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
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ── Edit these to match your environment ─────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/trl}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data/stage1}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/checkpoints/stage1_smoke}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
CONDA_ENV=${CONDA_ENV:-vlm}
# ─────────────────────────────────────────────────────────────────────────────

# ── Activate the conda env that has torch + transformers + trl[dev] installed.
#    Auto-detects miniconda/anaconda in the usual locations. Override CONDA_ENV
#    on the sbatch line if you use a different env name.
for _conda_root in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
  if [ -f "$_conda_root/etc/profile.d/conda.sh" ]; then
    source "$_conda_root/etc/profile.d/conda.sh"
    break
  fi
done
conda activate "$CONDA_ENV"

mkdir -p "$(dirname "${SLURM_SUBMIT_DIR:-$PWD}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out")"

cd "$PROJECT_DIR"
export HF_HOME

echo "[stage1_smoke] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES conda_env=$CONDA_DEFAULT_ENV"
python -c "import torch, trl; print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}  trl={trl.__file__}')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python examples/scripts/distill_vlm/train_stage1.py \
  --dataset_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --smoke_test
