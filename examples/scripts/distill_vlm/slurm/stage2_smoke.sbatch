#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Why this file exists:
#   SLURM batch script for the Stage 2 VLM-distillation smoke test. Runs
#   `train_stage2.py --smoke_test` (50 samples, 2 optimizer steps) and starts
#   the teacher (VILA-M3 in the medical example) on a co-located GPU on the
#   same node.
#
#   Layout: 1 node, 2 GPUs.
#       GPU 0 -> SmolVLM2 student training (`train_stage2.py`)
#       GPU 1 -> teacher vLLM server (`vllm serve ...`)
#
#   Edit the SBATCH header (account/partition/path placeholders) and PROJECT_DIR
#   / DATA_DIR / STAGE1_CKPT below to match your cluster, then submit with:
#       sbatch examples/scripts/distill_vlm/slurm/stage2_smoke.sbatch
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=trl_distill_vlm_stage2_smoke
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ── Edit these to match your environment ─────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-$HOME/trl}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data/stage2}
STAGE1_CKPT=${STAGE1_CKPT:-$PROJECT_DIR/checkpoints/stage1}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/checkpoints/stage2_smoke}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
TEACHER_MODEL=${TEACHER_MODEL:-Efficient-Large-Model/VILA-M3-7B}
TEACHER_PORT=${TEACHER_PORT:-8000}
# Activation: see stage1_smoke.sbatch for examples (conda / venv / module).
# ─────────────────────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"
export HF_HOME

echo "[stage2_smoke] node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── 1. Launch VILA-M3 teacher on GPU 1 (background) ─────────────────────────
TEACHER_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.teacher.log"
mkdir -p "$(dirname "$TEACHER_LOG")"

CUDA_VISIBLE_DEVICES=1 vllm serve "$TEACHER_MODEL" \
  --port "$TEACHER_PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  > "$TEACHER_LOG" 2>&1 &

TEACHER_PID=$!
trap 'kill $TEACHER_PID 2>/dev/null || true' EXIT

# ── 2. Wait for the teacher to accept connections ───────────────────────────
echo "[stage2_smoke] waiting for teacher on http://localhost:${TEACHER_PORT} ..."
for i in $(seq 1 90); do
  if curl -fs "http://localhost:${TEACHER_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[stage2_smoke] teacher ready after ${i}s"
    break
  fi
  if ! kill -0 "$TEACHER_PID" 2>/dev/null; then
    echo "[stage2_smoke] teacher died, see $TEACHER_LOG"; exit 1
  fi
  sleep 5
done

# ── 3. Launch the student smoke training on GPU 0 ───────────────────────────
CUDA_VISIBLE_DEVICES=0 python examples/scripts/distill_vlm/train_stage2.py \
  --stage1_ckpt "$STAGE1_CKPT" \
  --dataset_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --teacher_url "http://localhost:${TEACHER_PORT}" \
  --smoke_test
