#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Why this file exists:
#   SLURM batch script for the Stage 2 VLM-distillation smoke test. Runs
#   `train_stage2.py --smoke_test` (50 samples, 2 optimizer steps) and starts
#   the teacher (a public VLM, default `SmolVLM2-2.2B-Instruct`; override
#   `TEACHER_MODEL` to swap, e.g. `Qwen/Qwen2.5-VL-3B-Instruct` or the
#   gated MONAI VILA-M3 repos for the medical workflow) on a co-located
#   GPU on the same node.
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
PROJECT_DIR=${PROJECT_DIR:-$HOME/projects/trl}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/data/stage2}
STAGE1_CKPT=${STAGE1_CKPT:-$PROJECT_DIR/checkpoints/stage1}
OUT_DIR=${OUT_DIR:-$PROJECT_DIR/checkpoints/stage2_smoke}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
TEACHER_MODEL=${TEACHER_MODEL:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}
TEACHER_PORT=${TEACHER_PORT:-8000}
# Two conda envs are used because vLLM and the trl trainer have incompatible
# dependency requirements (e.g. transformers 4.x for vLLM 0.10.x vs. 5.x for
# the trainer; CUDA 12 vs 13 wheel mismatches; etc.). Keeping them separate
# is the only sane way to avoid an endless dependency-fight.
SERVE_ENV=${SERVE_ENV:-vlm-serve-v1}      # vLLM teacher server
TRAIN_ENV=${TRAIN_ENV:-vlm}               # trl student trainer
# ─────────────────────────────────────────────────────────────────────────────

# ── Source conda from the usual install locations so `conda activate` works.
for _conda_root in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
  if [ -f "$_conda_root/etc/profile.d/conda.sh" ]; then
    source "$_conda_root/etc/profile.d/conda.sh"
    break
  fi
done

cd "$PROJECT_DIR"
export HF_HOME
mkdir -p logs

echo "[stage2_smoke] node=$(hostname) serve_env=$SERVE_ENV train_env=$TRAIN_ENV"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── 1. Launch teacher on GPU 1 (background) under the serving env ───────────
#    The serving env is intentionally separate from the training env: vLLM
#    pins older transformers/torch combos that the trainer can't tolerate.
TEACHER_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.teacher.log"
conda activate "$SERVE_ENV"
python -c "import torch, vllm; print(f'  [serve] torch={torch.__version__} vllm={vllm.__version__}')"

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
    echo "[stage2_smoke] teacher ready after $((i*5))s"
    break
  fi
  if ! kill -0 "$TEACHER_PID" 2>/dev/null; then
    echo "[stage2_smoke] teacher died, see $TEACHER_LOG"; exit 1
  fi
  sleep 5
done

# ── 3. Switch to the training env and launch the student on GPU 0 ───────────
conda deactivate
conda activate "$TRAIN_ENV"
python -c "import torch, trl; print(f'  [train] torch={torch.__version__} trl={trl.__file__}')"

CUDA_VISIBLE_DEVICES=0 python examples/scripts/distill_vlm/train_stage2.py \
  --stage1_ckpt "$STAGE1_CKPT" \
  --dataset_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --teacher_url "http://localhost:${TEACHER_PORT}" \
  --smoke_test
