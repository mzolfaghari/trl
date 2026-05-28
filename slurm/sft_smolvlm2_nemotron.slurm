#!/bin/bash
#SBATCH --job-name=sft_smolvlm2
#SBATCH --chdir=/home/reza/projects/trl
#SBATCH --output=logs/sft_smolvlm2_nemotron/%A_%a.out
#SBATCH --error=logs/sft_smolvlm2_nemotron/%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=5
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=88
#SBATCH --partition=VLM
#SBATCH --qos=normal
#SBATCH --mem=500G

# ─────────────────────────────────────────────────────────────────────────────
# Thin SLURM submitter for examples/scripts/sft_smolvlm2_nemotron.py.
#
# Usage:
#   sbatch slurm/sft_smolvlm2_nemotron.slurm configs/sft_smolvlm2/<config>.yaml
#
# The YAML is the single source of truth for everything experiment-specific
# (model, data paths, LRs, batch size, epochs, intervals). Override #SBATCH
# directives on the sbatch CLI when running smaller-scale variants, e.g.:
#
#   sbatch --nodes=1 --gpus-per-node=1 --job-name=smoke \
#       slurm/sft_smolvlm2_nemotron.slurm configs/sft_smolvlm2/nemotron_smoke.yaml
# ─────────────────────────────────────────────────────────────────────────────

CONFIG="${1:?Usage: sbatch $0 <path/to/config.yaml>}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}"
  exit 1
fi

# ── EFA / NCCL settings ──────────────────────────────────────────────────────
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_HUGE_PAGE=0
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_SOCKET_IFNAME=eth0

export FI_EFA_ENABLE_SHM_TRANSFER=0
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# Disable NCCL heartbeat watchdog (large checkpoints over NFS can stall it).
export TORCH_NCCL_ENABLE_MONITORING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

# ── Environment ──────────────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate vlm-distill

export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)

mkdir -p logs/sft_smolvlm2_nemotron
cd /home/reza/projects/trl

# ── Rendezvous ───────────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) |
        shuf | head -n "$N"
}
export MASTER_PORT=$(unused_port)

ulimit -n 99999

echo "CONFIG=${CONFIG}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${SLURM_NNODES} GPUS_PER_NODE=${SLURM_GPUS_PER_NODE}"

# Free stale node-local JSONL caches from prior crashed jobs. These live in
# /dev/shm or /tmp and can fill the partition (breaking NCCL) or confuse
# JsonlIndex's "is the local copy up-to-date?" check. One task per node.
srun --ntasks-per-node=1 bash -c '
  rm -rf /tmp/nemotron_jsonl_cache /tmp/nemotron_*.jsonl.part 2>/dev/null || true
  rm -f /dev/shm/nemotron_*.jsonl /dev/shm/nemotron_*.jsonl.part /dev/shm/nemotron_*.ready 2>/dev/null || true
'

# Snapshot the resolved config alongside the job logs so the run is
# reproducible from the log directory alone.
RUN_LOG_DIR="logs/sft_smolvlm2_nemotron"
cp "${CONFIG}" "${RUN_LOG_DIR}/${SLURM_JOB_ID}.yaml"

srun --kill-on-bad-exit=1 torchrun \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf=read_timeout=1800,join_timeout=1800 \
    examples/scripts/sft_smolvlm2_nemotron.py --config "${CONFIG}"

echo "=== finetune complete ==="
