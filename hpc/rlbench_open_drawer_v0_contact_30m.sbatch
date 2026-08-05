#!/usr/bin/env bash
#SBATCH --job-name=rlbench_open_drawer_v0_contact_30m
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the action_bridge_policy directory.}}"
cd "${PROJECT_DIR}"
test -d action_bridge || { echo "Expected to run from action_bridge_policy; got ${PROJECT_DIR}" >&2; exit 2; }

mkdir -p outputs/hpc_logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

SEED="${SEED:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${RLBENCH_CACHE_ROOT:-data/rlbench_cache}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_STEPS="${MAX_STEPS:-300000}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-rlbench}"
WANDB_GROUP="${WANDB_GROUP:-open_drawer_v0_contact_30m}"
RUN_ID="rlbench_open_drawer_v0_contact_30m_seed${SEED}_${RUN_SUFFIX}"

echo "Project: ${PROJECT_DIR}"
echo "Cache: ${CACHE_ROOT}/open_drawer/variation0.h5"
echo "Run: ${RUN_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown-node}"

uv run --extra jax-cu13 python -m action_bridge.jax.training.train_rlbench \
  --config-name rlbench_jax_contact_bridge \
  run_id="${RUN_ID}" \
  seed="${SEED}" \
  data.cache_root="${CACHE_ROOT}" \
  data.tasks='["open_drawer"]' \
  data.variation_ids='[0]' \
  data.preload_to_memory=true \
  data.prefetch_workers=8 \
  data.prefetch_batches=8 \
  encoder.d_model=384 \
  encoder.n_heads=8 \
  bridge.hidden_dim=864 \
  bridge.z_embed_dim=96 \
  bridge.z_dim=4 \
  optim.batch_size="${BATCH_SIZE}" \
  optim.max_steps="${MAX_STEPS}" \
  logging.log_every_steps=100 \
  logging.val_every_steps=2000 \
  logging.val_batches=8 \
  logging.checkpoint_every_steps=20000 \
  logging.artifact_every_steps=10000 \
  logging.wandb.enabled=true \
  logging.wandb.entity=ricvalp \
  logging.wandb.project="${WANDB_PROJECT}" \
  logging.wandb.group="${WANDB_GROUP}" \
  logging.wandb.log_plotly=false \
  logging.wandb.tags='["rlbench","open_drawer","variation0","contact_bridge","30m"]' \
  > "outputs/hpc_logs/${RUN_ID}.log" 2>&1
