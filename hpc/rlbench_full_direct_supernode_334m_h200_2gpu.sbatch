#!/usr/bin/env bash
#SBATCH --job-name=rlb_full_direct_334m
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the action_bridge_policy directory.}"
test -d action_bridge || { echo "Expected action_bridge_policy as the submit directory." >&2; exit 2; }
ulimit -n 65536 2>/dev/null || echo "Warning: could not raise the open-file limit." >&2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cuda
export WANDB__SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=120
export WANDB_CONSOLE=off

SEED="${SEED:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${RLBENCH_CACHE_ROOT:-data/rlbench_cache}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_STEPS="${MAX_STEPS:-200000}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-rlbench}"
WANDB_GROUP="${WANDB_GROUP:-full_allvars_large_comparison}"
RUN_ID="rlbench_full_direct_supernode_334m_seed${SEED}_${RUN_SUFFIX}"

echo "Run: ${RUN_ID}"
echo "Dataset: all cached tasks and variations under ${CACHE_ROOT}"
echo "Global batch: ${BATCH_SIZE}; devices: 2"
nvidia-smi -L

uv run --extra jax-cu13 python -m action_bridge.jax.training.train_rlbench \
  --config-name rlbench_jax_direct_chunk_bc \
  run_id="${RUN_ID}" \
  seed="${SEED}" \
  data.cache_root="${CACHE_ROOT}" \
  data.tasks='[]' \
  data.variation_ids='[]' \
  data.sampling_strategy=task_uniform \
  data.point_count=1024 \
  data.preload_to_memory=false \
  data.prefetch_workers=4 \
  data.prefetch_batches=4 \
  encoder.type=supernode \
  encoder.d_model=1024 \
  encoder.n_heads=16 \
  encoder.mlp_mult=4 \
  encoder.supernode_layers=4 \
  encoder.history_layers=2 \
  encoder.query_layers=2 \
  encoder.query_num_latents=128 \
  encoder.use_task_tokens=true \
  decoder.num_layers=8 \
  bridge.hidden_dim=2048 \
  distributed.num_devices=2 \
  optim.batch_size="${BATCH_SIZE}" \
  optim.max_steps="${MAX_STEPS}" \
  logging.log_every_steps=100 \
  logging.val_every_steps=5000 \
  logging.val_batches=4 \
  logging.checkpoint_every_steps=50000 \
  logging.artifact_every_steps=10000 \
  logging.artifact_examples=4 \
  logging.wandb.enabled=true \
  logging.wandb.entity=ricvalp \
  logging.wandb.project="${WANDB_PROJECT}" \
  logging.wandb.group="${WANDB_GROUP}" \
  logging.wandb.log_plotly=false \
  logging.wandb.tags='["rlbench","full_dataset","all_variations","direct_chunk_bc","supernode","task_conditioned","334m","h200","2gpu"]' \
  checkpoint.save_best_val=false
