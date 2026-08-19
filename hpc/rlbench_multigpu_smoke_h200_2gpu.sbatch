#!/usr/bin/env bash
#SBATCH --job-name=rlbench_mgpu_smoke_h200
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
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
export WANDB_CONSOLE=off

SEED="${SEED:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${RLBENCH_CACHE_ROOT:-data/rlbench_cache}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-rlbench-smoke}"
RUN_ID="rlbench_mgpu_smoke_h200_seed${SEED}_${RUN_SUFFIX}"

echo "Run: ${RUN_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
nvidia-smi -L

uv run --extra jax-cu13 python -m action_bridge.jax.training.train_rlbench \
  --config-name rlbench_jax_contact_bridge \
  run_id="${RUN_ID}" \
  seed="${SEED}" \
  data.cache_root="${CACHE_ROOT}" \
  data.tasks='[]' \
  data.variation_ids='[]' \
  data.max_episodes_per_variation=1 \
  data.point_count=128 \
  data.preload_to_memory=false \
  data.prefetch_workers=2 \
  data.prefetch_batches=2 \
  encoder.d_model=128 \
  encoder.n_heads=4 \
  encoder.supernode_layers=1 \
  encoder.history_layers=1 \
  encoder.query_layers=1 \
  encoder.query_num_latents=16 \
  encoder.use_task_tokens=true \
  decoder.num_layers=2 \
  bridge.hidden_dim=256 \
  bridge.control_depth=2 \
  bridge.reference_depth=2 \
  bridge.auxiliary_depth=2 \
  bridge.z_dim=4 \
  bridge.z_embed_dim=32 \
  distributed.num_devices=2 \
  optim.batch_size=16 \
  optim.max_steps=4 \
  logging.log_every_steps=1 \
  logging.val_every_steps=2 \
  logging.val_batches=1 \
  logging.checkpoint_every_steps=4 \
  logging.artifact_every_steps=4 \
  logging.artifact_examples=2 \
  logging.wandb.enabled=true \
  logging.wandb.entity=ricvalp \
  logging.wandb.project="${WANDB_PROJECT}" \
  logging.wandb.group=rlbench_multigpu_smoke \
  logging.wandb.log_plotly=false \
  logging.wandb.tags='["rlbench","smoke","h200","multi_gpu"]'
