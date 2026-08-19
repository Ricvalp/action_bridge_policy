#!/usr/bin/env bash
#SBATCH --job-name=rlb_20task_allvar_64m
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
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
MAX_STEPS="${MAX_STEPS:-300000}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-rlbench}"
WANDB_GROUP="${WANDB_GROUP:-20tasks_allvars_medium}"
RUN_ID="rlbench_20tasks_allvars_contact_64m_seed${SEED}_${RUN_SUFFIX}"
TASKS='["basketball_in_hoop","close_jar","empty_dishwasher","insert_onto_square_peg","light_bulb_in","meat_off_grill","open_drawer","pick_and_lift","place_cups","place_shape_in_shape_sorter","push_button","push_buttons","put_groceries_in_cupboard","put_item_in_drawer","put_money_in_safe","reach_and_drag","slide_block_to_target","stack_blocks","stack_cups","turn_tap"]'

echo "Run: ${RUN_ID}"
echo "Tasks: ${TASKS}"
echo "Dataset: every cached variation of the selected tasks."
nvidia-smi -L

uv run --extra jax-cu13 python -m action_bridge.jax.training.train_rlbench \
  --config-name rlbench_jax_contact_bridge \
  run_id="${RUN_ID}" \
  seed="${SEED}" \
  data.cache_root="${CACHE_ROOT}" \
  data.tasks="${TASKS}" \
  data.variation_ids='[]' \
  data.sampling_strategy=task_uniform \
  data.point_count=1024 \
  data.preload_to_memory=true \
  data.prefetch_workers=2 \
  data.prefetch_batches=4 \
  encoder.type=supernode \
  encoder.d_model=512 \
  encoder.n_heads=8 \
  encoder.mlp_mult=4 \
  encoder.supernode_layers=3 \
  encoder.history_layers=2 \
  encoder.query_layers=2 \
  encoder.query_num_latents=64 \
  encoder.use_task_tokens=true \
  decoder.num_layers=4 \
  bridge.hidden_dim=1024 \
  bridge.control_depth=4 \
  bridge.reference_depth=3 \
  bridge.auxiliary_depth=3 \
  bridge.z_dim=8 \
  bridge.z_embed_dim=128 \
  distributed.num_devices=1 \
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
  logging.wandb.tags='["rlbench","20tasks","all_variations","contact_bridge","supernode","task_conditioned","64m","h200","1gpu"]' \
  checkpoint.save_best_val=false
