#!/usr/bin/env bash
#SBATCH --job-name=bc_contact_z16_latent
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the action_bridge_policy directory.}}"
cd "${PROJECT_DIR}"
test -d action_bridge || { echo "Expected to run from action_bridge_policy; got ${PROJECT_DIR}" >&2; exit 2; }

mkdir -p outputs/hpc_logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

SEED="${SEED:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
DATASET_PATH="${PUSHT_DATASET_PATH:-data/pusht/pusht_cchi_v7_replay.zarr}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-bc-ablation}"
WANDB_GROUP="${WANDB_GROUP:-contact_medium_z16_latent_use_seed${SEED}}"
RUN_ID="bc_contact_medium_z16_latent_use_seed${SEED}_${RUN_SUFFIX}"

echo "Project: ${PROJECT_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Run: ${RUN_ID}"
echo "W&B project/group: ${WANDB_PROJECT}/${WANDB_GROUP}"
echo "Node: ${SLURMD_NODENAME:-unknown-node}"

uv run python -m action_bridge.training.train_pusht \
  --config-name pusht_lowdim_contact_quadratic \
  run_id="${RUN_ID}" \
  seed="${SEED}" \
  device=cuda \
  obs_dim=5 \
  data.dataset_path="${DATASET_PATH}" \
  data.backend=auto \
  model.hidden_dim=704 \
  model.h_emb_dim=704 \
  model.encoder_depth=3 \
  model.control_depth=4 \
  model.time_emb_dim=64 \
  model.z_embed_dim=128 \
  model.z_dim=16 \
  reference.hidden_dim=192 \
  reference.time_emb_dim=64 \
  loss.beta_z_start=0.0 \
  loss.beta_z_end=0.0003 \
  loss.beta_z_warmup_steps=100000 \
  loss.free_nats=0.0 \
  loss.num_z_samples_train=4 \
  loss.vectorize_z_samples_train=true \
  optim.batch_size=512 \
  optim.max_steps=300000 \
  logging.log_every_steps=100 \
  logging.eval_every_steps=1000 \
  logging.checkpoint_every_steps=20000 \
  logging.full_eval_every_steps=10000 \
  logging.full_eval_split=val \
  logging.full_eval_max_batches=4 \
  logging.full_eval_closed_loop_episodes=64 \
  logging.full_eval_num_samples=128 \
  logging.sim_eval_enabled=true \
  logging.sim_eval_async=true \
  logging.sim_eval_device=cpu \
  logging.sim_eval_num_workers=6 \
  logging.sim_eval_worker_threads=1 \
  logging.sim_eval_max_pending=2 \
  logging.sim_eval_every_steps=5000 \
  logging.sim_eval_episodes=100 \
  logging.sim_eval_max_steps=500 \
  logging.sim_eval_n_exec=8 \
  logging.sim_eval_render_episodes=4 \
  logging.sim_eval_save_gifs=true \
  logging.sim_eval_collect_contact_diagnostics=true \
  logging.sim_eval_keep_checkpoints=false \
  logging.wandb.enabled=true \
  logging.wandb.project="${WANDB_PROJECT}" \
  logging.wandb.group="${WANDB_GROUP}" \
  logging.wandb.log_images=false \
  logging.wandb.tags="[\"bc_ablation\",\"contact_medium\",\"z16\",\"latent_use\"]" \
  > "outputs/hpc_logs/${RUN_ID}.log" 2>&1
