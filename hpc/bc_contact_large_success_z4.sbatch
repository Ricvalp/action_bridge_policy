#!/usr/bin/env bash
#SBATCH --job-name=bc_contact_large_success
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}}"
cd "${PROJECT_DIR}"
test -d action_bridge || { echo "Expected action_bridge_policy; got ${PROJECT_DIR}" >&2; exit 2; }
mkdir -p hpc/logs outputs/hpc_logs

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
WANDB_GROUP="${WANDB_GROUP:-contact_large_success_z4_seed${SEED}}"
RUN_ID="bc_contact_large_success_z4_seed${SEED}_${RUN_SUFFIX}"

uv run python -m action_bridge.training.train_pusht \
  --config-name pusht_lowdim_contact_quadratic \
  run_id="${RUN_ID}" \
  seed="${SEED}" \
  device=cuda \
  obs_dim=5 \
  data.dataset_path="${DATASET_PATH}" \
  data.backend=auto \
  inference.latent_commitment=chunk \
  model.hidden_dim=2560 \
  model.h_emb_dim=2560 \
  model.encoder_depth=4 \
  model.control_depth=6 \
  model.time_emb_dim=128 \
  model.z_embed_dim=320 \
  model.z_dim=4 \
  reference.hidden_dim=640 \
  reference.time_emb_dim=128 \
  loss.beta_z_start=0.0 \
  loss.beta_z_end=0.0003 \
  loss.beta_z_warmup_steps=100000 \
  loss.free_nats=0.0 \
  loss.num_z_samples_train=4 \
  loss.vectorize_z_samples_train=true \
  loss.lambda_unroll=1.0 \
  loss.lambda_unroll_warmup_steps=5000 \
  optim.lr=0.0002 \
  optim.batch_size=1024 \
  optim.max_steps=500000 \
  optim.grad_clip=1.0 \
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
  logging.sim_eval_num_workers=8 \
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
  logging.wandb.tags="[\"success_optimization\",\"contact_bridge\",\"large\",\"z4\",\"batch1024\"]" \
  > "outputs/hpc_logs/${RUN_ID}.log" 2>&1
