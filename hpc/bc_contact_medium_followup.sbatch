#!/usr/bin/env bash
#SBATCH --job-name=bc_contact_medium
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
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

SEED="${SEED:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
DATASET_PATH="${PUSHT_DATASET_PATH:-data/pusht/pusht_cchi_v7_replay.zarr}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-bc-ablation}"
WANDB_GROUP="${WANDB_GROUP:-contact_medium_followup_seed${SEED}}"

COMMON_OVERRIDES=(
  "obs_dim=5"
  "data.dataset_path=${DATASET_PATH}"
  "data.backend=auto"
  "optim.batch_size=512"
  "optim.max_steps=300000"
  "logging.log_every_steps=100"
  "logging.eval_every_steps=1000"
  "logging.checkpoint_every_steps=10000"
  "logging.full_eval_every_steps=10000"
  "logging.full_eval_split=val"
  "logging.full_eval_max_batches=4"
  "logging.full_eval_closed_loop_episodes=64"
  "logging.full_eval_num_samples=32"
  "logging.sim_eval_enabled=true"
  "logging.sim_eval_async=true"
  "logging.sim_eval_device=cpu"
  "logging.sim_eval_num_workers=3"
  "logging.sim_eval_worker_threads=1"
  "logging.sim_eval_max_pending=1"
  "logging.sim_eval_every_steps=10000"
  "logging.sim_eval_episodes=100"
  "logging.sim_eval_max_steps=500"
  "logging.sim_eval_n_exec=8"
  "logging.sim_eval_render_episodes=4"
  "logging.sim_eval_save_gifs=true"
  "logging.sim_eval_collect_contact_diagnostics=true"
  "logging.wandb.enabled=true"
  "logging.wandb.project=${WANDB_PROJECT}"
  "logging.wandb.group=${WANDB_GROUP}"
  "logging.wandb.tags=[\"bc_ablation\",\"contact_medium\",\"followup\"]"
)

MEDIUM_CONTACT_OVERRIDES=(
  "model.hidden_dim=512"
  "model.h_emb_dim=512"
  "model.encoder_depth=3"
  "model.control_depth=4"
  "model.time_emb_dim=48"
  "model.z_embed_dim=32"
  "model.z_dim=2"
  "reference.hidden_dim=160"
  "reference.time_emb_dim=48"
)

NO_LATENT_CONTACT_OVERRIDES=(
  "model.hidden_dim=640"
  "model.h_emb_dim=640"
  "model.encoder_depth=3"
  "model.control_depth=4"
  "model.time_emb_dim=48"
  "model.latent_type=none"
  "model.z_embed_dim=0"
  "model.z_dim=0"
  "reference.hidden_dim=192"
  "reference.time_emb_dim=48"
)

pids=()
names=()

run_exp() {
  local name="$1"
  shift
  local run_id="${name}_seed${SEED}_${RUN_SUFFIX}"
  names+=("${run_id}")
  (
    set -Eeuo pipefail
    echo "=== START ${run_id} on ${SLURMD_NODENAME:-unknown-node} ==="
    uv run python -m action_bridge.training.train_pusht \
      --config-name pusht_lowdim_contact_quadratic \
      run_id="${run_id}" \
      seed="${SEED}" \
      device=cuda \
      "${COMMON_OVERRIDES[@]}" \
      "$@" \
      > "outputs/hpc_logs/${run_id}.log" 2>&1
    echo "=== DONE ${run_id} ==="
  ) &
  pids+=("$!")
}

echo "Project: ${PROJECT_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "W&B project/group: ${WANDB_PROJECT}/${WANDB_GROUP}"

run_exp "bc_contact_medium_full_z2" \
  "${MEDIUM_CONTACT_OVERRIDES[@]}"

run_exp "bc_contact_medium_no_damping_z2" \
  "${MEDIUM_CONTACT_OVERRIDES[@]}" \
  "reference.gamma_mode=constant" \
  "reference.gamma_const=0.0"

run_exp "bc_contact_medium_no_latent" \
  "${NO_LATENT_CONTACT_OVERRIDES[@]}" \
  "loss.beta_z_start=0.0" \
  "loss.beta_z_end=0.0"

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "=== FAILED ${names[$i]} ==="
    status=1
  fi
done

exit "${status}"
