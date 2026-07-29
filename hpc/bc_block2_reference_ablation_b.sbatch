#!/usr/bin/env bash
#SBATCH --job-name=bc_block2_ref_b
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
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

SEED="${SEED:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
DATASET_PATH="${PUSHT_DATASET_PATH:-data/pusht/pusht_cchi_v7_replay.zarr}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-bc-ablation}"
WANDB_GROUP="${WANDB_GROUP:-block2_reference_ablation_seed${SEED}}"

COMMON_OVERRIDES=(
  "obs_dim=5"
  "data.dataset_path=${DATASET_PATH}"
  "data.backend=auto"
  "inference.latent_commitment=chunk"
  "model.hidden_dim=1024"
  "model.h_emb_dim=1024"
  "model.encoder_depth=4"
  "model.control_depth=5"
  "model.time_emb_dim=64"
  "model.z_embed_dim=64"
  "model.z_dim=4"
  "reference.hidden_dim=256"
  "reference.time_emb_dim=64"
  "optim.batch_size=512"
  "optim.max_steps=300000"
  "logging.log_every_steps=100"
  "logging.eval_every_steps=1000"
  "logging.checkpoint_every_steps=10000"
  "logging.full_eval_every_steps=10000"
  "logging.full_eval_split=val"
  "logging.full_eval_max_batches=4"
  "logging.full_eval_closed_loop_episodes=64"
  "logging.full_eval_num_samples=64"
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
  "logging.wandb.enabled=true"
  "logging.wandb.project=${WANDB_PROJECT}"
  "logging.wandb.group=${WANDB_GROUP}"
  "logging.wandb.tags=[\"bc_ablation\",\"block2\",\"reference_structure\"]"
)

pids=()
names=()

run_exp() {
  local name="$1"
  local config_name="$2"
  shift 2
  local run_id="${name}_seed${SEED}_${RUN_SUFFIX}"
  names+=("${run_id}")
  (
    set -Eeuo pipefail
    echo "=== START ${run_id} on ${SLURMD_NODENAME:-unknown-node} ==="
    uv run python -m action_bridge.training.train_pusht \
      --config-name "${config_name}" \
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

run_exp "bc_block2_no_potential" "pusht_lowdim_contact_quadratic" \
  "reference.potential_type=none"

run_exp "bc_block2_fixed_damped_reference" "pusht_lowdim_contact_fixed_damping"

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "=== FAILED ${names[$i]} ==="
    status=1
  fi
done

exit "${status}"
