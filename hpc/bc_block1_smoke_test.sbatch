#!/usr/bin/env bash
#SBATCH --job-name=bc_block1_smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

mkdir -p hpc/logs outputs/slurm outputs/hpc_logs

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
RUN_SUFFIX="${RUN_SUFFIX:-smoke_$(date +%Y%m%d_%H%M%S)}"
DATASET_PATH="${PUSHT_DATASET_PATH:-data/pusht/pusht_cchi_v7_replay.zarr}"
DATA_BACKEND="${DATA_BACKEND:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-bc-ablation-smoke}"
WANDB_GROUP="${WANDB_GROUP:-block1_smoke_seed${SEED}}"

COMMON_OVERRIDES=(
  "obs_dim=5"
  "data.dataset_path=${DATASET_PATH}"
  "data.backend=${DATA_BACKEND}"
  "optim.batch_size=64"
  "optim.max_steps=12"
  "logging.log_every_steps=1"
  "logging.eval_every_steps=4"
  "logging.checkpoint_every_steps=4"
  "logging.full_eval_every_steps=4"
  "logging.full_eval_split=val"
  "logging.full_eval_max_batches=1"
  "logging.full_eval_closed_loop_episodes=4"
  "logging.full_eval_num_samples=4"
  "logging.sim_eval_enabled=true"
  "logging.sim_eval_async=true"
  "logging.sim_eval_device=cpu"
  "logging.sim_eval_num_workers=2"
  "logging.sim_eval_worker_threads=1"
  "logging.sim_eval_max_pending=1"
  "logging.sim_eval_every_steps=4"
  "logging.sim_eval_episodes=4"
  "logging.sim_eval_max_steps=40"
  "logging.sim_eval_n_exec=8"
  "logging.sim_eval_render_episodes=1"
  "logging.sim_eval_save_gifs=true"
  "logging.sim_eval_collect_contact_diagnostics=false"
  "logging.sim_eval_continue_on_error=false"
  "logging.wandb.enabled=true"
  "logging.wandb.project=${WANDB_PROJECT}"
  "logging.wandb.group=${WANDB_GROUP}"
  "logging.wandb.tags=[\"smoke\",\"bc_ablation\",\"block1\",\"async_sim_eval\"]"
)

SMALL_CONTACT_OVERRIDES=(
  "model.hidden_dim=384"
  "model.h_emb_dim=384"
  "model.encoder_depth=2"
  "model.control_depth=3"
  "model.time_emb_dim=32"
  "model.z_embed_dim=16"
  "model.z_dim=2"
  "reference.hidden_dim=128"
  "reference.time_emb_dim=32"
)

SMALL_BC_OVERRIDES=(
  "model.hidden_dim=560"
  "model.h_emb_dim=560"
  "model.depth=3"
  "model.time_emb_dim=32"
  "model.latent_type=none"
  "model.z_embed_dim=0"
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
      device="${DEVICE}" \
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
nvidia-smi || true

run_exp "smoke_block1_contact_small" "pusht_lowdim_contact_quadratic" "${SMALL_CONTACT_OVERRIDES[@]}"
run_exp "smoke_block1_ar_small" "pusht_lowdim_small_ar_bc" "${SMALL_BC_OVERRIDES[@]}" "model.policy_type=autoregressive_bc"
run_exp "smoke_block1_direct_small" "pusht_lowdim_small_ar_bc" "${SMALL_BC_OVERRIDES[@]}" "model.policy_type=direct_bc"

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "=== FAILED ${names[$i]} ==="
    status=1
  fi
done

echo "Smoke run directories:"
for name in "${names[@]}"; do
  echo "  outputs/${name}"
  test -f "outputs/${name}/checkpoints/latest.pt"
  test -f "outputs/${name}/metrics/train_metrics.csv"
  test -f "outputs/${name}/metrics/val_metrics.csv"
  test -f "outputs/${name}/metrics/periodic_sim_eval_metrics.csv"
done

exit "${status}"
