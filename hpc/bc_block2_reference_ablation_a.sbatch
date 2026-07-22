#!/usr/bin/env bash
#SBATCH --job-name=bc_block2_ref_a
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_DIR}"

mkdir -p outputs/slurm outputs/hpc_logs

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
DATASET_PATH="${PUSHT_DATASET_PATH:-data/pusht/pusht_cchi_v7_replay.zarr}"
DATA_BACKEND="${DATA_BACKEND:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-bc-ablation}"
WANDB_GROUP="${WANDB_GROUP:-block2_reference_ablation_seed${SEED}}"

COMMON_OVERRIDES=(
  "obs_dim=5"
  "data.dataset_path=${DATASET_PATH}"
  "data.backend=${DATA_BACKEND}"
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
  "logging.sim_eval_every_steps=10000"
  "logging.sim_eval_episodes=10"
  "logging.sim_eval_max_steps=500"
  "logging.sim_eval_n_exec=8"
  "logging.sim_eval_render_episodes=0"
  "logging.sim_eval_save_gifs=false"
  "logging.sim_eval_collect_contact_diagnostics=false"
  "logging.sim_eval_continue_on_error=true"
  "logging.wandb.enabled=true"
  "logging.wandb.project=${WANDB_PROJECT}"
  "logging.wandb.group=${WANDB_GROUP}"
  "logging.wandb.tags=[\"bc_ablation\",\"block2\",\"reference_structure\"]"
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

run_exp "bc_block2_full_contact"
run_exp "bc_block2_no_damping" \
  "reference.gamma_mode=constant" \
  "reference.gamma_const=0.0"

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "=== FAILED ${names[$i]} ==="
    status=1
  fi
done

exit "${status}"
