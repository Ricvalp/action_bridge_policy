#!/usr/bin/env bash
#SBATCH --job-name=bc_contact_z4_resume_best
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
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/bc_contact_medium_z4_batch512_seed0_20260727_183922/checkpoints/best_sim_success.pt}"
DATASET_PATH="${PUSHT_DATASET_PATH:-data/pusht/pusht_cchi_v7_replay.zarr}"
MAX_STEPS="${MAX_STEPS:-500000}"
WANDB_PROJECT="${WANDB_PROJECT:-action-bridge-policy-bc-ablation}"
WANDB_RESUME_SAME_RUN="${WANDB_RESUME_SAME_RUN:-true}"
WANDB_RUN_ID="${WANDB_RUN_ID:-39sbdwqi}"

test -f "${CHECKPOINT_PATH}" || { echo "Missing checkpoint: ${CHECKPOINT_PATH}" >&2; exit 2; }
RUN_NAME="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
LOG_NAME="resume_best_sim_${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "Project: ${PROJECT_DIR}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Max steps: ${MAX_STEPS}"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B resume same run: ${WANDB_RESUME_SAME_RUN}"
echo "W&B run id override: ${WANDB_RUN_ID:-<infer from checkpoint/run dir>}"
echo "Node: ${SLURMD_NODENAME:-unknown-node}"

EXTRA_OVERRIDES=(
  "device=cuda"
  "data.dataset_path=${DATASET_PATH}"
  "data.backend=auto"
  "optim.max_steps=${MAX_STEPS}"
  "logging.wandb.enabled=true"
  "logging.wandb.project=${WANDB_PROJECT}"
  "logging.wandb.resume_same_run=${WANDB_RESUME_SAME_RUN}"
  "logging.wandb.log_images=false"
  "logging.sim_eval_enabled=true"
  "logging.sim_eval_async=true"
  "logging.sim_eval_device=cpu"
  "logging.sim_eval_num_workers=6"
  "logging.sim_eval_worker_threads=1"
  "logging.sim_eval_max_pending=2"
  "logging.sim_eval_every_steps=5000"
  "logging.sim_eval_episodes=100"
  "logging.sim_eval_max_steps=500"
  "logging.sim_eval_n_exec=8"
  "logging.sim_eval_render_episodes=4"
  "logging.sim_eval_save_gifs=true"
  "logging.sim_eval_collect_contact_diagnostics=true"
  "logging.sim_eval_keep_checkpoints=false"
)

if [[ "${WANDB_RESUME_SAME_RUN}" == "true" && -n "${WANDB_RUN_ID}" ]]; then
  EXTRA_OVERRIDES+=("logging.wandb.id=${WANDB_RUN_ID}")
fi

uv run python -m action_bridge.training.train_pusht \
  --resume-from "${CHECKPOINT_PATH}" \
  "${EXTRA_OVERRIDES[@]}" \
  > "outputs/hpc_logs/${LOG_NAME}.log" 2>&1
