#!/usr/bin/env bash
#SBATCH --job-name=eval_int_no_potential
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the action_bridge_policy directory.}}"
cd "${PROJECT_DIR}"
test -d action_bridge || { echo "Expected to run from action_bridge_policy; got ${PROJECT_DIR}" >&2; exit 2; }

mkdir -p outputs/hpc_logs

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-outputs/bc_contact_medium_z4_batch512_seed0_20260727_183922/checkpoints/best_sim_success.pt}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EPISODES="${EPISODES:-200}"
MAX_STEPS="${MAX_STEPS:-500}"
N_EXEC="${N_EXEC:-8}"
SEED="${SEED:-0}"
RENDER_EPISODES="${RENDER_EPISODES:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
WORKER_THREADS="${WORKER_THREADS:-1}"
NAME="interventions_no_potential_${RUN_SUFFIX}"

test -f "${CHECKPOINT}" || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }

echo "Project: ${PROJECT_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Variants: full_policy, no_potential"
echo "Episodes: ${EPISODES} | Workers: ${NUM_WORKERS} | Device: ${EVAL_DEVICE}"

uv run python -m action_bridge.scripts.run_checkpoint_only_experiments \
  --checkpoint "${CHECKPOINT}" \
  --name "${NAME}" \
  --device "${EVAL_DEVICE}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --n-exec "${N_EXEC}" \
  --render-episodes "${RENDER_EPISODES}" \
  --save-gifs \
  --seed "${SEED}" \
  --num-workers "${NUM_WORKERS}" \
  --worker-threads "${WORKER_THREADS}" \
  --variants "full_policy,no_potential" \
  --skip-extra-diagnostics \
  --continue-on-error \
  > "outputs/hpc_logs/${NAME}.log" 2>&1
