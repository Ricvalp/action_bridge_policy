#!/usr/bin/env bash
# Submit from the repository root after creating workspace/hpc-logs.
#SBATCH --job-name=mujoco_planar_reach
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=workspace/hpc-logs/%x_%j.out
#SBATCH --error=workspace/hpc-logs/%x_%j.err

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy root.}}"
cd "$PROJECT_DIR"

test -f pyproject.toml || {
  echo "Expected action_bridge_policy at PROJECT_DIR=$PROJECT_DIR" >&2
  exit 2
}
test -f workspace/phi-mujoco/pyproject.toml || {
  echo "Missing the pinned phi-mujoco checkout at workspace/phi-mujoco" >&2
  exit 2
}

COLLECTION_ROOT="${COLLECTION_ROOT:?Export the absolute completed collection path.}"
case "$COLLECTION_ROOT" in
  /*) ;;
  *)
    echo "COLLECTION_ROOT must be absolute: $COLLECTION_ROOT" >&2
    exit 2
    ;;
esac
test -f "$COLLECTION_ROOT/manifest.json" || {
  echo "COLLECTION_ROOT is not a completed phi-mujoco bundle: $COLLECTION_ROOT" >&2
  exit 2
}
export COLLECTION_ROOT

PYTHON="$PROJECT_DIR/.venv/bin/python"
test -x "$PYTHON" || {
  echo "Missing locked parent environment. Run uv sync on a compatible login node first." >&2
  exit 2
}

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/workspace/experiments/mujoco/hpc}"
RUN_ID="${RUN_ID:-planar_reach_${SLURM_JOB_ID:?SLURM_JOB_ID is unset}}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
CONFIG_NAME="${CONFIG_NAME:-mujoco_planar_reach_no_latent}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_STEPS="${MAX_STEPS:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_SEED="${EVAL_SEED:-1000000}"

test ! -e "$RUN_DIR" || {
  echo "Run directory already exists; choose a new RUN_ID: $RUN_DIR" >&2
  exit 2
}

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export UV_CACHE_DIR="$PROJECT_DIR/workspace/.uv-cache"
export XDG_CACHE_HOME="$PROJECT_DIR/workspace/.cache"
export TORCH_HOME="$PROJECT_DIR/workspace/.torch"
export TMPDIR="$PROJECT_DIR/workspace/tmp/${SLURM_JOB_ID}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p "$OUTPUT_ROOT" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"

"$PYTHON" - <<'PY'
import os

from phi_mujoco.dataset import validate_collection_bundle

bundle = validate_collection_bundle(os.environ["COLLECTION_ROOT"])
if not bundle.runtime_cleanup_succeeded:
    raise SystemExit("collection records unsuccessful runtime cleanup")
if bundle.successful_episodes != bundle.requested_episodes:
    raise SystemExit("collection does not contain only successful demonstrations")
print(
    "validated collection",
    bundle.manifest_sha256,
    bundle.successful_episodes,
    bundle.total_steps,
    flush=True,
)
PY

if [[ "$DEVICE" == cuda* ]]; then
  "$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("DEVICE requests CUDA but torch.cuda.is_available() is false")
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY
fi

echo "project=$PROJECT_DIR"
echo "collection=$COLLECTION_ROOT"
echo "config=$CONFIG_NAME"
echo "run=$RUN_DIR"
echo "device=$DEVICE"

"$PYTHON" -m action_bridge.training.train_mujoco \
  --config-name "$CONFIG_NAME" \
  data.collection_root="$COLLECTION_ROOT" \
  device="$DEVICE" \
  seed="$SEED" \
  output_dir="$OUTPUT_ROOT" \
  run_id="$RUN_ID" \
  optim.batch_size="$BATCH_SIZE" \
  optim.max_steps="$MAX_STEPS" \
  eval.clip_actions=true

CHECKPOINT="$RUN_DIR/checkpoints/best.pt"
test -f "$CHECKPOINT" || {
  echo "Training completed without the expected best checkpoint: $CHECKPOINT" >&2
  exit 3
}
sha256sum "$CHECKPOINT"

# This is a technical native-runtime gate. Learned success is recorded but is
# not required for the Slurm job to complete successfully.
NATIVE_RUN="$RUN_DIR/native_eval_seed${EVAL_SEED}"
MUJOCO_GL=disable \
  "$PYTHON" -m action_bridge.eval.mujoco_online \
    --checkpoint "$CHECKPOINT" \
    --trusted-checkpoint \
    --device "$DEVICE" \
    --run-dir "$NATIVE_RUN" \
    --episodes "$EVAL_EPISODES" \
    --max-steps 200 \
    --actions-per-plan 1 \
    --seed "$EVAL_SEED" \
    --no-require-success \
    --json
