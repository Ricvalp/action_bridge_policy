#!/usr/bin/env bash
# Train the small low-dimensional Franka cube-lift Action Bridge policy.
#
# One-time setup on the login node, from the action_bridge_policy root:
#   mkdir -p hpc/logs
#   uv sync --locked --extra cu128
#
# The processed schema-2 cache and workspace/phi-isaaclab checkout must be
# visible on the compute node. Submit with, for example:
#   sbatch --export=ALL,COLLECTION_ROOT=/absolute/path/to/processed/cache \
#     hpc/isaaclab_franka_cube_lift_small_h200_1gpu.sbatch
#
# Existing jobs on this cluster request H200s through gpuq + gpu:1. This job
# verifies the allocated device name at runtime. If the site later publishes
# an H200 Slurm feature, it can additionally be submitted with
# `sbatch --constraint=h200 ...`.
#SBATCH --job-name=isaac_small_ab_h200
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

# Uncomment and edit only if the cluster requires an account.
##SBATCH --account=rvalperga

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy root.}}"
cd "$PROJECT_DIR"

test -f pyproject.toml && test -d action_bridge || {
  echo "Expected action_bridge_policy at PROJECT_DIR=$PROJECT_DIR" >&2
  exit 2
}
test -f workspace/phi-isaaclab/pyproject.toml || {
  echo "Missing the pinned phi-isaaclab checkout at workspace/phi-isaaclab" >&2
  exit 2
}

COLLECTION_ROOT="${COLLECTION_ROOT:?Export the absolute processed phi-isaaclab cache path.}"
case "$COLLECTION_ROOT" in
  /*) ;;
  *)
    echo "COLLECTION_ROOT must be absolute: $COLLECTION_ROOT" >&2
    exit 2
    ;;
esac
test -f "$COLLECTION_ROOT/manifest.json" || {
  echo "COLLECTION_ROOT is not a completed phi-isaaclab cache: $COLLECTION_ROOT" >&2
  exit 2
}
export COLLECTION_ROOT

PYTHON="$PROJECT_DIR/.venv/bin/python"
test -x "$PYTHON" || {
  echo "Missing the locked parent environment." >&2
  echo "Run 'uv sync --locked --extra cu128' on a compatible login node first." >&2
  exit 2
}

SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/workspace/experiments/isaaclab/hpc}"
RUN_ID="${RUN_ID:-isaaclab_franka_cube_lift_small_no_latent_seed${SEED}_${SLURM_JOB_ID:?SLURM_JOB_ID is unset}}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

case "$RUN_ID" in
  "" | */*)
    echo "RUN_ID must be one non-empty path component: $RUN_ID" >&2
    exit 2
    ;;
esac
test ! -e "$RUN_DIR" || {
  echo "Run directory already exists; choose a new RUN_ID: $RUN_DIR" >&2
  exit 2
}

# The default identity is the corrected reactive-expert schema-2 collection.
# Override only when deliberately training on another strictly validated cache.
EXPECTED_COLLECTION_SHA256="${EXPECTED_COLLECTION_SHA256:-4eb6659825a01c7e1480f0295faf9f461bb5f0ec37be3fc49d937c0e562ca0c3}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-2048}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-12.8}"
REQUIRE_H200="${REQUIRE_H200:-1}"
export EXPECTED_COLLECTION_SHA256 EXPECTED_EPISODES EXPECTED_TORCH_CUDA REQUIRE_H200

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export UV_CACHE_DIR="$PROJECT_DIR/workspace/.uv-cache"
export XDG_CACHE_HOME="$PROJECT_DIR/workspace/.cache"
export TORCH_HOME="$PROJECT_DIR/workspace/.torch"
export TMPDIR="${SLURM_TMPDIR:-$PROJECT_DIR/workspace/tmp/${SLURM_JOB_ID}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
mkdir -p "$OUTPUT_ROOT" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"

ulimit -n 65536 2>/dev/null || echo "Warning: could not raise the open-file limit." >&2

echo "job_id=$SLURM_JOB_ID"
echo "node=${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-unknown}}"
echo "project=$PROJECT_DIR"
echo "collection=$COLLECTION_ROOT"
echo "run=$RUN_DIR"
echo "seed=$SEED batch_size=$BATCH_SIZE max_steps=$MAX_STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "parent_commit=$(git rev-parse HEAD)"
echo "phi_isaaclab_commit=$(git -C workspace/phi-isaaclab rev-parse HEAD)"
sha256sum pyproject.toml uv.lock "$COLLECTION_ROOT/manifest.json"
nvidia-smi -L

"$PYTHON" - <<'PY'
import os

from phi_isaaclab.dataset import validate_collection_bundle

bundle = validate_collection_bundle(os.environ["COLLECTION_ROOT"])
expected_sha = os.environ["EXPECTED_COLLECTION_SHA256"].lower()
expected_episodes = int(os.environ["EXPECTED_EPISODES"])
if bundle.manifest_sha256 != expected_sha:
    raise SystemExit(
        "collection manifest identity mismatch: "
        f"expected={expected_sha}, actual={bundle.manifest_sha256}"
    )
if bundle.episode_count != expected_episodes:
    raise SystemExit(
        "collection episode count mismatch: "
        f"expected={expected_episodes}, actual={bundle.episode_count}"
    )
if bundle.successful_episodes != bundle.episode_count:
    raise SystemExit(
        "collection contains unsuccessful demonstrations: "
        f"successful={bundle.successful_episodes}, total={bundle.episode_count}"
    )
print(
    "validated collection",
    bundle.manifest_sha256,
    bundle.episode_count,
    bundle.total_steps,
    flush=True,
)
PY

"$PYTHON" - <<'PY'
import os

import torch

if not torch.cuda.is_available():
    raise SystemExit("the locked Torch environment cannot access CUDA")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        "this single-GPU job expects exactly one visible CUDA device, got "
        f"{torch.cuda.device_count()}"
    )
name = torch.cuda.get_device_name(0)
cuda_version = str(torch.version.cuda)
expected_cuda = os.environ["EXPECTED_TORCH_CUDA"]
if "+cu128" not in torch.__version__:
    raise SystemExit(f"expected a +cu128 Torch build, got {torch.__version__}")
if cuda_version != expected_cuda:
    raise SystemExit(
        f"expected the tested cu128 profile ({expected_cuda}), got CUDA {cuda_version}"
    )
require_h200 = os.environ["REQUIRE_H200"].lower() not in {"0", "false", "no"}
if require_h200 and "H200" not in name.upper():
    raise SystemExit(f"the allocated GPU is not an H200: {name}")
capability = torch.cuda.get_device_capability(0)
if require_h200 and capability != (9, 0):
    raise SystemExit(f"expected H200 compute capability (9, 0), got {capability}")

probe = torch.ones((64, 64), device="cuda")
probe = probe @ probe
torch.cuda.synchronize()
if not torch.isfinite(probe).all():
    raise SystemExit("CUDA matrix-multiply preflight produced non-finite output")
print(
    "validated CUDA",
    {"torch": torch.__version__, "cuda": cuda_version, "device": name,
     "capability": capability},
    flush=True,
)
PY

# This is the small (~366k parameter) no-latent Action Bridge baseline. The
# transition-aware sampler and tube perturbation reproduce the strongest
# current offline recipe; they do not imply that closed-loop success is solved.
"$PYTHON" -m action_bridge.training.train_isaaclab \
  --config-name isaaclab_franka_cube_lift_no_latent \
  data.collection_root="$COLLECTION_ROOT" \
  data.first_close_sampling_weight=16 \
  device=cuda:0 \
  seed="$SEED" \
  output_dir="$OUTPUT_ROOT" \
  run_id="$RUN_ID" \
  model.control_scale=2.5 \
  loss.tube_training=true \
  loss.tube_noise_std_start=0.0 \
  loss.tube_noise_std_end=0.02 \
  loss.tube_noise_warmup_steps=5000 \
  optim.batch_size="$BATCH_SIZE" \
  optim.max_steps="$MAX_STEPS" \
  logging.log_every_steps=100 \
  logging.eval_every_steps=1000 \
  logging.checkpoint_every_steps=2500 \
  logging.full_eval_every_steps=0 \
  logging.wandb.enabled=false \
  eval.offline_max_batches=0

for artifact in \
  "$RUN_DIR/config.json" \
  "$RUN_DIR/checkpoints/best.pt" \
  "$RUN_DIR/checkpoints/latest.pt" \
  "$RUN_DIR/metrics/train_metrics.csv" \
  "$RUN_DIR/metrics/val_metrics.csv" \
  "$RUN_DIR/metrics/test_metrics.json" \
  "$RUN_DIR/metrics/isaaclab_offline_metrics.json"
do
  test -f "$artifact" || {
    echo "Training completed without expected artifact: $artifact" >&2
    exit 3
  }
done

sha256sum \
  "$RUN_DIR/config.json" \
  "$RUN_DIR/checkpoints/best.pt" \
  "$RUN_DIR/checkpoints/latest.pt" \
  "$RUN_DIR/metrics/test_metrics.json" \
  "$RUN_DIR/metrics/isaaclab_offline_metrics.json"

echo "training_complete=$RUN_DIR"
