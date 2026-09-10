#!/usr/bin/env bash
# Submit from action_bridge_policy after the environment setup in MUJOCO.md.
# Optional argument: joint (default) or stopgrad. Both have 5.15M trainable params.
#SBATCH --job-name=square_dissipative
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"

variant="${1:-joint}"
case "$variant" in
  joint) config=mujoco_robomimic_square_dissipative ;;
  stopgrad) config=mujoco_robomimic_square_dissipative_stopgrad ;;
  *) echo "Expected joint or stopgrad, got: $variant" >&2; exit 2 ;;
esac

test "$(git -C ../phi-mujoco rev-parse HEAD)" = 528bdf97471493a9d63da424872141ff1279be4b
test -z "$(git -C ../phi-mujoco status --porcelain)"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PWD/workspace/.cache"
mkdir -p "$XDG_CACHE_HOME"

.venv/bin/python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; install the cu128 environment first.")
name = torch.cuda.get_device_name(0)
if "H200" not in name.upper():
    raise SystemExit(f"Expected an H200 allocation, got {name}")
print(f"GPU: {name}; PyTorch: {torch.__version__}", flush=True)
PY

run_id="square-dissipative-${variant}-h8-exec4-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%S%NZ)"

uv run --frozen --no-sync --extra cu128 --extra robomimic \
  python -m action_bridge.training.train_mujoco \
  --config-name "$config" \
  data.cache_root="$PWD/../phi-mujoco/datasets/processed/robomimic_square-20260908T122032Z" \
  run_id="$run_id" \
  device=cuda \
  logging.wandb.enabled=true \
  logging.sim_eval_enabled=true \
  logging.sim_eval_success_videos="${SIM_EVAL_VIDEOS:-0}" \
  logging.sim_eval_failure_videos="${SIM_EVAL_VIDEOS:-0}"
