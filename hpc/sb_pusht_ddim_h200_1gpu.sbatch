#!/usr/bin/env bash
# Independent DDIM baseline; submit after prepare (not after the reference).
# Setup and submission: docs/SB_PUSHT.md, "HPC (Peano)".
#SBATCH --job-name=sb_pusht_ddim
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"
: "${SB_PUSHT_RUN_ROOT:?Export one shared absolute run directory for this comparison.}"
PUSHT_DATASET="${PUSHT_DATASET:-$PWD/workspace/datasets/pusht/pusht_cchi_v7_replay.zarr}"
test -f "$SB_PUSHT_RUN_ROOT/windows.pt"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$PWD/workspace/.cache"
mkdir -p "$XDG_CACHE_HOME"

.venv-sb-pusht/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; prepare the cu128 environment first.")
name = torch.cuda.get_device_name(0)
if "H200" not in name.upper():
    raise SystemExit(f"Expected an H200 allocation, got {name}")
print(f"GPU: {name}; PyTorch: {torch.__version__}", flush=True)
PY

exec .venv-sb-pusht/bin/python -m action_bridge.scripts.sb_pusht ddim \
  --dataset "$PUSHT_DATASET" \
  --run-root "$SB_PUSHT_RUN_ROOT" \
  --device cuda \
  --threads 4 \
  --wandb \
  --wandb-project action-bridge-policy \
  --wandb-mode "${WANDB_MODE:-online}" \
  --sim-eval \
  --eval-every 10000 \
  --eval-device cpu \
  --eval-threads 2 \
  --eval-videos
