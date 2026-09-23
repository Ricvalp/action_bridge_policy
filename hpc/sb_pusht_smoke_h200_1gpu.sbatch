#!/usr/bin/env bash
# Actual async Push-T rollout + MP4 check on an H200 node; no dataset needed.
# Setup and submission: docs/SB_PUSHT.md, "HPC (Peano)".
#SBATCH --job-name=sb_pusht_smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$PWD/workspace/.cache"
mkdir -p "$XDG_CACHE_HOME"

exec .venv-sb-pusht/bin/python -m action_bridge.scripts.sb_pusht_smoke \
  --require-h200 \
  --output-dir "$PWD/workspace/sb_pusht/hpc-smoke-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%S%NZ)"
