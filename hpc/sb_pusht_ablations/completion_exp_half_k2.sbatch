#!/usr/bin/env bash
# OU-SB with exponential recency weights (half-span half-life) for learned completion.
# Fresh campaign root required; preparation and reference fitting are included.
#SBATCH --job-name=pusht_completion_exp_half_k2
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"
source hpc/sb_pusht_ablations/common.sh completion_exp_half_k2

# Refuse an existing run; freeze the preset for all three stages.
mkdir -p "$SB_PUSHT_CAMPAIGN_ROOT"
mkdir "$run_root"
cp "$config" "$run_root/config.json"
common_args=(--dataset "$PUSHT_DATASET" --run-root "$run_root"
             --config "$run_root/config.json" --threads 4)

"$python" -m action_bridge.scripts.sb_pusht prepare \
  "${common_args[@]}" --device cpu

"$python" -m action_bridge.scripts.sb_pusht reference \
  "${common_args[@]}" "${tracking_args[@]}" --device cuda

exec "$python" -m action_bridge.scripts.sb_pusht sb_ou \
  "${common_args[@]}" "${tracking_args[@]}" --device cuda \
  --sim-eval \
  --eval-every 10000 \
  --eval-episodes 20 \
  --eval-device cpu \
  --eval-threads 2 \
  --eval-videos
