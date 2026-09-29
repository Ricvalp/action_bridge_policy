#!/usr/bin/env bash
# DDIM on 5% of the training demos; H=16, deployment K=8.
# Fresh preparation is included; DDIM does not need a reference fit.
#SBATCH --job-name=pusht_scarcity5_ddim
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
source hpc/sb_pusht_ablations/common.sh scarcity5_ddim

# Refuse an existing run; freeze the preset for every stage.
mkdir -p "$SB_PUSHT_CAMPAIGN_ROOT"
mkdir "$run_root"
cp "$config" "$run_root/config.json"
common_args=(--dataset "$PUSHT_DATASET" --run-root "$run_root"
             --config "$run_root/config.json" --threads 4)

"$python" -m action_bridge.scripts.sb_pusht prepare \
  "${common_args[@]}" --device cpu

exec "$python" -m action_bridge.scripts.sb_pusht ddim \
  "${common_args[@]}" "${tracking_args[@]}" --device cuda \
  --sim-eval \
  --eval-every 10000 \
  --eval-episodes 20 \
  --eval-device cpu \
  --eval-threads 2 \
  --eval-videos
