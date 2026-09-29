#!/usr/bin/env bash
# Kinetic-SB on 10% of training demos; K=1 -> 2 -> 4 -> 8, H=16.
# Learned dissipative completion; preparation and reference fitting are included.
#SBATCH --job-name=pusht_scarcity10_k_curriculum_sb_kinetic
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
source hpc/sb_pusht_ablations/common.sh scarcity10_k_curriculum_sb_kinetic

# Refuse an existing run; freeze the preset for every stage.
mkdir -p "$SB_PUSHT_CAMPAIGN_ROOT"
mkdir "$run_root"
cp "$config" "$run_root/config.json"
common_args=(--dataset "$PUSHT_DATASET" --run-root "$run_root"
             --config "$run_root/config.json" --threads 4)

"$python" -m action_bridge.scripts.sb_pusht prepare \
  "${common_args[@]}" --device cpu

"$python" -m action_bridge.scripts.sb_pusht reference \
  "${common_args[@]}" "${tracking_args[@]}" --device cuda

exec "$python" -m action_bridge.scripts.sb_pusht sb_kinetic \
  "${common_args[@]}" "${tracking_args[@]}" --device cuda \
  --sim-eval \
  --eval-every 10000 \
  --eval-episodes 20 \
  --eval-device cpu \
  --eval-threads 2 \
  --eval-videos
