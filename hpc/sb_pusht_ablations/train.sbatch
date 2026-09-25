#!/usr/bin/env bash
# Usage: sbatch hpc/sb_pusht_ablations/train.sbatch VARIANT METHOD
# DDIM needs prepare; FM/SB also need this variant's reference job.
#SBATCH --job-name=pusht_ablation_train
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
if [[ $# != 2 ]]; then
  echo "Usage: sbatch $0 VARIANT METHOD" >&2
  exit 2
fi
method="$2"
case "$method" in
  ddim|fm_paired|fm_local_ot|sb_ou|sb_kinetic) ;;
  *) echo "Unknown method: $method" >&2; exit 2 ;;
esac
case "$1:$method" in
  brownian:sb_ou|isotropic_ou:sb_ou) ;;
  brownian:*|isotropic_ou:*)
    echo "$1 is an sb_ou reference ablation." >&2; exit 2 ;;
  direct_mlp:ddim)
    echo "DDIM has no completion mechanism; use an FM/SB method for direct_mlp." >&2; exit 2 ;;
  expert_sources_only:ddim)
    echo "DDIM has no previous-plan source; use an FM/SB method for expert_sources_only." >&2; exit 2 ;;
esac
cd "${SLURM_SUBMIT_DIR:?Submit from the action_bridge_policy directory.}"
source hpc/sb_pusht_ablations/common.sh "$1"
test -f "$run_root/windows.pt"
if [[ "$method" != ddim ]]; then
  test -f "$run_root/reference/latest.pt"
fi

exec "$python" -m action_bridge.scripts.sb_pusht "$method" \
  "${common_args[@]}" \
  "${tracking_args[@]}" \
  --device cuda \
  --sim-eval \
  --eval-every 10000 \
  --eval-episodes 20 \
  --eval-device cpu \
  --eval-threads 2 \
  --eval-videos
