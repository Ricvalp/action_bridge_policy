#!/usr/bin/env bash
# 20 policy jobs: 8 seed replications, 6 horizon jobs, 6 width jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${SB_PUSHT_CAMPAIGN_ROOT:?Export a fresh absolute campaign directory first.}"
for variant in baseline_seed1 baseline_seed2 h32_k8 h64_k8 wide18m wide42m; do
  if [[ -e "$SB_PUSHT_CAMPAIGN_ROOT/$variant" ]]; then
    echo "First batch requires fresh variant directories; already exists: $variant" >&2
    exit 1
  fi
done
for variant in baseline_seed1 baseline_seed2; do
  bash hpc/sb_pusht_ablations/submit.sh "$variant" ddim fm_paired sb_ou sb_kinetic
done
for variant in h32_k8 h64_k8 wide18m wide42m; do
  bash hpc/sb_pusht_ablations/submit.sh "$variant" ddim fm_paired sb_kinetic
done
