#!/usr/bin/env bash
# 17 policy jobs: 12 scarcity, 3 direct-tail, and 2 reference-process ablations.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${SB_PUSHT_CAMPAIGN_ROOT:?Export a fresh absolute campaign directory first.}"
for variant in scarcity50 scarcity25 scarcity10 direct_mlp brownian isotropic_ou; do
  if [[ -e "$SB_PUSHT_CAMPAIGN_ROOT/$variant" ]]; then
    echo "Second batch requires fresh variant directories; already exists: $variant" >&2
    exit 1
  fi
done
for variant in scarcity50 scarcity25 scarcity10; do
  bash hpc/sb_pusht_ablations/submit.sh "$variant" ddim fm_paired sb_ou sb_kinetic
done
bash hpc/sb_pusht_ablations/submit.sh direct_mlp fm_paired sb_ou sb_kinetic
for variant in brownian isotropic_ou; do
  bash hpc/sb_pusht_ablations/submit.sh "$variant" sb_ou
done
