#!/usr/bin/env bash
# One fresh comparison; jobs run independently once their prerequisites finish.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
variant="${1:?Usage: bash hpc/sb_pusht_ablations/submit.sh CONFIG [METHOD ...]}"
shift
case "$variant" in
  ''|*[!a-z0-9_]*) echo "Invalid config name: $variant" >&2; exit 1 ;;
esac
: "${SB_PUSHT_CAMPAIGN_ROOT:?Export an absolute directory for this campaign.}"
if [[ "$SB_PUSHT_CAMPAIGN_ROOT" != /* ]]; then
  echo "SB_PUSHT_CAMPAIGN_ROOT must be absolute." >&2
  exit 1
fi
job_dir="hpc/sb_pusht_ablations"
test -f "$job_dir/configs/$variant.json"
methods=("$@")
if [[ ${#methods[@]} -eq 0 ]]; then
  case "$variant" in
    brownian|isotropic_ou) methods=(sb_ou) ;;
    direct_mlp) methods=(fm_paired sb_ou sb_kinetic) ;;
    *) methods=(ddim fm_paired sb_ou sb_kinetic) ;;
  esac
fi
needs_reference=false
declare -A seen=()
for method in "${methods[@]}"; do
  case "$variant:$method" in
    brownian:sb_ou|isotropic_ou:sb_ou) ;;
    brownian:*|isotropic_ou:*)
      echo "$variant is an sb_ou reference ablation; do not change the policy dynamics too." >&2; exit 1 ;;
    direct_mlp:ddim)
      echo "DDIM has no completion mechanism; use an FM/SB method for direct_mlp." >&2; exit 1 ;;
  esac
  case "$method" in
    ddim) ;;
    fm_paired|fm_local_ot|sb_ou|sb_kinetic) needs_reference=true ;;
    *) echo "Unknown policy method: $method" >&2; exit 1 ;;
  esac
  if [[ -n "${seen[$method]:-}" ]]; then
    echo "Duplicate method: $method" >&2
    exit 1
  fi
  seen[$method]=1
done
command -v sbatch >/dev/null
run_root="$SB_PUSHT_CAMPAIGN_ROOT/$variant"
mkdir -p hpc/logs "$SB_PUSHT_CAMPAIGN_ROOT"
if ! mkdir "$run_root"; then
  echo "Run already exists. Resume individual stage jobs; do not submit duplicate writers: $run_root" >&2
  exit 1
fi
cp "$job_dir/configs/$variant.json" "$run_root/config.json"
printf 'Run: %s\n' "$run_root"
prepare_job=$(sbatch --parsable --job-name="ab_${variant}_prepare" "$job_dir/prepare.sbatch" "$variant")
prepare_job=${prepare_job%%;*}
printf 'prepare %s\n' "$prepare_job" | tee -a "$run_root/jobs.txt"
if "$needs_reference"; then
  reference_job=$(sbatch --parsable --job-name="ab_${variant}_reference" \
    --dependency="afterok:$prepare_job" "$job_dir/reference.sbatch" "$variant")
  reference_job=${reference_job%%;*}
  printf 'reference %s\n' "$reference_job" | tee -a "$run_root/jobs.txt"
fi
for method in "${methods[@]}"; do
  dependency="$prepare_job"
  if [[ "$method" != ddim ]]; then dependency="$reference_job"; fi
  job=$(sbatch --parsable --job-name="ab_${variant}_${method}" \
    --dependency="afterok:$dependency" "$job_dir/train.sbatch" "$variant" "$method")
  job=${job%%;*}
  printf '%s %s\n' "$method" "$job" | tee -a "$run_root/jobs.txt"
done
