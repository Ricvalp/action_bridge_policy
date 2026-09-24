#!/usr/bin/env bash
# Sourced by the three stage jobs, after changing to SLURM_SUBMIT_DIR.
variant="${1:?Pass a config name, e.g. h32_k8.}"
case "$variant" in
  ''|*[!a-z0-9_]*) echo "Invalid config name: $variant" >&2; exit 1 ;;
esac
: "${SB_PUSHT_CAMPAIGN_ROOT:?Export an absolute directory for this campaign.}"
if [[ "$SB_PUSHT_CAMPAIGN_ROOT" != /* ]]; then
  echo "SB_PUSHT_CAMPAIGN_ROOT must be absolute." >&2
  exit 1
fi
run_root="$SB_PUSHT_CAMPAIGN_ROOT/$variant"
config="$PWD/hpc/sb_pusht_ablations/configs/$variant.json"
test -f "$config"
# submit.sh freezes the config once, so queued jobs and resumes agree.
if [[ -f "$run_root/config.json" ]]; then
  config="$run_root/config.json"
fi
python="$PWD/.venv-sb-pusht/bin/python"
PUSHT_DATASET="${PUSHT_DATASET:-$PWD/workspace/datasets/pusht/pusht_cchi_v7_replay.zarr}"
common_args=(--dataset "$PUSHT_DATASET" --run-root "$run_root" --config "$config" --threads 4)
tracking_args=(--wandb --wandb-project sb-pusht-ablations --wandb-mode "${WANDB_MODE:-online}")

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MPLBACKEND=Agg
export XDG_CACHE_HOME="$PWD/workspace/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$XDG_CACHE_HOME"
printf 'Variant: %s\nRun: %s\nConfig: %s\nW&B project: sb-pusht-ablations\n' "$variant" "$run_root" "$config"

"$python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; prepare .venv-sb-pusht first.")
name = torch.cuda.get_device_name(0)
if "H200" not in name.upper():
    raise SystemExit(f"Expected an H200 allocation, got {name}")
print(f"GPU: {name}; PyTorch: {torch.__version__}", flush=True)
PY
