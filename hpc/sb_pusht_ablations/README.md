# Push-T ablations on Peano

Separate H200 jobs for the experiments supported by the current training code.
Training and reference fits log to the dedicated W&B project
**`sb-pusht-ablations`** in your usual W&B account/team.

Use the existing `.venv-sb-pusht` environment and replay dataset described in
[the HPC setup instructions](../../docs/SB_PUSHT.md#hpc-peano). The Push-T
simulation/video smoke test must have passed. No new dependencies are needed.
Commit/push these files here, then pull on Peano before submitting.

## Start the first batch

From the repository root on Peano:

```bash
mkdir -p hpc/logs
export PUSHT_DATASET="$PWD/workspace/datasets/pusht/pusht_cchi_v7_replay.zarr"
export SB_PUSHT_CAMPAIGN_ROOT="$PWD/workspace/sb_pusht/ablations-$(date -u +%Y%m%dT%H%M%S%NZ)"
printf 'Keep this campaign path: %s\n' "$SB_PUSHT_CAMPAIGN_ROOT"

bash hpc/sb_pusht_ablations/submit_first_batch.sh
```

This submits **20 policy jobs**, plus their preparation/reference jobs:

- Seeds 1 and 2: DDIM, paired FM, OU-SB, kinetic-SB (8 jobs).
- H32/K8 and H64/K8: DDIM, paired FM, kinetic-SB (6 jobs).
- Approximately 18M and 42M parameters: those same three methods (6 jobs).

Each variant has its own `prepare → reference → revisers` dependency chain.
DDIM needs only `prepare`, so it can run alongside reference fitting. Independent
methods and variants do not wait for each other; Slurm controls concurrency.
Nothing is submitted until you run a submission command.

Every policy job requests one H200 on `gpuq` (never `aiq`), eight CPUs, 64 GB RAM,
and ten hours. Larger/longer runs may need resubmission; ten hours is not a measured
completion guarantee. Seed 0's existing results remain the initial baseline;
submit `baseline_seed0` below if you want a fresh copy in this campaign.

## Choose comparisons individually

`submit.sh VARIANT [METHOD ...]` submits preparation, the reference when needed,
and just the requested policies. With no methods it uses `ddim fm_paired sb_ou
sb_kinetic`. `fm_local_ot` is also supported.

```bash
# Four core methods, one new variant.
bash hpc/sb_pusht_ablations/submit.sh h32_k16

# The reference-temperature change affects SB, not DDIM or FM.
bash hpc/sb_pusht_ablations/submit.sh temperature001 sb_ou sb_kinetic

# Kinetic damping affects only kinetic SB.
bash hpc/sb_pusht_ablations/submit.sh damping05 sb_kinetic

# The self-source curriculum affects revisers, not DDIM.
bash hpc/sb_pusht_ablations/submit.sh self_sources_all fm_paired sb_ou sb_kinetic
```

The JSON files in `configs/` contain only overrides of
[`get_config()`](../../action_bridge/configs/sb_pusht.py). Unspecified defaults
are H16/K8, approximately 5M deployed parameters, seed 0, 300k optimizer updates,
and 32 inference network evaluations. Use the same code commit throughout a
comparison. The submission helper snapshots the overrides into each variant's
`config.json`; jobs and resumed jobs use that snapshot.

| Proposed comparison | Variant(s) / command | Methods to compare |
| --- | --- | --- |
| 1. Baseline repetitions | `baseline_seed0`, `baseline_seed1`, `baseline_seed2` | Four core methods; optionally local-OT FM |
| 2. H16/K4 | `h16_k4` | Four core methods |
| 3. H16/K16 | `h16_k16` | Four core methods |
| 4. H32/K8 | `h32_k8` | Four core methods |
| 5. H32/K16 | `h32_k16` | Four core methods |
| 6. H64/K8 | `h64_k8` | Four core methods |
| 7. H64/K32 | `h64_k32` | Four core methods |
| 8. Wider, approximately 18M | `wide18m` | Four core methods |
| 9. Wider, approximately 42M | `wide42m` | Four core methods |
| 10. Baseline size, 600k updates | `long600k` | Four core methods |
| 11. Approximately 18M, 600k updates | `wide18m_long600k` | Four core methods |
| 12. Deeper versus wider | `deep18m` versus `wide18m` | Four core methods |
| 13. Completion mechanisms | `completion_eval.sbatch` below | Existing FM/SB checkpoints |
| 14. SB temperature | `temperature001`, `baseline_seed0`, `temperature020` | OU-SB and kinetic-SB |
| 15. Kinetic damping | `damping05`, `baseline_seed0`, `damping8` | Kinetic-SB |
| 16. Self-generated sources from the start | `self_sources_all` versus `baseline_seed0` | FM and SB; DDIM is unchanged |

For temperature, the baseline is `0.05`; for damping, it is `2.0`. The 600k
configs keep four training/source blocks and four SB rounds, with 75k updates
per direction/phase. SB's budget includes both forward and reverse optimization:
300k total updates means 150k forward updates, unlike DDIM/FM's 300k.

H/K variants prepare their own windows and fit their own reference. Longer
horizons reduce the number of eligible training windows; compare the recorded
window counts as well as success. H16/K16 changes both overlap and replanning
frequency, so it is not a clean test of removing plan memory alone.

## Logging, outputs, and resuming

Outputs are under `$SB_PUSHT_CAMPAIGN_ROOT/<variant>/<method>/`; Slurm logs are
in `hpc/logs/`. W&B retains the existing training metrics and action-chunk/T-pose
images. CPU simulation runs asynchronously on the training node every 10k
updates with **20 fixed validation episodes**, retaining selected videos locally.
Busy evaluation intervals are skipped, and outstanding evaluations finish after
optimization. No GPU renderer or extra Slurm job is needed for training evaluation.

These scripts do **not** change the evaluation or checkpoint-selection rules:
`best.pt` still uses the legacy, looser `sim_eval/success_rate`. Inspect the
stricter native `sim_eval/env_success_rate` separately. This campaign does not
silently implement the proposed native-metric selection change.

After a timeout, restore the **same** campaign path and resubmit only the
unfinished stage, for example:

```bash
export SB_PUSHT_CAMPAIGN_ROOT="/absolute/path/printed/when/you/submitted"
sbatch hpc/sb_pusht_ablations/train.sbatch wide42m sb_kinetic
```

Policy training resumes from `latest.pt`. Do not edit the saved configuration,
run duplicate writers, or rerun preparation while trainers are active.
`submit.sh` refuses an existing variant directory to avoid duplicate submissions;
use the direct training command above for resuming. If a reference job fails,
inspect its log before launching dependents. Failed prerequisites require
resubmitting their affected dependent jobs after the prerequisite succeeds.
If compute nodes cannot reach W&B, export `WANDB_MODE=offline` before submission.

## Completion comparison without retraining

After reviser training has finished, evaluate all three completion modes on the
same 200 held-out seeds. Keep the checkpoint unchanged throughout the comparison:

```bash
sbatch hpc/sb_pusht_ablations/completion_eval.sbatch \
  "$SB_PUSHT_CAMPAIGN_ROOT/baseline_seed1" sb_kinetic

# Or choose latest.pt explicitly:
sbatch hpc/sb_pusht_ablations/completion_eval.sbatch \
  "$SB_PUSHT_CAMPAIGN_ROOT/h32_k8" fm_paired latest.pt
```

The first argument is the **variant/run root containing method directories**,
not a checkpoint file. An older run root with the same layout also works.
For SB, use `best.pt` or a completed forward `latest.pt`, not a checkpoint saved
partway through a reverse phase.
The job runs repeat, fixed damping, and learned dissipative completion, each with
four CPU workers and identical seeds starting at 1000000. Results and selected
MP4s go in a fresh timestamped `completion_eval/` directory below that run root.
This standalone evaluation writes local metrics, not W&B runs. Completion modes
were mixed during reviser training; no retraining is required. DDIM does not use
completion and is intentionally excluded from this comparison.
