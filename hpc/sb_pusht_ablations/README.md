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

## Start the second batch: scarcity, direct completion, and SB references

Pull the updated code on Peano and use a **fresh campaign root**:

```bash
export PUSHT_DATASET="$PWD/workspace/datasets/pusht/pusht_cchi_v7_replay.zarr"
export SB_PUSHT_CAMPAIGN_ROOT="$PWD/workspace/sb_pusht/ablations-second-$(date -u +%Y%m%dT%H%M%S%NZ)"
printf 'Keep this campaign path: %s\n' "$SB_PUSHT_CAMPAIGN_ROOT"
bash hpc/sb_pusht_ablations/submit_second_batch.sh
```

This queues **29 jobs: 17 policies, 6 preparations, and 6 reference fits**, with
the same dependencies, H200 allocation, W&B project, and asynchronous evaluation
as the first batch. No additional environment setup is needed.

| Variant | Change | Policies |
| --- | --- | --- |
| `scarcity50` | 82 training demonstrations (50%) | DDIM, paired FM, OU-SB, kinetic-SB |
| `scarcity25` | 41 demonstrations (25%) | Same four |
| `scarcity10` | 16 demonstrations (10%, rounded down) | Same four |
| `direct_mlp` | Direct learned prediction of the missing K targets | Paired FM, OU-SB, kinetic-SB |
| `brownian` | Brownian reference: noise without attraction | `sb_ou` policy with the reference replaced |
| `isotropic_ou` | Isotropic rather than structured OU attraction | `sb_ou` policy with the reference replaced |

All six use H16/K8, seed 0, the baseline model size, and 300k policy updates.
Compare them with `baseline_seed0` or your existing full-data seed-0 baseline;
the second batch does not rerun that baseline. These are initial single-seed
comparisons, not yet evidence of a statistically reliable improvement.

Scarcity subsets are **nested whole episodes** from the original training split,
chosen with `subset_seed=0`, independently of the policy training seed. Validation
and test episode IDs do not change. Normalization, reference/completer fitting,
innovation statistics, and source caches use only the selected training episodes.
Do not copy full-data `windows.pt`, references, or source caches into these runs.
The prepared metadata records the original split and selected episode IDs.

The direct-tail MLP learns from adjacent **expert** chunks: it sees the retained
old suffix, current observations, and executed-action history, and predicts only
the missing targets. It is fitted for 20k updates during the reference stage and
then frozen; the original dissipative reference is fitted and kept unchanged.
Reviser training mixes repeat, fixed damping, and direct-MLP completion
(`training_completion_modes: [0, 1, 3]`), replacing learned dissipative completion
in that mixture. Evaluation uses direct completion by default. This tests
completion, not a simultaneous change to the SB reference.

Conversely, the two reference variants retain the baseline completion. Brownian
removes deterministic attraction; isotropic OU keeps the learned center but uses
the mean precision eigenvalue in every direction (same precision trace). They use
the existing `sb_ou` trainer/architecture; `reference_kind` selects the process.
The launchers reject other methods for these two named variants.

To launch only one comparison, use `submit.sh scarcity25`, `submit.sh direct_mlp`,
`submit.sh brownian`, or `submit.sh isotropic_ou`. Restore the campaign variable
when resuming jobs, but never change a configuration inside an existing run.

## Choose comparisons individually

`submit.sh VARIANT [METHOD ...]` submits preparation, the reference when needed,
and just the requested policies. With no methods it uses `ddim fm_paired sb_ou
sb_kinetic`. `fm_local_ot` is also supported. The exceptions are `direct_mlp`,
which defaults to its three revisers, and `brownian`/`isotropic_ou`/
`expert_sources_only`, which default to `sb_ou` only.

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
| Expert previous-plan control | `expert_sources_only` versus `baseline_seed0` | OU-SB by default; other revisers also supported |
| 17–19. Data scarcity | `scarcity50`, `scarcity25`, `scarcity10` | Four core methods |
| 23. Direct learned completion | `direct_mlp` | FM and SB |
| 24. Brownian reference | `brownian` | `sb_ou` trainer |
| 25. Isotropic OU reference | `isotropic_ou` | `sb_ou` trainer |

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

## Control: expert previous chunks instead of self-generated chunks

`expert_sources_only` changes only `source_self_probabilities` to `[0, 0, 0, 0]`:
every non-startup training source retains the previous **expert** chunk's suffix,
then uses the same completion and source noise. The baseline instead progresses
from 10% to 100% generated previous chunks. Both use H16/K8, seed 0, the same
network, reference/completer settings, and 300k updates.

For the cleanest comparison, reuse a **finished, matching `baseline_seed0`**
run's windows and frozen reference, not its policy or source caches. On Peano,
restore `PUSHT_DATASET` and `SB_PUSHT_CAMPAIGN_ROOT`, then:

```bash
baseline="$SB_PUSHT_CAMPAIGN_ROOT/baseline_seed0"
control="$SB_PUSHT_CAMPAIGN_ROOT/expert_sources_only"
mkdir "$control" && mkdir "$control/reference" && \
cp "$baseline/windows.pt" "$control/" && \
cp "$baseline/reference/latest.pt" "$control/reference/" && \
cp hpc/sb_pusht_ablations/configs/expert_sources_only.json "$control/config.json" && \
sbatch hpc/sb_pusht_ablations/train.sbatch expert_sources_only sb_ou
```

Use the same dataset path and code commit, and compare at equal training budgets
on identical evaluation seeds. This keeps the reference artifact exactly the
same. If starting from scratch, `bash hpc/sb_pusht_ablations/submit.sh
expert_sources_only` submits its preparation, reference, and OU-SB training jobs.

Evaluation still carries the policy's **own** generated previous plan; there is
no expert at deployment. A smaller offline state–plan gap is expected by
construction, so success in closed-loop—not that smaller gap alone—determines
whether expert-only training helps. This is a single-seed diagnostic, not proof
that the replay mismatch causes the performance difference. Other FM/SB methods
can use this variant; DDIM is excluded because it has no previous-plan source.

## Completion comparison without retraining

After reviser training has finished, evaluate its trained completion modes on the
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
The job reads the completion-mode list from the checkpoint. Baselines run repeat,
fixed damping, and learned dissipative completion; `direct_mlp` runs repeat,
fixed damping, and direct-MLP completion instead. Each uses
four CPU workers and identical seeds starting at 1000000. Results and selected
MP4s go in a fresh timestamped `completion_eval/` directory below that run root.
This standalone evaluation writes local metrics, not W&B runs. Completion modes
were mixed during reviser training; no retraining is required. DDIM does not use
completion and is intentionally excluded from this comparison.
