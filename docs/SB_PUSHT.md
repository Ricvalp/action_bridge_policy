# Whole-plan revision on Push-T

The default experiment is **self_source_v1**, defined in
[SB-PUSHT-ADDENDUM.md](../SB-PUSHT-ADDENDUM.md). It replaces the fixed-DDIM-source
protocol; old runs remain untouched and are not rows in this comparison.
The [edited plan](../SB-PUSHT_EDIT.md) describes the reusable interfaces and
OU/kinetic mathematics. This is offline behavior cloning, separate from the
existing Action Bridge loss.

## K curriculum (Addendum B)

Opt into **`self_source_k_curriculum_v1`** to train with K=1,2,4,8 in four
75k-update blocks, keeping H=16. Each SB block is a complete 37.5k reverse /
37.5k forward round. The expert/self mixture stays unchanged. At boundaries,
the current forward EMA is frozen for a fresh sequence replay on the new K grid;
old-K source and coupling samples are not relabeled or reused.

The main evaluation always executes **K=8**, including checkpoints trained so
far only at smaller K. It measures deployment-K transfer, not performance at
the current training K. K conditioning is a fixed-width input to all revisers;
the model is not rebuilt between stages. DDIM training itself is unchanged.

The learned completer/reference is fitted once, then frozen. The supplied
curriculum presets follow Addendum B's **last-pair initialization** from the
full previous plan, including when K=H has no retained overlap. The existing
whole-suffix weighting choices remain available as separate overrides.
The three completion modes are still balanced in training; the main evaluation
uses learned dissipative completion. Startup sampling is explicitly held at
10% instead of varying accidentally with the number of replay decisions.

This is still **offline imitation**: observation and executed-command histories
come from demonstrations, not execution of generated plans. The curriculum does
not make replay on-policy or supply recovery labels for unseen physical states.
Smaller K also requires more source-generation calls; equal optimization budgets
are not equal total compute. Source/curriculum logs report this separately.
The optional simulator-mismatch probe and gated advancement are not implemented:
the replay observations alone are not exact restorable simulator snapshots.

### Short wiring smoke

From the repository root with the existing environment and replay dataset:

```bash
export PUSHT_DATASET="$PWD/workspace/datasets/pusht/pusht_cchi_v7_replay.zarr"
run_root="$PWD/workspace/sb_pusht/k-smoke-$(date -u +%Y%m%dT%H%M%S%NZ)"
config="$PWD/docs/configs/sb_pusht_k_curriculum_smoke.json"

for stage in prepare reference sb_ou; do
  .venv-sb-pusht/bin/python -m action_bridge.scripts.sb_pusht "$stage" \
    --dataset "$PUSHT_DATASET" --run-root "$run_root" --config "$config" \
    --device cpu --threads 2 --no-sim-eval
done
```

This fits a tiny model for eight updates across all four stages (plus two
reference updates), using 1% of the training episodes. It tests wiring, not
learning quality. Replay still traverses the selected episodes; it is not eight
simulator steps. `sb_kinetic` can be run in the same smoke root after the reference.

### Substantive run, resume, and evaluation

For the H200 scripts and matched fixed-K controls, see
[the curriculum jobs](../hpc/sb_pusht_ablations/README.md#k-curriculum-and-matched-fixed-k-controls).
The equivalent direct OU-SB run is:

```bash
run_root="$PWD/workspace/sb_pusht/k-curriculum-$(date -u +%Y%m%dT%H%M%S%NZ)"
config="$PWD/hpc/sb_pusht_ablations/configs/k_curriculum_sb_ou.json"
for stage in prepare reference sb_ou; do
  .venv-sb-pusht/bin/python -m action_bridge.scripts.sb_pusht "$stage" \
    --dataset "$PUSHT_DATASET" --run-root "$run_root" --config "$config" \
    --device cuda --wandb --wandb-project sb-pusht-ablations \
    --eval-every 10000 --eval-episodes 20 --eval-device cpu --eval-videos
done

# Resume an interrupted policy: use the SAME run_root and config, not a new date.
.venv-sb-pusht/bin/python -m action_bridge.scripts.sb_pusht sb_ou \
  --dataset "$PUSHT_DATASET" --run-root "$run_root" --config "$config" \
  --device cuda --wandb --wandb-project sb-pusht-ablations \
  --eval-every 10000 --eval-episodes 20 --eval-device cpu --eval-videos

# Standalone held-out evaluation: actual execution and K conditioning both use 8.
.venv-sb-pusht/bin/python -m action_bridge.scripts.eval_pusht_sb_ou \
  --checkpoint "$run_root/sb_ou/best.pt" --execute 8 \
  --completion learned_dissipative --device cpu --workers 4 --threads 1 \
  --episodes 50 --seed 1000000 --save-videos
```

Do not change the schedule or initialization inside an existing run. Resumption
restores the active source snapshot, stage, direction, optimizer and RNG state.
Keep the run's `sources/` artifacts with its checkpoint when transferring a
resumable training run; an evaluation-only checkpoint is self-contained.

For an isolated curriculum comparison, repeat the substantive command in a
**fresh root** with `configs/k_fixed8_sb_ou.json` instead: the K-conditioned model,
source-mixture schedule, startup sampling, learned components and update budget
match, but every block uses K=8. Use the analogous `sb_kinetic` files/module to
repeat both conditions for kinetic-SB. Existing fixed-K results without the new
conditioning branch are a broader comparison, not this matched control.

For the unchanged legacy fixed-K protocol, use a separate fresh root and
`--config hpc/sb_pusht_ablations/configs/baseline_seed0.json` in the same
prepare/reference/train commands. Existing legacy runs continue to resume with
their original saved configuration. There is no automatic checkpoint migration
or cross-protocol cache reuse; select the curriculum for a new run explicitly.

## HPC (Peano)

The files `hpc/sb_pusht_*_h200_1gpu.sbatch` submit **separate jobs**, not an
`all` job. Every file uses the existing H200 partition `gpuq`, never the B200
partition `aiq`, and refuses to run if CUDA reports a non-H200 GPU.
Each main policy job requests one H200, eight CPUs, 32 GB RAM and ten hours.
The default scientific settings remain 300k updates, H=16, K=8 and 32 NFE.

For the horizon, capacity, training-budget, data-scarcity, completion, and reference campaigns,
see [hpc/sb_pusht_ablations/README.md](../hpc/sb_pusht_ablations/README.md).
Those jobs use the separate W&B project `sb-pusht-ablations`.
The second batch adds nested 50%/25%/10% training subsets, a direct MLP tail
predictor, and Brownian/isotropic-OU references. Use fresh run directories:
each scarcity subset fits its own normalizer and learned dependencies; held-out
episodes stay fixed. Existing environments need no additional packages.

### 1. Update code and transfer the replay dataset

Commit/push the changes here, then pull them in the policy clone on Peano.
Transfer the **entire** `pusht_cchi_v7_replay.zarr/` directory, including its
hidden Zarr metadata, to:

```text
<HPC action_bridge_policy>/workspace/datasets/pusht/pusht_cchi_v7_replay.zarr/
```

The local copy is about 32 MB. No previous run, checkpoint, window cache,
virtual environment or simulator installation needs transferring. Use your
laptop as an intermediary if workstation-to-Peano SSH is still unavailable.
The job scripts also accept another absolute path through `PUSHT_DATASET`.

### 2. Set up the environment on the login node

From the policy clone, create a separate environment so existing MuJoCo jobs
keep their current environment. Push-T does not need any `phi-*` checkouts:

```bash
cd /path/to/action_bridge_policy
mkdir -p hpc/logs workspace/.uv-cache
export UV_CACHE_DIR="$PWD/workspace/.uv-cache"

UV_PROJECT_ENVIRONMENT=.venv-sb-pusht \
uv sync --frozen --python 3.11 \
  --extra cu128 --extra diffusion --extra pusht-sim \
  --no-install-package phi-isaaclab \
  --no-install-package phi-coppeliasim \
  --no-install-package phi-mujoco

.venv-sb-pusht/bin/wandb login
```

These exclusions use the existing lock without editing it. Jobs execute this
environment directly; they do not install anything on compute nodes.

### 3. Check simulation and video on an H200 node

```bash
sbatch hpc/sb_pusht_smoke_h200_1gpu.sbatch
```

Wait for it to finish. Its `hpc/logs/sb_pusht_smoke_<job-id>.out` must contain:

```text
PASS: H200 CUDA, asynchronous CPU Push-T simulation and headless MP4.
```

This checks CUDA computation, then runs the **actual evaluation subprocess**,
steps the simulator, writes an MP4 and decodes a frame. Artifacts go under
`workspace/sb_pusht/hpc-smoke-<job-id>-<timestamp>/`. No dataset is needed.
Push-T uses Pymunk and software rendering: **no EGL, X server, Isaac Sim or
MuJoCo installation is involved**. Headless SDL and Matplotlib are set in all
the scripts. Passing local tests is not a substitute for passing this job on Peano.

### 4. Submit the independent experiment jobs

Set these variables once in the login shell. Keep the printed run path for
resuming; every job in this comparison must use the same shared filesystem path.

```bash
export PUSHT_DATASET="$PWD/workspace/datasets/pusht/pusht_cchi_v7_replay.zarr"
export SB_PUSHT_RUN_ROOT="$PWD/workspace/sb_pusht/self-source-hpc-$(date -u +%Y%m%dT%H%M%S%NZ)"
printf 'Run directory: %s\n' "$SB_PUSHT_RUN_ROOT"
```

After the smoke passes, submit all jobs with their required dependencies:

```bash
prepare_job=$(sbatch --parsable hpc/sb_pusht_prepare_h200_1gpu.sbatch)

reference_job=$(sbatch --parsable --dependency=afterok:$prepare_job hpc/sb_pusht_reference_h200_1gpu.sbatch)
ddim_job=$(sbatch --parsable --dependency=afterok:$prepare_job hpc/sb_pusht_ddim_h200_1gpu.sbatch)

paired_job=$(sbatch --parsable --dependency=afterok:$reference_job hpc/sb_pusht_fm_paired_h200_1gpu.sbatch)
ot_job=$(sbatch --parsable --dependency=afterok:$reference_job hpc/sb_pusht_fm_local_ot_h200_1gpu.sbatch)
ou_job=$(sbatch --parsable --dependency=afterok:$reference_job hpc/sb_pusht_sb_ou_h200_1gpu.sbatch)
kinetic_job=$(sbatch --parsable --dependency=afterok:$reference_job hpc/sb_pusht_sb_kinetic_h200_1gpu.sbatch)

sbatch --dependency=afterok:$ddim_job:$paired_job:$ot_job:$ou_job:$kinetic_job \
  hpc/sb_pusht_evaluate_h200_1gpu.sbatch

squeue -u "$USER"
```

DDIM and reference fitting can start together after preparation. The four
revisers can start together after reference fitting; none waits for DDIM or
another reviser. Slurm decides actual concurrency from available resources.
The final job waits for all five trainers, then generates the held-out
comparison, common-source diagnostic and `REPORT.md`.

You may also submit any individual file after its prerequisites finish, e.g.:

```bash
sbatch hpc/sb_pusht_sb_ou_h200_1gpu.sbatch
```

### Evaluation, logging and resuming

All five policy jobs enable W&B in `action-bridge-policy`, including the
action-chunk/T-pose images. Every 10k updates the trainer starts an asynchronous
CPU evaluator **on the same allocated node**, using two threads from the job's
eight CPUs. The worker hides CUDA, so training retains the GPU. MP4 encoding
uses one thread. No extra Slurm allocation or nested `sbatch` is needed.
These five training jobs pass `--eval-episodes 20`: each evaluation uses the
same 20 validation seeds. The local default remains five episodes, and the
final held-out comparison remains 50 seeds. The 10k-update cadence,
busy-worker skipping, final drain and local video defaults are unchanged.
Videos are not uploaded to W&B. Already running jobs keep their submitted
settings; the 20-episode setting takes effect only after resubmission.

Slurm stdout/stderr are in `hpc/logs/`; metrics, checkpoints, figures and videos
are below `$SB_PUSHT_RUN_ROOT/<method>/`. The smoke must pass before starting
long jobs; inspect `<method>/sim_eval/errors.jsonl` if later workers fail.
If compute nodes cannot reach W&B, export `WANDB_MODE=offline` **before**
submission and sync the saved W&B runs later from a network-enabled node.

Ten hours is a requested allocation, not a measured completion guarantee.
After a timeout, export the **same** `SB_PUSHT_RUN_ROOT` and resubmit only the
unfinished method's file. It resumes from its last saved checkpoint (normally
every 5k updates). Do not regenerate the timestamp, rerun preparation while
training jobs are active, or run two writers for the same method. If a
prerequisite job failed, dependent jobs still refer to that old job ID: cancel
and resubmit the affected dependent jobs with the new ID, or submit them after
their prerequisites have finished.

To copy checkpoints through your laptop, run this **on the laptop**, connected
to the VPN (pull the repository there, or copy just this script):

```bash
bash hpc/rsync_sb_pusht_checkpoints.sh YOUR_RUN_DIRECTORY YOUR_WORKSTATION_SSH_ALIAS
```

The first argument is the existing run's directory name, not its full path.
The script uses the laptop's `peano` SSH alias and the workstation alias you
supply. It copies `best.pt`, `latest.pt`, and their summary files, retaining
method directories. The laptop copy stays in `~/Downloads/sb-pusht/<run>/`;
the workstation copy goes under `workspace/sb_pusht/from-hpc/<run>/`.

## Local sequential run

Use the policy project's environment:

```bash
uv sync --frozen --extra cu128 --extra diffusion --extra pusht-sim --extra test

PUSHT_DATASET="$PWD/workspace/datasets/pusht/pusht_cchi_v7_replay.zarr"
run_root="$PWD/workspace/sb_pusht/self-source-$(date -u +%Y%m%dT%H%M%S%NZ)"
test -e "$PUSHT_DATASET"

uv run --frozen --no-sync python -m action_bridge.scripts.sb_pusht all \
  --dataset "$PUSHT_DATASET" --run-root "$run_root" --device cuda --wandb
```

Set the dataset path to your actual replay. Zarr reads `data/state` (5D), not
keypoints; the existing NPZ/PT formats also work. Use a **new run root** for this
protocol. This command runs five 300k-update trainings, not a smoke test.

To inspect the manifest without training or writing files:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.sb_pusht all \
  --run-root "$run_root" --dry-run
```

Stages:

1. `prepare`: split whole episodes, fit normalization on train only, create
   full-horizon windows on the t=0,K,2K… replan grid, including startup.
2. `reference`: one shared 20k-update supervised dissipative completer/prior fit.
3. `ddim`, `fm_paired`, `fm_local_ot`, `sb_ou`, `sb_kinetic`: five main jobs.
4. `evaluate`: 50 common held-out seeds, two additional kinetic completion
   evaluations, and one shared-source offline diagnostic.
5. `report`: main and kinetic-completion tables, CSV and success plot.

Run individual stages with the same arguments:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.sb_pusht prepare \
  --dataset "$PUSHT_DATASET" --run-root "$run_root" --device cpu

uv run --frozen --no-sync python -m action_bridge.scripts.sb_pusht reference \
  --dataset "$PUSHT_DATASET" --run-root "$run_root" --device cuda --wandb

uv run --frozen --no-sync python -m action_bridge.scripts.sb_pusht sb_ou \
  --dataset "$PUSHT_DATASET" --run-root "$run_root" --device cuda --wandb
```

DDIM is independent. Each reviser needs only the frozen reference—not a DDIM
checkpoint or a shared `sources` stage. Different methods may run independently
after reference fitting. Do not run two writers in the same method directory.

Interrupted jobs resume from `latest.pt`; completed jobs are skipped. A JSON
file passed through `--config` overrides the flat settings in
[action_bridge/configs/sb_pusht.py](../action_bridge/configs/sb_pusht.py).
For shortened software checks, keep `updates` divisible by `training_blocks`,
provide one `source_self_probabilities` entry per block, and for SB keep
`rounds == training_blocks` and `updates == 2 * rounds * phase_updates`.

Legacy DDIM/reference artifacts are not imported automatically. Their dataset,
window eligibility, normalization, settings and completed budget must be checked
and reuse explicitly documented before using them in a new comparison. In
particular, old reviser checkpoints and shared fixed-DDIM source caches are
incompatible. Nothing is automatically submitted to Slurm.

## What the models learn

| Method | Generator | Correspondence/reference |
| --- | --- | --- |
| `ddim` | Diffusers DDIM, 100 cosine training levels | Ordinary diffusion BC |
| `fm_paired` | Midpoint flow-matching ODE | Natural source/expert pairs |
| `fm_local_ot` | Same ODE | Observation-local discrete OT re-pairing |
| `sb_ou` | Stochastic directional DSBM | Exact affine OU reference |
| `sb_kinetic` | Stochastic directional DSBM | Exact augmented kinetic reference |

Defaults: two observations, two actual executed commands, H=16, execute K=8,
32 network evaluations per decision. Actions are normalized absolute pusher
targets, not the physical pusher trajectory. The simulator bounds executed
commands; intermediate plans are not clipped. DDIM disables sample clipping.

For each reviser, four 75k-update blocks use self-source probabilities
**0.10, 0.50, 1.00, 1.00**. At each block boundary:

- Freeze an independent copy of that model's EMA (the forward model for SB).
- Replay logged episodes separately for all three completion modes, keeping
  the mode fixed within each replay.
- At ordinary decisions, select the whole previous generated chunk or the
  previous replan's expert chunk according to that block's probability.
- Align and complete it, add source noise once, and pair it with the current
  expert future. Sample the frozen model and carry its generated plan onward,
  even when the current source used an expert old plan.

Recorded observations and actual executed-command histories never change.
The first decision uses a repeated observed action anchor, perturbed once and
revised by the model itself; no expert or external generator supplies its first
executed prefix. Startup examples remain in every block. Completion ID and
previous-plan availability are model inputs; expert/self origin is audit-only.

The three completers preserve retained overlap and fill only the tail:
repeat, fixed damping (0.8), or the shared learned dissipative continuation.
Source noise is normalized std 0.01; expert endpoint smoothing is std 0.001
during training only. Local OT never crosses completion modes or
startup/ordinary regimes and never averages endpoint coordinates.

SB keeps the block's prescribed source population **S** separate from its
revision-time coupling cache **Pi**. Each round fits reverse then forward
(37,500 updates each), with frozen opposite-direction rollouts to refresh Pi.
Later rounds warm-start from the existing forward sampler on the new S, not
from natural pairs again. OU/kinetic bridges retain exact Gaussian reference
transitions and stochastic sampling; generated kinetic endpoints retain their
auxiliary velocities. Frozen reference coefficients depend only on observed
context. This is a succession of changing-boundary bridge problems, not a
claim of fixed-boundary convergence.

### Whole-chunk initialization for learned completion

New runs initialize **learned dissipative completion** with a weighted line fit
to all `H-K` retained targets versus action time. Its slope replaces the noisy
last-pair velocity. The continuation still starts at the exact final old target;
retained targets, learned forces, reference fitting and the SB reference process
are unchanged. Repeat, fixed damping and direct-MLP completion are also unchanged.

Choose `--completion-velocity-weighting NAME` in the training command, or set
`completion_velocity_weighting` in its JSON configuration:

| Name | Weights, oldest to newest |
| --- | --- |
| `linear` (new default) | `1, 2, ..., H-K` |
| `uniform` | Equal weight for every retained target |
| `exp_half` | Exponential increase; oldest has 1/4 the newest weight |
| `exp_quarter` | Stronger exponential increase; oldest has 1/16 the newest weight |
| `last_pair` | Original two-target velocity, for the baseline |

The exponential half-lives are half/quarter of the retained time span, so their
endpoint weight ratios do not change with H. With only one retained target the
fits return zero velocity; with two they equal the last-pair estimate. The fit
has a free intercept, but its fitted position is **not** used as the tail anchor.

Use a fresh run root and the same setting for prepare/reference/policy stages.
Changing it while resuming is rejected because it changes the training sources.
Existing checkpoints without this setting retain `last_pair` automatically.
To explicitly probe a pretrained OU-SB checkpoint without modifying its weights:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.eval_pusht_sb_ou \
  --checkpoint "$checkpoint" --device cpu --episodes 20 --workers 4 \
  --completion learned_dissipative --completion-velocity-weighting exp_half
```

The same flags work with rollout/candidate visualization and source-gap
diagnostics. Evaluation records the trained setting and explicit override;
this is an inference-only ablation, not a substitute for training with the new
source construction. Symmetric synthetic probes use fixed damping and therefore
do not accept this learned-completion override. For matched HPC training runs,
see the `completion_*` presets in the [ablation instructions](../hpc/sb_pusht_ablations/README.md#completion-velocity-ablations).

## Logging and asynchronous evaluation

All five policy trainers evaluate every **10k updates** in a separate process
(default CPU, two threads). Training continues while it runs. Change this with
`--eval-every 5000 --eval-device cpu --eval-threads 2`.
Use `--eval-episodes 20` for 20 consecutive seeds starting at 500000 (the HPC
training jobs already set this). This evaluation-only option can change when
resuming without rebuilding training caches. Changing the seed panel restarts
`best.pt` selection at the first trained-policy result on the new panel;
previous checkpoints are not automatically re-evaluated.
One worker runs per trainer; busy intervals are skipped, not queued. At training
end, pending evaluation finishes and final weights are evaluated if necessary.
Use `--no-sim-eval` to disable it.

W&B is opt-in with `--wandb` (project `action-bridge-policy`).
It logs losses, gradient norms, source/coupling timing and origin diagnostics,
OT statistics where applicable, and validation success/coverage/smoothness.
Action-chunk images show predictions, expert targets and the T pose every 5k
updates by default. Use `--wandb-images-every`, `--wandb-image-count`, or
`--wandb-mode offline` as needed. Diagnostic sampling preserves training RNG.
Reviser images distinguish unexecuted previous targets (black), appended
completion (purple), and the actual noisy source (gray). The retained targets
and appended tail together form the clean completed old plan.

MP4s stay **local**, never on W&B: up to two successes and two failures per
evaluation. They are enabled by default, including standalone checkpoint
evaluation. `--no-eval-videos` disables training-evaluation video recording.
Worker logs, metrics and videos live in
`<run-root>/<method>/sim_eval/step_<step>-<timestamp>/`.
Failures are logged as errors, never disguised as zero success.

Artifacts inside each method directory:

- `latest.pt`: exact optimizer/EMA/RNG/phase resume state and frozen components.
- `best.pt`: EMA inference snapshot selected by closed-loop validation success;
  `best_eval.json` records the checkpoint step and score.
- `sources/block_*/`: immutable source-producing snapshot and versioned,
  once-perturbed replay cache. A missing cache is rebuilt from that original
  snapshot and seed, never from a newer EMA.
- `train.jsonl`, `validation.jsonl`, `source_replay.jsonl`, and `figures/`.

SB evaluation always uses the forward field. Initial reverse-phase results
cannot select best weights before any forward updates. Inference checkpoints
do not need the training dataset or replay cache; resumable training does.
Treat Torch checkpoints as trusted local files only.

## Evaluate a checkpoint

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.eval_pusht_sb_ou \
  --checkpoint "$run_root/sb_ou/best.pt" --device cpu --episodes 10
```

Use `eval_pusht_ddim`, `eval_pusht_fm_paired`, `eval_pusht_fm_local_ot`,
`eval_pusht_sb_ou`, or `eval_pusht_sb_kinetic`.
Options include `--execute 4`, `--seed 1000000`, `--max-steps 300`,
`--completion repeat`, and `--no-save-videos`. Raw SB `latest.pt` must be
from a forward phase; asynchronous snapshots explicitly select the forward field.

Evaluation is serial by default (`--workers 1`, four Torch threads). To run
episodes concurrently on CPU:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.eval_pusht_sb_ou \
  --checkpoint "$run_root/sb_ou/best.pt" \
  --device cpu --workers 8 --threads 1 --episodes 50 --seed 1000000
```

Multiple workers require `--device cpu`; each worker loads its own CPU policy
and needs its own RAM. The worker count is capped by the episode count.
With multiple workers, `--threads` defaults to one per worker; an explicit
value overrides either default. Keep the same held-out seeds across methods;
`--seeds 1000000 1000007 1000019` selects an explicit, unique seed list.
Parallel evaluation uses one parent progress bar. Selected videos/GIFs are
rendered in a second pass, with the usual limit of two successes and two
failures across the whole evaluation, not per worker.

Each invocation creates a timestamped directory below
`workspace/sb_pusht/evaluation/`, or use `--output-dir` with a new path.
An episode progress bar with elapsed time and ETA is enabled by default;
use `--no-progress` to hide it. Background training evaluators disable the bar
in their worker logs.
Rollout overlays distinguish old/completed/revised targets from actual motion.
`--save-gifs` additionally saves GIFs.

After all training, run the full comparison:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.sb_pusht evaluate \
  --run-root "$run_root" --device cuda
```

The main table uses learned completion. The kinetic completion table reuses one
checkpoint for three modes, without retraining. The common-source diagnostic
pools equal numbers of self-only logged-replay proposals from all four revisers,
freezes the history-aligned set, and tests each model on those exact sources.
Error-bin thresholds come from validation, not test data. It reports overlap
and tail errors plus revision magnitude; one logged expert is not the only
valid behavior. Closed-loop success remains the primary result. With K=H,
every decision restarts from its observed anchor; the old-plan diagnostic is
marked not applicable because no overlap remains.

### Check the pusher / previous-plan mismatch

Every closed-loop evaluation now records the following distances in **pixels**.
Here `old[K]` is the first unexecuted target, and `last_command` is the command
actually sent at the preceding robot step.

| Metric | Euclidean distance between |
| --- | --- |
| `retained_pusher_gap_px` | `old[K]` and current pusher position |
| `previous_command_mismatch_px` | `old[K-1]` and `last_command` |
| `command_tracking_gap_px` | `last_command` and current pusher position |
| `retained_plan_jump_px` | `old[K]` and `old[K-1]` |

Summaries append `_mean`, `_median`, `_p95`, and `_max`. They pool replan events,
not episode averages. Startup and K=H have no retained plan and are excluded;
`gap_samples=0` means not applicable, not perfect agreement. Targets are the
clean, **unclipped** old commands, before completion noise or SB/FM revision.
Thus online command mismatch can reflect clipping, and pusher gap can reflect
tracking lag or a jump within the old plan—not just offline replay mismatch.

For a pretrained checkpoint, compare offline generated and expert previous
plans on identical held-out states, without running simulation:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.diagnose_pusht_sources \
  --checkpoint "$run_root/sb_ou/best.pt" \
  --windows "$run_root/windows.pt" --device cpu --episodes 8 --replans 16
```

This writes a new timestamped `workspace/sb_pusht/source_gaps/` directory with
`source_gaps.json`: per-replan measurements, summaries and checkpoint/data
identities. `--replans` includes the excluded startup. The default panel is
eight seeded validation episode prefixes, not all validation replans.
Windows must match the checkpoint's split, normalization, H and K; copying
them from another machine is fine. Training caches/checkpoints are not changed.

For offline **and** closed-loop measurements of the same checkpoint:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.eval_pusht_sb_ou \
  --checkpoint "$run_root/sb_ou/best.pt" --device cpu \
  --episodes 20 --workers 4 --threads 1 --no-save-videos \
  --source-gap-windows "$run_root/windows.pt"
```

The usual `metrics.json` now includes the online metrics and the same metrics
prefixed `offline_self_` / `offline_expert_`. Offline details are in
`source_gaps.json`; online details are in each episode's `plan_gaps` list.
Use `--source-gap-episodes` and `--source-gap-replans` to enlarge the offline panel.
The online panel uses simulator seeds, not the offline demonstration states.

During training, revisers automatically compute these diagnostics in the
**existing asynchronous evaluation worker**, using the run's `windows.pt`.
When W&B is enabled, compare `sim_eval/retained_pusher_gap_px_mean` with
`sim_eval/offline_self_retained_pusher_gap_px_mean` and
`sim_eval/offline_expert_retained_pusher_gap_px_mean` (also check p95 and counts).
The offline self panel always uses the evaluated forward EMA with `p_self=1`;
it is not the mixed/frozen source cache originally used to train that checkpoint.

Gaps diagnose a distribution mismatch, not its effect on success. For the
controlled retraining comparison, use the
[expert-previous-chunk control](../hpc/sb_pusht_ablations/README.md#control-expert-previous-chunks-instead-of-self-generated-chunks).

## Visualize completion and revision

Generate a short diagnostic MP4 and per-decision PNG storyboards from a trusted
EMA checkpoint, without training or access to its original dataset:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.visualize_pusht_revision \
  --checkpoint "$run_root/sb_kinetic/best.pt" --device cpu --threads 2 \
  --seed 1000000 --start-replan 1 --replans 3 --execute 8
```

The method is inferred from the checkpoint: `fm_paired`, `fm_local_ot`, `sb_ou`,
or `sb_kinetic`; SB requires forward weights. DDIM has no old-plan completion/
revision process and is not supported here. Swap in an FM checkpoint and add
`--completion repeat`, `--completion fixed_damped`, or
`--completion learned_dissipative` to inspect the different tail rules. Omit
`--completion` and `--execute` to inherit the checkpoint settings. Use
`--start-replan 10` to inspect later decisions nearer contact (seed-dependent),
or `--start-replan 0` for startup. Decisions are zero-indexed; warmup is simulated
from the same seed. With K=H every decision uses the startup anchor because no
old-plan overlap remains, so completion is explicitly marked not applicable.
The startup anchor is the last executed command (initially the observed pusher
position), not an unexecuted old target.

The robot stays frozen while retained overlap, tail completion, one-time source
noise, and actual sampler intermediates are shown; only the execution phase
shows physical motion. Candidate targets are not the pusher trajectory.
This is a diagnostic clip, **not a success-rate benchmark**; it may stop early
on environment termination or at the checkpoint's episode limit.

Outputs go to a fresh timestamped directory under
`workspace/sb_pusht/visualizations/`, or a new `--output-dir`. The
`visualization.json` manifest records checkpoint/runtime identities, settings,
phase semantics and artifacts. `--save-gif` adds a GIF, `--fps 10` controls
playback, and `--no-progress` hides rollout progress. CPU/two threads are the
defaults. The command collects a fresh full trace: older evaluation JSON files
contain only the first three coarse traces and cannot reconstruct full FM/SB
generation histories.

### Many candidate revisions at one state

Generate 200 alternatives from one **learner-reached** decision, without a dataset:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.visualize_pusht_candidates \
  --checkpoint "$run_root/sb_ou/best.pt" --device cpu \
  --candidates 200 --seed 1234 --replan 6 --candidate-seed 42
```

`--seed` changes the simulator reset and policy warm-up. `--replan` selects the
zero-based decision: at H16/K8, decision 6 follows 48 executed commands.
`--candidate-seed` changes only the alternative draws at that same decision.
The observation history, actual executed-command history, old plan, and completed
source are shared across candidates. No candidate is scored or executed.

Default `--source-noise fixed` shares the original decision's perturbed source;
diversity comes from SB sampling (also fresh auxiliary velocities for kinetic SB).
`--source-noise independent` perturbs the same completed plan separately per draw;
`--source-noise none` removes source noise, not SB process noise. FM is deterministic
for a fixed source: use `independent` to inspect FM's source-induced diversity.
`--batch-size 32` bounds memory. Keep it fixed for reproducible candidate draws.

Outputs go to a new timestamped `workspace/sb_pusht/candidate_batches/` directory:
`candidate_overlay.png` / `.svg` (all draws, full view and next-K zoom),
`candidate_grid.png` (up to 16 evenly spaced, unranked draws), `candidates.npz`
(all raw command arrays and policy context), and `candidates.json` (seeds,
checkpoint identity and spread statistics). Curves are **unclipped command
targets, not simulated physical paths**. This does not yet implement simulator
cloning, candidate-outcome evaluation, or MPC; the saved observation is not a
complete physics snapshot. Existing rollout/checkpoint files are unchanged.

For a **synthetic symmetric wrong-side probe**, use:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.visualize_pusht_candidates \
  --checkpoint "$run_root/sb_ou/best.pt" --device cpu \
  --scene symmetric --candidates 1000 --candidate-seed 42
```

The T and goal share their symmetry axis; the pusher starts on the goal-facing
side and must go around the T to push toward the goal. The hand-built previous
chunk retreats along that axis. Histories assume ideal command tracking, not
simulated motion. This mode defaults to `fixed_damped` completion and no source
perturbation, keeping the whole starting chunk symmetric; SB sampling remains
stochastic, and its learned reference is unchanged. `--seed` and `--replan` do
not apply here. The additional `candidate_lateral.png` shows signed offsets of
commands K and H from the symmetry axis—not measured go-around success.

To probe source sensitivity, add `--source-variant right_tilt_15`,
`--source-variant right_turn_90`, or `--source-variant right_route` to that command
(`axial` is the default). The first two rotate the unexecuted old suffix 15° or
90° toward the opposite side of the T; they retain its short length. The last
uses a longer right-side go-around template designed for H16/K8. Each source is
completed with the selected completion mode. The scene, observation/action
histories, executed old prefix and learned reference stay fixed. Keep the same
candidate seed and batch size to use paired generation noise across conditions.
“Right” is positive lateral displacement from the T's symmetry axis, not screen
horizontal. These hand-built sources do not establish simulated feasibility.

## Another dataset

Generic code lives in `action_bridge/plan_revision/`, without simulator imports.
The adapter supplies normalized `obs_hist` (tensor or mapping), `act_hist`,
`future_actions`, `valid_mask`, episode/time IDs on a complete replan grid, and
`startup_actions` defined from observed information in that action representation.
Provide an encoder returning [batch, history_dim], action codec, local-OT context
metric/compatibility rule, and environment evaluation attachment.

Push-T coordinate slicing belongs only in its adapters. Included completion
semantics are absolute Euclidean targets; other representations need an explicit
compatible adapter. Partial-horizon Gaussian endpoints are rejected.
Custom encoder factories must be supplied when rebuilding their checkpoints.

```bash
uv run --frozen --no-sync pytest -q tests/test_revision_*.py \
  tests/test_diffusion_policy.py tests/test_pusht_training.py \
  --basetemp workspace/test-artifacts/sb-pusht
```

Tests and reduced-budget smoke runs verify implementation, not scientific
performance. Reports distinguish pending jobs from completed results; one
training seed cannot establish seed robustness or exact SB optimality.
