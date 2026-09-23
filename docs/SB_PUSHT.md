# Whole-plan revision on Push-T

The active experiment is **self_source_v1**, defined in
[SB-PUSHT-ADDENDUM.md](../SB-PUSHT-ADDENDUM.md). It replaces the fixed-DDIM-source
protocol; old runs remain untouched and are not rows in this comparison.
The [edited plan](../SB-PUSHT_EDIT.md) describes the reusable interfaces and
OU/kinetic mathematics. This is offline behavior cloning, separate from the
existing Action Bridge loss.

## HPC (Peano)

The files `hpc/sb_pusht_*_h200_1gpu.sbatch` submit **separate jobs**, not an
`all` job. Every file uses the existing H200 partition `gpuq`, never the B200
partition `aiq`, and refuses to run if CUDA reports a non-H200 GPU.
Each main policy job requests one H200, eight CPUs, 32 GB RAM and ten hours.
The default scientific settings remain 300k updates, H=16, K=8 and 32 NFE.

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
The same five validation seeds, busy-worker skipping, final drain and local
video defaults described below apply. Videos are not uploaded to W&B.

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

## Logging and asynchronous evaluation

All five policy trainers evaluate every **10k updates** in a separate process
(default CPU, two threads). Training continues while it runs. Change this with
`--eval-every 5000 --eval-device cpu --eval-threads 2`.
One worker runs per trainer; busy intervals are skipped, not queued. At training
end, pending evaluation finishes and final weights are evaluated if necessary.
Use `--no-sim-eval` to disable it.

W&B is opt-in with `--wandb` (project `action-bridge-policy`).
It logs losses, gradient norms, source/coupling timing and origin diagnostics,
OT statistics where applicable, and validation success/coverage/smoothness.
Action-chunk images show predictions, expert targets and the T pose every 5k
updates by default. Use `--wandb-images-every`, `--wandb-image-count`, or
`--wandb-mode offline` as needed. Diagnostic sampling preserves training RNG.

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

Each invocation creates a timestamped directory below
`workspace/sb_pusht/evaluation/`, or use `--output-dir` with a new path.
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
