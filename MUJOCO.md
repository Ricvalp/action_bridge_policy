# MuJoCo demonstrations and Action Bridge

This document is the executable handoff for the first PHI native-MuJoCo
experiment. The implemented task is `planar_reach`, variation 0. It uses an
8-dimensional policy-visible state, 2-dimensional direct joint torque at
50 Hz, and a deterministic analytic-IK/PD demonstration policy.

The first goal is an end-to-end acceptance run, not a claim that planar reach
is the final research benchmark:

```text
phi-mujoco scripted collection
  -> immutable validated episode bundle
  -> loader-time normalized temporal windows
  -> Action Bridge or direct chunk BC training
  -> trusted checkpoint reconstruction
  -> phi-mujoco closed-loop native evaluation
```

## Ownership boundary

`phi-mujoco` owns simulator semantics and simulator-derived data:

- task construction, reset, step, success, truncation, seeding, rendering, and
  lifecycle;
- the named observation and action profiles;
- scripted demonstration collection, retry/resume, manifests, checksums, and
  bundle validation;
- deterministic episode splits, train-derived normalization, and loader-time
  temporal windows;
- framework-neutral policy and evaluation protocols and structured native
  evaluation artifacts.

Action Bridge owns model-specific work:

- Torch models, losses, optimization, and checkpoints;
- conversion of NumPy windows to Torch batches;
- checkpoint trust decisions and reconstruction;
- online observation/action histories, normalization, latent state, and the
  `phi-mujoco` policy adapter;
- offline learned-policy metrics and experiment configuration.

Consequently, `phi-mujoco` alone can generate and validate demonstrations and
construct training windows. It deliberately cannot train Action Bridge or load
its Torch checkpoints. Conversely, offline Action Bridge training never needs
to launch MuJoCo. Native MuJoCo is launched only for collection or closed-loop
evaluation.

The integration consumes `phi_mujoco.windows.PlanarReachWindowDataset`
directly. It must not use `PushTLowDimDataset`: Push-T's initial-action padding
would interpret the first two observation components as actions, whereas the
MuJoCo reset action history is physical zero torque.

## Local source and environment setup

The current parent lock uses this editable source:

```toml
phi-mujoco = { path = "workspace/phi-mujoco", editable = true }
```

Therefore both directories must exist before syncing the parent environment.
From the parent repository root:

```bash
test -f workspace/phi-mujoco/pyproject.toml

(
  cd workspace/phi-mujoco
  scripts/bootstrap.sh --with-gym --with-viz
  scripts/with_mujoco_env.sh --gl disable -- \
    uv run --frozen phi-mujoco doctor --launch-test --json
)

# Select the one Torch extra appropriate for this machine. CPU is the most
# portable local acceptance profile.
uv sync --locked --extra cpu
```

The official `mujoco` Python wheel includes the native engine; no CoppeliaSim,
separate MuJoCo archive, licence server, global Python installation, or `sudo`
operation is involved.

## 1. Generate one immutable collection

Every collection target is create-once. Choose a new path; do not delete or
reuse a completed bundle. The example below generates 1,024 successful
single-mode demonstrations. A 64-episode bundle is sufficient for a plumbing
smoke, while 1,024 is a reasonable first learned-policy baseline.

```bash
export ACTION_BRIDGE_ROOT="$PWD"
export PHI_MUJOCO_ROOT="$ACTION_BRIDGE_ROOT/workspace/phi-mujoco"
export COLLECTION_ROOT="$PHI_MUJOCO_ROOT/runs/datasets/planar-reach-pd-seed0-1024"

mkdir -p "$(dirname "$COLLECTION_ROOT")"
test ! -e "$COLLECTION_ROOT"

(
  cd "$PHI_MUJOCO_ROOT"
  scripts/with_mujoco_env.sh --gl disable -- \
    uv run --frozen phi-mujoco collect \
      --output-dir "$COLLECTION_ROOT" \
      --episodes 1024 \
      --max-steps 200 \
      --attempts-per-episode 2 \
      --seed 0 \
      --policy pd \
      --require-success \
      --json
)
```

If collection is interrupted before `manifest.json` is published, repeat the
identical command with `--resume`. Resume verifies the existing prefix and
does not overwrite a completed bundle. Do not add `--resume` to an ordinary
new or completed run.

## 2. Validate before training

The semantic validator checks the manifest, hashes, paths, metadata, arrays,
dtypes, profiles, seeds, terminal flags, and action bounds. Opening an NPZ
directly is not an equivalent validation.

```bash
(
  cd "$PHI_MUJOCO_ROOT"
  COLLECTION_ROOT="$COLLECTION_ROOT" \
    scripts/with_mujoco_env.sh --gl disable -- \
    uv run --frozen python - <<'PY'
import os

from phi_mujoco.dataset import validate_collection_bundle

bundle = validate_collection_bundle(os.environ["COLLECTION_ROOT"])
print(
    {
        "manifest_sha256": bundle.manifest_sha256,
        "episodes": bundle.requested_episodes,
        "successful_episodes": bundle.successful_episodes,
        "total_steps": bundle.total_steps,
        "runtime_cleanup_succeeded": bundle.runtime_cleanup_succeeded,
    }
)
PY
)
```

Retain the printed manifest SHA-256 with the experiment record. Training also
validates the bundle and embeds its identity, split, normalization, profile,
history, horizon, and action-execution metadata in the checkpoint.

### Why there is no second cache

The collection bundle is the immutable simulator-derived asset. Training
windows are cheap deterministic views with the alignment

```text
obs[t-history+1 : t+1]
actions[t-action_history : t]
actions[t : t+horizon]
observations[t+1 : t+horizon+1]
```

Reset observations and physical `0 N*m` actions provide masked left padding;
terminal targets are never fabricated. Splits happen by episode and
normalization is derived from the training split only. Materializing those
views into another cache would duplicate data and introduce an additional
artifact that could drift from the manifest, split, or normalization. Add a
versioned derived cache only if a measured loading bottleneck later justifies
one; never rewrite the source collection.

## 3. One-step training acceptance

Use a new run ID. This command exercises bundle validation, split and
normalization construction, collation, model/loss/optimizer code, checkpoint
metadata, checkpoint writing, and bounded offline evaluation on CPU:

```bash
export MUJOCO_RUNS_ROOT="$ACTION_BRIDGE_ROOT/workspace/experiments/mujoco"
export RUN_ID="planar_reach_continuous_smoke_seed0"
test ! -e "$MUJOCO_RUNS_ROOT/$RUN_ID"

uv run --frozen --extra cpu python -m action_bridge.training.train_mujoco \
  --config-name mujoco_planar_reach_continuous \
  data.collection_root="$COLLECTION_ROOT" \
  device=cpu \
  output_dir="$MUJOCO_RUNS_ROOT" \
  run_id="$RUN_ID" \
  optim.max_steps=1 \
  optim.batch_size=8 \
  logging.log_every_steps=1 \
  logging.eval_every_steps=1 \
  logging.checkpoint_every_steps=0 \
  eval.offline_max_batches=1 \
  eval.clip_actions=true
```

Expected files include:

```text
workspace/experiments/mujoco/<run-id>/
├── config.json
├── checkpoints/
│   ├── best.pt
│   └── latest.pt
├── metrics/
│   ├── train_metrics.csv
│   ├── val_metrics.csv
│   ├── mujoco_offline_metrics.json
│   └── test_metrics.json
```

The normalized online-evaluation contract is embedded in both checkpoints;
there is no separate mutable online-metadata sidecar in a normal training run.

For the direct chunk BC control, change only the config name:

```bash
--config-name mujoco_planar_reach_direct_chunk_bc
```

For the Action Bridge reference/controller without a latent bottleneck, use:

```bash
--config-name mujoco_planar_reach_no_latent
```

For a real GPU run, remove the smoke overrides, select the appropriate locked
Torch extra (`cu118`, `cu126`, `cu128`, or `cu130`), and set `device=cuda`.
The default horizon is four actions. This retains near-goal transitions in
these roughly 30-step demonstrations; the audited horizon-16 run excluded the
last 15 transitions from every full-horizon window and generalized much more
poorly. The current continuous-latent default is 50,000 optimizer steps and is
an experiment configuration, not a tuned claim of latent-policy superiority.

## 4. Trusted checkpoint native evaluation

Torch checkpoint loading uses pickle and can execute code. Evaluate only a
checkpoint produced locally or transferred through a trusted channel after
verifying its checksum:

```bash
export CHECKPOINT="$MUJOCO_RUNS_ROOT/$RUN_ID/checkpoints/best.pt"
sha256sum "$CHECKPOINT"

mkdir -p "$MUJOCO_RUNS_ROOT/$RUN_ID/eval"
export NATIVE_RUN="$MUJOCO_RUNS_ROOT/$RUN_ID/eval/native-seed1000000"
test ! -e "$NATIVE_RUN"

MUJOCO_GL=disable \
uv run --frozen --extra cpu python -m action_bridge.eval.mujoco_online \
  --checkpoint "$CHECKPOINT" \
  --trusted-checkpoint \
  --device cpu \
  --run-dir "$NATIVE_RUN" \
  --episodes 50 \
  --max-steps 200 \
  --actions-per-plan 1 \
  --seed 1000000 \
  --no-require-success \
  --json
```

`--no-require-success` is appropriate for the technical smoke: simulator,
policy, artifact, or cleanup failures still make the command fail, while a
poor one-step model can finish and report its honest success rate. Omit that
flag for an experiment whose acceptance criterion is 100% success.

The run directory must be new and its parent must already exist. The loader
reads a stable regular-file snapshot, validates embedded or explicitly
provided online metadata, reconstructs the exact model, and only then imports
and launches native MuJoCo. The v1 contract fixes `--actions-per-plan=1` and
rejects non-finite or out-of-bound torque predictions by default. The training
command above opts into torque clipping explicitly through
`eval.clip_actions=true`; that choice is embedded in the checkpoint and
clipping is diagnosed by the adapter. It is not a runtime-side silent clip.

For video, use another new run directory and a separately validated rendering
backend:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --frozen --extra cpu python -m action_bridge.eval.mujoco_online \
  --checkpoint "$CHECKPOINT" \
  --trusted-checkpoint \
  --device cpu \
  --run-dir "$MUJOCO_RUNS_ROOT/$RUN_ID/eval/native-egl-seed1000100" \
  --episodes 3 \
  --seed 1000100 \
  --record-video \
  --no-require-success \
  --json
```

Do not infer EGL support from GPU visibility; first pass the `phi-mujoco
doctor --render-test` gate on that host.

## HPC portability

The parent branch does not, by itself, transport the ignored
`workspace/phi-mujoco` source tree. A parent clone on the HPC is therefore not
ready for `uv sync` until the child project is reproduced at the pinned path.
Do not copy an unversioned working directory and call it reproducible.

The preferred handoff is an independent private `phi-mujoco` repository:

1. select an explicit repository licence before any public release (the
   current `NOTICE.md` records provenance but grants no redistribution rights);
2. push the already-independent child history to the PHI Lab organization;
3. record the exact child commit in the experiment notes;
4. clone or check out that commit at `workspace/phi-mujoco` on the HPC;
5. run `uv sync --locked --extra <matching-torch-extra>` in the parent.

Until a remote exists, an independent child Git bundle is acceptable:

```bash
test "$(git -C workspace/phi-mujoco rev-parse --show-toplevel)" = \
  "$PWD/workspace/phi-mujoco"

PHI_MUJOCO_SHA="$(git -C workspace/phi-mujoco rev-parse HEAD)"
PHI_MUJOCO_BUNDLE="$PWD/workspace/phi-mujoco-${PHI_MUJOCO_SHA:0:12}.bundle"
git -C workspace/phi-mujoco bundle create "$PHI_MUJOCO_BUNDLE" --all
git bundle verify "$PHI_MUJOCO_BUNDLE"
sha256sum "$PHI_MUJOCO_BUNDLE"
```

Transfer the bundle and its SHA-256 through the institute-approved channel,
then on the HPC:

```bash
mkdir -p workspace
git clone /approved/path/phi-mujoco-<sha>.bundle workspace/phi-mujoco
git -C workspace/phi-mujoco switch --detach <full-child-sha>
test "$(git -C workspace/phi-mujoco rev-parse HEAD)" = "<full-child-sha>"
uv sync --locked --extra cu130  # replace with the verified Peano Torch profile
```

A Git bundle transports source history, not demonstrations. Transfer each
completed collection independently, preserve every byte, and run
`validate_collection_bundle` on the HPC before training. Record its manifest
SHA-256. Checkpoints and evaluation outputs are separate experiment artifacts
and must not be added to either source repository.

The template [`hpc/mujoco_planar_reach.sbatch`](hpc/mujoco_planar_reach.sbatch)
assumes the parent and child checkouts, locked parent environment, and
validated collection are already present. It performs no network install and
runs offline training followed by a physics-only native evaluation. Before
submission:

```bash
mkdir -p workspace/hpc-logs
export COLLECTION_ROOT=/absolute/read-only/path/to/planar-reach-collection
sbatch --export=ALL,PROJECT_DIR="$PWD",COLLECTION_ROOT="$COLLECTION_ROOT" \
  hpc/mujoco_planar_reach.sbatch
```

The Peano Torch/CUDA profile, filesystem location, account/QoS, time request,
and GPU partition must still be checked for the target allocation. Rendering
requires a separate EGL audit; the supplied job intentionally performs no
rendering.

## Current validation truth

As of 2026-08-19, the current workspace implementation has recorded these
backend results:

- `phi-mujoco` default suite: 162 passed, 8 skipped;
- native physics suite: 169 passed, 1 skipped;
- native EGL suite: 170 passed;
- Ruff, format, mypy, and wheel-content checks passed;
- a real three-episode collection produced 45 valid horizon-16 windows.

The parent online checkpoint/adapter targeted suite passed 27 tests before the
final aggregate regression. A synthetic zero-output direct-BC checkpoint also
passed a one-episode, one-step loader-to-adapter-to-real-MuJoCo smoke and shared
provenance validation.

The independent child source used here is commit
`22775aeae421214be8d27de1e702ed8ad86398f9`. A complete local handoff bundle is
`workspace/phi-mujoco-22775ae.bundle`, with SHA-256
`b098a81239dc2347517133380382b5bf4b0b2c99d7d4798a06c1af680cbc7b94`.
The bundle is a transport fallback until an approved PHI Lab remote exists.

The complete learned-policy path was then run against a new 2,048-episode
collection (58,402 true transitions; all episodes successful; manifest
`7b6be28339b790932c5cd9efa971f7c0a8d779e0f5de42dd286c5e8fff58021f`).
On 100 fixed held-out seeds starting at 1,000,000:

| Policy | Training | Success | Exceptions | Cleanup |
| --- | ---: | ---: | ---: | --- |
| direct chunk BC, horizon 4 | 5,000 steps | 99/100 | 0 | passed |
| no-latent Action Bridge, horizon 4 | 20,000 steps | 94/100 | 0 | passed |
| continuous-latent Action Bridge, horizon 4 | 5,000 steps | 18/100 | 0 | passed |

The direct-BC test-set physical first-action MSE was about `0.0387 N^2 m^2`
at 5,000 steps; the 20,000-step no-latent bridge reached about
`0.0203 N^2 m^2`. The continuous result used the learned prior at inference
and remains under-tuned; its gap is an experimental result, not a missing
runtime integration. A horizon-16 direct-BC audit reached only 65/100 on the
same held-out seeds, which is why four is now the default.

Interactive GLFW operation remains unvalidated on the audited workstation.
Physics-only native MuJoCo and EGL rendering passed there. No claim is made
for an unaudited HPC rendering stack.

## Experiment sequence

### A. Single-mode acceptance now

Use the implemented `pd` collector first. Compare
`mujoco_planar_reach_direct_chunk_bc` with
`mujoco_planar_reach_no_latent` and
`mujoco_planar_reach_continuous`. Report, at minimum:

- held-out physical torque MSE and first-action MSE;
- predicted bound-violation and target-saturation rates;
- closed-loop success over fixed, collection-disjoint seeds;
- return, steps to success, invalid actions, exceptions, and runtime cleanup.

Planar reach is fully observed and the current expert is deterministic. This
experiment validates the pipeline; it should not be used to argue that a
latent Action Bridge is superior to deterministic BC.

### B. Paired elbow modes later

The smallest meaningful multimodal extension is to expose explicit
`elbow_up` and `elbow_down` IK modes and collect two bundles over the same
2,048 reset seeds. Both branches succeeded for all 512 seeds in a local
in-memory audit, with mean episode length about 30.2 steps, p95 39, and maximum
42. The current CLI does **not** implement these expert modes yet.

When implemented, split by reset seed before combining the two bundles so the
paired initial state cannot leak across train and test. Start with a
two-category latent, commit the sampled latent for the episode, and compare it
with direct chunk BC and a no-latent bridge. Add best-of-K error, mode coverage,
branch consistency, and success per sampled mode to the ordinary closed-loop
metrics.

MuJoCo Playground remains a later, distinct integration. Its MJX/Warp/JAX
execution path and assets should be preserved, and it does not currently
provide the scripted demonstration source needed for this first experiment.
