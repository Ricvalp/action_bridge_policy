# Action Bridge with PHI Isaac Lab

This is the executable handoff for the first Isaac Lab experiment:

```text
batched Warp expert collection in phi-isaaclab
  -> immutable validated HDF5 episode bundle
  -> normalized transition windows in Action Bridge
  -> direct BC or Action Bridge training
  -> trusted checkpoint reconstruction
  -> batched closed-loop Isaac evaluation and optional video
```

The implemented contract is `franka_cube_lift` variation 0:

- upstream task `Isaac-Lift-Cube-Franka-IK-Abs-v0`;
- observation profile `phi.isaaclab.franka_cube_lift.state.v2`, width 35;
- action profile
  `phi.isaaclab.franka_cube_lift.ee_pose_abs_gripper.v2`, width 8;
- raw acquisition schema 2 and processed cache schema 2;
- demonstration expert
  `phi.isaaclab.franka_cube_lift.reactive_pick_lift.v1`;
- observation history 2, action history 2, prediction horizon 4;
- 50 Hz control and one executed action per plan.

Profile v2 normalizes each XYZW quaternion, then makes its first
largest-absolute-value component non-negative; exact ties use the lowest XYZW
index. This replaces the historical non-negative-w rule. Version-1 caches and
checkpoints are incompatible with this workflow and must not be relabelled or
silently upgraded.

## Ownership and environments

`phi-isaaclab` owns the task, expert, raw recording, immutable cache, splits,
normalization, windows, strict action boundary, and generic batched evaluator.
This repository owns Torch models, losses, training, checkpoints, and the
device-native policy adapter.

Use two environments:

- the Action Bridge `.venv` for cache loading and training;
- `workspace/phi-isaaclab/.native/venv` for Isaac Sim collection/evaluation.

Do not `uv sync` the Action Bridge project into the native environment: it can
replace Isaac's exact Torch/CUDA stack. Data bundles and trusted checkpoints
are the handoff between environments.

Set paths once from the Action Bridge repository root:

```bash
export ACTION_BRIDGE_ROOT="$PWD"
export PHI_ISAACLAB_ROOT="$ACTION_BRIDGE_ROOT/workspace/phi-isaaclab"
export ISAAC_DATA_ROOT="$ACTION_BRIDGE_ROOT/workspace/datasets/isaaclab"
export ISAAC_RUNS_ROOT="$ACTION_BRIDGE_ROOT/workspace/experiments/isaaclab"
mkdir -p "$ISAAC_DATA_ROOT/raw" "$ISAAC_DATA_ROOT/processed" "$ISAAC_RUNS_ROOT"
```

## 1. Install and validate Isaac

Isaac Sim and NVIDIA-hosted assets are proprietary. Review NVIDIA's terms
before installing or using them. Each operator must explicitly set the
official process-local acceptance variable; neither repository sets it:

```bash
(
  cd "$PHI_ISAACLAB_ROOT"
  scripts/bootstrap_native.sh
)

# Do this only after you have reviewed and accepted NVIDIA's terms.
export OMNI_KIT_ACCEPT_EULA=YES

(
  cd "$PHI_ISAACLAB_ROOT"
  scripts/with_native_env.sh -- \
    phi-isaaclab doctor \
      --isaaclab-source-root "$PHI_ISAACLAB_ROOT/.native/IsaacLab" \
      --writable-root "$PHI_ISAACLAB_ROOT/.native" \
      --portable-root "$PHI_ISAACLAB_ROOT/.native/kit/doctor-launch" \
      --device cuda:0 \
      --launch-test \
      --json
)
```

The exact profile is Python 3.12, Isaac Sim 6.0.1.0, Isaac Lab commit
`ffff603eafc6b74264a5261cc0183d6a65390d78`, and the
Torch/torchvision/torchaudio
`2.10.0+cu128`/`0.25.0+cu128`/`2.10.0+cu128` distributions. Current Isaac Sim
6.0.1 metadata asks for the corresponding 2.11/0.26/2.11 family and also
disagrees with the source install's `coverage` pin. The backend install guide
records all four audited conflicts, and the bootstrap rejects any additional
`uv pip check` disagreement.

Do not continue to data collection after a failing doctor. Rendering is a
separate gate: add `--render-test` with a new portable root only when cameras
are needed.

## 2. Collect demonstrations

Use a new output and Kit root. This baseline requests 2,048 successful demos
across eight collection-seed streams and keeps all live tensors vectorized on
the GPU:

```bash
export RAW_ID=franka-cube-lift-reactive-v1-2048-seed0-v1
export RAW_ROOT="$ISAAC_DATA_ROOT/raw/$RAW_ID"
test ! -e "$RAW_ROOT"

(
  cd "$PHI_ISAACLAB_ROOT"
  scripts/with_native_env.sh -- \
    phi-isaaclab collect \
      --output-directory "$RAW_ROOT" \
      --episodes 2048 \
      --num-shards 8 \
      --num-envs 64 \
      --base-seed 0 \
      --device cuda:0 \
      --isaaclab-source-root "$PHI_ISAACLAB_ROOT/.native/IsaacLab" \
      --writable-root "$ACTION_BRIDGE_ROOT/workspace" \
      --portable-root "$PHI_ISAACLAB_ROOT/.native/kit/collect-$RAW_ID"
)
```

The collector records only successful episodes from the timer-free reactive
expert. Each decision uses current O35 and, after the first command, prior
policy-visible A8; the first waypoint is anchored at the observed TCP. Its
exact behavior descriptor and thresholds are bound into configuration and
provenance. One true simulator RNG stream belongs to each shard; vector lanes
are not mislabeled as independent seeds. Batched completion may produce
slightly more successes than requested, and `summary.json` records the actual
count. A strict `manifest.json` is atomically published last; its presence and
successful validation identify a complete collection. A failed or interrupted
collection is not resumed by this release—inspect it for diagnosis and choose a
new ID.

On a new host, first use a fresh ID with `--episodes 12`, `--num-shards 3`,
`--num-envs 4`, and a new `--base-seed`, then convert and validate that
schema-2 smoke. Twelve episodes are a systems check, not a useful corpus.
Earlier 12-episode and 2,048-episode acquisitions used superseded expert,
reset, or profile semantics and are diagnostic artifacts only.

## 3. Convert and validate the immutable cache

The raw output contains one `raw_episodes.hdf5` in each seed directory. Select
those completed shards explicitly and convert them with the offline backend
environment:

```bash
export PROCESSED_ID=franka-cube-lift-reactive-v1-2048-seed0-v1
export COLLECTION="$ISAAC_DATA_ROOT/processed/$PROCESSED_ID"
test ! -e "$COLLECTION"

raw_args=()
while IFS= read -r shard; do
  raw_args+=(--raw "$shard")
done < <(find "$RAW_ROOT/raw-shards" -type f -name raw_episodes.hdf5 -print | sort)
test "${#raw_args[@]}" -ge 6

(
  cd "$PHI_ISAACLAB_ROOT"
  scripts/with_offline_env.sh -- uv run --frozen phi-isaaclab convert-raw \
    "${raw_args[@]}" \
    --raw-provenance "$RAW_ROOT/provenance.json" \
    --raw-manifest "$RAW_ROOT/manifest.json" \
    --output-directory "$COLLECTION"
  scripts/with_offline_env.sh -- \
    uv run --frozen phi-isaaclab validate-cache "$COLLECTION"
)
```

Retain the printed manifest SHA-256. Both raw and processed destinations are
create-once. Conversion writes a complete sibling, validates transition
alignment and content hashes, then publishes it atomically; it never appends
to or overwrites a cache. `--raw-manifest` must bind the exact complete shard
set and the supplied `--raw-provenance`; missing, extra, repeated-path,
byte-identical, or content-mismatched shards are rejected. Portable cache
metadata identifies each source by stable ordinal, SHA-256, size, collection
seed, and episode count; it does not retain workstation paths or filenames.

The resulting manifest must identify `phi.isaaclab.episode_hdf5` schema 2 and
the v2 observation/action profiles. Conversion rejects raw provenance that
does not bind the current reactive expert descriptor, including any changed
threshold.

## 4. Install the Action Bridge training environment

The parent lock currently uses editable backend checkouts under `workspace/`:

```bash
test -f "$PHI_ISAACLAB_ROOT/pyproject.toml"
uv sync --locked --extra cu128
```

Select exactly one Torch extra supported by the training machine. The training
environment does not import Isaac Sim and can run on another host after the
validated collection is transferred.

## 5. One-step training acceptance

Start with direct chunk BC and a new run ID. This exercises cache validation,
split/normalization construction, batching, model/loss/optimizer code,
checkpoint metadata, checkpoint writing, and bounded offline evaluation:

```bash
export RUN_ID=isaaclab_cube_lift_reactive_v1_direct_smoke_seed0_v1
test ! -e "$ISAAC_RUNS_ROOT/$RUN_ID"

uv run --frozen --extra cu128 \
  python -m action_bridge.training.train_isaaclab \
    --config-name isaaclab_franka_cube_lift_direct_chunk_bc \
    data.collection_root="$COLLECTION" \
    device=cuda \
    output_dir="$ISAAC_RUNS_ROOT" \
    run_id="$RUN_ID" \
    optim.max_steps=1 \
    optim.batch_size=8 \
    logging.log_every_steps=1 \
    logging.eval_every_steps=1 \
    logging.checkpoint_every_steps=0 \
    eval.offline_max_batches=1
```

Then run substantive training with one of:

```text
isaaclab_franka_cube_lift_direct_chunk_bc
isaaclab_franka_cube_lift_no_latent
isaaclab_franka_cube_lift_continuous
```

Use direct BC as the control, then the no-latent bridge, then the continuous
latent model. Every checkpoint embeds the exact cache manifest, split,
train-derived normalization, profile names, temporal geometry, and policy-side
action projection. Resume with a different identity is rejected.

## 6. Prepare trusted native checkpoint inference

Action Bridge checkpoints use PyTorch pickle and may execute code. Use only a
locally produced or trusted transferred file and verify it first:

```bash
export CHECKPOINT="$ISAAC_RUNS_ROOT/$RUN_ID/checkpoints/best.pt"
test -f "$CHECKPOINT"
sha256sum "$CHECKPOINT"
```

Install the lightweight model-config dependency and this trusted source
checkout into the existing native environment without resolving the Action
Bridge dependency set over Isaac's Torch:

```bash
native_python="$PHI_ISAACLAB_ROOT/.native/venv/bin/python"
native_uv_cache="$PHI_ISAACLAB_ROOT/.native/uv-cache"

UV_CACHE_DIR="$native_uv_cache" \
  uv pip install --python "$native_python" --no-deps 'ml-collections==1.1.0'
UV_CACHE_DIR="$native_uv_cache" \
  uv pip install --python "$native_python" --no-deps --editable "$ACTION_BRIDGE_ROOT"
```

The parent build configuration limits the editable wheel to `action_bridge/`;
the install does not package ignored simulator checkouts or experiment data.
Do not omit `--no-deps`. A later released policy adapter should be packaged
separately with explicit native-compatible dependencies.

## 7. Evaluate closed loop

The output directory must not exist. A UTC timestamp prevents accidental
reuse. Start with a small smoke and no video:

```bash
export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export NATIVE_RUN="$ISAAC_RUNS_ROOT/$RUN_ID/eval/seed1000000-5eps-$STAMP"
mkdir -p "$(dirname "$NATIVE_RUN")"
test ! -e "$NATIVE_RUN"

(
  cd "$PHI_ISAACLAB_ROOT"
  scripts/with_native_env.sh -- \
    python -m action_bridge.eval.isaaclab_online \
      --checkpoint "$CHECKPOINT" \
      --trusted-checkpoint \
      --collection-manifest "$COLLECTION/manifest.json" \
      --device cuda:0 \
      --isaaclab-source-root "$PHI_ISAACLAB_ROOT/.native/IsaacLab" \
      --writable-root "$ACTION_BRIDGE_ROOT/workspace" \
      --portable-root "$PHI_ISAACLAB_ROOT/.native/kit/eval-$STAMP" \
      --run-dir "$NATIVE_RUN" \
      --episodes 5 \
      --num-envs 5 \
      --max-steps 250 \
      --seed 1000000 \
      --no-require-success \
      --json
)
```

`--no-require-success` lets an undertrained smoke finish with an honest success
rate; checkpoint, policy, simulator, artifact, and cleanup errors still fail.
For the held-out experiment, use a new timestamped directory, 100 episodes,
and the validated vector-environment count.

The CLI reads the supplied immutable manifest before native launch, verifies
its exact SHA-256 against checkpoint metadata, and carries the shared bounded
cache identity into evaluation provenance. It never scans or rewrites the
collection.

The adapter keeps histories and latent state per lane on the GPU. It clamps
XYZ to the checkpoint-declared workspace, normalizes XYZW quaternions and
applies the profile-v2 largest-absolute-component sign rule, thresholds gripper
output to exactly `-1` or `+1`, and then hands the action to the backend's
strict validator. Policy and simulator devices must match; silent CUDA-to-CPU
fallback is rejected.

## 8. Record video

Use a separate new run, enable `--record-video`, and normally reduce
`--num-envs` to one. The current evaluator records the first vector lane into
one `videos/rollout.mp4`:

```bash
export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export VIDEO_RUN="$ISAAC_RUNS_ROOT/$RUN_ID/eval/video-seed1000100-$STAMP"
test ! -e "$VIDEO_RUN"

(
  cd "$PHI_ISAACLAB_ROOT"
  scripts/with_native_env.sh -- \
    python -m action_bridge.eval.isaaclab_online \
      --checkpoint "$CHECKPOINT" \
      --trusted-checkpoint \
      --collection-manifest "$COLLECTION/manifest.json" \
      --device cuda:0 \
      --isaaclab-source-root "$PHI_ISAACLAB_ROOT/.native/IsaacLab" \
      --writable-root "$ACTION_BRIDGE_ROOT/workspace" \
      --portable-root "$PHI_ISAACLAB_ROOT/.native/kit/video-$STAMP" \
      --run-dir "$VIDEO_RUN" \
      --episodes 1 \
      --num-envs 1 \
      --max-steps 250 \
      --seed 1000100 \
      --record-video \
      --video-fps 50 \
      --no-require-success \
      --json
)
```

Rendering is host-specific. A passing physics doctor does not establish camera
or video support; run the doctor's `--render-test` first.

## Artifacts

Training follows the usual Action Bridge layout:

```text
workspace/experiments/isaaclab/<run-id>/
├── config.json
├── checkpoints/{best.pt,latest.pt}
├── metrics/
└── eval/<evaluation-id>/
```

Each native evaluation directory contains resolved config, bounded shared
provenance, summary, episode and exception JSONL, and optional video. The
shared provenance binds the exact processed manifest and a canonical digest of
online preprocessing; native runtime and policy details live in
`resolved_config.json`. Isaac's current video is one batch-level
`videos/rollout.mp4`, so shared per-episode `video_path` values remain `null`
and each affected episode carries an artifact warning. Dataset/cache,
checkpoint, native Kit state, logs, and videos are experiment artifacts and
must not be committed to any source repository.

## Sharing and HPC

The parent checkout does not transport ignored `workspace/phi-isaaclab`
source. Before a shared run:

1. select a repository licence and publish the independent backend history;
2. pin the Action Bridge dependency to one full backend commit, not a branch or
   editable path, then regenerate and commit `uv.lock`;
3. transfer a current schema-2 processed collection separately, preserve its
   bytes, and run `phi-isaaclab validate-cache` on the destination;
4. build `.native/` on the destination rather than copying it;
5. have the operator review/accept NVIDIA's terms before scheduling native
   work, and rerun doctor on that host;
6. pre-stage the trusted checkpoint and record its SHA-256;
7. use job-specific writable Kit and run roots and keep source/data inputs
   read-only.

Record the Action Bridge/backend commits, lock digest, native distribution and
GPU/driver versions, collection manifest, checkpoint SHA-256, resolved config,
seeds, and exact doctor/evaluation results. An HPC CUDA allocation does not by
itself establish RTX rendering or video support.

## Current acceptance truth

Simulator-independent tests cover schema-2 conversion/windows, checkpoint
metadata, profile-v2 projection, device-native adapter behavior, and artifact
logic. On 2026-08-20, the pinned native stack passed physics and cleanup gates.
A separate RGB doctor also remains accepted for that audited host: it returned
one `[720,1280,3]` frame and cleaned up successfully.
A corrected reset probe then recorded two episodes in each of four lanes: all
8/8 O0 records exactly matched fresh live post-reset observations, and the
maximum O0-to-O1 TCP movement was 0.000222280 m.

The current reactive expert subsequently passed 256/256 first episodes across
four independent seeds 20000–20003 with 64 lanes per seed. Episode lengths were
128–176; every lane visited all four phases and toggled the gripper exactly
once. First and successive commands respected the declared 0.02 m/0.20 rad
caps within the 5e-5 native tolerance, and cleanup passed. The expert semantic
identity SHA-256 is
`59ecf807f4efbcf847bf79a059e2cbcc6cff34a575def2fa85cc081d766e43ee`;
the gate summary SHA-256 is
`0d46cf57f5353b9e4e4cbc00060e23e1208733c3136c2476a0eac008175b4376`.

Earlier collection/cache/training/evaluation and H.264 video runs proved the
infrastructure but used stale-reset, timed-expert, or profile-v1 semantics.
They and their derived checkpoints are quarantined diagnostics, not current
training evidence. A fresh schema-2 multi-shard corpus and conversion,
substantive training, current learned-policy video, and held-out success
evaluation remain pending. Exact paths and audit details are in
`workspace/phi-isaaclab/docs/milestones.md`.
