# MuJoCo datasets and Action Bridge

Use `phi-mujoco` to download/convert demonstrations and run the simulator.
Use this project to train Action Bridge and load its checkpoints.

The backend dependency is the standalone sibling checkout `../phi-mujoco`
(/home/rvalperga/phi-mujoco here), installed editable. The tested backend is
version 0.2.0, commit `528bdf97471493a9d63da424872141ff1279be4b`.

## 1. Prepare the backend and data

Follow the standalone [phi-mujoco README](../phi-mujoco/README.md) for the
Robomimic profile and the Square/Tool Hang download and conversion commands.
This can be done independently of the Action Bridge environment.

Each processed directory contains `manifest.json` and `episodes.hdf5`.
Use that directory as `data.cache_root` below; do not pass the original
`low_dim_v15.hdf5` file to training. Data may live anywhere: an absolute cache
path is enough. Neither repository needs to copy the other's source code.

## 2. Prepare the policy environment

From this repository:

```bash
cd /home/rvalperga/action_bridge_policy
uv sync --locked --extra cu128 --extra robomimic --extra test

uv run --frozen --extra cu128 --extra robomimic python -c \
  'import phi_mujoco; print(phi_mujoco.__file__)'
```

The printed path should be under /home/rvalperga/phi-mujoco. Here `cu128`
selects CUDA PyTorch and `robomimic` adds the pinned simulator and video
dependencies to the policy environment. Use `cpu` instead of `cu128` for a CPU
smoke test. Always select the same Torch extra in subsequent `uv run` commands.
Offline training does not start or render a simulator.

The other backend checkouts referenced in `pyproject.toml` are still required
by the project environment; this change only replaces the MuJoCo source path.
For a shared/HPC experiment, pin the tested backend commit in the source
dependency and regenerate the lock. An editable path is convenient locally,
but does not freeze the source checkout.

## 3. Train

For the learned-potential/damping Action Bridge, use the
[dissipative Square configs](#dissipative-action-bridge-square) below.
The original `mujoco_robomimic_square` and `mujoco_robomimic_tool_hang`
configs described here use the **older continuation reference**.

Choose a task and its converted cache:

| Config | State | Action | Default horizon / executed actions |
| --- | ---: | ---: | ---: |
| `mujoco_robomimic_square` | 23 | 7 | 8 / 4 |
| `mujoco_robomimic_tool_hang` | 53 | 7 | 8 / 4 |

Both use state and action histories of length two. The state already includes
end-effector pose, gripper position, and task objects; these are state-based
experiments with no point clouds. Actions are Robosuite's normalized
pose-delta/gripper commands. The backend defines their exact meaning.

```bash
config=mujoco_robomimic_square
cache="/home/rvalperga/phi-mujoco/datasets/processed/robomimic_square-<timestamp>"
run_id="square-bridge-$(date -u +%Y%m%dT%H%M%S%NZ)"

uv run --frozen --extra cu128 --extra robomimic \
  python -m action_bridge.training.train_mujoco \
  --config-name "$config" \
  data.cache_root="$cache" \
  run_id="$run_id" \
  device=cuda \
  optim.max_steps=20000 \
  optim.batch_size=256
```

Replace `<timestamp>` with the actual directory created by conversion. For
Tool Hang, change both the config and cache. The 20,000-step command is a
starting experiment, not a validated training budget or performance claim.

First check the connection with a short run: use `cpu`/`device=cpu`, a new
`run_id`, and add these overrides to the same command:

```text
optim.max_steps=2 optim.batch_size=8
logging.eval_every_steps=1 logging.validation_max_batches=1
eval.batch_size=8 eval.offline_max_batches=1
```

Useful training overrides:

```text
chunk_horizon=16                   # number of predicted actions
eval.actions_per_plan=4            # default execution count stored in checkpoint
obs_history=2 action_history=2     # action_history must be at least two
model.latent_type=continuous       # optional latent; default is none
logging.wandb.enabled=true         # optional loss logging, off by default
logging.progress=false            # disable the training progress bar
```

These older task configs default to a no-latent continuation-reference model
with width 256. They are starting
settings, not tuned contact-task baselines.

For a roughly 5-million-parameter **continuous-latent** Square model, add:

```text
model.latent_type=continuous model.hidden_dim=736 model.h_emb_dim=736
chunk_horizon=8 eval.actions_per_plan=4
```

This has 5,069,631 total parameters, including the training-only posterior;
3,395,223 are in the inference path. The trainer prints the parameter count.
The continuous latent remains four-dimensional. A larger model is an
experiment, not a guarantee of higher task success.

Before training, Action Bridge copies the backend's normalized windows into
CPU RAM, with a short loading progress bar. Training and validation then read
these arrays, not HDF5. Square at horizon eight needs about 13 MiB of window
arrays across both splits; the trainer prints the actual size. This adds a
one-time startup cost but avoids rereading whole episodes every minibatch.
No reconversion, extra cache files, or backend changes are needed. This simple
in-memory path is for small state datasets, not large image/point-cloud data.

Training preserves the official 180 train / 20 validation episodes. It fits
normalization only on those 180 training episodes. There is no offline test
split, so final offline metrics are labelled validation metrics. By default,
periodic validation and final offline evaluation use the complete validation
loader. Validation generates the full action chunk autoregressively from the
history-conditioned prior (by default its deterministic mean), feeding back
its own predicted actions. It compares these with held-out expert actions in
the original action coordinates, before clipping. It never uses the posterior
or future expert actions as prediction inputs. For quicker smoke tests,
`logging.validation_max_batches` and
`eval.offline_max_batches` cap them respectively; zero means all batches.

Runs are written under `workspace/experiments/mujoco/$run_id/` with config,
source/lock provenance, training and validation metrics, and
`checkpoints/best.pt` plus `checkpoints/latest.pt`. Checkpoints contain the
integration specification, dataset hashes and splits, normalization, histories,
and action horizon. `best.pt` means lowest **validation action MSE** across the
full predicted horizon, not lowest training loss or highest simulator success.
`latest.pt` is always the most recently saved model. Validation scores are in
`metrics/val_metrics.csv` (`action_mse`) and, when enabled, W&B `val/action_mse`.
The final offline report uses the separate W&B prefix `offline_val` (or
`offline_test` for datasets with a test split).
There is no step-zero validation-loss calculation in this trainer anymore.
New checkpoints record `checkpoint_metric=val_action_mse`; start a fresh run
instead of resuming a checkpoint selected using the old loss metric.

## 4. Evaluate in simulation

From the Action Bridge project and its environment:

```bash
checkpoint="$PWD/workspace/experiments/mujoco/$run_id/checkpoints/best.pt"
eval_run="$PWD/workspace/experiments/mujoco/eval-$(date -u +%Y%m%dT%H%M%S%NZ)"

MUJOCO_GL=egl uv run --frozen --extra cu128 --extra robomimic \
  python -m action_bridge.eval.mujoco_online \
  --checkpoint "$checkpoint" \
  --trusted-checkpoint \
  --device cuda \
  --run-dir "$eval_run" \
  --episodes 10 \
  --seed 1000000 \
  --actions-per-plan 4 \
  --record-video \
  --progress \
  --json
```

The task is read from the checkpoint. Defaults are 400 steps for Square and
700 for Tool Hang; `--max-steps` overrides the limit. Omit `--record-video`
and use `MUJOCO_GL=disable` for evaluation without rendering. Each rollout,
including a failed one, gets its own MP4 when recording is enabled. An
unsuccessful policy is reported normally; `--require-success` optionally makes
any failed episode result in a nonzero exit code.

`--actions-per-plan` may range from one to the training horizon. The policy
adapter buffers the selected actions while receiving every simulator
observation, so its observation and executed-action histories stay aligned
with training. Actions are denormalized and projected through the backend's
public API before execution.

This connection uses the current backend API and new checkpoint metadata.
Old planar-reach caches/checkpoints need new conversion/training; there is no
compatibility mode. The three planar-reach config names still work with a
current processed planar cache and retain horizon four/execution one.

Earlier planar experiments reached 99/100 successes for direct chunk BC
(5,000 steps), 94/100 for no-latent Action Bridge (20,000 steps), and 18/100 for
continuous-latent Action Bridge (5,000 steps). Those are historical results
from the previous connection, not Robomimic results.

### Background success-rate evaluation

MuJoCo training can evaluate checkpoint snapshots in separate CPU processes
while GPU training continues. No RLlib or additional dependencies are needed.
Add these overrides to a training command:

```text
logging.sim_eval_enabled=true
logging.sim_eval_every_steps=2000
logging.sim_eval_episodes=40
logging.sim_eval_num_workers=8
logging.sim_eval_worker_threads=1
logging.sim_eval_n_exec=8
```

Each worker owns its simulator and CPU policy copy. The same 40 seeds starting
at `logging.sim_eval_seed=2000000` are used for every snapshot. Keep the final
evaluation seeds (default `1000000`) separate from this checkpoint-selection
set. Set `sim_eval_n_exec` explicitly when comparing execution horizons; it
otherwise follows `eval.actions_per_plan`.

Only one evaluation batch runs at a time. If it is still busy at the next
interval, that interval is skipped. Training polls results without waiting;
after the last optimizer step it waits for outstanding work and evaluates the
final checkpoint too. Checkpoint saving itself still has a small cost.

When W&B is enabled, `sim_eval/success_rate` is plotted against
`sim_eval/checkpoint_step`, not the training step when results arrive. Local
results are in `metrics/sim_eval_metrics.csv` and `eval/sim_step_*/summary.json`.
`checkpoints/best_success.pt` stores the exact snapshot with highest measured
success rate (ties keep the earlier snapshot). `best.pt` still means best
offline validation MSE. Temporary evaluation snapshots are removed after use;
set `logging.sim_eval_keep_checkpoints=true` to retain them. Simulator errors
are reported separately, never silently scored as failed episodes.

Videos are optional:

```text
logging.sim_eval_success_videos=3
logging.sim_eval_failure_videos=3
logging.sim_eval_video_backend=egl
```

Scoring runs without rendering. Afterwards, up to three successful and three
failed seeds are rerun with rendering; only the selected clips are uploaded to
W&B. Rerun outcomes are checked and labelled as actually observed, without
changing the original success rate. Zero successes means no successful videos.
Rendering errors do not discard the success-rate result. EGL uses the allocated
GPU; `osmesa` uses software rendering if its system libraries are installed.
Leave both quotas at zero to avoid rendering entirely.

## 5. Run on the HPC

Keep the two clones as siblings: `action_bridge_policy/` and `phi-mujoco/`.
The backend must be clean at commit
`528bdf97471493a9d63da424872141ff1279be4b`. Copy only `manifest.json` and
`episodes.hdf5` into:

```text
phi-mujoco/datasets/processed/robomimic_square-20260908T122032Z/
```

No separate index, raw dataset, simulator installation, or virtual environment
needs transferring. If direct workstation-to-Peano SSH is unavailable, transfer
the cache via the laptop. Commit/push policy changes and pull them on Peano.

From the policy clone on Peano's login node, prepare the environment once:

```bash
export UV_CACHE_DIR="$PWD/workspace/.uv-cache"
uv sync --frozen --python 3.11 --extra cu128 --extra robomimic \
  --no-install-package phi-isaaclab \
  --no-install-package phi-coppeliasim

.venv/bin/phi-mujoco validate-cache \
  "$PWD/../phi-mujoco/datasets/processed/robomimic_square-20260908T122032Z"
.venv/bin/wandb login
mkdir -p hpc/logs
```

The two exclusions skip unused backends without changing `pyproject.toml` or
`uv.lock`. The jobs use the prepared `.venv`; no package installation happens
on compute nodes. Existing training environments need no new dependencies.

First test EGL **inside a GPU allocation**, not on the login node:

```bash
sbatch hpc/mujoco_egl_smoke.sbatch
```

Check `hpc/logs/mujoco_egl_smoke_<job-id>.out` and `.err`. A passing test prints
`PASS: headless EGL rendering and MP4 encoding/decoding succeeded.` It also
saves a frame and a short Square video under
`workspace/experiments/mujoco/egl-smoke-<job-id>/`. This test uses the same
Robosuite rendering path as evaluation. If Slurm GPU indices are remapped or
UUID-based, it stops rather than guessing a device; share both logs so we can
check the cluster's EGL mapping.

To reproduce the older **continuation-reference** latent baseline, submit:

```bash
sbatch hpc/mujoco_robomimic_square_5m_h200_1gpu.sbatch
```

This requests **one H200, 16 CPUs, 32 GiB RAM, and 10 hours**, using the existing
`gpuq`/`gpu:1` allocation and verifying the GPU model after allocation. It trains
the 5.07M-parameter latent policy for 100,000 steps, horizon eight/execution four,
and enables W&B, eight CPU evaluation workers, 40 episodes every 5,000 steps,
and up to three successful/three failed videos. These workers share the same
Slurm allocation; no additional jobs are submitted.

If EGL is not ready, keep success-rate evaluation but disable videos:

```bash
SIM_EVAL_VIDEOS=0 sbatch hpc/mujoco_robomimic_square_5m_h200_1gpu.sbatch
```

Logs are `hpc/logs/square_latent_5m_<job-id>.out` and `.err`. Runs are under
`workspace/experiments/mujoco/square-latent-5m-h8-exec4-<job-id>-<timestamp>/`.
Use `squeue -u "$USER"` for status. Changing the sbatch file does not change
an already-running job or its CPU allocation.

### Dissipative Action Bridge: Square

These configs define their model, reference, and loss explicitly; neither
imports a Push-T config. They reuse the existing MuJoCo data and runtime setup.

| Config | Reference training |
| --- | --- |
| `mujoco_robomimic_square_dissipative` | Potential, damping, and control learn jointly from demonstrations. |
| `mujoco_robomimic_square_dissipative_stopgrad` | Control fits demonstrations against a detached EMA reference; the live reference fits passive targets separately. |

Both learn a quadratic potential (attractor and diagonal stiffness) and scalar
damping, with reference acceleration `-K * (q - m) - gamma * p`. The controlled
path adds `sigma * u`, without the old `tanh` bound. Its path-KL/control-energy
penalty is `0.5 * sum(dt * ||u||²)`, weighted by **`reference.beta_kl`** (default
0.001), not `loss.beta_R`.

Here `q` is the standardized seven-dimensional controller command and `p` is
its finite difference. These are **not physical end-effector poses/momenta**:
XYZ delta commands, axis-angle delta commands, and gripper commands retain the
dataset's original meaning. We use `sigma=0.5` in standardized command space.

The stop-gradient variant's `passive_target=damped_continuation` projects each
demonstrated command change onto a damped version of the previous change
(`passive_alpha_max=0.4`). This is a training target, **not** the old continuation
reference. Its reference has EMA decay 0.995, target weight 0.5, slow-change
weight 0.01, and dissipation weight 0.0001. Task fitting does not backpropagate
through that reference; the history encoder remains shared. Inference uses the
EMA reference too, and checkpoints preserve both live and EMA weights.

Both defaults have **5,148,502 trainable parameters**, no latent, horizon 8,
execution 4, batch size 256, AdamW learning rate 0.0002, and 100,000 steps.
Both use unroll-MSE weight 1 with a 1,000-step warmup; the stop-gradient version
blocks unroll gradients through the reference. These are starting settings,
not tuned results. Optional `model.latent_type=continuous` enables a
four-dimensional latent and increases the parameter count.

Train locally with either config name:

```bash
config=mujoco_robomimic_square_dissipative
run_id="${config}-$(date -u +%Y%m%dT%H%M%S%NZ)"
uv run --frozen --no-sync --extra cu128 --extra robomimic \
  python -m action_bridge.training.train_mujoco \
  --config-name "$config" \
  data.cache_root="$PWD/../phi-mujoco/datasets/processed/robomimic_square-20260908T122032Z" \
  run_id="$run_id" \
  logging.wandb.enabled=true \
  logging.sim_eval_enabled=true
```

On Peano, after the environment setup above, submit either or both:

```bash
mkdir -p hpc/logs
sbatch hpc/mujoco_robomimic_square_dissipative_h200_1gpu.sbatch joint
sbatch hpc/mujoco_robomimic_square_dissipative_h200_1gpu.sbatch stopgrad
```

Each job requests one H200 for 10 hours, enables W&B in `action-bridge-policy`,
and scores 40 episodes on eight CPU workers every 5,000 steps (busy intervals
are skipped). Videos default to off until EGL is verified. No new dependencies,
backend changes, or data conversion are needed; start fresh runs, not resumes
of continuation checkpoints. Evaluation uses the same command in section 4.

W&B logs the task losses, path KL, unroll MSE, damping/stiffness statistics,
validation action MSE, and background success rate; the stop-gradient variant
also logs its passive-reference losses. Use `best_success.pt` for the best
evaluated success rate; `best.pt` still selects validation action MSE.

### DDIM diffusion baseline

`mujoco_robomimic_square_diffusion` replaces Action Bridge with a small
conditional temporal UNet with 5,258,455 parameters. It uses the same processed
cache, train/validation split, observation/action histories, training loop, and
closed-loop evaluator.
No backend changes or data reconversion are needed. The trainer prints the
exact parameter count.

The model learns to predict added Gaussian noise with MSE. Sampling uses
`diffusers` DDIM with 100 training noise levels, 20 denoising steps, and
`eta=0`; it predicts the whole eight-action chunk and executes four actions
before replanning. `train/loss` is **noise MSE**, not Action Bridge's training
loss or action MSE. Compare policies using `val/action_mse` and, especially,
`sim_eval/success_rate`. Validation samples chunks with DDIM rather than
teacher forcing or a future-conditioned latent.

Actions retain the same train-fitted mean/std normalization. DDIM does **not**
clip standardized actions to `[-1, 1]`; physical action limits are applied
after denormalization by the existing simulator adapter.

The scheduler uses DDIM's default `leading` timestep spacing. With 100 noise
levels and 20 sampling steps, it starts at level 95, avoiding the near-zero
signal level 99 where small noise-prediction errors are greatly amplified.
Training still samples all 100 levels. We do not add EMA or change optimizer
settings for this first controlled comparison.

Fresh runs use a separate seeded batch-shuffling generator, so model
initialization and diffusion noise do not affect sample order. Validation uses
its own fixed sampling seed (`eval.sampling_seed=0`). This does not change an
already-running Action Bridge job or make resumed runs bitwise identical.

On Peano's login node, update the policy environment with the pinned optional
dependency (`diffusers==0.39.0`), then submit:

```bash
uv sync --frozen --python 3.11 --extra cu128 --extra robomimic --extra diffusion \
  --no-install-package phi-isaaclab \
  --no-install-package phi-coppeliasim

.venv/bin/wandb login
mkdir -p hpc/logs
sbatch hpc/mujoco_robomimic_square_diffusion_h200_1gpu.sbatch
```

The job requests one H200, 16 CPUs, 32 GiB RAM, and 10 hours. It trains for
100,000 steps with batch size 256, and evaluates 40 episodes with eight CPU
workers every 5,000 training steps. W&B is enabled. Videos default to **zero**
while EGL is being checked on Peano; after a passing smoke test, submit with
`SIM_EVAL_VIDEOS=3` to request up to three clips per outcome.
These settings apply to new submissions, not to already-running jobs.

DDIM requires multiple UNet passes per prediction, so CPU evaluations may take
longer than Action Bridge evaluations. Busy evaluation intervals are skipped
as described above; there is no growing queue of checkpoints.
Full offline validation also requires DDIM sampling and remains synchronous.

Scheduler settings are configurable in the training command:

```text
model.num_train_timesteps=100
model.num_inference_steps=20
```

Runs are saved under
`workspace/experiments/mujoco/square-diffusion-h8-exec4-<job-id>-<timestamp>/`;
Slurm logs use `hpc/logs/square_diffusion_<job-id>.out` and `.err`.
`best.pt` still selects validation action MSE; `best_success.pt` selects
closed-loop success. To evaluate a diffusion checkpoint, use the command in
section 4, adding `--extra diffusion --no-sync` to `uv run` so the prepared
environment, including `diffusers`, is retained.
