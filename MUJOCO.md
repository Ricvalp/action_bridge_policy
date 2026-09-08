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

The task configs use the existing Action Bridge model. Their default is the
no-latent reference/controller variant, with width 256. They are starting
settings, not tuned contact-task baselines.

Training preserves the official 180 train / 20 validation episodes. It fits
normalization only on those 180 training episodes. There is no offline test
split, so final offline metrics are labelled validation metrics. By default,
periodic validation and final offline evaluation use the complete validation
loader. For quicker smoke tests, `logging.validation_max_batches` and
`eval.offline_max_batches` cap them respectively; zero means all batches.

Runs are written under `workspace/experiments/mujoco/$run_id/` with config,
source/lock provenance, training and validation metrics, and
`checkpoints/best.pt` plus `checkpoints/latest.pt`. Checkpoints contain the
integration specification, dataset hashes and splits, normalization, histories,
and action horizon. `best.pt` means lowest validation loss, not best simulator
success rate.

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
