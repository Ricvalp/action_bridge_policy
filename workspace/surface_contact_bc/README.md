# Surface-contact BC toy

A point robot approaches a compliant line, makes contact, slides to a random
endpoint, and stops. This tests learned **command-space** reference dynamics,
not physical passivity. No rewards, penetration penalties, or contact costs
enter training. See [EXPERIMENT.md](EXPERIMENT.md) for the comparison logic.
See [VALIDATION.md](VALIDATION.md) for what has actually been tested: this is
an implemented experiment, not a completed scientific comparison.

Everything new is here. The models/losses import existing `action_bridge` code;
no MuJoCo, Isaac Lab, CoppeliaSim, or `phi-*` package is used.

## Setup

Run these commands from the **action_bridge_policy repository root**. Its
existing environment works if it already has PyTorch and the diffusion extra:

```bash
toy=workspace/surface_contact_bc
python="$PWD/.venv/bin/python"
```

Alternatively, create a small isolated CPU environment (does not change your
policy or simulator environments):

```bash
toy=workspace/surface_contact_bc
uv venv --python 3.11 "$toy/.venv"
python="$PWD/$toy/.venv/bin/python"
uv pip install --python "$python" torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$python" -r "$toy/requirements.txt"
```

CPU is the default, with two threads. Use `--device cuda` only on an available
GPU with an appropriate PyTorch installation. No renderer/GPU is needed for GIFs.

## Collect and inspect demonstrations

```bash
dataset="$toy/datasets/demo-$(date -u +%Y%m%dT%H%M%S%NZ)"
"$python" -m workspace.surface_contact_bc.data \
  --output "$dataset" --train-episodes 128 --val-episodes 16 --test-episodes 16

"$python" -m workspace.surface_contact_bc.visualize \
  --dataset "$dataset" --split train --episode 0 \
  --output "$toy/runs/expert-$(date -u +%Y%m%dT%H%M%S%NZ)" --gif
```

The collection contains `manifest.json` and `episodes.npz`. Splits are disjoint
episodes, not overlapping windows. Failed expert episodes are reported, never
silently discarded. Output directories must be new; nothing is appended or
overwritten. `--train-episodes 16` at training time takes the same first 16 demos
for every method and fits normalization on only those demos.

## Train

```bash
stamp=$(date -u +%Y%m%dT%H%M%S%NZ)
bridge="$toy/runs/bridge-$stamp"
diffusion="$toy/runs/diffusion-$stamp"

"$python" -m workspace.surface_contact_bc.train \
  --dataset "$dataset" --method action_bridge_contact_frame_full \
  --output "$bridge" --steps 10000 --batch-size 128 --horizon 8 --n-exec 2

"$python" -m workspace.surface_contact_bc.train \
  --dataset "$dataset" --method diffusion_world --output "$diffusion" \
  --steps 10000 --batch-size 128 --horizon 8 --n-exec 2 \
  --inference-candidates 10 20 50
```

These are starting configurations: ~303k bridge versus ~304k diffusion parameters
(within 1%), not a claim of optimal hyperparameters. Use validation to choose capacity and sampling steps for both
families. Training prints a progress bar/loss, writes `metrics.csv`, and saves
`best.pt`, `latest.pt`, resolved config and provenance. By default it evaluates
the selected checkpoint on eight ID episodes; `--skip-eval` separates the steps.

`best.pt` minimizes held-out **generated-chunk target MSE in m²**: previous
observations/actions come from held-out demonstrations, but the predicted chunk
uses no future expert actions. This is not teacher-forced MSE and not simulator
success. Both generators use whole-model EMA for selection/evaluation.
Diffusion also supports `--prediction-type sample` for standard clean-target
denoising instead of the default `epsilon`; select it on validation, not OOD.

Available methods, including the full model and its ablations:

| Method | Description |
| --- | --- |
| `action_bridge_contact_frame_full` | Full Action Bridge: learned normal potential, normal/tangential damping, residual control and path KL |
| `action_bridge_contact_frame_no_kl` | Zero path-KL coefficient |
| `action_bridge_isotropic` | One learned damping value instead of normal/tangential values |
| `diffusion_world` | Existing conditional U-Net, Diffusers DDIM, absolute targets |
| `diffusion_contact_frame` | Same diffusion model in invertible normal/tangent coordinates |
| `diffusion_residual` | DDIM controls decoded by the full bridge's frozen reference |
| `reference_only` | Evaluation of that same bridge with its residual disabled |

The last two reuse the **exact selected full-bridge reference**:

```bash
"$python" -m workspace.surface_contact_bc.train \
  --dataset "$dataset" --method diffusion_residual \
  --reference-checkpoint "$bridge/best.pt" \
  --output "$toy/runs/residual-$stamp" --steps 10000 \
  --inference-candidates 10 20 50

"$python" -m workspace.surface_contact_bc.evaluate \
  --checkpoint "$bridge/best.pt" --reference-only --episodes 8
```

Reference and residual diffusion must have matching seed, training subset,
normalization, histories and horizon. The residual checkpoint contains its
frozen reference, so later evaluation needs no second checkpoint.

## Evaluate and watch

```bash
"$python" -m workspace.surface_contact_bc.evaluate \
  --checkpoint "$bridge/best.pt" --episodes 16 --n-exec 2 --suites id --gif

# Run robustness tests AFTER checking convergence and ID rollouts for both models.
"$python" -m workspace.surface_contact_bc.evaluate \
  --checkpoint "$bridge/best.pt" --episodes 16 --suites all
```

Repeat with the diffusion checkpoint. `all` includes disjoint orientations,
hidden mass/stiffness/damping/friction/controller-gain shifts, and signed normal
impulses during sliding. Each impulse rollout is paired with an undisturbed
rollout using the same context and sampling seed. No best-of-N action selection.

To compare runs trained separately:

```bash
"$python" -m workspace.surface_contact_bc.plotting \
  --inputs "$bridge/eval/comparison.json" "$diffusion/eval/comparison.json" \
  --output "$toy/runs/comparison-$stamp"
```

Each evaluation saves per-episode metrics, complete traces, summary JSON, task/
contact robustness figures, and a representative trajectory. Bridge runs also
show learned damping, normal/tangential reference/residual forces, and a
reference-only overlay. `--gif` writes a CPU-rendered animation. Missing suites
are labeled “Not evaluated,” not presented as results.

`--n-exec` controls executed targets per replan, independently of the trained
`--horizon`. Histories always contain actual simulator observations and previously
executed targets. Internal reference positions are never substituted for physical
state. After copying a checkpoint, `--dataset NEW_PATH` overrides its dataset path.

## Matched sweep and tests

```bash
"$python" -m workspace.surface_contact_bc.sweep \
  --dataset "$dataset" --train-sizes 16 64 128 --seeds 0 1 2 \
  --steps 10000 --eval-episodes 16

"$python" -m pytest -q "$toy/tests" --basetemp "$toy/runs/pytest"

# Includes the two longer CPU tiny-data overfit checks (~one minute here).
CONTACT_BC_SLOW_TESTS=1 "$python" -m pytest -q "$toy/tests" \
  --basetemp "$toy/runs/pytest-overfit"
```

The sweep runs all seven comparisons and writes `comparison.json`, selection
records and cross-seed figures. This is a research run, not a quick smoke test.
`--methods`, `--suites` and `--train-sizes` reduce the matrix.
An optional `--grid file.json` maps each method to a list of config overrides;
the sweep chooses the candidate by validation MSE **before** held-out evaluation.
Give competing methods comparable tuning budgets and report all parameter counts.
The residual-diffusion comparison has additional frozen-reference parameters
and pretraining cost; both are reported separately.

Code map: `env.py` physics/expert; `data.py` episodes/windows;
`reference.py` geometry; `policies.py` existing-model adapters;
`train.py` BC; `evaluation.py` rollout/metrics; `plotting.py` figures.
