# Whole-plan revision on Push-T

Implementation of [SB-PUSHT.md](../SB-PUSHT.md), separate from the existing
Action Bridge loss and training pipeline. It shares the existing diffusion
policy's temporal U-Net; simulator physics and old checkpoints are unchanged.

## Run

Use the policy project's environment, not a shared simulator installation:

```bash
cd /home/rvalperga/action_bridge_policy
uv sync --frozen --extra cu128 --extra diffusion --extra pusht-sim --extra test
```

The initial audit found **no real Push-T replay dataset or compatible trained
Push-T DDIM checkpoint on this workstation**. Supply the dataset used by your
existing Push-T experiments; do not use test fixtures. The existing `.zarr`,
`.npz` and `.pt` formats are supported. Standard replay zarr reads `data/state`
(5D), not keypoints.

```bash
read -rp 'Absolute path to the Push-T replay dataset: ' PUSHT_DATASET
test -e "$PUSHT_DATASET"
run_root="$PWD/workspace/sb_pusht/revision-v1"

uv run --frozen --no-sync python -m action_bridge.scripts.sb_pusht all \
  --dataset "$PUSHT_DATASET" --run-root "$run_root" --device cuda
```

This is a **long training run**, not a smoke test. Stages are:

1. `prepare`: episode splits, train-only normalization, causal full-horizon windows.
2. `reference`: 20k updates for the standalone dissipative reference.
3. `ddim`: 300k updates; its selected frozen EMA supplies earlier plan proposals.
4. `sources`: two proposals per earlier logged history; frozen reference priors.
5. `fm_paired`, `fm_local_ot`, `sb_ou`, `sb_kinetic`: 300k updates each.
6. `evaluate`: five main results plus two kinetic completion-mode evaluations.
7. `report`: `REPORT.md`, `results.csv`, and `success.png` when results exist.

Run any individual stage with the same arguments once its dependencies exist:

```bash
uv run --frozen --no-sync python -m action_bridge.scripts.sb_pusht sb_kinetic \
  --dataset "$PUSHT_DATASET" --run-root "$run_root" --device cuda

uv run --frozen --no-sync python -m action_bridge.scripts.sb_pusht evaluate \
  --run-root "$run_root" --device cuda
```

Interrupted training resumes from `latest.pt`; completed jobs are skipped.
Reference and DDIM jobs can run independently. Revision jobs can run independently
after `sources`. Do not run two writers for the same stage/directory. Use a new
root when changing an experiment. Nothing is submitted to Slurm automatically.

`--config file.json` overrides the flat settings in
[`configs/sb_pusht.py`](../action_bridge/configs/sb_pusht.py). DSBM defaults to four
rounds of 37,500 reverse plus 37,500 forward updates. Keep
`updates == 2 * rounds * phase_updates` if changing this. Reduced budgets are
software checks, not the requested substantive experiment.

## Methods and shared data

| Method | Initial chunk | Learned process | Network calls |
| --- | --- | --- | ---: |
| `ddim` | Gaussian noise | Diffusers DDIM, 100 cosine levels, eta=0 | 32 |
| `fm_paired` | Aligned/completed previous plan | Linear-pair flow matching | 32 (16 midpoint steps) |
| `fm_local_ot` | Same | Observation-local, discrete OT pair flow matching | 32 |
| `sb_ou` | Same | Alternating DSBM with affine OU reference | 32 |
| `sb_kinetic` | Same plus fresh revision velocity | Alternating DSBM with kinetic reference | 32 |

All use two observations, two **executed** commands, H=16 and K=8. Actions are
mean/std-normalized absolute pusher targets, not physical pusher positions.
Only commands sent to the simulator are bounded; no intermediate clipping.
DDIM uses `clip_sample=False` and random initial Gaussian noise.

An old plan is predicted at logged history `t-K`, then aligned/completed at
logged history `t`. No simulation changes the recorded history and no expert
overlap is used as the old prediction. Training uniformly samples repeat,
fixed-damped and learned-dissipative completion, with mode as an input. All
three use the same underlying proposal cache; evaluation holds mode fixed.

Each reviser executes the same seeded DDIM bootstrap prefix, then uses **its own**
previous predictions. There is no repeated proposal query except explicit K=H
bootstrap. Predicted plans and actual executed history are separate. The resulting
training/deployment source-distribution shift is intentional and is not hidden
by relabeling or online data collection.

Source perturbation is std=0.01 in normalized coordinates during training and
deployment. Teacher dequantization is std=0.001 at training only, including DDIM.
Pixel equivalents are saved. Local OT uses observed geometry/command histories,
train-calibrated neighbourhoods, hard compatibility masks and discrete pairs.
Targets remain uniformly sampled original records. It is **approximate local
OT**, not exact conditional OT; marginal errors and context displacement are logged.

## Bridge training and outputs

The small frozen reference predicts robot-index attractor, stiffness and damping
from observed history and index. Train residual variances, precision scaling and
explicit spectral limits define its whole-plan quadratic prior. `robot_dt=1` and
revision time tau in [0,1] are different clocks; temperature=.05 is not physical
temperature. Default mobility is identity; banded SPD mobility is configurable.

OU/kinetic transitions, bridges and scores use exact affine-Gaussian calculations;
identity mobility uses scalar/2x2 modal algebra. Kinetic instantaneous control/noise
enters velocity only. Finite-time noise correctly correlates position and velocity.
Endpoint revision velocities are independent Gaussians and are discarded after
sampling. Reverse time does not flip velocity parity.

Training follows reverse fit → reverse rollout from real targets → forward fit
→ forward rollout from real sources, following
[DSBM Algorithm 1](https://arxiv.org/html/2303.16852v3#S4) and checked against the
[official implementation](https://github.com/yuyang-shi/dsbm-pytorch).
Opposite-direction EMA snapshots include encoders; detached coupling caches
refresh during each phase. Sampling is stochastic, with exact linear steps
holding control constant on cosine-grid intervals. Matching samples tau in
[.01,.99], with positive endpoint-distance weights (cubed for kinetic).
Sampler endpoint calls use the nearest trained time. There is no extra latent
KL, control penalty, candidate selection or deterministic SB shortcut.

Each method saves:

- `latest.pt`: models, optimizers, EMA, RNG, frozen dependencies, normalizers,
  scheduler/reference specifications and full directional/cache resume state.
- `best.pt`: highest success on five fixed validation seeds, every 75k updates
  (after complete forward phases for DSBM), never selected on held-out test seeds.
- `train.jsonl`, `validation.jsonl`: losses, timing and pairing diagnostics.

These are trusted local Torch artifacts; do not load untrusted checkpoints.
The final panel uses 50 separate shared seeds and one rollout per seed. Metrics
include success, true max/final coverage, revision size, command smoothness,
clipping, NFE, latency and control energy. The inherited success rule is retained;
`env_success_rate` additionally gives native environment success. Control energy
is a diagnostic, not a certificate of exact path KL or optimal coupling.

Evaluation saves the first two successes/failures as annotated GIFs, episode
traces and a few internal candidate plans: gray old overlap, orange completion,
purple revised plan, white actual pusher trail. Candidate plans are not physical
trajectories. Fixed held-out probes compare before/after overlap and tail error,
using low/high source-error thresholds fitted on validation records.

`audit.json` records checkout/lock, discovered code/checkpoints and dataset location.
`manifest.json` records jobs and splits. Reference training, source generation and
directional cache time are extra compute, not free parts of the update comparison.
One training seed cannot establish seed robustness or exact SB optimality.

## Another dataset

Algorithms are in `action_bridge/plan_revision/`, without simulator imports.
Only `data/revision_pusht.py` and `eval/revision_pusht.py` know Push-T specifics.
Supply tensor records with `obs_hist` (tensor or mapping), `act_hist`,
`future_actions`, `valid_mask` and causal earlier histories for source generation.
Keep episode/time/split IDs as metadata, not learned inputs. Partial horizons
currently fail explicitly; masking a Gaussian endpoint is not implemented.

Provide a codec specifying dimensions, units, normalization and bounds. The
included completers support absolute Euclidean targets only. Deltas, torques,
rotations and categorical actions need their own codec/completion semantics.

`build_policy(config, encoder=...)` accepts a replaceable history encoder returning
`[batch, history_dim]`. Custom encoder architecture metadata belongs in config,
and the encoder must be supplied on reload; the built-in flat encoder rebuilds
entirely from checkpoint metadata. Pass `encoder=` to the generic `train` function
and `completion_encoder=` for an injected reference encoder when needed.
`LearnedCompletion(..., encoder=...)` likewise
accepts a standalone reference encoder. Inject your context-distance/pairing
callback and simulator validation callback into the generic trainer. Each
environment gets its own `PlanCache`.

Tests include H=9, action_dim=6, different histories and mapping observations:

```bash
uv run --frozen --no-sync pytest -q tests/test_revision_*.py \
  tests/test_diffusion_policy.py tests/test_pusht_training.py \
  --basetemp workspace/test-artifacts/sb-pusht
```

Code/tests or untrained simulator-smoke videos do **not** mean the five training
runs completed. Consult the generated report for what actually ran. No policy
performance comparison is claimed until the real jobs finish.
