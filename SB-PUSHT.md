# SB-PUSHT: conditional Schrödinger-bridge revision of Push-T action chunks

## 0. Assignment and scope

Implement **one substantive, closed-loop, low-dimensional Push-T experiment** in the existing `action_bridge_policy` repository. Implement its policies and training algorithms as reusable, dataset-independent components; Push-T is the first configuration, not the boundary of their implementation. The mandatory portability contract is in Section 1.1.

Question: does revising the previous action chunk with a learned conditional Schrödinger bridge produce a useful imitation policy, and does the reference process improve the trade-off between preserving useful continuity and correcting a stale plan?

Start on actual Push-T. Do not require a successful synthetic toy, a low-data sweep, many training seeds, or an exhaustive ablation grid before running this experiment. Run one training seed per main configuration initially. Short mathematical/unit tests are implementation checks, not a preliminary research programme.

This is **offline imitation learning**. Expert chunks supply the terminal data distribution. Do not add RL, a critic, reward-weighted learning, simulator task-cost gradients, best-of-N action selection, or an inverse-RL objective. Environment reward is used only for validation/evaluation.

This brief specifies a new generative-time model. Older project notes saying “do not implement diffusion/Sinkhorn/IPF” describe the original path-KL pilot; they do not prohibit the new DSBM and OT baselines requested here. Preserve those existing models and their interfaces. Respect any applicable repository agent instructions.

The concrete first comparison has **five model configurations**:

| ID | Generator | Training correspondence/reference |
|---|---|---|
| `ddim` | Gaussian-to-chunk diffusion policy | Standard diffusion BC; DDIM inference through `diffusers` |
| `fm_paired` | Old-chunk-to-new-chunk ODE | Natural within-rollout/context pairing |
| `fm_local_ot` | Same ODE architecture | Observation-local OT re-pairing; explicitly an approximate conditional-OT baseline |
| `sb_ou` | Conditional DSBM | Tractable affine OU revision process induced by a damped action-plan prior |
| `sb_kinetic` | Conditional DSBM on augmented plan state | Underdamped, temporally coupled mechanical revision process with analytic Gaussian reference bridges |

The kinetic model is part of the first experiment, not postponed until a ladder of simpler experiments succeeds. Implement a Brownian special case of the reference API for tests and debugging, but do not add another full training run by default.

Implement three tail-completion choices. To avoid multiplying training runs, train each chunk-revision model on a balanced mixture of these choices, **with the completion identifier explicitly included in the conditioning**, then evaluate each identifier separately. This is a shared conditional model for three source distributions, not three separately optimized specialist policies. The primary comparison uses learned dissipative completion; the completion comparison reuses the same checkpoints.

## 1. Repository audit: inspect locally and reuse

### What was verified when writing this brief

The public default branch of `Ricvalp/action_bridge_policy` was inspected at commit:

`d8bdc2c357c1388690833a38917bd44a8cc94dc7`.

At that revision:

- `action_bridge/models/baselines.py` contains direct-chunk BC, autoregressive BC, and reference-only policies; it does not contain DDIM.
- Searches for `diffusers` and `DDIM` returned no matches. `pyproject.toml` has no `diffusers` dependency. These observations do not establish what is in a newer/local/unpushed checkout.
- Existing relevant paths include `action_bridge/data/pusht_adapter.py`, `data/chunking.py`, `data/action_coordinates.py`, `models/encoders.py`, `models/references.py`, `models/action_bridge_policy.py`, `training/train_pusht.py`, and `scripts/eval_pusht_sim.py`.
- There are already geometric Push-T helpers/configuration, including `models/geometric_pusht.py` and `configs/pusht_lowdim_geometric_reference.py`. Their existence does not mean they define the Gaussian whole-plan bridge needed below.
- Configuration uses Python `ml_collections` files and dotted overrides. The environment setup uses `uv`, with a `pusht-sim` extra including `gym-pusht` and a `pymunk` compatibility constraint.

### Required local audit

Before writing a new denoiser, search the entire checkout, including current worktree changes, for diffusion policies, `DDIMScheduler`, `DDPMScheduler`, `diffusers`, flow matching, EMA, checkpoint factories, and existing Push-T training/evaluation wrappers. Reuse a working implementation if present. Do not interpret an empty filename search alone as proof that none exists.

Record the actual checked-out commit, relevant reused paths, discovered checkpoint paths, dataset path, and missing components in an experiment manifest. Do not invent checkpoint names or absolute dataset paths. Reuse the current environment, observation extraction, success definition, normalizers, split logic, plotting, and checkpoint conventions where possible.

Do not vendor an external research repository wholesale, introduce containers, rewrite the RLBench/JAX branch, or modify Push-T physics. External code may inform a small implementation subject to its licence.

### 1.1 Mandatory reusable implementation: Push-T is a configuration, not the policy library

**Implement reusable policy and training components, with Push-T as their first dataset/environment adapter. This is a required deliverable, not a later refactor.** The user will add other datasets and benchmarks. All five generators must be reusable without copying their implementation into another task-specific script.

The numeric settings below (`obs_dim=5`, `action_dim=2`, `H=16`, `K=8`, pixel units, and the Push-T observation ordering) are **experiment configuration values only**. They must not be assumptions inside denoisers, flow networks, DSBM losses, bridge samplers, tail-completion algorithms, or the generic training loop. Changing compatible continuous-action datasets must require an adapter/configuration and possibly an observation encoder, not rewriting those algorithms.

This requirement concerns **code reuse**, not one checkpoint operating zero-shot across arbitrary dimensions, horizons, embodiments, or action semantics. Construct models from an explicit shape/action specification; it is acceptable for an instantiated model/checkpoint to have fixed dimensions and horizon. Mixed-embodiment batches and new benchmark runs are not required here.

#### Separation of responsibilities

Keep these boundaries clear, using existing modules where possible. These are conceptual interfaces, not a demand for a separate class or package for every row.

| Layer | Reusable implementation | Dataset/task-specific attachment or boundary |
|---|---|---|
| Dataset pipeline | Episode-safe windowing, expert chunks, masks, and provenance under one schema | The dataset adapter owns raw field names, file formats, observation ordering and padding conventions |
| Observation encoder | A documented conditioning interface for tensor/dictionary observations and executed-action history | A state-history encoder now; another encoder later, without changing matching losses; no expert future inputs |
| Action-coordinate adapter and normalizer | Explicit encoding/decoding, dimensions, normalization and capability checks | The adapter/configuration declares units, coordinate semantics, anchors and bounds; no universal position assumption |
| Source builder and tail completer | Alignment by actual execution count, then repeat/fixed-damp/learned-damp completion | Semantic information supplied by the action adapter; no hardcoded Push-T observation slices |
| Generator and algorithm | DDIM, FM, local-OT pairing, and directional DSBM fitting/sampling | Dataset loaders, validation callbacks and OT context metrics are injected; no simulator dependency |
| Reference backend | State lifting, drift/noise channels, transition moments, bridge samples and score targets | A plan-prior builder supplies coefficients and boundary data; no mandatory Push-T potential |
| Rollout and metrics | Plan-cache management and calls to the common chunk-policy interface | An environment adapter handles execution/reset, task metrics and rendering; no duplicated policy mathematics |

Avoid a large plugin framework or a wholesale repository redesign. Small modules, explicit configuration, and a few injected functions/modules are sufficient. Push-T-specific entry points may remain, but should assemble and call reusable components rather than contain their mathematics.

#### Common data and policy contracts

Extend the repository's existing batch schema rather than inventing a parallel Push-T-only schema. It should express:

```text
obs_hist          configured tensor or mapping of observation-history tensors
act_hist          [B, L_a, d_a] in explicitly specified model action coordinates
future_actions    [B, H, d_a] expert endpoint; training/evaluation-label access only
source_actions    [B, H, d_a] aligned/completed old plan, when required
valid_mask        [B, H], plus history masks where needed
context           observed task conditioning and completion identifier
metadata          dataset/split/episode/time/proposal provenance; not learned inputs
```

Names may follow existing conventions. Raw environment action dimensions can differ from `d_a` only through an explicit action-coordinate adapter; for Push-T both are two. Distinguish the raw stored sample, encoded model batch, and environment command. Do not normalize twice or silently mix coordinate conventions.

DDIM, FM and DSBM must expose a consistent chunk-generation boundary: observed conditioning, optional source chunk, sampling settings/RNG in; generated `[B,H,d_a]` chunk and optional diagnostics out. DDIM does not require a source. FM/DSBM must either receive a valid source or use the configured bootstrap outside the core sampler. Loss entry points may differ by algorithm, but must consume the same encoded-batch contract. Sampling must not require `future_actions`, a dataset object, or a simulator.

Keep stateful execution outside the mathematical generator: a reusable plan-cache wrapper owns the previous chunk, executed count, startup state, and per-environment reset. Changing `K` in configuration must change alignment/tail length without changing policy source code; it need not leave a checkpoint's statistical source distribution unchanged. Handle `K=H` explicitly with bootstrap/full completion rather than indexing a nonexistent retained tail. The first experiment remains `K=8`.

For this implementation, primary training may keep the existing full-valid-horizon restriction. Carry masks through the interface and reject unsupported partial-endpoint patterns explicitly. Merely masking a loss does not define a correct Gaussian bridge with partially constrained endpoints; do not claim such support without implementing its semantics.

#### Swappable conditioning, action semantics, and reference geometry

Use the existing state-history encoder for Push-T, but inject it through an encoder interface instead of flattening Push-T observations inside every algorithm. A later image/multimodal encoder should supply the same documented conditioning interface without changing diffusion/FM/DSBM losses or Gaussian bridge algebra. Do not implement a collection of visual backbones for this task. Rollout snapshots must include the relevant encoder weights as already required in Section 6.

Supply the local-OT context distance and compatibility mask as adapter/configuration-selected functions. The pusher/block/circular-angle distance in Section 5.3 is the **Push-T implementation** of those functions, not the generic definition of conditional OT. A new dataset without a valid context-distance definition must not silently fall back to arbitrary cross-context matching.

Likewise, a plan-prior builder supplies the quadratic coefficients and boundary data used by the Gaussian reference backend. The backend must operate at dimension `H*d_a` or `2*H*d_a`, with explicit state packing and noise-channel shapes. The DSBM training loop should dispatch through that reference interface rather than branch on a dataset name. Tail completion and revision-reference selection remain separate configuration choices.

Declare the supported action representation for each completer/prior. The primary implementation uses continuous Euclidean model coordinates with absolute-target semantics. Do not silently interpret repeated deltas as holding position, damping torques as dissipating robot kinetic energy, or arbitrary quaternion/gripper channels as Cartesian positions. A later dataset may use an identity continuous-vector codec or an explicit absolute/delta/pose codec; geometrically constrained or categorical channels require an appropriate adapter/model extension. Unsupported combinations must fail clearly. No new manifold or discrete-action SB implementation is required now.

Store separate robot-index `dt` and revision-time settings. Normalization, diffusion scale, prior gains, and action bounds belong to the saved representation/configuration. A new dataset's units must not inherit Push-T pixel constants.

#### Reusable training, caching, and checkpoint loading

The natural-pair/source-cache builder should operate on generic episode windows and a proposal-policy interface. The Push-T adapter supplies the current/earlier histories and labels. Include dataset/split IDs, action/observation schema, horizon/history lengths, normalizer, completion/reference versions and proposal hash in cache keys or manifests so incompatible datasets/checkpoints cannot be mixed accidentally.

Implement one reusable training path per algorithm, with dataset loaders and validation callbacks supplied from configuration. Do not duplicate the DSBM outer loop for each benchmark. Keep `gym-pusht`, `pymunk`, and rendering imports in the Push-T adapter/evaluator; importing and sampling a generic policy must not require those simulator packages. Do not hide DDIM scheduler/version settings inside a Push-T script.

Checkpoints must contain enough information to rebuild the encoder, generator, action codec/normalizer, completer, reference and sampler without importing a training dataset or simulator. Include shape/representation checks and explicit errors on incompatible inputs. Document how to add another dataset adapter and select each policy through configuration.

#### Lightweight portability acceptance check

In the existing short test suite, instantiate the same core policy/trainer implementations with both `(H=16,d_a=2)` and a non-Push-T continuous Euclidean specification such as `(H=9,d_a=6)`, with different observation dimensions/history lengths. Respect documented backbone horizon restrictions or use internal padding/cropping; do not turn a Push-T horizon into an implicit universal constraint. Check a finite loss/backward pass, seeded sampling/output shapes, generic alignment, reference-state packing, and checkpoint reload as applicable. Test that changing padded labels cannot change supported masked losses, or assert the explicit full-horizon rejection where masking is unsupported. Verify that policy imports do not require Push-T simulator dependencies.

These are interface/unit checks using synthetic tensors, **not new learning experiments or extra benchmark runs**. A dimension change in a configuration must not require edits to core source files. The scientific run manifest remains the five Push-T configurations and two completion evaluations below.

## 2. Fixed Push-T experimental contract (configuration only)

Use the existing low-dimensional Push-T task:

- Observation history: two frames; use the same actual observation vector for all five methods. The documented state representation has five values: pusher position, block position, and block angle. Verify the runtime representation.
- Executed-action history: two commands.
- Action: **absolute 2D pusher target position**, not force, physical pusher position, or a delta action.
- Prediction horizon `H = 16`; execute `K = 8` actions before replanning.
- Same train/validation episode split, train-only normalization, action bounds, simulator steps, termination, and rollout initial states for all methods.
- Use the full available training split. Do not run a demonstration-count sweep.
- Keep raw pixels and normalized coordinates distinguishable in every cache/plot. Disable DDIM's default `[-1,1]` sample clipping if using mean/std normalization rather than bounded normalization.

Name the two time axes explicitly:

- `k`: robot-time action index inside a chunk;
- `tau`: internal revision time from zero to one.

`Y_tau` has shape `[B,H,d_a]` (`d_a=2` in this Push-T configuration): each state of the SB/FM process is a **whole plan**. The old sequential `(q_k,p_k)` policy is not the new DSBM model.

All unexecuted predictions are editable. Remove already executed actions before alignment. There is no non-cancellable asynchronous action queue in this experiment. Do not hard-inpaint the overlap, blend old/new outputs after generation, or average stochastic candidates.

## 3. Conditional endpoint distributions and data construction

Let `h_t` contain current observations and executed-action history. Let `c` identify the completion method. Set `z_cond = (h_t,c)`; this is not the old VAE latent.

The population-level target is:

```math
Q^*_{h,c} = argmin_Q KL(Q || R_h)
subject to Q_0 = mu_{h,c}, Q_1 = nu_h,
```

where `mu_{h,c}` is the aligned/completed old-plan law and `nu_h` is the expert new-chunk law. Observations and completion identifier stay fixed throughout every revision and every DSBM cache update.

The deployed model induces `K_theta(A_new | A_old,h,c)`. The intended endpoint constraint averages this kernel over old plans drawn from `mu_{h,c}`. It does not require the same output distribution separately for every fixed old plan.

### 3.1 Use a frozen proposal policy, not future expert overlap

Use a compatible existing DDIM BC checkpoint if available; otherwise the `ddim` baseline trained for this experiment supplies the proposal checkpoint. Freeze it, including its EMA weights and normalizers.

For a logged episode at current time `t`, query that frozen policy at the **earlier recorded history `h_{t-K}`**. This means an earlier robot observation, not an earlier optimizer step. Generate the old 16-action prediction using only information available at `t-K`, then align it to `t` and complete its missing tail.

Pair that proposal with the recorded current expert chunk `A*_t` and current recorded history `h_t`. The pairing is natural because both belong to the same logged episode/current-context transition. It is not an OT solution.

Do not execute the proposal to obtain `h_t` during dataset preparation: `h_t` remains the logged observation. Executing a different action sequence generally invalidates the original future expert label. Likewise, do not move the block/change observations and keep the original expert actions as ground truth.

Do not use overlapping expert future actions as the primary old-plan source. That would disclose the correct overlap at training and make copying appear to solve revision. An `old_source=logged_expert` adapter may exist for debugging but is not a primary result.

### 3.2 Cache contents and leakage safeguards

Cache records keyed by `(episode_id,t,proposal_seed)` containing:

- current and earlier input histories;
- uncompleted earlier policy prediction;
- current expert chunk and valid-horizon mask;
- source/target provenance, normalizer ID, proposal checkpoint hash;
- any frozen reference coefficients needed below.

Two stochastic proposals per eligible history are sufficient initially; avoid a separate large sampling experiment. Form all three completions from the same underlying earlier prediction. Add modest source noise as described below at training/inference, not by contaminating observations.

Split by complete episode before constructing windows, proposals, normalizers, the learned completion model, or OT neighbourhoods. No cross-episode windows. Use the same eligible current expert windows for the five main learners. Handle episode-boundary padding explicitly; do not score padded tail entries as real expert labels. For simplicity, use complete horizons for primary training when available.

### 3.3 Continuous endpoint convention

Repeated-tail source chunks can be low-rank, and a deterministic FM ODE cannot turn a point source into a genuinely multimodal continuous target. Add a small, full-rank Gaussian proposal perturbation in normalized chunk space for **all** chunk-to-chunk methods, during both training and deployment. A starting standard deviation of `0.01` is reasonable; record the equivalent pixel scale. This is perturbation around the old plan, not restarting from pure Gaussian noise.

Use a tiny common endpoint dequantization/smoothing, e.g. normalized standard deviation `0.001`, when sampling training expert endpoints. Apply it consistently to all learners, including DDIM, and evaluate against unsmoothed expert commands. This makes the continuous-density idealization explicit rather than claiming finite continuous path KL to an atomic empirical endpoint. These are engineering starting values, not physical noise estimates.

One expert future per exact observation does not identify every possible conditional mode. Shared-network generalization is required. Do not report exact conditional SB recovery or full mode coverage merely because a matching loss is small.

### 3.4 Keep the source fixed for the first comparison

Do not continuously update the proposal policy with an EMA of each learner. Keep the source dataset fixed through the comparison. DSBM still refreshes its **intermediate coupling caches**, but its prescribed source sampler does not change.

At deployment, old plans come from each learner's own previous outputs. This source-distribution mismatch is real; measure it through held-out source/revision diagnostics and closed-loop results. Do not hide it by querying the proposal policy at every evaluation replan. Self-refreshing proposal curricula and DAgger are failure-investigation options, not default additions.

## 4. Three tail completers, kept separate from the revision reference

After executing eight actions of an old 16-action plan, preserve the remaining eight predictions at their aligned indices and append eight targets. Tail completion must never overwrite the retained overlap. The following are **robot-index** operations; they do not define the SB's `tau`-time dynamics.

### `repeat`

Append the last predicted target eight times.

### `fixed_damped`

Initialize `q` and `p` from the last two old target commands, not from the measured pusher position. Continue with:

```math
p_{j+1} = alpha p_j,
q_{j+1} = q_j + dt p_{j+1}.
```

Use `dt=1` in the documented command convention and fixed `alpha=0.8` initially. It is not learned. No potential, task goal, or added tail noise beyond the common full-chunk proposal perturbation.

### `learned_dissipative`

Reuse or fit a small **reference-only** damped model with bounded, history/index-conditioned attractor, stiffness and damping:

```math
p_{j+1} = p_j + dt[-K_eta(h,j)(q_j-m_eta(h,j)) - gamma_eta(h,j)p_j],
q_{j+1} = q_j + dt p_{j+1}.
```

There is no residual controller or VAE posterior in this completer. Learn from train-split action sequences by ordinary supervised transition fitting; use the existing reference regularization/smoothness conventions where helpful. Freeze the completer and its encoder before training FM/DSBM. Reuse an existing compatible reference only after auditing its training split, units, and future-information access; a jointly trained old reference may not be a useful standalone predictor.

Condition coefficients on the **current observed history** and the current-horizon index. Start the rollout at the end of the retained overlap and evaluate coefficients at current tail indices 8 through 15. Do not query an untrained step embedding beyond its supported range or initialize the tail from an expert state.

Use a limited-capacity reference, not an unrestricted second full policy disguised as a “prior”. Log reference-only prediction error and tail speed/acceleration. Bounded gains and integration checks are required; no blanket passivity claim for a time-varying learned attractor.

### Shared training across completers

Sample `c` uniformly per example; input its identifier to all FM/DSBM networks, and retain it in every coupling cache. Each identifier defines its own conditional source law, with the same expert target. Evaluate a fixed identifier throughout each episode. Do not mix identifiers inside OT matching blocks.

This design makes all three completions in-distribution for a single trained revision model. It does not measure the optimum achievable by training three separate specialists.

## 5. Main baselines

### 5.1 `ddim`: ordinary observation-conditioned diffusion BC

Reuse the local implementation if it exists. Otherwise add a compact, competent low-dimensional diffusion policy using `diffusers.DDPMScheduler` (or the compatible forward-noising API) and `diffusers.DDIMScheduler` for inference.

Use a shared forward-noise schedule, matching `prediction_type`, standard noise-prediction or sample-prediction training, and EMA. The baseline starts from independent Gaussian chunk noise at each replan. `eta=0` DDIM is deterministic given that initial noise; do not replace the initial noise with zeros.

A reasonable initial setup is a cosine training noise schedule, 100 training noise levels, and 32 inference model evaluations. Reuse an already validated repository configuration instead if available and document it. Do not import image-model guidance defaults or clipping incompatible with action normalization. Save/reload scheduler configuration with the checkpoint.

### 5.2 `fm_paired`: same-data chunk continuation without SB coupling optimization

Use the natural records `(h,c,A0,A1)` above. Train a conditional velocity network with straight interpolation:

```math
Y_tau = (1-tau) A0 + tau A1,
L_FM = E ||v_theta(Y_tau,tau,h,c) - (A1-A0)||^2.
```

At inference integrate `dY/dtau=v_theta` from the aligned/completed old proposal. It is an ODE, not a Brownian bridge. Source noise supplies stochasticity.

Do not input the expert endpoint. Do not keep the sampled initial plan as an additional side-conditioning tensor in the primary comparison; it enters through `Y_0`, as it does for the SB. A mode mask/horizon index is allowed equally for both.

The regression is trained on the natural pairing; its induced ODE endpoint coupling is not guaranteed to reproduce that pairing exactly. Do not label this “SB without iterations”.

### 5.3 `fm_local_ot`: practical OT pairing without silently crossing scenes

Use the same architecture and FM loss, but re-pair source and target chunks before drawing interpolation states.

**The conditional issue is real:** standard Push-T does not provide large sets of expert futures at each identical `h`. Global action-only minibatch OT would associate an old plan from one scene with expert actions for another scene. Do not implement that and call it conditional OT.

Implement observation-local, context-penalized minibatch entropic OT, explicitly labelled **approximate local conditional OT**:

```math
C_ij = ||A0_i-A1_j||^2/(2 H d_a) + lambda_h d_ctx(h_i,h_j)^2,
```

subject to hard matching restrictions across incompatible contexts and completion identifiers. Use observed quantities only in `d_ctx`: pusher position, block position, circular block-angle distance, and recent executed commands. Do not use future expert actions to decide whether contexts are compatible. Distances must be normalized explicitly; the angle must be periodic.

Use small neighbourhood minibatches anchored uniformly in the training records, and the same target-record sampling distribution for the paired baseline. Select a conservative context radius from training-data distances, keep each record's original diagonal edge feasible, and use log-domain Sinkhorn with uniform row/column masses. A starting neighbour block size of 8 is sufficient; this is not a nearest-neighbour imitation method at inference.

Draw discrete pairs from the transport matrix. **Do not barycentrically average expert chunks.** Keep the target endpoint's actual history `h_j` as conditioning for the resulting pair. The source `A0_i` is then a local approximation to a source sample at `h_j`, not an exactly matched conditional sample. This changes the empirical conditional source law slightly and must be reported.

Log the context displacement of selected pairs, off-diagonal pairing mass, action transport cost before/after, and Sinkhorn marginal errors. If only identity matches are admissible, report that OT was effectively inactive; do not silently relax the context restriction until unrelated scenes mix. Exact repeated-context groups, if genuinely available, can use exact within-context OT.

This baseline asks whether **approximate geometry-aware re-pairing** helps in practical conditional BC. It is not a theorem-level comparison with exact conditional OT, and a poor result under large context mismatch cannot establish SB superiority.

## 6. Shared network and budget choices

Use the same sensible temporal backbone/observation encoder family across DDIM, FM, and each directional SB model, reusing a competent existing denoiser when possible. A temporal Transformer or conditional 1D U-Net is appropriate; choose one rather than implementing both. Use enough capacity for actual Push-T, not a deliberately tiny baseline. Do not inherit the old 52M-parameter MLP configuration blindly.

Inputs are the evolving chunk (plus revision velocity for the kinetic model), internal time, observed history, and completion ID. Output dimensions follow the relevant noise-channel control or FM velocity. There is no VAE latent/posterior in the new model.

Freeze the learned action-reference encoder. Forward and reverse model snapshots must include their own observation encoders when caching rollouts; changes to an encoder must not silently alter a supposedly frozen opposite-direction model.

Default to **one training seed**, a substantial training budget using existing successful Push-T conventions, and five main configurations. As an initial explicit budget, use 300,000 optimizer updates per main configuration. For DSBM, count updates to both directions in that total: four reverse/forward rounds with 37,500 updates per direction per round. Preserve weights between rounds. These are starting budgets, not claims that four rounds solve the exact bridge.

Count DSBM rollout/cache cost and backward-network memory separately. Equal update counts are not equal wall-clock training compute. Report both. Do not conduct automatic grids over all hyperparameters.

## 7. Reference A: `path_ou`, a tractable plan-prior diffusion

Construct a prior from the learned frozen damped **robot-index** action model, then use it to define a process in **revision time**.

For a candidate plan `A=(q_1,...,q_H)`, take `q_0` and `q_{-1}` from the two executed target commands. Define `p_k=(q_k-q_{k-1})/dt` and innovations:

```math
e_k(A;h) = p_{k+1}-p_k-dt f_eta(q_k,p_k,h,k),  k=0,...,H-1.
```

With a frozen positive innovation covariance `S_k`, define:

```math
U_h(A) = (1/2) sum_k e_k^T S_k^{-1} e_k
         + (ridge/2) ||A - a_anchor(h)||^2.
```

All coefficients depend only on observed `h` and fixed robot-index `k`, not on an expert endpoint, a VAE posterior, or the initial sampled source plan. Estimate `S_k` from train-only innovation residuals with a covariance floor. Use a weak anchor derived from the last executed command. The ridge is numerical confinement, not a task reward.

Since the frozen quadratic reference is affine in `(q,p)`, stack `e=B_h A-d_h`. Then:

```math
U_h(A) = (1/2)(A-mu_h)^T P_h(A-mu_h) + constant,
P_h = B_h^T S^{-1} B_h + ridge I,
mu_h = solve(P_h, B_h^T S^{-1} d_h + ridge a_anchor).
```

Use a bounded, explicit rescaling of the precision/relaxation rate so the reference does not erase the old plan before revision time 1. Record this rescaling and the eigenvalue range. Do not import the old `sigma=7` into generative time: it has different units and semantics.

The reference is:

```math
dY_tau = -D P_h(Y_tau-mu_h) d tau + sqrt(2 T_rev D) dW_tau.
```

Use `D=I`, `T_rev=0.05`, and `gamma_rev=2.0` (kinetic only) as the initial normalized-coordinate settings. These are revision-process hyperparameters, not estimates of physical temperature or friction. Normalize reference precision rates using a train-only scale and record any spectral stabilization. Start with a fixed SPD mobility `D`; `D=I` is the primary setting. A banded mobility induced by `I + lambda_s D2^T D2`, with `D2` the second-difference operator over robot indices, is also supported as a configuration choice, not another mandatory sweep. Keep the same choice in the kinetic comparison.

This is affine OU on an `H*d_a`-dimensional chunk (32 dimensions in this Push-T configuration). It favours plans compatible with learned damped command dynamics. It is not a controller for the physical pusher and does not guarantee physical passivity or collision avoidance. The non-diagonal `P_h` couples future command coordinates; this is not merely independent shrinkage of each target toward zero.

Freeze `P_h`, `mu_h`, `D`, and `T_rev` during a revision. Recompute context-dependent coefficients at the next real replan. An old-source-dependent anchor would require a different augmented/non-Markov formulation; do not add one secretly.

## 8. Reference B: `kinetic_path`, underdamped revision of whole plans

Use the same plan potential `U_h` and mobility `D`, but introduce an auxiliary revision velocity `V_tau` of dimension `H*d_a`:

```math
dY_tau = V_tau d tau,
dV_tau = [-P_h(Y_tau-mu_h) - gamma_rev D V_tau] d tau
         + sqrt(2 gamma_rev T_rev D) dW_tau.
```

The complete state `X_tau=(Y_tau,V_tau)` has dimension `2*H*d_a` (64 in this Push-T configuration). The revision velocity is **not** the command velocity `p_k` and is **not** measured robot velocity.

This reference allows momentum in the direction of plan changes, with damping and a temporally coupled potential. For fixed `h`, its deterministic unforced energy is `U_h(Y)+||V||^2/2` and its time derivative is `-gamma_rev V^T D V`. This is an internal reference property, not a guarantee about the controlled bridge or executed robot.

### Explicit auxiliary endpoint convention

Demonstrations specify a terminal plan, not a terminal revision velocity. Make the modelling choice explicit:

```math
bar_mu_{h,c} = mu_{h,c} x N(0,T_rev I),
bar_nu_h     = nu_h     x N(0,T_rev I).
```

Sample the auxiliary velocities independently when drawing real source/target endpoints. Solve DSBM for these augmented marginals and discard `V_1` after generating the plan. At every real replan draw a fresh `V_0` from the prescribed source; do not carry the previous solver velocity across replans.

Do not impose a Dirac `V_1=0` and then pretend the standard full-state endpoint theory applies unchanged. A position-only terminal constraint would be another algorithm; it is not required here.

This is an engineering choice that adds an auxiliary endpoint distribution and can affect the resulting plan coupling. A gain is evidence for this complete kinetic formulation, not proof that inertia alone caused it.

### Tractability

The reference is still a linear Gaussian SDE. Noise acts only on `V`, but the `(F,G)` system is controllable when `D` is SPD: noise reaches `Y` through integration. Positive-duration transition covariance is full rank even though instantaneous `GG^T` is singular.

Use the correct full-state Gaussian bridge and noise-channel scores. Exact finite-step transition noise will have correlated components in both `Y` and `V`, because velocity noise is integrated into position; this is consistent with instantaneous noise acting only on `V`. Do not add noise/control directly to `Y`, invert the singular instantaneous covariance, or substitute Brownian interpolation. The conditional kinetic DSBM is our adaptation of the matching construction; cite linear controllable SB theory and do not imply the original isotropic-neural DSBM convergence theorem automatically covers the numerical implementation.

## 9. One exact linear-Gaussian reference API for both SB variants

Both references have the form:

```math
dX = (F_h X + g_h) d tau + G_h dW.
```

For OU: `X=Y`, `F=-D P_h`, `g=D P_h mu_h`, `G=sqrt(2 T_rev D)`.

For kinetic:

```math
F = [[0, I],[-P_h, -gamma_rev D]],
g = [0, P_h mu_h],
G = [0; sqrt(2 gamma_rev T_rev D)].
```

Provide transition moments, transition-score targets, endpoint-conditioned intermediate sampling, and stable stochastic stepping. For duration `t-s`:

```math
Phi_{t,s} = exp(F(t-s)),
m_{t|s}(x) = Phi_{t,s} x + int_s^t Phi_{t,r} g dr,
Q_{t,s} = int_s^t Phi_{t,r} GG^T Phi_{t,r}^T dr.
```

Use stable matrix-exponential/Lyapunov or equivalent linear-Gaussian computations, symmetry restoration, and Cholesky solves. Avoid explicit inverses in code. Cache by context and sampled time-grid where economical; batched 64D algebra must not dominate every neural minibatch unnecessarily. With the primary `D=I` and scalar kinetic damping, diagonalizing the SPD `P_h` once per context reduces OU transitions to scalar modes and kinetic transitions to independent 2-by-2 modes; exploit this or an equally efficient implementation rather than repeating dense exponentials for every training sample.

Given endpoint states `(x0,x1)`, the reference bridge at time `tau` has:

```math
C_tau = Q_{tau,0} Phi_{1,tau}^T,
mean = m_{tau|0}(x0) + C_tau solve(Q_{1,0}, x1-m_{1|0}(x0)),
cov  = Q_{tau,0} - C_tau solve(Q_{1,0}, C_tau^T).
```

The endpoint-directed score targets at intermediate state `x` are:

```math
s_plus  = Phi_{1,tau}^T solve(Q_{1,tau}, x1-m_{1|tau}(x)),
s_minus = -solve(Q_{tau,0}, x-m_{tau|0}(x0)).
```

These are derivatives with respect to the **current full state**. The corresponding full forward drift is `b_R + GG^T s_plus`; the backward drift in increasing reverse time is `-b_R + GG^T s_minus`.

For a common implementation, predict noise-channel controls:

```math
u_plus_target  = G^T s_plus,
u_minus_target = G^T s_minus.
```

Then forward drift is `b_R + G u_plus`, reverse drift is `-b_R + G u_minus`. This handles the kinetic rank deficiency correctly and enforces its deterministic kinematics.

## 10. Conditional DSBM training: use actual iterative matching

Implement separate forward and backward conditional controls. In the OU case the input state is a chunk; in the kinetic case it is a chunk plus revision velocity. Expert endpoints appear in target construction only, never as network inputs.

The losses are weighted bridge-matching regressions:

```math
L_plus  = E[w_plus(tau) ||u_theta(X_tau,tau,h,c)-G^T s_plus||^2],
L_minus = E[w_minus(tau)||u_phi(X_tau,tau,h,c)-G^T s_minus||^2].
```

Choose positive time weights and an endpoint-safe parameterization/cutoff consistent with the reference. Kinetic endpoint scores are more singular than Brownian ones. Stable endpoint prediction is acceptable if it is mapped to these drifts correctly and the equivalence is documented. Do not clip away all endpoint learning or fit arbitrary straight-line targets for the non-Brownian models.

The regression optimum is a conditional expectation given current state, observation, completion ID, and time: the Markovian projection. Reconstructing intermediate reference bridges from endpoint pairs is the reciprocal projection.

### Outer loop

```text
Initialize endpoint pairs from natural proposal/expert records.
For each reverse/forward round:
    Fit the reverse control on reference-bridge samples from current pairs.
    Freeze a snapshot of the reverse model.
    Start at real expert endpoints (plus fresh auxiliary V for kinetic).
    Roll backward and retain (generated source, real target, h, c).
    Fit the forward control on reference bridges between these endpoints.
    Freeze a snapshot of the forward model.
    Start at real source endpoints from the fixed proposal dataset.
    Roll forward and retain (real source, generated target, h, c).
    Use those pairs for the next reverse phase.
Deploy a selected forward checkpoint.
```

Refresh finite rollout caches during a phase using the frozen opposite-direction snapshot and newly sampled training contexts. Store detached endpoints, not computation graphs. Use the reference bridge sampler for training intermediates, not an unrelated heuristic interpolation. Preserve `h,c`, reference coefficients, and endpoint-state conventions throughout.

Use the official DSBM implementation/Algorithm 1 to check iteration order and direction conventions. The Brownian reduction must give forward target `(A1-Y)/(sigma(1-tau))` and reverse target `(A0-Y)/(sigma tau)` for whitened controls.

Do not add the old variational teacher-forced loss, a latent KL, a hand-designed task loss, or another arbitrary control-energy penalty to make training “more SB-like”. Here the matching algorithm approximates the constrained SB problem. Log on-policy `0.5*int||u||^2` as a control-energy diagnostic; finite-step estimates and finite neural iterates are not certificates of exact path KL or optimal coupling.

No terminal-density oracle is needed online. No online Sinkhorn is needed. The target law is learned from the demonstration endpoints.

## 11. Closed-loop sampler and coupling fidelity

Use the existing Push-T simulator and common receding-horizon wrapper. At a normal replan:

1. Read actual observations; update **executed command** history separately from the cached plan.
2. Align the previous generated chunk and complete its missing tail using the selected `c`.
3. Apply the small source perturbation used in training.
4. Initialize the FM/SB state, including fresh auxiliary velocity for kinetic.
5. Integrate from `tau=0` to `1`, output `Y_1`, and execute the first eight commands.
6. Cache the entire decoded new chunk for the next replan.

At the first decision, bootstrap all chunk-revision methods using the same frozen DDIM proposal at the initial observation. Execute the same bootstrap prefix before beginning revision; include this cost and explicitly identify it in evaluation. Thereafter do not call the proposal policy. This avoids training an extra first-plan generator for this experiment.

DDIM's own first decision should use the same seeded initial sample. Histories and caches reset completely at episode termination.

**Do not turn off SB diffusion for the headline result.** A drift-only ODE is not the same process. Even a properly constructed probability-flow ODE may preserve marginals but change the endpoint coupling we are testing. Use one stochastic SB sample per replan, not averaging, repeated draws until success, or a critic.

Use a stable stochastic integrator for the controlled linear process, with finer time intervals near the endpoints when needed. A fixed cosine-spaced time grid is a reasonable first choice; late numerical noise must not overwhelm the narrow target distribution. Do not compensate by clamping to an unavailable expert endpoint. Exact linear-reference stepping with a held neural control over each interval is a reasonable implementation; make its discretization and control-energy diagnostic explicit. Ensure backward stepping uses the actual reverse drift, with backward `dY=-V ds` for kinetic when velocity parity is not flipped. Do not reset `Y` or hard-copy its overlap inside the solver.

Compare at 32 neural function evaluations per replan initially. Count all evaluations, not just ODE solver steps. A fixed-step midpoint FM integrator with 16 steps uses 32 evaluations; DDIM uses 32 denoising steps; SB can use 32 control evaluations. Cache-generation rollouts may use finer internal steps if needed and must be accounted for. There is no mandatory sampling-step sweep. Report measured latency because kinetic matrix work and stochastic integration can make equal NFE unequal in time.

Only the usual shared action-bound handling applies at the final environment interface. Log pre-clipping outputs and clipping frequency. Do not project intermediate bridge states onto the workspace, which would change the reference law.

## 12. Minimal evaluation, on the real task

Use one common list of **50 held-out simulator initial seeds**, the same episode-length limit as the existing benchmark, and one sampled policy rollout per initial seed for the main table. Keep a separate small fixed validation panel for checkpoint selection. One training seed cannot establish training-seed robustness; say so without delaying the experiment for more seeds.

Primary outcome: simulator success and maximum/final target coverage. Secondary outcome: continuity/revision at comparable task performance and inference cost. Offline action MSE alone is not a successful Push-T result.

Record:

- `sim_success_rate`, mean maximum coverage, mean final coverage, episode length;
- disagreement between the new chunk's overlap and the aligned old prediction;
- executed-command boundary jump, command acceleration/jerk, and clipping rate;
- inference latency and NFE, plus training/cache-generation time;
- on-policy revision control energy for SB and relevant reference coefficients.

Overlap revision is:

```math
revision_rms = sqrt(mean_{j < H-K} ||A_new[j]-A_prev[j+K]||^2).
```

A small value is not automatically good: copying a stale plan can be smooth and fail. Always interpret it alongside task performance.

### Cheap held-out revision probe, using the same dataset

On a fixed set of held-out records, evaluate how each reviser changes the original old-plan error relative to the current expert chunk. Separate records into low-error and high-error source bins, with bin thresholds selected from training/validation records. Compare before/after overlap error, tail error, and revision magnitude. The same sources are used across models.

This tests preservation versus correction without creating a new environment or requiring new expert labels. MSE against one expert is only a diagnostic in multimodal contexts, not proof of semantic correctness.

Save a few full closed-loop videos with old aligned plan, completed source, final revised plan, realized pusher motion, block pose, and target overlaid. Include representative successes and failures selected by fixed rules, not hand-picked best rollouts. Save a small number of internal revision states for inspection; these are whole candidate plans, not physical pusher trajectories.

No mandatory obstacle-removal benchmark, surface-contact toy, parameter randomization suite, homotopy labels, multi-seed sweep, or RL experiment in this brief.

## 13. Compact run manifest

### Main table: five rows

Evaluate the five configurations using `learned_dissipative` completion for the four chunk-revision models. DDIM has no tail completer after startup.

### Completion comparison: no retraining

Using the trained `sb_kinetic` checkpoint, also evaluate `fixed_damped` and `repeat` on the same 50 initial seeds. Its learned-completion result is already in the main table. All completion variants were present during its balanced training.

Thus the initial requested report contains **five main model runs and two additional completion evaluations**, not a full algorithm-by-completion-by-reference grid. All other checkpoint/mode evaluations are available through config but are not automatically launched.

Reuse a valid existing DDIM checkpoint and a compatible learned reference when possible; otherwise training those is a real dependency. Do not hide auxiliary reference/proposal pretraining from the compute report.

Evaluate validation at a few common milestones (for DSBM, after complete forward phases), not hundreds of simulator rollouts every few optimizer steps. A manifest/driver should be resumable, run independent jobs in any available order, and avoid rerunning completed jobs.

## 14. Concrete deliverables for Codex

Use existing modules where they fit; choose clean names for new components. Implement the reusable boundaries in Section 1.1. Deliver:

- a generic episode-window/source-pair cache builder with causal proposal provenance, connected to the existing Push-T dataset through an adapter;
- all three tail completers and a frozen learned reference fitter/loader;
- dataset-independent DDIM, paired-FM, local-OT-FM, OU-DSBM, and kinetic-DSBM policy factories, with configurable shapes and replaceable encoders/action adapters;
- a common linear-Gaussian reference bridge implementation;
- DSBM directional training, endpoint-cache refresh, and resumable checkpoints;
- a stateful closed-loop plan cache that is distinct from executed-action history;
- a small manifest for the five primary jobs plus the two completion evaluations;
- an evaluation/report script producing one main table, one completion table, and a small set of revision/rollout figures;
- exact documented commands, using this repository's actual CLI and dataset paths rather than guessed interfaces;
- a brief developer note showing how another dataset supplies the common batch/action specifications, context-distance callback and evaluator without copying policy or trainer code.

Save resolved configs, optimizer/EMA/RNG state, normalizers, source/reference hashes, outer iteration and direction, scheduler/reference parameters, and split IDs. Mid-DSBM-phase resumption must restore the coupling-cache provenance. Invalid cache/checkpoint mixtures should fail clearly.

The report must state what was reused, what was implemented, what actually ran, and what remains unexecuted. Do not claim success based only on a CPU smoke test.

## 15. Focused correctness checks, not extra research experiments

Add short automated checks before long training:

1. Alignment maps old index `j+K` to new index `j`; all completers leave the retained overlap unchanged and cannot read future expert actions.
2. Dataset windows/splits are episode-safe; train/inference coordinates and checkpoint normalization round-trip.
3. DDIM scheduler training/inference settings and clipping agree; reload reproduces seeded sampling.
4. OU transition/bridge moments match Gaussian-conditioning identities; Brownian limit recovers its known mean, covariance, and both score signs.
5. Kinetic transition covariance is positive definite for positive duration; control/noise act only on revision velocity; forward/reverse kinematic signs are correct.
6. Frozen reference coefficients depend on `h,k` only; changing the teacher endpoint cannot alter the reference or conditioning features.
7. DSBM caches preserve contexts/completion IDs and use real source/target endpoint refreshes in the correct directions.
8. OT preserves minibatch marginal masses, samples discrete pairs, respects completion/context masks, and reports any context mismatch.
9. Episode reset clears every cached plan/auxiliary state; no expert chunk is accessed by simulator evaluation.
10. The Section 1.1 portability checks pass for a non-Push-T shape/action specification; generic policy imports and checkpoint sampling do not require a Push-T dataset or simulator. These are unit checks, not additional benchmark training.

These checks should take minutes, not require learning separate synthetic tasks to convergence.

## 16. How to read the result

- If FM continuation beats DDIM but SB does not improve on FM, old-plan initialization is useful; the added SB machinery is not yet justified.
- If SB improves task performance or the correction/continuity trade-off over both FM baselines, there is evidence that the reference-aware coupling/bridge training adds value. It is not proof of global SB optimality.
- If OU and kinetic match, prefer OU. Do not keep kinetic terminology solely because it sounds physics-informed.
- If learned completion wins but the generator does not matter, the result belongs to the completion prior, not specifically SBs.
- If a policy reduces revision and jerk but stalls or fails more, it has over-preserved stale decisions.
- If OT is effectively identity or crosses materially different contexts, the OT result is inconclusive about exact conditional transport.
- If offline revision works but closed-loop Push-T fails, first investigate source/history shift and alignment rather than immediately adding a new architecture or benchmark.

The output of this experiment is a go/no-go assessment of conditional SB plan revision on Push-T. Do not promise a win, a physics guarantee for the robot, or a publishable result before seeing closed-loop evidence.

## 17. Source map and references

Repository facts above were checked against the public snapshot listed in Section 1 and the supplied project architecture/README/reference notes. The concrete five-way experiment, shared completion-mode training, local-OT approximation, and kinetic augmentation are **proposed design choices**, not algorithms already present in those notes.

Core mathematical/algorithmic sources:

1. Shi, De Bortoli, Deligiannidis, Doucet, *Conditional Simulation Using Diffusion Schrödinger Bridges*, UAI 2022. Sections 4 and 5.1–5.2: conditional problem, joint samples, conditional sources/references.
   https://proceedings.mlr.press/v180/shi22a.html
   https://github.com/vdeborto/cdsb
2. Shi, De Bortoli, Campbell, Doucet, *Diffusion Schrödinger Bridge Matching*, NeurIPS 2023. Sections 2–4, especially Algorithm 1: bridge matching, Markovian/reciprocal projections, actual iterative training.
   https://arxiv.org/abs/2303.16852
   https://github.com/yuyang-shi/dsbm-pytorch
3. Chen, Georgiou, Pavon, *Optimal Transport over a Linear Dynamical System*. Sections III–IV: controllable linear references and dynamics-induced endpoint transport costs.
   https://arxiv.org/abs/1502.01265
4. Sophia Tang, *Foundations of Schrödinger Bridges for Generative Modeling*, March 2026 uploaded edition. Sections 2.6–2.7, 4.1, 4.4–4.5, and 6.3; Section 6.4 for why arbitrary paired interpolation is not automatically the optimal SB. Use original papers to check formulas with ambiguous time/sign conventions.
   https://arxiv.org/abs/2603.18992
5. Lipman et al., *Flow Matching for Generative Modeling*; Tong et al., *Simulation-Free Schrödinger Bridges via Score and Flow Matching*. These motivate matching and coupling baselines; our `fm_local_ot` is not claimed to be exact conditional SF2M.
   https://arxiv.org/abs/2210.02747
   https://arxiv.org/abs/2307.03672
6. Chi et al., *Diffusion Policy*, and official `diffusers` scheduler documentation. Use these for a competent diffusion-BC baseline and current scheduler behaviour.
   https://diffusion-policy.cs.columbia.edu/
   https://huggingface.co/docs/diffusers/api/schedulers/ddim

Relevant supplied/repository documents: `ARCHITECTURES.md`, `README.md`, `contact_hamiltonian_reference_implementation.md`, and `dissipative_sb_policy_derivation.md` (filenames can differ in the local checkout). Their original robot-index reference model is reusable; their original teacher-forced path-KL objective is not DSBM. Do not import the unrelated BC-to-RL plan.
