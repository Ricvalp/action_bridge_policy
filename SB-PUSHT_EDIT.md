# SB-PUSHT: conditional Schrödinger-bridge revision of Push-T action chunks

> **ACTIVE PROTOCOL: `self_source_v1`. Read [Addendum A](#active-self-source-addendum) before implementing or launching runs.**
> It replaces the former frozen-DDIM-proposal experiment and DDIM-bootstrap evaluation. Train revisers on their own detached EMA-snapshot plans with an early expert-plan mixture. DDIM is an independent baseline, not a proposal dependency. Do not run both protocols.
> The reusable implementation contract and OU/kinetic bridge definitions below remain in force where Addendum A does not override them. Superseded passages are retained as historical context, not executable instructions.

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

> **SUPERSEDED — do not implement the source policy below.** Addendum A.3–A.4 replaces this with own-policy EMA snapshots, early expert-old-plan mixing, and no external bootstrap. The recorded-observation/valid-label safeguards still apply.

Use a compatible existing DDIM BC checkpoint if available; otherwise the `ddim` baseline trained for this experiment supplies the proposal checkpoint. Freeze it, including its EMA weights and normalizers.

For a logged episode at current time `t`, query that frozen policy at the **earlier recorded history `h_{t-K}`**. This means an earlier robot observation, not an earlier optimizer step. Generate the old 16-action prediction using only information available at `t-K`, then align it to `t` and complete its missing tail.

Pair that proposal with the recorded current expert chunk `A*_t` and current recorded history `h_t`. The pairing is natural because both belong to the same logged episode/current-context transition. It is not an OT solution.

Do not execute the proposal to obtain `h_t` during dataset preparation: `h_t` remains the logged observation. Executing a different action sequence generally invalidates the original future expert label. Likewise, do not move the block/change observations and keep the original expert actions as ground truth.

Do not use overlapping expert future actions as the primary old-plan source. That would disclose the correct overlap at training and make copying appear to solve revision. An `old_source=logged_expert` adapter may exist for debugging but is not a primary result.

### 3.2 Cache contents and leakage safeguards

> **CACHE PROTOCOL UPDATED.** Keep split and leakage safeguards, but use Addendum A.3/A.8 for per-model, per-block snapshot provenance and source-cache construction. A single external-DDIM cache is not the active source.

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

> **SUPERSEDED — source fixation now applies within each training block, not the entire experiment.** Follow Addendum A.1/A.3/A.5; self-generated source refresh is part of the main protocol, not a deferred ablation.

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

> **REPLAY SEMANTICS UPDATED.** Balance completion IDs across sequence replays and keep each ID fixed within a replay/episode, as in Addendum A.3/A.6. Do not generate a cached plan under one completion regime and silently relabel it as another.

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

> **OUTER SOURCE UPDATE REPLACED BY ADDENDUM A.5.** Preserve the matching mathematics above. The loop below describes the retired fixed-source schedule; use the current block source cache and the explicit source-refresh/coupling-refresh distinction in the addendum.

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

> **STARTUP OVERRIDE.** The DDIM bootstrap paragraphs below are retired. Addendum A.4/A.7 requires observed-information-only startup by each reviser, with no externally generated or executed prefix. The reference-consistent stochastic sampling requirements remain.

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

> **REPLACED MANIFEST.** Addendum A.1/A.8 defines the only active jobs and dependencies. Keep five main methods and two completion evaluations, but do not launch old fixed-DDIM-source jobs or require DDIM training before revisers.

### Main table: five rows

Evaluate the five configurations using `learned_dissipative` completion for the four chunk-revision models. DDIM has no tail completer after startup.

### Completion comparison: no retraining

Using the trained `sb_kinetic` checkpoint, also evaluate `fixed_damped` and `repeat` on the same 50 initial seeds. Its learned-completion result is already in the main table. All completion variants were present during its balanced training.

Thus the initial requested report contains **five main model runs and two additional completion evaluations**, not a full algorithm-by-completion-by-reference grid. All other checkpoint/mode evaluations are available through config but are not automatically launched.

Reuse a valid existing DDIM checkpoint and a compatible learned reference when possible; otherwise training those is a real dependency. Do not hide auxiliary reference/proposal pretraining from the compute report.

Evaluate validation at a few common milestones (for DSBM, after complete forward phases), not hundreds of simulator rollouts every few optimizer steps. A manifest/driver should be resumable, run independent jobs in any available order, and avoid rerunning completed jobs.

## 14. Concrete deliverables for Codex

> **DELIVERABLES EXTENDED/OVERRIDDEN BY ADDENDUM A.8.** Implement the reusable interfaces below with self-source snapshot/replay support and block-versioned caches; retire external-DDIM-proposal dependencies.

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

> **CHECKS UPDATED BY ADDENDUM A.9.** Preserve applicable numerical/portability checks and add the self-source, startup, curriculum, and active-manifest tests. Expert-old-plan access is allowed only through the declared early-training branch.

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

---

<a id="active-self-source-addendum"></a>

## Addendum A — Active self-generated-source experiment

**Status: ACTIVE REPLACEMENT. Protocol ID: `self_source_v1`.**

This addendum implements the latest agreed experiment. It replaces the former fixed-DDIM-proposal training and DDIM-bootstrap protocol in `SB-PUSHT.md`; it is **not an extra experiment to run alongside that protocol**. Keep the reusable implementation contract and the mathematical definitions that are not explicitly changed below.

The primary question is now:

> Can an observation-conditioned chunk reviser learn from its own detached previous predictions, and does conditional SB matching with a structured revision reference outperform ordinary chunk-to-chunk flow matching and independent DDIM generation on closed-loop Push-T?

The five generators remain DDIM, paired FM, local-OT FM, OU-DSBM, and kinetic-DSBM. What changes is how the four revisers obtain their old plans during training and at startup.

### A.0 Precedence, retired instructions, and scope

**Do not launch the old protocol.** Remove it from the active/default run manifest and dependency graph. Preserve reusable code and previous artifacts; do not delete checkpoints, caches, or results. Legacy configurations may remain clearly marked as inactive, but must never be selected by the new default comparison command.

| Earlier instruction | Replacement |
|---|---|
| Section 3.1: obtain all training old plans from a separate frozen DDIM | Each reviser obtains its own old predictions from an immutable snapshot of its current EMA policy |
| Section 3.1: expert old chunks are debugging-only | Expert old chunks are an explicit early-training curriculum component; remove them from the final half of training |
| Sections 3.2 and 3.4: one fixed external proposal cache throughout training | Refresh per-model source caches at four training-block boundaries; hold each source snapshot fixed within its block |
| Section 10: start forward DSBM rollouts from the fixed external proposal dataset | Start from the prescribed source cache for the current model and block; retain actual iterative matching |
| Section 11: DDIM generates and executes the common bootstrap prefix | Each reviser generates its own first plan from an observed-information-only initial proposal |
| Section 13: DDIM training is a dependency of all revisers | DDIM is an independent baseline and can run in parallel with them |
| Section 4: choose completion mode independently for every replay step | Balance the three modes across sequence replays; keep the mode fixed within a replay and an evaluation episode |
| Sections 12–15: external-proposal provenance and associated checks | Use the cache, evaluation, checkpoint, and acceptance requirements below |

The learned completer and the revision-reference dynamics **remain fixed during main-policy training in this experiment**. They are not the source-producing policy. This addendum does not introduce joint reference learning or end-to-end continuation learning.

Preserve Sections 1.1 and 2: dataset-independent policies/trainers, replaceable encoders and action adapters, configurable dimensions/horizons, and the existing Push-T observation/action interface. Preserve the DDIM/FM objectives, local-OT safeguards, and OU/kinetic Gaussian bridge mathematics in Sections 5 and 7–10 unless an explicit override below applies. Keep this experiment offline BC; do not import the older BC-to-RL plan.

### A.1 The only active scientific run manifest

| Run ID | Generator | Source during policy training | Main evaluation completion |
|---|---|---|---|
| `ddim` | Observation-conditioned diffusion BC; `diffusers` DDIM inference | Gaussian chunk noise, as in ordinary diffusion BC | Not applicable |
| `fm_paired` | Conditional FM with natural within-rollout pairing | Own detached previous predictions, mixed with expert old chunks early | `learned_dissipative` |
| `fm_local_ot` | Same FM architecture with approximate local conditional OT | Same curriculum rule, using this model's own predictions | `learned_dissipative` |
| `sb_ou` | Conditional iterative DSBM with the plan-prior OU reference | Same curriculum rule, using this model's own forward predictions | `learned_dissipative` |
| `sb_kinetic` | Conditional iterative DSBM with the kinetic whole-plan reference | Same curriculum rule, using this model's own forward predictions | `learned_dissipative` |

Train all four revisers with the three recognized completion modes. After the five main runs, evaluate the **same selected `sb_kinetic` checkpoint** with `repeat` and `fixed_damped`. Its learned-completion result is already in the main table. Do not retrain for these two comparisons.

Use one training seed, the full training split, and the existing substantive budget. Initially retain **300,000 optimizer updates per main configuration**, partitioned as follows:

| Block | Update interval, zero-based | Probability of using a self-generated old plan | Expert-old-plan probability |
|---|---:|---:|---:|
| 1 | `[0, 75000)` | `0.10` | `0.90` |
| 2 | `[75000, 150000)` | `0.50` | `0.50` |
| 3 | `[150000, 225000)` | `1.00` | `0.00` |
| 4 | `[225000, 300000)` | `1.00` | `0.00` |

For each SB block, run one full reverse/forward round: 37,500 reverse updates and 37,500 forward updates. FM uses 75,000 updates in each block. DDIM uses ordinary training; it does not need this source curriculum. These are the agreed initial engineering settings, not optimal schedules or guarantees of SB convergence.

Auxiliary work is at most one shared supervised dissipative-model fit when a compatible checkpoint is unavailable. Source replay and DSBM coupling refresh are part of each model's training cost, not additional benchmark experiments. Reuse a compatible DDIM baseline implementation/checkpoint after the local audit, but never make it a required proposal producer.

Do not add a preliminary toy-training ladder, more environments, demonstration-count sweeps, multiple training seeds, extra source-policy baselines, or a completion-by-generator grid.

### A.2 Keep the three roles separate

| Role | Parameters/update policy in this experiment |
|---|---|
| Previous-plan producer | The reviser's own forward/generative network keeps learning; an immutable copy of its EMA weights generates each block's source cache |
| Tail completer | `repeat`, fixed damping, or one shared learned dissipative model; the learned model is fit on train-only demonstrations and then frozen |
| SB revision reference | The OU or kinetic dynamics from the base plan, constructed from the frozen plan prior; no policy-loss gradients update this reference |

For fixed observed context, the reference drift/diffusion coefficients are unchanged by source refreshes. Strictly, its full path law is initialized with the current source distribution, so changing that boundary changes the full reference law even though the transition dynamics remain fixed. Do not confuse this with learning new reference coefficients.

Fit the learned completer using its existing supervised transition objective, with no residual policy or VAE posterior. Share its checkpoint and normalizer across FM and SB methods. The same frozen damped model may supply the OU/kinetic plan potential as in the base document. Record its cost separately. No simulator reward or expert task-cost gradient is used.

### A.3 Define the self-generated source curriculum precisely

Let `h_t` be the recorded current observation and executed-action history, `c` the completion mode, and `bar_theta_r` an immutable snapshot of the model's EMA at the start of block `r`. For SB this snapshot is the **forward** revision policy, including its encoder and sampling configuration, not the reverse model.

At a recorded replan after startup, the source is:

```math
B_t ~ Bernoulli(p_self[r]),
A_old = B_t ? Ahat_previous : Astar_previous,
A0_t = stopgrad(Complete_c(Align_K(A_old), h_t) + source_noise),
A1_t = Astar_current + training_endpoint_smoothing.
```

Here `Ahat_previous` is the output of the snapshot at the preceding recorded replan, not a fresh prediction at `h_t`. `Astar_previous` is the chunk starting at that preceding replan. It overlaps the current expert chunk by design; this is intentional teacher forcing during the first two blocks.

Select a whole generated or expert old plan. Do not average their coordinates. Source origin (`self`, `expert`, or `startup`) is audit metadata, not a privileged learned feature. Completion ID is conditioning as before. A `has_previous_plan` bit or equivalent availability mask is permitted for startup and must be available to all four revisers at both training and inference.

The endpoint law during a block is a mixture of expert-old and self-generated proposals. In the early blocks, the self-generated cache itself is produced by the specified mixture-assisted replay; do not describe it as a fully autonomous rollout. Blocks 3 and 4 use only generated old plans after startup.

The four models use the same curriculum, recorded history keys, normalization, and replay/sampling budget, but their actual source distributions differ because their predictions differ. This is a comparison of complete learning systems, not an isolated experiment with identical endpoint couplings.

#### Logged observations remain fixed

During source-cache construction, **do not execute the predicted commands in the simulator**. Read the recorded observation and recorded executed-command history at the next timestamp. Maintain two separate objects:

- `act_hist`: commands actually executed in the demonstration;
- `plan_cache`: the model's unexecuted predicted future.

The policy-generated plan cache never replaces `act_hist` during offline replay. At deployment both observations and executed-command history instead come from actual environment execution.

This curriculum addresses old-plan-input mismatch. It does not eliminate BC state-distribution shift. Do not claim on-policy state training or DAgger, and do not modify the block pose/current observation while retaining an invalid expert future label.

#### Refresh through sequence replay, not independent isolated queries

At the beginning of every block:

1. Copy the current EMA generative policy to an immutable source-producing snapshot. Include encoder parameters, buffers, normalizers, sampler settings, and RNG provenance. It must not share mutable storage with the student or its live EMA.
2. Replay the train-split logged episodes on the chosen replanning grid. Use complete valid expert horizons for primary training and handle the initial histories explicitly.
3. Balance `c` across sequence replay passes, keeping it fixed for a whole replay. If necessary, replay an episode separately for each completion mode; this is cache generation, not three policy trainings.
4. At each replan, choose expert/generated old plan using the block's probability, align and complete it, and store its detached pairing with the current expert target.
5. Run the snapshot on that source and the recorded context, then cache the generated chunk for the next replan. Update this cache even when an expert old chunk was selected for the current input.

The current expert endpoint may be stored as a training label, but cannot be an input to the snapshot's sampler. Do not replace the generated output with the expert target before carrying it forward. The explicit Bernoulli decision at the next replan is the only teacher-forcing replacement.

Preserve the full-rank source perturbation, expert endpoint smoothing, split hygiene, and action-coordinate conventions from Section 3.3. Do not add source noise twice. The primary source cache stores the completed perturbed source realization and its provenance; coupling refreshes resample from that block's prescribed source population. All seeded replay outputs must be reproducible.

### A.4 Startup without a pretrained generator

Each reviser must generate its own first chunk. There is no DDIM bootstrap and no common externally executed prefix.

For Push-T absolute-target actions, form a length-`H` initial proposal by repeating the last executed target; when there is no command history, use the observed pusher position. Apply the prescribed source perturbation once, condition on the current observation and `has_previous_plan=false`, and run the model's normal FM/SB sampler. Execute only the resulting generated chunk's prefix, not the unrevised initial proposal.

Do not seed startup with an expert future chunk, arbitrary future block position, or the output of another model. For the kinetic model, sample the initial auxiliary revision velocity using the already specified source convention, including at startup.

Store startup/source-to-expert records during every block and ensure they receive nonzero training mass throughout. After the first generated chunk, normal alignment and completion apply. This is startup coverage within the same training run, not a separate first-plan model or pretraining task.

Block 1 uses a snapshot of the initialized policy; it need not already be competent. Early expert-old-plan mixing supplies the agreed stabilization mechanism for later decisions. Numerical failures must be surfaced and counted, not silently repaired with a DDIM/expert fallback. Standard initialization and the base plan's stable integration/output handling remain available.

The initialization rule belongs in the action/environment adapter. Do not embed Push-T pusher-coordinate slices in a generic policy implementation.

### A.5 Learner updates and the two different cache refreshes

#### Paired FM and local-OT FM

`fm_paired` fits the existing straight-interpolation FM objective on the new block's natural `(h,c,A0,A1)` records. Keep model weights, EMA, optimizer, and normalizers across blocks; replace the source snapshot/cache, not the learner.

`fm_local_ot` applies the base plan's context-restricted re-pairing to its **own current-block records**. Retain target-context conditioning, sample endpoint pairs rather than averaging targets, and log context displacement/off-diagonal mass. Also prevent matching across completion IDs or startup/ordinary-source regimes. Do not relax context restrictions to manufacture a nontrivial OT result.

#### DSBM: keep iterative matching, but update its prescribed source between rounds

Use two distinct cache types:

- **Source cache `S_r`**: generated by logged robot-time replay under `bar_theta_r` and the block curriculum. It specifies the source boundary for this round.
- **Coupling cache `Pi`**: generated by forward/backward simulations in revision time. It supplies endpoint pairs for Markovian/reciprocal matching and may refresh within the round.

A reverse-generated revision endpoint is not automatically a new old-plan boundary sample. Only `S_r` defines the prescribed source for the block. Expert endpoints remain the prescribed target.

A consistent update order is:

```text
Initialize forward/reverse networks, optimizers, and EMAs.
For block r in 1..4:
    Snapshot the current forward EMA and build S_r by logged sequence replay.
    Keep S_r and the reference transition dynamics fixed for this round.

    If r == 1:
        Initialize Pi from S_r's natural source / current expert pairs.
    Else:
        Warm-start Pi by simulating the existing forward snapshot
        from refreshed S_r endpoints, retaining (source, generated target).
        Do not reuse old-source pairs as though they came from S_r.

    Fit reverse matching on reference bridges between Pi endpoints.
    Snapshot the fitted reverse sampler.
    Start reverse rollouts at real expert targets and retain
        (generated source, real target, unchanged conditioning).
    Fit forward matching on reference bridges between these pairs.
    Snapshot the fitted forward sampler.
    Start forward rollouts at prescribed S_r endpoints and retain
        (real source, generated target, unchanged conditioning).
    Validate/save the forward model and preserve all learner weights.
```

This warm-start at a changed boundary is an explicit implementation choice for the evolving-source experiment. It avoids repeatedly discarding the learned correspondence by treating every round as a new one-pass supervised interpolation model. It is not a claimed convergence theorem for a moving-source SB.

Within matching phases, retain the base plan's detached opposite-direction rollouts, exact OU/kinetic reference-bridge intermediate sampling, correct score/noise-channel targets, and real-boundary refreshes. Do not regenerate the prescribed source with the continuously changing student halfway through a round.

For kinetic DSBM, lift real source/expert endpoints with fresh independent auxiliary velocities as prescribed in Section 8. Generated augmented endpoints keep their sampled velocity components in coupling caches; do not replace them by position finite differences or zeros.

Changing the source between rounds means the global procedure fits a succession of conditional bridge problems. Standard fixed-endpoint convergence statements do not automatically apply. Keep that qualification in experiment documentation, while still running the actual iterative algorithm.

### A.6 Completion training and reference settings stay separate

The three tail algorithms in Section 4 are unchanged: repetition; fixed decay starting at `0.8`; and bounded learned spring–damper continuation without a task residual. They preserve the retained overlap, then fill only missing indices.

Freeze the shared learned completer before the five-way policy comparison. Detach the completed source from the main matching loss. Joint continuation learning is not a new run in this addendum. Do not mistake these stopped gradients for a prohibition on the main policy learning from its own generated inputs.

Retain both reference processes exactly as defined in Sections 7–9, including the quadratic plan potential, context-dependent frozen coefficients, fixed diffusion settings, Gaussian bridge sampler, and kinetic endpoint convention. Updating an EMA source producer does not justify using Brownian interpolation for a non-Brownian reference or turning off SB noise.

For all models, fix the completion identifier for each deployed episode. Because all three identifiers were represented during training, the two extra completion evaluations are supported source modes; they are not new specialist checkpoints.

### A.7 Evaluation replaces the old DDIM-bootstrap evaluation

Run genuine closed-loop low-dimensional Push-T from the beginning of each episode. Retain `H=16`, `K=8`, two-frame observation history, two-command executed history, the same absolute-target representation, and the same simulator/task definitions. Those numbers remain configuration values under the portability contract.

Use one training seed per method; 50 common held-out simulator initial states; a separate fixed validation panel; and initially 32 neural function evaluations per replan. Retain the base document's equal-NFE accounting and report actual latency and training/source/coupling-cache cost. Reuse existing competent architecture/scheduler settings rather than adding a tuning sweep.

DDIM generates independently from Gaussian noise at every replan. Each reviser uses its own observed-information-only startup and then its actual previous generated chunk. There are no expert old plans, no external proposal calls, no candidate scoring, and no best-of-N selection at evaluation. Do not exclude the first generated prefix or startup failures from success/coverage scores.

Evaluate the checkpoint's stated sampler: FM uses its ODE; DDIM uses its seeded initial noise and DDIM schedule; SB uses one stochastic revision sample. Do not average samples, turn off SB diffusion, or substitute a marginal-preserving ODE and assume its old/new coupling is unchanged.

The main table reports task success, maximum/final coverage, and episode length, together with overlap revision, executed-command boundary discontinuity, command acceleration/jerk, clipping rate, and latency. Keep physical pusher trajectories separate from commanded targets. Small revision/jerk alone is not success if the policy stalls or preserves wrong decisions.

#### One inexpensive common-source revision diagnostic

After training, replay a fixed held-out set of logged histories using the four learned revisers, with self-only sources and no expert-old-plan mixing. Pool an equal, predetermined number of old-plan proposals from each model; complete them with the shared learned completer. Freeze this probe set, preserving the conditioning/history key and origin of every proposal.

Evaluate every reviser on exactly that same set. Bin sources by low/high original error to the logged current expert using thresholds selected on training/validation data; report before/after overlap and tail errors plus revision magnitude. This is a diagnostic using logged labels, not another training run or a simulator expert query. No pooled proposal may cross to another history.

At a given history, one expert is not the only possible valid behavior; report this limitation of MSE-based correction diagnostics. The main decision remains closed-loop task performance. This probe partly separates revision quality from the fact that the models trained on different self-generated source laws.

### A.8 Config, implementation, and artifact changes for Codex

Use the existing configuration system. The following are **required semantics**, not claims about flags already implemented:

```yaml
protocol: self_source_v1
methods: [ddim, fm_paired, fm_local_ot, sb_ou, sb_kinetic]
source:
  producer: own_forward_ema_snapshot
  stop_gradient: true
  refresh: training_block_boundary
  self_probability_by_block: [0.10, 0.50, 1.00, 1.00]
  recorded_observations_and_executed_history: true
  teacher_choice: whole_old_chunk
  external_proposal_checkpoint: null
startup:
  mode: observed_anchor_repeat
  train_startup_examples: true
  external_bootstrap: false
completion:
  train_modes: [repeat, fixed_damped, learned_dissipative]
  balance: across_sequence_replays
  fixed_within_replay_and_episode: true
  learned_model_shared_and_frozen: true
revision_reference:
  parameter_updates_during_main_training: false
budget:
  training_seeds: 1
  total_optimizer_updates_per_model: 300000
  training_blocks: 4
  dsbm_updates_per_direction_per_block: 37500
  evaluation_initial_states: 50
  inference_nfe: 32
report:
  main_completion: learned_dissipative
  additional_evaluations:
    - {method: sb_kinetic, completion: repeat, retrain: false}
    - {method: sb_kinetic, completion: fixed_damped, retrain: false}
```

The active driver must list only these five main configurations, the optional shared auxiliary fit, and two extra completion evaluations. DDIM has no source-snapshot dependency; the four revisers have no DDIM-checkpoint dependency. The SB jobs can reuse the frozen auxiliary plan-prior model, while FM only needs the completer for its learned-completion sources.

Extend the reusable source builder, not a Push-T-only trainer. Support immutable model snapshots, sequence replay, plan-cache resets, completion/startup conditioning, and block-specific mixture probabilities through generic interfaces. Policy imports and checkpoint sampling must not require a simulator or training dataset.

Version source caches by protocol, model ID, block, snapshot hash, episode/time, completion mode, startup status, normalizer/schema, reference/completer hashes, and RNG seed. Store expert/self/startup origin and its prescribed probability as metadata. Coupling caches additionally record direction and generating snapshot; never load a fixed-DDIM-source cache into this protocol without an explicit, reported conversion that is not used for primary results.

Checkpoints must restore live forward/reverse weights, their EMAs, the immutable source snapshot or its exact artifact, current block/direction/update count, mixture schedule, source/coupling-cache provenance, optimizers, RNGs, and shared frozen components. Resuming within a block must not silently take a newer source snapshot. Rebuild a lost cache only reproducibly from its recorded snapshot and replay seeds.

Revise the repository's default run commands and README so `self_source_v1` is the sole active comparison. Existing frozen-DDIM-source checkpoints/results are legacy, not completed rows in the replacement report. A compatible ordinary DDIM baseline or shared reference checkpoint can still be reused when its data, settings and budget are documented. Do not rerun legacy jobs just to fill a comparison table.

### A.9 Focused acceptance checks and interpretation

Keep the base plan's short geometry/bridge/normalization tests. Add these source-protocol checks; they are unit checks, not extra experiments:

1. No reviser requires or queries DDIM, including on the first decision.
2. Source and completer tensors carry no main-policy gradient; snapshot parameters/buffers do not change when the student or live EMA updates.
3. In self-only mode, replacing current/future expert labels cannot change the generated source or sampler output. In early expert-mix mode, expert access occurs only through the declared old-chunk branch and training target.
4. Offline replay changes only the predicted-plan cache, not recorded observations or executed-action histories; reset prevents cross-episode leakage.
5. The last two blocks have zero expert-old-plan selections after startup, and the empirical source-origin rates match the intended schedule over non-startup records.
6. Source versions stay fixed within an SB round; coupling refreshes remain distinct from source refreshes and retain all conditional/auxiliary state information.
7. All completion modes appear during training, stay fixed within replay/episode, and leave retained overlap unchanged before the common source perturbation.
8. Startup sampling and mid-block checkpoint resumption work without labels or external proposal artifacts at inference.
9. A dry-run of the active manifest enqueues no legacy fixed-DDIM-source experiment and no unintended factorial sweep.
10. The existing non-Push-T shape/encoder/action-adapter portability tests continue to pass.

Produce one main five-row table, one three-row kinetic completion table, and the small common-source revision diagnostic with representative rollout overlays. State what actually ran and which artifacts were reused. Do not present implementation completion or a smoke test as a positive scientific result.

Interpret the outcome as follows:

- FM beating DDIM supports the complete self-conditioned action-to-action approach; it does not isolate the advantage of SB.
- SB outperforming FM on task success or the continuity/correction trade-off supports its added machinery under this training procedure, not exact coupling optimality.
- Kinetic and OU matching favors the simpler OU implementation.
- A completion effect belongs to the continuation component; it is not automatically evidence for SB.
- Low offline error with poor closed-loop performance motivates checking initialization, alignment, and state/source mismatch before adding another benchmark.

**Final execution instruction: run this replacement Push-T comparison only. The former frozen-DDIM-source experiment and DDIM-bootstrap evaluation are retired, not additional required baselines.**

#### Basis of this addendum

The source curriculum and replacement run manifest implement the latest decisions in this conversation. They are proposed experiment/training choices, not results reported by the cited SB papers. Sections 1.1, 2, 5, and 7–10 of the base document remain the implementation and mathematical basis where not overridden here; its Section 17 retains the algorithmic references. The legacy architecture and reference notes define reusable command-coordinate components, not this new training protocol. No RL experiment or new theoretical guarantee is introduced.
