# Addendum B — Curriculum over the robot-time replanning interval K

**Applies to:** `SB-PUSHT.md`, particularly its active **Addendum A: `self_source_v1`**.

**New protocol identifier:** `self_source_k_curriculum_v1`.

## B.0 Assignment and precedence

Implement a curriculum that increases the execution/replanning interval **K** while keeping the predicted chunk horizon **H** fixed. For the current Push-T configuration, use:

```text
H = 16 throughout
training K: 1 -> 2 -> 4 -> 8
deployment/main evaluation K: 8
```

This is an addendum to the **new chunk-to-chunk plan-revision pipeline**, not the older robot-index latent contact-Langevin policy. Inspect the current checkout and its agent instructions first. The user reported source replay in `action_bridge/plan_revision/data.py`; treat that as a navigation hint, not a verified current API or line number. Locate the active source builder, trainers, coupling buffers, completers, and evaluation wrapper before editing.

When this protocol is selected, override Addendum A's fixed **training** `K=8` and its protocol/cache identifier. Preserve everything else unless explicitly changed below:

- The same five configurations: `ddim`, `fm_paired`, `fm_local_ot`, `sb_ou`, `sb_kinetic`.
- Each reviser uses its own **immutable forward/generative EMA snapshot** to produce detached old plans, with Addendum A's early expert-old-plan mixture. Do not reintroduce a frozen external DDIM proposal policy.
- The learned completer and OU/kinetic reference coefficients remain shared/frozen as specified in Addendum A. The main policy continues learning.
- The three completion modes remain balanced across sequence replays and fixed within each replay/evaluation episode.
- Preserve observed-information-only startup, source perturbation, target smoothing, action normalization, and kinetic auxiliary-state conventions.
- Preserve the actual FM/DSBM objectives. Do not substitute the original teacher-forced action-path loss for DSBM.
- This remains **offline imitation learning**. No RL, reward-weighted fitting, planner-generated training targets, expert querying, or simulator gradients are introduced.
- Keep the implementation dataset-independent. Push-T dimensions, units, state fields, and simulator capabilities belong to adapters/configuration.

Retain the original fixed-K protocol as a reproducible ablation and for old checkpoint compatibility. Do not delete old artifacts or automatically launch both protocols.

### Hypothesis, not a guarantee

Starting at small K may keep the recorded next observation closer to what would result from executing a short prefix of the model's chunk. As the model improves, increasing K exposes it to older proposals, less overlap, and longer completion tails.

This does **not** make offline replay physically on-policy. Recorded states still come from expert execution. K=1 can already teach correction of inaccurate *unexecuted plans*; increasing K does not suddenly create self-correction supervision. Do not claim that this curriculum solves state-distribution shift or guarantees recovery from states absent from the data.

## B.1 Keep the meanings and shapes separate

| Symbol/setting | Meaning | Curriculum behavior |
|---|---|---|
| `H` | Number of robot commands in every predicted chunk | Fixed for the instantiated model |
| `K` | Number of robot steps between consecutive replans; offline replay advances the recorded timestamp by this amount | Changes by training block |
| `k` or `j` | Robot-index position within a chunk | Remains in `0 .. H-1` |
| `tau` | Internal FM/SB revision time | Remains on the existing interval, normally `[0,1]` |
| Revision NFE / integrator steps | Number of neural evaluations/steps during chunk revision | Unchanged by the K curriculum |
| Consecutive replay decisions | Length of recursive plan-cache replay through an episode | Not a separately scheduled curriculum here |

In ordinary source construction:

```text
old decoded plan:     [B, H, d_a]
retained suffix:      [B, H-K, d_a]
completion tail:      [B, K, d_a]
completed source:     [B, H, d_a]
expert endpoint:      [B, H, d_a]
new decoded plan:     [B, H, d_a]
```

For `sb_kinetic`, preserve its existing augmented revision-state shape, e.g. a packed state of dimension `2*H*d_a`. The auxiliary **revision velocity** is not the target-command finite difference used by the tail completer.

No layer, positional table, optimizer, or model head should be rebuilt when a curriculum stage changes. Any existing fixed-length encoding of the *uncompleted suffix* must be replaced by encoding the completed source, or by a fixed-size padded/masked representation. The ordinary whole-chunk input/output sizes stay unchanged.

Keep robot control interval `dt`, physics substeps, observation-history length, and executed-command-history length fixed. K is not a change in physical sampling frequency.

## B.2 Default schedule: reuse the existing four blocks

Use the established total budget and source-mixture schedule. Add K to those same blocks:

| Block | Global optimizer-update interval | K | Self-old-plan probability | Expert-old-plan probability |
|---|---:|---:|---:|---:|
| 1 | `[0, 75000)` | 1 | 0.10 | 0.90 |
| 2 | `[75000, 150000)` | 2 | 0.50 | 0.50 |
| 3 | `[150000, 225000)` | 4 | 1.00 | 0.00 |
| 4 | `[225000, 300000)` | 8 | 1.00 | 0.00 |

For FM, each block contains 75,000 optimizer updates. For each SB model, one block contains a **complete reverse/forward round**, with 37,500 reverse updates and 37,500 forward updates. Count both directions in the global total, as in Addendum A.

Changing K and changing the self-source mixture are separate controls. Preserve both in configuration and logs. The default changes them at the same boundaries; therefore, a comparison against an unrelated training schedule cannot isolate the effect of K alone.

Requirements:

- All K values must be integers satisfying `1 <= K <= H`.
- Validate a nondecreasing schedule and compatible block/update counts; reject invalid configurations instead of silently clipping K.
- Make `[1,2,4,8]` a configuration, not a hard-coded algorithmic constant.
- Hold K fixed within a block, within every sequence replay, and throughout each complete DSBM round.
- Use optimizer-update counts or explicit complete-round counts, not epoch counts. Dataset/cache size changes with K, so epochs are not a comparable unit.
- Reserve the final configured block for deployment K. Do not label an interrupted run that stopped at K=2 as a completed K=8 curriculum run.
- Keep the existing learning-rate schedule and all model weights across stages. Do not restart learning at each K.

**Mandatory first implementation:** this deterministic, block-based schedule. Optional metric-gated advancement is described in B.9 and must not block implementation of the schedule.

## B.3 Alignment and variable-length completion

Use zero-based chunk indices. A plan generated at recorded time `t_prev` is:

```text
A_old[j] proposes the command for robot time t_prev + j
```

At the next recorded decision, `t = t_prev + K`:

```text
A_source[j] = A_old[j + K]      for j = 0, ..., H-K-1
A_source[H-K:H] = completion   # exactly K new commands
A_target = expert_actions[t:t+H]
```

Before the existing common source perturbation, completion must leave the retained overlap unchanged. Do not blend it with the current pusher position, overwrite it with the expert target, or hard-freeze it during subsequent SB/FM revision. The **reviser** can modify the whole source.

For example, with `H=16`:

| K | Retained old indices | Current-horizon indices filled by completion |
|---|---|---|
| 1 | `1:16` | `15:16` |
| 2 | `2:16` | `14:16` |
| 4 | `4:16` | `12:16` |
| 8 | `8:16` | `8:16` |

### Completion initialization and indexing

For absolute-target actions and a damped completer, initialize from the **last two commands of the full old plan**, not from the beginning of the retained suffix or the measured pusher position:

```math
q_0 = A_old[H-1],
p_0 = (A_old[H-1] - A_old[H-2]) / dt.
```

Run the same one-step continuation update exactly K times. A deterministic fixed-damping example is:

```math
p_{r+1} = alpha p_r,
q_{r+1} = q_r + dt p_{r+1},   r = 0, ..., K-1.
```

Append `q_1, ..., q_K` in that order. The recurrence permits a variable number of completion actions; dissipation shapes those actions, but is not what makes the length variable.

For the learned dissipative completer, preserve the existing force/coordinate convention. Evaluate history/index-dependent coefficients using the **current history** and the **current horizon indices**:

```text
j = H-K, ..., H-1
```

Do not use `H, ..., H+K-1` or reset every completion to coefficient indices `0, ..., K-1`. The old plan's final command corresponds to the end of the retained overlap in the current horizon. Update both command position and command velocity recursively after every completion step.

A learned fixed-H coefficient head may still be used: select its relevant tail coefficients while keeping its H fixed. Audit whether the frozen learned model was trained for every requested index. Do not silently extrapolate a learned index embedding beyond its support. Refit a shared auxiliary model only if genuinely necessary, using training data only and documenting the cost.

### Edge cases

- `K=H-1`: the overlap has only one action; obtain velocity initialization from the full old plan, not that one-action slice.
- `K=H`: there is no overlap. For continuous absolute-target actions, support full completion from the old plan's last two commands; do not treat an empty overlap as missing plan history. Another adapter-specific reset convention must be explicit and versioned, not silently substituted.
- No previous plan: preserve Addendum A's full-H observed-anchor startup proposal and `has_previous_plan=false`. There is no suffix to shift. The next decision occurs after the stage's K steps.
- Too short a plan to initialize a two-command recurrence: use an explicitly supported history/adapter fallback or reject it. Do not invent physical velocity from unavailable data.
- Stop at episode boundaries. For primary training, preserve the full-valid-expert-horizon restriction rather than padding arbitrary terminal endpoints and claiming exact Gaussian-bridge semantics.

Apply the existing source noise exactly once after source construction. Do not add noise once per completed step unless that was already part of the declared completer. Do not rescale source noise or SB diffusion solely because K changes.

## B.4 Expose K without variable-width model inputs

Always record the alignment shift, overlap length, and stage K in batch/cache metadata. Provide an overlap mask with shape `[B,H]`, where retained entries are true and completed entries are false; startup has no retained entries. Keep this distinct from a target-validity mask.

For new curriculum runs, condition all four revisers on the scheduled interval through a small **fixed-width** feature such as `K/H`, using the existing conditioning interface. A fixed-width K embedding added to the context is sufficient. The completion ID and `has_previous_plan` conditioning remain unchanged. Supplying the overlap mask as another learned feature is optional; it is already determined by K for ordinary full chunks.

This is an explicit one-time conditioning change, not a shape change at every stage. Version the model/checkpoint schema accordingly. Preserve the old fixed-K model path when loading legacy checkpoints. Any warm-start migration to a K-conditioned model must be explicit, documented, and tested; a zero-initialized new conditioning branch is one possible implementation.

Do not supply source origin (`self` versus `expert`), future labels, or simulator diagnostic outcomes as privileged learned features. Those remain metadata.

For a stateful execution wrapper, distinguish:

- **elapsed count since the cached plan was generated**, which determines alignment;
- **next execution count**, which determines how long the new chunk will be applied.

They coincide in an ordinary fixed-K episode. At curriculum transitions, rebuild/reset replay rather than shifting an old cache using a count that was never elapsed. For startup, store actual elapsed count as zero and use the configured K as the planned interval, with the startup bit disambiguating it.

Do not modify the frozen OU/kinetic coefficients or revision-time horizon merely because K has changed. Any explicitly K-dependent reference is a separate design change, outside this addendum.

## B.5 Rebuild the source through sequence replay at each stage

Follow Addendum A's self-source replay, but use the new block's K everywhere the replay advances, aligns, or completes.

At every block boundary:

1. Finish the current block/complete DSBM round and save a checkpoint.
2. Set the next block's K and its existing expert/self mixture probability.
3. Snapshot the current generative EMA; for SB, this is the forward model. Include its encoder, conditioning, normalizers, sampler configuration, and immutable buffers.
4. Rebuild a detached source cache by replaying training episodes on the new K grid.
5. Keep that source snapshot and prescribed source cache fixed for the block.

Within an episode replay:

```text
Initialize from observed information only; retain the resulting generated plan.
At every subsequent recorded decision:
    Read recorded observations and recorded executed-command history at time t.
    Select the whole self-generated or expert old chunk using p_self[block].
    Align by the actual elapsed K and complete exactly K commands.
    Apply the prescribed source perturbation once.
    Store source, current expert endpoint, conditioning, and provenance.
    Run the immutable snapshot on the source and recorded observations.
    Carry its generated full chunk to the next recorded decision.
    Advance t by K within the same episode.
```

Even when the current input used an expert old chunk, carry the **snapshot's output** to the next decision. The next Bernoulli selection is the only teacher-forcing replacement. Preserve the balanced completion modes across entire replay passes.

Do not mutate only a `K` field on cached old sources. A recursive replay at K=1 and a recursive replay at K=8 generally have different generated ancestors, contexts, completion tails, and source distributions. Replaying/rebuilding is necessary.

Keep observation and executed-action histories at their original contiguous robot timesteps. Increasing K does not mean subsampling history frames at spacing K. During offline replay, generated commands must never replace the recorded executed history. The replay does not step the simulator.

### Sampling and computational cost

Smaller K creates more replay decisions per episode. Budget and log this explicitly; the source-generation cost at K=1 can be substantially larger than at K=8.

Preserve train/validation episode splits and train-only normalization. Use the existing episode/time sampler where compatible, with reproducible offsets if needed to avoid always selecting one residue class of timestamps. Report eligible and cached record counts at each K. A changed timestamp grid may change empirical context coverage; do not claim the target sampling distribution stayed exactly identical if it did not.

Keep startup examples represented throughout; avoid allowing their sampling fraction to change accidentally only because the number of non-startup records changes. Use the existing startup sampling policy if present, or an explicit, logged stratification rule.

A bounded cache may subsample records for storage, but skipped intermediate decisions must still be simulated by the **snapshot generator in logged replay** if needed to obtain later recursive plan caches. Do not replace sequence replay with unrelated independent queries to save cost without labeling that as a different protocol.

## B.6 DSBM source and coupling caches must remain distinct

Continue to distinguish:

- `S_r`: the prescribed old-plan source population produced by logged robot-time replay for block r;
- `Pi`: endpoint pairs produced by forward/reverse **revision-time** rollouts for matching.

A generated reverse endpoint is not a newly prescribed old-plan source sample.

For a new K block, use Addendum A's update rule:

```text
Preserve learned forward/reverse networks, EMAs, optimizers, and global LR schedule.
Build the new S_r with the new K and current immutable forward snapshot.

First block:
    Initialize Pi from S_r's natural source / expert pairs.
Later blocks:
    Build a fresh Pi using the existing forward snapshot started at new S_r sources.
    Retain (real new source, generated target, unchanged conditioning).

Fit the reverse phase.
Generate reverse coupling pairs from real expert endpoints for current contexts/K.
Fit the forward phase.
Generate forward coupling pairs from prescribed S_r sources.
Validate/save the forward model before any further K transition.
```

Invalidate obsolete coupling pairs when K or the prescribed source cache changes. Do not re-label old-K pairs as new-K pairs, or reuse a cached bridge mean/target associated with the wrong endpoint/context. Reference transition-matrix caches that truly depend only on unchanged geometry and time can be reused through correct keys; sample-specific bridge quantities cannot.

During a block, retain the existing opposite-direction coupling refreshes. Do **not** freeze Pi merely because S_r is fixed, and do **not** refresh S_r continuously with the changing student. Never change K between a block's reverse and forward phases.

Preserve the correct OU/kinetic bridge samplers and whitened-control targets. K does not alter revision time or license Brownian interpolation for a non-Brownian reference. For the kinetic model, carry generated auxiliary velocities in coupling caches and retain the prescribed source/target velocity sampling; do not reinterpret them as physical pusher velocities.

For `fm_local_ot`, add K/overlap/startup compatibility to existing context/completion restrictions. Never pair endpoints across incompatible K regimes to increase off-diagonal OT mass. All source and target conditioning must retain the same intended K.

Changing K/source boundaries means training on a succession of conditional problems. Do not claim fixed-endpoint DSBM convergence for this moving curriculum.

## B.7 Configuration and checkpoint semantics

The following is illustrative configuration, **not a claim that these flags already exist**. Map the semantics into the actual repository's configuration conventions without creating competing authorities for K.

```yaml
protocol: self_source_k_curriculum_v1
chunk_horizon: 16
k_curriculum:
  enabled: true
  mode: scheduled
  values_by_block: [1, 2, 4, 8]
  optimizer_updates_per_block: [75000, 75000, 75000, 75000]
  transition_boundary: complete_training_block
  condition_on_k: true
source:
  producer: own_forward_ema_snapshot
  self_probability_by_block: [0.10, 0.50, 1.00, 1.00]
  refresh: training_block_boundary
  recorded_observations_and_executed_history: true
completion:
  train_modes: [repeat, fixed_damped, learned_dissipative]
  fixed_within_replay_and_episode: true
  learned_model_shared_and_frozen: true
revision_reference:
  parameter_updates_during_main_training: false
evaluation:
  n_exec: 8
  condition_on_actual_evaluation_k: true
mismatch_probe:
  enabled: false
  use_as_training_data: false
```

Add K, curriculum identity, stage, elapsed-count convention, overlap-mask convention, and K-conditioning schema to existing cache provenance. Keep snapshot, completer/reference, normalizer, dataset/split, episode/time, completion ID, startup/origin, sampler, and RNG hashes.

Checkpoint enough state to resume inside either DSBM direction without taking a new source snapshot: active stage/K, updates completed, direction/outer round, source snapshot/artifact, source/coupling versions, RNGs, all networks/EMAs/optimizers, and any pending stage transition. Make boundary resume idempotent: no skipped or double-applied promotion/cache generation.

Changing a schedule in a resumed run must require an explicit new-run/override operation. Old fixed-K checkpoints remain loadable with their saved semantics; they are not automatically curriculum results. Existing main policies should sample without importing a dataset or simulator.

## B.8 Evaluation and experiment accounting

Keep the base five-way comparison. DDIM training is unchanged because it has no previous-plan source. Its main closed-loop execution K must still match the revisers' deployment K.

The primary checkpoint-selection/evaluation panel uses **K=8 for all methods**, including checkpoints produced during earlier training stages. These early scores test deployment-K transfer; label them accordingly. Do not compare a K=1 rollout score with a K=8 score and attribute the difference to learning rather than feedback frequency.

Optional diagnostics may evaluate the active training K or another K, but use separate namespaced metrics. At each evaluation, alignment and K-conditioning must use the actual executed count, not the checkpoint's most recent training K.

Preserve:

- fixed validation initial states separate from the final held-out test panel;
- one training seed and existing substantive budget initially;
- the same revision NFE and sampler conventions;
- stochastic SB execution, ordinary FM ODE execution, and the existing DDIM sampling;
- no expert old plans, candidate scoring, or external bootstrap during deployment;
- same selected kinetic checkpoint for the existing extra completion evaluations.

The useful isolating ablation is **fixed K=8 versus the K curriculum**, with the same expert/self mixture, conditioning capability, completion model, architecture, data split, and update budget. Provide configuration support, but do not automatically multiply the main manifest into a new grid. Existing fixed-K results with incompatible conditioning must be labeled as a broader comparison, not a perfectly isolated ablation.

Report data-generation wall time, snapshot sampler calls, cache size, training updates in each direction, and inference cost. Equal optimizer updates do not mean equal total compute. Source-cache growth must not silently increase batch size or reduce the optimization budget.

## B.9 Optional simulator mismatch probe and gated promotion

Implement this as an adapter/evaluation callback only when the existing simulator supports valid state restoration. It is diagnostic; it must not feed new training targets, rewards, or gradients to the learner.

At a fixed validation anchor corresponding to the earlier replan:

1. Restore the same complete simulator state in two branches.
2. Use a self-generated full plan from that anchor and execute its first K commands in the model branch, without intermediate replanning.
3. Execute the recorded expert commands from the same anchor in the expert branch.
4. Compare physical trajectories through all K steps, not just final target-command coordinates.

Report maximum and terminal pusher-position discrepancy, block-position discrepancy, periodic block-angle error, and velocity discrepancies when available. Keep components in explicitly defined units; do not combine pixels and radians without documented scales. Report means, upper quantiles, and early termination/failure rates.

Also compare expert re-execution with the recording to assess simulator reconstruction error. If observations omit velocities/controller/contact state, do not call a reset from position alone an exact snapshot. Prefer recovering states by replaying recorded commands from a reproducible episode initialization. When that is unavailable, mark the probe approximate or unavailable and skip automatic gating.

The probe must not create `(policy-reached state, original recorded future)` training pairs. Those targets are not automatically valid from the reached state. It must not snap the old plan to the physical pusher as an unreported repair.

**Optional gated mode:** after the scheduled implementation works, allow promotion only at complete block/round boundaries and after configured minimum training plus consecutive passing validation checks. Test the proposed **next K**, not only the current K. Require explicit task-scaled thresholds and a reliable state-restoration probe; do not invent universal thresholds. A failed/missing probe holds the stage or raises a clear configuration error, never silently counts as a pass.

Any extra complete rounds needed to hold a stage consume an explicitly configured total budget. Report if final K was never reached; do not force advancement and call the result gated. Gate decisions and probe RNG/provenance must be saved for reproducible resume. This optional mode is not an additional default scientific run.

## B.10 Logging and focused tests

Log at least:

```text
curriculum/block, active_k, deployment_k, overlap_length, completion_length
updates_in_block, global_updates, dsbm_direction, dsbm_round
self_probability, empirical_self_fraction, startup_fraction
source_version, source_snapshot_id, coupling_source_version
source_records, source_replay_calls, source_build_seconds
source_error_overlap, source_error_tail, revised_error_overlap, revised_error_tail
revision_magnitude_overlap, revision_magnitude_tail
closed_loop/deployment_k_8/*
optional_mismatch_probe/*
```

Use per-coordinate/per-valid-action reductions for overlap and tail diagnostics so differing region lengths are not mistaken for progress. Empty overlap at K=H must yield a masked/unavailable metric, not NaN or a fabricated zero-error result. Single-demonstration MSE is a diagnostic, not proof that every different plan is wrong.

Add short tests, not new learning benchmarks:

1. Fixed output/source shapes for K=1,2,4,8 and edge cases K=H-1,H; include a non-Push-T H/action dimension.
2. Exact alignment of timestamp-labeled synthetic commands; no off-by-one target shift.
3. Bitwise/equivalent overlap preservation before the prescribed common perturbation.
4. Completion length exactly K; step coefficients queried at `H-K .. H-1`.
5. Fixed-damping zero-velocity case, alpha=0 case, and alpha=1 linear extrapolation match hand calculations.
6. Velocity initialization uses the full old plan when overlap is shorter than two actions.
7. Noise is applied once; K changes neither physical dt nor SB integration settings.
8. Stage changes occur at the specified boundaries; for DSBM there is no change midway through a reverse/forward round.
9. New K rebuilds source ancestry/cache provenance; stale-K coupling/source artifacts are rejected.
10. Student/EMA updates cannot mutate the immutable source snapshot or frozen completer/reference.
11. In self-only replay, changing future-label-only data cannot change sources/sampler outputs when observed histories are held fixed. Early expert access is limited to the declared old-plan mixture branch and fitting labels.
12. Offline histories remain recorded; genuine evaluation histories contain actually executed commands. No cache crosses episode boundaries.
13. Startup works at every K without an expert future or external generator.
14. Resume immediately before/after a stage boundary and midway through each DSBM direction preserves the intended schedule, sources, and seeded behavior.
15. Disabled curriculum reproduces the fixed-K protocol; explicit checkpoint migration is tested separately.
16. A tiny end-to-end smoke run traverses all four stages with reduced budgets and finite losses; it tests wiring, not learning quality.
17. Optional mismatch-probe branches start identically, expert replay fidelity is reported, and diagnostic states never enter the supervised training buffer.

## B.11 Deliverables and acceptance

Deliver reusable schedule handling, variable-K source construction/completion, K-conditioning support, boundary-safe cache refreshes, checkpoint/resume integration, and deployment-K evaluation. Extend the current code rather than creating a second trainer per K or per dataset.

Add exact commands using the repository's actual CLI for a short curriculum smoke test, a substantive curriculum run, resuming it, fixed-K compatibility, and fixed deployment-K evaluation. Update the active manifest/header documentation so it is unambiguous when `self_source_k_curriculum_v1` is selected and which earlier fixed-training-K instructions it overrides.

The implementation is accepted when the source trace visibly changes from retaining 15 actions to 14, 12, and 8 at H=16; every completed source remains H long; stage transitions refresh the right caches without resetting learned weights; and all headline evaluations still execute the declared common K.

Report what was implemented, what actually ran, and any unavailable simulator diagnostics. Do not claim improved closed-loop performance from a smoke test, decreasing logged-history MSE, or reduced plan-to-pusher distance alone.

### Basis

The existing protocol is `SB-PUSHT.md`, Addendum A, especially A.1–A.8: four training blocks, self-source mixture, immutable EMA snapshots, startup, separate source/coupling caches, and frozen completion/reference models. Base Sections 1.1 and 4 provide the reusable fixed-H interface and three completion conventions. The present conversation adds the K curriculum and the mismatch interpretation.

The older `ARCHITECTURES(1).md` and `contact_hamiltonian_reference_implementation.md` document command coordinates, fixed-H outputs, and recursive damping. They are not specifications of the newer DSBM trainer. The BC-to-RL document is outside this addendum's scope.

**Scientific interpretation:** this is a proposed curriculum over source age, overlap, and completion length. It may help the reviser learn under manageable mismatch. It does not make unexecuted offline proposals generate the recorded state, and it supplies no new recovery labels after physical deviations.
