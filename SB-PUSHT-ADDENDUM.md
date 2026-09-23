# SB-PUSHT addendum: self-generated-source experiment

**Status: ACTIVE REPLACEMENT. Protocol ID: `self_source_v1`.**

This addendum implements the latest agreed experiment. It replaces the former fixed-DDIM-proposal training and DDIM-bootstrap protocol in `SB-PUSHT.md`; it is **not an extra experiment to run alongside that protocol**. Keep the reusable implementation contract and the mathematical definitions that are not explicitly changed below.

The primary question is now:

> Can an observation-conditioned chunk reviser learn from its own detached previous predictions, and does conditional SB matching with a structured revision reference outperform ordinary chunk-to-chunk flow matching and independent DDIM generation on closed-loop Push-T?

The five generators remain DDIM, paired FM, local-OT FM, OU-DSBM, and kinetic-DSBM. What changes is how the four revisers obtain their old plans during training and at startup.

## A.0 Precedence, retired instructions, and scope

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

## A.1 The only active scientific run manifest

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

## A.2 Keep the three roles separate

| Role | Parameters/update policy in this experiment |
|---|---|
| Previous-plan producer | The reviser's own forward/generative network keeps learning; an immutable copy of its EMA weights generates each block's source cache |
| Tail completer | `repeat`, fixed damping, or one shared learned dissipative model; the learned model is fit on train-only demonstrations and then frozen |
| SB revision reference | The OU or kinetic dynamics from the base plan, constructed from the frozen plan prior; no policy-loss gradients update this reference |

For fixed observed context, the reference drift/diffusion coefficients are unchanged by source refreshes. Strictly, its full path law is initialized with the current source distribution, so changing that boundary changes the full reference law even though the transition dynamics remain fixed. Do not confuse this with learning new reference coefficients.

Fit the learned completer using its existing supervised transition objective, with no residual policy or VAE posterior. Share its checkpoint and normalizer across FM and SB methods. The same frozen damped model may supply the OU/kinetic plan potential as in the base document. Record its cost separately. No simulator reward or expert task-cost gradient is used.

## A.3 Define the self-generated source curriculum precisely

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

### Logged observations remain fixed

During source-cache construction, **do not execute the predicted commands in the simulator**. Read the recorded observation and recorded executed-command history at the next timestamp. Maintain two separate objects:

- `act_hist`: commands actually executed in the demonstration;
- `plan_cache`: the model's unexecuted predicted future.

The policy-generated plan cache never replaces `act_hist` during offline replay. At deployment both observations and executed-command history instead come from actual environment execution.

This curriculum addresses old-plan-input mismatch. It does not eliminate BC state-distribution shift. Do not claim on-policy state training or DAgger, and do not modify the block pose/current observation while retaining an invalid expert future label.

### Refresh through sequence replay, not independent isolated queries

At the beginning of every block:

1. Copy the current EMA generative policy to an immutable source-producing snapshot. Include encoder parameters, buffers, normalizers, sampler settings, and RNG provenance. It must not share mutable storage with the student or its live EMA.
2. Replay the train-split logged episodes on the chosen replanning grid. Use complete valid expert horizons for primary training and handle the initial histories explicitly.
3. Balance `c` across sequence replay passes, keeping it fixed for a whole replay. If necessary, replay an episode separately for each completion mode; this is cache generation, not three policy trainings.
4. At each replan, choose expert/generated old plan using the block's probability, align and complete it, and store its detached pairing with the current expert target.
5. Run the snapshot on that source and the recorded context, then cache the generated chunk for the next replan. Update this cache even when an expert old chunk was selected for the current input.

The current expert endpoint may be stored as a training label, but cannot be an input to the snapshot's sampler. Do not replace the generated output with the expert target before carrying it forward. The explicit Bernoulli decision at the next replan is the only teacher-forcing replacement.

Preserve the full-rank source perturbation, expert endpoint smoothing, split hygiene, and action-coordinate conventions from Section 3.3. Do not add source noise twice. The primary source cache stores the completed perturbed source realization and its provenance; coupling refreshes resample from that block's prescribed source population. All seeded replay outputs must be reproducible.

## A.4 Startup without a pretrained generator

Each reviser must generate its own first chunk. There is no DDIM bootstrap and no common externally executed prefix.

For Push-T absolute-target actions, form a length-`H` initial proposal by repeating the last executed target; when there is no command history, use the observed pusher position. Apply the prescribed source perturbation once, condition on the current observation and `has_previous_plan=false`, and run the model's normal FM/SB sampler. Execute only the resulting generated chunk's prefix, not the unrevised initial proposal.

Do not seed startup with an expert future chunk, arbitrary future block position, or the output of another model. For the kinetic model, sample the initial auxiliary revision velocity using the already specified source convention, including at startup.

Store startup/source-to-expert records during every block and ensure they receive nonzero training mass throughout. After the first generated chunk, normal alignment and completion apply. This is startup coverage within the same training run, not a separate first-plan model or pretraining task.

Block 1 uses a snapshot of the initialized policy; it need not already be competent. Early expert-old-plan mixing supplies the agreed stabilization mechanism for later decisions. Numerical failures must be surfaced and counted, not silently repaired with a DDIM/expert fallback. Standard initialization and the base plan's stable integration/output handling remain available.

The initialization rule belongs in the action/environment adapter. Do not embed Push-T pusher-coordinate slices in a generic policy implementation.

## A.5 Learner updates and the two different cache refreshes

### Paired FM and local-OT FM

`fm_paired` fits the existing straight-interpolation FM objective on the new block's natural `(h,c,A0,A1)` records. Keep model weights, EMA, optimizer, and normalizers across blocks; replace the source snapshot/cache, not the learner.

`fm_local_ot` applies the base plan's context-restricted re-pairing to its **own current-block records**. Retain target-context conditioning, sample endpoint pairs rather than averaging targets, and log context displacement/off-diagonal mass. Also prevent matching across completion IDs or startup/ordinary-source regimes. Do not relax context restrictions to manufacture a nontrivial OT result.

### DSBM: keep iterative matching, but update its prescribed source between rounds

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

## A.6 Completion training and reference settings stay separate

The three tail algorithms in Section 4 are unchanged: repetition; fixed decay starting at `0.8`; and bounded learned spring–damper continuation without a task residual. They preserve the retained overlap, then fill only missing indices.

Freeze the shared learned completer before the five-way policy comparison. Detach the completed source from the main matching loss. Joint continuation learning is not a new run in this addendum. Do not mistake these stopped gradients for a prohibition on the main policy learning from its own generated inputs.

Retain both reference processes exactly as defined in Sections 7–9, including the quadratic plan potential, context-dependent frozen coefficients, fixed diffusion settings, Gaussian bridge sampler, and kinetic endpoint convention. Updating an EMA source producer does not justify using Brownian interpolation for a non-Brownian reference or turning off SB noise.

For all models, fix the completion identifier for each deployed episode. Because all three identifiers were represented during training, the two extra completion evaluations are supported source modes; they are not new specialist checkpoints.

## A.7 Evaluation replaces the old DDIM-bootstrap evaluation

Run genuine closed-loop low-dimensional Push-T from the beginning of each episode. Retain `H=16`, `K=8`, two-frame observation history, two-command executed history, the same absolute-target representation, and the same simulator/task definitions. Those numbers remain configuration values under the portability contract.

Use one training seed per method; 50 common held-out simulator initial states; a separate fixed validation panel; and initially 32 neural function evaluations per replan. Retain the base document's equal-NFE accounting and report actual latency and training/source/coupling-cache cost. Reuse existing competent architecture/scheduler settings rather than adding a tuning sweep.

DDIM generates independently from Gaussian noise at every replan. Each reviser uses its own observed-information-only startup and then its actual previous generated chunk. There are no expert old plans, no external proposal calls, no candidate scoring, and no best-of-N selection at evaluation. Do not exclude the first generated prefix or startup failures from success/coverage scores.

Evaluate the checkpoint's stated sampler: FM uses its ODE; DDIM uses its seeded initial noise and DDIM schedule; SB uses one stochastic revision sample. Do not average samples, turn off SB diffusion, or substitute a marginal-preserving ODE and assume its old/new coupling is unchanged.

The main table reports task success, maximum/final coverage, and episode length, together with overlap revision, executed-command boundary discontinuity, command acceleration/jerk, clipping rate, and latency. Keep physical pusher trajectories separate from commanded targets. Small revision/jerk alone is not success if the policy stalls or preserves wrong decisions.

### One inexpensive common-source revision diagnostic

After training, replay a fixed held-out set of logged histories using the four learned revisers, with self-only sources and no expert-old-plan mixing. Pool an equal, predetermined number of old-plan proposals from each model; complete them with the shared learned completer. Freeze this probe set, preserving the conditioning/history key and origin of every proposal.

Evaluate every reviser on exactly that same set. Bin sources by low/high original error to the logged current expert using thresholds selected on training/validation data; report before/after overlap and tail errors plus revision magnitude. This is a diagnostic using logged labels, not another training run or a simulator expert query. No pooled proposal may cross to another history.

At a given history, one expert is not the only possible valid behavior; report this limitation of MSE-based correction diagnostics. The main decision remains closed-loop task performance. This probe partly separates revision quality from the fact that the models trained on different self-generated source laws.

## A.8 Config, implementation, and artifact changes for Codex

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

## A.9 Focused acceptance checks and interpretation

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

### Basis of this addendum

The source curriculum and replacement run manifest implement the latest decisions in this conversation. They are proposed experiment/training choices, not results reported by the cited SB papers. Sections 1.1, 2, 5, and 7–10 of the base document remain the implementation and mathematical basis where not overridden here; its Section 17 retains the algorithmic references. The legacy architecture and reference notes define reusable command-coordinate components, not this new training protocol. No RL experiment or new theoretical guarantee is introduced.
