# Codex Implementation Brief: Surface-Contact Imitation-Learning Toy Experiment

## 0. Purpose

Implement a new closed-loop toy benchmark inside the existing Action Bridge Policy repository to test the following hypothesis:

> A behavior-cloned policy with a learned, contact-frame dissipative reference can match the demonstrated tangential task while producing more stable normal contact behavior, especially under limited data and out-of-distribution contact perturbations, than a standard action-chunk diffusion policy trained on the same demonstrations.

This is an **imitation-learning / behavior-cloning experiment**. The learner must not receive a hand-designed task reward or trajectory cost during training.

A task cost may be used internally to generate expert demonstrations, and task/contact quantities are used for evaluation only.

The experiment must isolate which component helps:

- the contact-frame coordinate structure;
- learned anisotropic damping;
- the reference-plus-residual parameterization;
- the path-KL/control-energy penalty;
- the choice of Action Bridge rather than diffusion as the action generator.

Do not treat a win over one weak diffusion configuration as sufficient. Build the experiment so that the result can be interpreted mechanistically.

---

## 1. Existing repository context

Inspect the repository before implementing and reuse existing abstractions whenever possible.

The current main policy already follows the pattern

```text
policy path = reference path + learned control deformation
```

with:

- `ActionBridgePolicy` as the main policy class;
- reference processes under the existing reference-model module;
- teacher-forced path losses under the current training-loss module;
- receding-horizon rollout/evaluation utilities;
- common dataset fields such as observation history, action history, and future action chunks;
- existing output conventions for configs, checkpoints, CSV/JSON metrics, and figures.

The current contact-Langevin policy uses absolute target actions as its internal `q` coordinate, finite-difference target velocity as `p`, semi-implicit updates, a learned reference force, and a whitened residual control with a path-KL/control-energy term. Preserve those semantics where applicable rather than creating an unrelated implementation.

Do not break the existing delayed-branch, annular, or Push-T experiments.

---

## 2. Scientific scope and non-goals

### Primary question

Does a learned contact-frame reference improve closed-loop contact stability and OOD recovery while preserving tangential task performance?

### Secondary questions

1. Does the model actually learn larger or otherwise different damping in the normal and tangential directions?
2. Does the residual use the reference, or simply override it?
3. Is any advantage explained only by expressing the policy in the contact frame?
4. Is the path-KL term necessary once the structured reference exists?
5. Would a diffusion policy using the same reference recover the same benefit?

### Non-goals

- Do not implement reinforcement learning.
- Do not expose the learner to the expert trajectory cost.
- Do not implement a full Schrödinger Bridge solver, Sinkhorn, IPF, score matching, or contact-Wasserstein optimization.
- Do not hard-code `gamma_normal > gamma_tangent`.
- Do not constrain the learned residual to be tangential.
- Do not put the tangential goal into a reference force that directly drives the agent to the goal.
- Do not claim physical passivity if the implementation only damps command-space targets; report precisely what state is being damped.
- Do not add tactile sensing only to one method. All learned policies must receive the same observations.

---

## 3. Toy task: approach, engage, slide, and stop

Create a two-dimensional point end-effector interacting with a compliant flat surface.

### 3.1 Physical state

Use a physical state containing at least:

```text
position x in R^2
velocity v in R^2
```

The surface is an oriented line described by:

```text
unit normal n
unit tangent t
surface offset c
signed distance d(x) = n^T x - c
```

Choose and document one sign convention for free space, penetration, and the desired contact offset.

### 3.2 Policy action

For the primary implementation, use the action representation that best matches the current Action Bridge architecture:

```text
absolute Cartesian target position in R^2
```

The environment should track that target through one fixed low-level PD/impedance controller shared by every learned policy.

This makes the policy's internal `q` coordinate the target position and `p` the finite-difference target velocity, matching the existing contact-Langevin architecture. The physical point position and the target position must remain distinct in the code and diagnostics.

If the existing abstractions make direct acceleration commands substantially cleaner, Codex may implement that instead, but then the action/state adapters, theory-facing diagnostics, and all baselines must use the same action representation. Do not mix representations across models.

### 3.3 Contact model

Implement a numerically stable compliant unilateral contact model with:

- normal spring force under penetration;
- normal contact damping;
- configurable tangential friction;
- configurable end-effector mass;
- semi-implicit or otherwise stable integration.

The precise penalty/contact formula is left to Codex, but it must be:

- deterministic under a fixed seed;
- stable over the intended parameter ranges;
- differentiability is not required because training is offline BC;
- instrumented to report normal force, penetration, contact state, and frictional force.

### 3.4 Episode task

Each episode should require four phases:

1. approach the surface from free space;
2. establish stable contact;
3. move tangentially toward a context-dependent target;
4. stop at the target while maintaining contact.

The main version should use a randomly sampled tangential endpoint. A harder optional version may use a stop-go-reverse tangential profile.

A successful trajectory must not be defined only by final position. Define:

- **task success:** tangential goal reached within tolerance and low terminal speed;
- **contact success:** desired contact maintained near the end;
- **clean success:** task success with no excessive penetration, force spike, or repeated contact loss.

Keep these components separate in the metrics.

---

## 4. Context, observations, and hidden variables

Every policy must receive the same information.

The observation/context should contain enough information to solve the task, including:

- current physical position and velocity;
- surface normal and surface offset, or an equivalent contact-frame description;
- current signed distance to the surface;
- tangential goal coordinate;
- recent policy actions/targets according to the repository's history format;
- optionally the current normal contact force or contact flag, but only if supplied to all methods.

Do not expose randomized physical parameters such as mass, friction, or contact stiffness unless an explicit observed-parameter experiment is being run. For the principal robustness experiment, these should be latent environment variations.

Condition and normalize context consistently across all models.

---

## 5. Expert demonstration generation

Generate an offline dataset using a controller that reliably completes the task.

The expert may be:

- a scripted phase-based Cartesian impedance controller;
- a small MPC controller;
- or another deterministic closed-loop controller that Codex judges robust.

The expert should:

- approach smoothly;
- regulate contact without a large impact;
- track the tangential target;
- stop with low terminal speed;
- remain robust over the nominal training parameter range.

The expert must be independent of the learned Action Bridge decomposition. In particular, do not generate data by simply running the same learned-reference equations with the desired answer hard-coded.

A hand-designed trajectory/task cost may be used inside an MPC expert, but it must not be passed to the learned policies.

### 5.1 Dataset diversity

Randomize at least:

- initial position and velocity;
- tangential goal;
- surface orientation;
- surface location;
- modest nominal contact-physics parameters;
- episode timing or desired sliding speed when useful.

Support multiple dataset sizes so that low-data behavior can be measured.

### 5.2 Train/validation/test split

Split by episode context, not by overlapping windows from the same episode.

Create:

- an ID test split drawn from the training distribution;
- unseen surface-orientation splits;
- OOD contact-physics splits;
- disturbance evaluation scenarios that are never included in demonstrations.

### 5.3 Chunked training data

Window expert episodes into the repository's existing BC batch format. Include, directly or through `context`, all geometry required to reconstruct the surface frame and evaluate contact metrics.

Store enough physical state to perform closed-loop replay and diagnostics, even if the policy action remains an absolute target position.

---

## 6. Correct learning pipeline

The experiment is offline imitation learning.

### 6.1 Action Bridge training

Train the Action Bridge from demonstrations using the existing teacher-forced path objective:

```text
expert next-state/action fitting
+ path-KL / whitened control energy
+ existing optional rollout/reference regularizers
```

Do not add the evaluation task cost, force penalty, penetration penalty, or success reward to the learner's loss.

For the primary task, disable the latent variable unless it is required by the current implementation. This experiment is intended to isolate contact/dissipation rather than multimodal path commitment.

Start without tube perturbation training. Add a second robustness variant with local off-path/tube perturbations only after the main BC pipeline works, and document that this adds extra robustness supervision.

### 6.2 Diffusion Policy training

Train a conditional action-chunk diffusion policy on exactly the same demonstration windows.

Requirements:

- predict the same action representation;
- condition on the same history and task context;
- use the same chunk horizon and receding-horizon execution schedule;
- use standard diffusion BC training, not reward or contact-cost supervision;
- use a competent low-dimensional denoiser and standard training practices such as EMA when appropriate;
- select denoising steps and model capacity through validation rather than intentionally weakening the baseline.

Codex should choose a simple, appropriate denoiser for low-dimensional action chunks after inspecting the repository and available dependencies. Avoid unnecessary architecture complexity.

### 6.3 No privileged training signals

Both methods may use:

- demonstrations;
- shared normalization statistics;
- shared context and surface geometry;
- identical data augmentation when applicable.

Neither method may use evaluation-only OOD conditions or hand-designed task/contact costs during BC training.

---

## 7. Contact-frame Action Bridge reference

Extend the existing contact-Langevin reference with a contact-frame parameterization.

Let:

```text
P_N = n n^T
P_T = I - P_N
```

Use a damping matrix of the form

```text
Gamma = gamma_N * P_N + gamma_T * P_T
```

where `gamma_N` and `gamma_T` are learned nonnegative values.

Do **not** impose an ordering between them.

Use a normal-only potential based on signed target distance to the surface, for example a learned quadratic form around a learned contact offset:

```text
V_N(q) proportional to k_N * (d(q) - d_star)^2
```

The reference may learn:

- normal stiffness `k_N`;
- desired normal target offset `d_star`;
- normal damping `gamma_N`;
- tangential damping `gamma_T`;
- optional dependence on history and chunk phase.

The reference must not include a tangential attractor to the task goal. It may damp tangential motion, but the task-specific tangential progress must come from the learned residual.

The residual control remains a general two-dimensional control. Do not project it onto the tangent.

Return and log the decomposed quantities:

```text
gamma_N
gamma_T
k_N
d_star
normal reference force
tangential reference force
normal residual control
tangential residual control
control/reference norm ratio
```

Keep the reference positive/stable through the same type of bounded parameterization used elsewhere in the repository.

---

## 8. Required baselines and ablations

Implement the experiment in stages, but the final comparison should include enough baselines to identify the mechanism.

### Required for the first complete result

1. **Action Bridge, learned contact-frame damping + path KL**
2. **Action Bridge, same architecture but path-KL weight zero**
3. **Action Bridge, learned scalar/isotropic damping**
4. **Vanilla conditional Diffusion Policy on the same action chunks**
5. **Reference-only rollout** with residual control disabled

### Strong fairness baselines

6. **Contact-frame Diffusion Policy**
   - express observations and/or output chunks in the same local normal/tangent frame;
   - decode back to world coordinates for execution.

7. **Residual Diffusion Policy using the same reference**
   - diffusion predicts a residual/control sequence;
   - the common learned or frozen contact reference decodes it into the action chunk.

This is the strongest test of whether any gain is due to the reference rather than the non-diffusion generator. If implementing it in the first pass is too disruptive, make it a clearly marked second-stage deliverable rather than silently omitting it.

### Useful structural ablations

8. world-coordinate learned diagonal damping;
9. fully learned positive-semidefinite damping without fixed contact-frame eigenvectors;
10. direct/autoregressive BC with anisotropic smoothness regularization;
11. optional tube-training variants for both structured and baseline models.

Do not interpret a result until at least the isotropic, no-KL, reference-only, and vanilla diffusion comparisons are available.

---

## 9. Closed-loop evaluation

Use the same receding-horizon wrapper for all policies.

For the contact experiment, default to frequent replanning:

```text
n_exec = 1 or 2
```

Use a common chunk horizon, preferably consistent with the repository defaults unless validation shows that the toy needs a shorter horizon.

At every replan:

1. read the actual simulated state;
2. update observation and action histories;
3. generate an action-target chunk;
4. execute only the first `n_exec` targets through the shared low-level controller;
5. repeat until termination.

Do not propagate the model's internal state in place of the actual simulator state across replans.

Use deterministic evaluation as the primary result. If stochastic candidate sampling is reported, use the same number of samples and selection rule for all applicable methods.

---

## 10. Evaluation conditions

Implement at least the following evaluation suites.

### 10.1 ID evaluation

Same distribution of geometry and physics used for demonstrations.

Purpose: ensure the structured model does not sacrifice basic imitation accuracy.

### 10.2 Low-data evaluation

Train every method on several demonstration counts.

Purpose: test whether the geometric reference improves sample efficiency rather than expressivity.

### 10.3 Unseen surface orientations

Train on a sparse orientation set or range, evaluate on disjoint orientations.

All methods receive the true surface normal.

Purpose: test whether the contact-frame parameterization yields reusable equivariance.

### 10.4 Contact-physics shift

Vary outside the demonstration range:

- mass;
- contact stiffness;
- contact damping;
- friction;
- low-level controller gains.

Purpose: test robustness to unobserved dynamics.

### 10.5 Normal impulse recovery

During the sliding phase, apply a controlled impulse along the surface normal.

Sweep impulse magnitude and sign when appropriate.

Purpose: directly test recovery of normal stability while the tangential task continues.

### 10.6 Optional observation noise and delay

Add only after the core experiment is stable.

---

## 11. Metrics

Report task quality and contact stability separately.

### Task metrics

- tangential terminal error;
- tangential tracking RMSE if a profile is used;
- completion time;
- low terminal speed;
- task success rate.

### Contact metrics

- clean success rate;
- peak and RMS normal force;
- maximum penetration;
- contact-loss count and duration;
- normal velocity energy;
- number of normal-velocity sign changes/chatter events;
- recovery time after the impulse;
- post-impulse tangential-progress loss.

### Action/path metrics

- target velocity, acceleration, and jerk;
- chunk-boundary discontinuity;
- path-KL/control energy;
- normal and tangential residual-control energy;
- normal and tangential reference-force energy;
- ratio of residual to reference magnitude.

### Learned-reference diagnostics

- distributions and time profiles of `gamma_N`, `gamma_T`, `k_N`, and `d_star`;
- `gamma_N - gamma_T`, reported without assuming its sign;
- parameter behavior before contact, during contact, and during stopping;
- reference-only trajectory overlays.

### Primary summary plots

Produce at least:

1. task performance versus peak/RMS normal force;
2. success and clean success versus impulse magnitude;
3. recovery time versus impulse magnitude;
4. low-data learning curves;
5. unseen-orientation robustness curves;
6. normal distance, force, velocity, tangential position, and action target versus time for representative episodes;
7. learned normal/tangential damping over an episode;
8. reference force versus residual force decomposition.

Prefer Pareto plots and robustness curves over a single weighted score.

---

## 12. Model selection and fairness

Use validation data only for:

- early stopping;
- model checkpoint selection;
- diffusion sampling-step selection;
- loss-weight/hyperparameter selection;
- architecture-size selection.

Use the same number of random seeds for all primary methods.

Report:

- parameter counts;
- training updates;
- inference time per replan;
- diffusion denoising steps;
- number of demonstrations;
- low-level controller and simulator settings.

Do not tune only the Action Bridge extensively while leaving the diffusion baseline at one default.

---

## 13. Recommended experiment matrix

A compact first paper-quality matrix is:

```text
methods:
  - diffusion_world
  - diffusion_contact_frame
  - action_bridge_isotropic
  - action_bridge_contact_frame_no_kl
  - action_bridge_contact_frame_full
  - reference_only

train_sizes:
  - small
  - medium
  - full

eval_suites:
  - id
  - unseen_orientation
  - contact_physics_shift
  - normal_impulse_sweep

seeds:
  - at least 3 for development
  - increase for final reporting
```

Add residual diffusion as the next mandatory comparison before making a claim that the Action Bridge generator, rather than the reference, is responsible for the gain.

---

## 14. Tests

Add focused tests consistent with the repository's existing test style.

### Geometry tests

- `P_N + P_T = I`;
- projectors are symmetric and idempotent;
- normal and tangential projections are orthogonal;
- rotating the surface rotates the contact-frame damping matrix equivariantly.

### Reference tests

- learned damping/stiffness are in configured bounds;
- zero residual yields zero path-KL;
- the normal-only potential produces no tangential spring force;
- reference-only dynamics reduce normal target velocity/energy in a simple fixed-surface case;
- scalar/isotropic mode reproduces the existing behavior when configured equivalently.

### Environment tests

- no contact force in free space;
- nonnegative normal contact force under penetration;
- friction direction opposes tangential motion;
- integration is stable for configured parameter ranges;
- seeded rollouts are deterministic.

### Data tests

- chunk windows do not cross episode boundaries;
- context/geometry fields align with the trajectory;
- train/validation/test contexts do not leak;
- action and state normalization round-trips correctly.

### Model/baseline smoke tests

- Action Bridge and diffusion baseline can overfit a tiny dataset;
- each training entry point runs briefly on CPU;
- each checkpoint can be reloaded for closed-loop evaluation;
- all required metrics and figures are generated.

---

## 15. Outputs and repository integration

Follow the repository's existing output conventions. Every run should save:

```text
resolved config
latest and best checkpoints
training and validation metrics
closed-loop ID metrics
closed-loop OOD metrics
per-episode or aggregate robustness data
figures
representative rollout videos/GIFs when practical
```

Add:

- a dataset/environment visualizer;
- a training entry point or extension to the existing toy trainer;
- a standalone evaluation entry point;
- a compact sweep/comparison script;
- configs for all required methods;
- README documentation with exact commands;
- a short experiment note describing the hypothesis and the interpretation of every ablation.

Use existing modules and naming conventions where possible. Codex should inspect the codebase and decide whether a new environment/data module, a specialized adapter, or a modest extension to current modules is cleanest.

---

## 16. Implementation order

Implement in this order:

1. contact environment and deterministic simulator tests;
2. expert demonstration generator and dataset visualizations;
3. common closed-loop rollout/evaluation wrapper;
4. vanilla diffusion BC baseline;
5. contact-frame Action Bridge reference with learned `gamma_N`, `gamma_T`, `k_N`, and `d_star`;
6. core BC training and ID evaluation;
7. required ablations;
8. unseen-orientation and physics-shift evaluation;
9. normal-impulse sweep;
10. strong contact-frame and residual-diffusion baselines;
11. multi-seed sweep and final figures.

Do not begin with the hardest OOD sweeps before both primary policies can overfit a small dataset and complete ID closed-loop rollouts.

---

## 17. Acceptance criteria

### Code acceptance

- all existing tests still pass;
- new CPU smoke tests pass;
- data generation, Action Bridge training, diffusion training, and evaluation each run from documented commands;
- checkpoints are reloadable;
- no method uses privileged evaluation costs during BC training.

### Scientific acceptance

The implementation is complete when it can answer all of these questions with quantitative evidence:

1. Does learned contact-frame Action Bridge match ID tangential task performance?
2. Does it reduce normal-force/penetration/contact-loss metrics?
3. Does it degrade more slowly under normal impulses and unseen contact physics?
4. Does learned anisotropy emerge without enforcing `gamma_N > gamma_T`?
5. Does the reference contribute nontrivially, or is it overridden by the residual?
6. Does removing path KL remove part of the robustness benefit?
7. Does contact-frame diffusion close the gap?
8. Does residual diffusion with the same reference close the gap?
9. Are gains still present at equal model capacity, data, and replanning frequency?

A valid negative result is acceptable. The scripts and metrics should make the following conclusions distinguishable:

- the reference helps, but diffusion versus Action Bridge does not matter;
- the contact frame helps, but path KL does not;
- path KL improves OOD recovery;
- all methods perform similarly;
- the structured reference harms tangential task performance;
- the residual ignores the reference.

---

## 18. What not to overclaim in documentation

Do not write that diffusion policies fundamentally cannot handle contact.

Use the narrower framing:

> Vanilla action-chunk diffusion has no built-in contact-aligned passive dynamics. This experiment tests whether a learned, contact-frame dissipative reference improves data efficiency and perturbation robustness under otherwise matched behavior-cloning conditions.

If residual diffusion with the same reference matches Action Bridge, report that the useful contribution is the reference/process structure rather than the choice of non-diffusion generator.

If anisotropic smoothness or contact-frame BC matches the full model, report that the geometric inductive bias is sufficient and that the path-KL formulation did not provide a measurable extra benefit in this toy.
