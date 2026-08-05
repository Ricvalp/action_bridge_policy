# Action Bridge Policy Architecture Notes

Living notes for the policy architectures used in this sandbox. Add a new section for each architecture variant we try, and keep the implementation/config links current.

## 1. Latent Contact-Langevin Action Bridge

Current main Push-T architecture. Implementation entry point: `ActionBridgePolicy` in `action_bridge/models/action_bridge_policy.py`; reference process in `action_bridge/models/references.py`; loss in `action_bridge/training/losses.py`.

### Purpose

Generate a chunk of low-dimensional robot actions by deforming a learned reference action process with a learned control residual. The intended inductive bias is:

$$
\text{policy path} = \text{reference path} + \text{minimum-control deformation}.
$$

For Push-T, the action is the **absolute 2D pusher target position** in pixel coordinates after denormalization. The simulator then follows that target with its own PD controller; the policy does not directly output forces.

### Inputs And Outputs

- Observation history: `obs_hist` with shape `[B, obs_history, obs_dim]`.
- Action history: `act_hist` with shape `[B, action_history, action_dim]`.
- For Push-T lowdim state: `obs_dim=5`, usually `[pusher_x, pusher_y, block_x, block_y, block_theta]`.
- Action dimension: `action_dim=2`, absolute pusher target position.
- Chunk horizon: currently `H=16`.
- Output: a chunk `actions` with shape `[B, H, 2]`.

Push-T data is normalized with train-set mean/std for observations and actions. Plots and simulator execution denormalize actions back to pixels.

### History Encoder

The history encoder is an MLP:

$$
h = f_\phi(\mathrm{flatten}(o_{t-L_o:t}, a_{t-L_a:t-1})).
$$

Implementation: `HistoryEncoder`.

- Input dimension: `obs_history * obs_dim + action_history * action_dim`.
- Output: `h_emb` of size `h_emb_dim`.
- Activation: SiLU.
- Optional layer norm exists but is usually off.

Current big Push-T run:

- `obs_history=2`
- `action_history=2`
- `hidden_dim=2048`
- `h_emb_dim=2048`
- `encoder_depth=4`

### Latent Variable

The current Push-T bridge uses a continuous latent:

$$
z \sim p_\theta(z \mid h), \qquad q_\psi(z \mid h, a_{t:t+H-1})
$$

Implementation: `ContinuousLatent`.

- Prior: learned diagonal Gaussian from `h`.
- Posterior: learned diagonal Gaussian from `[h, future_action_embedding]`.
- Future-action encoder: MLP over the flattened expert chunk.
- Reparameterized samples during training.
- At inference, sample from the prior unless using sticky/episode latent commitment.
- Latent embedding: MLP maps `z` to `z_emb`.

Current big Push-T run:

- `z_dim=4`
- `z_embed_dim=128`
- prior type: `learned_conditional_gaussian`

Known empirical issue: later checkpoints can keep broad prior std while the action generator becomes nearly insensitive to `z`. We track this with the wrong-side go-around diagnostic.

### Contact-Langevin Reference Process

The contact reference lives in a coordinate space `q` and velocity-like state `p`.

For Push-T contact runs:

- `reference.type=contact_langevin`
- `reference.coordinate_mode=absolute_action`
- `q` is the absolute pusher target position.
- `p = q_k - q_{k-1}` with `dt=1`.

The reference dynamics are:

$$
p_{k+1} = p_k + dt \, f_R(q_k, p_k, h, k)
$$

$$
q_{k+1} = q_k + dt \, p_{k+1}
$$

For the quadratic contact reference:

$$
V(q,h,k)=\frac{1}{2}\sum_i K_i(h,k)(q_i - m_i(h,k))^2
$$

$$
f_R(q,p,h,k) = -\nabla_q V(q,h,k) - \gamma(h,k)p.
$$

Learned reference components:

- `m(h,k)`: attractor point in action space.
- `K(h,k)`: diagonal stiffness, bounded by `[k_min, k_max]`.
- `gamma(h,k)`: damping, currently learned scalar in the Push-T contact quadratic config.

Current Push-T contact quadratic config:

- `potential_type=quadratic`
- `attractor_mode=learned`
- `stiffness_mode=learned_diag`
- `gamma_mode=learned_scalar`
- `k_min=0.0`, `k_max=2.0`
- `gamma_min=0.0`, `gamma_max=0.95`
- `sigma=7.0`
- `beta_kl=0.001`
- `control_is_whitened=true`

### Learned Control Residual

The control network predicts a residual acceleration/control:

$$
u_\theta(q_k,p_k,h,k,z).
$$

Implementation: `ContactControlNet`.

Inputs:

- current `q`
- current `p`
- history embedding `h`
- sinusoidal chunk-step embedding for `k`
- latent embedding `z_emb`

Output:

- `u` in action-coordinate dimension.
- If `control_is_whitened=true`, the applied acceleration residual is `sigma * u`.

Controlled dynamics:

$$
p_{k+1} = p_k + dt \left(f_R(q_k,p_k,h,k) + \sigma u_\theta(q_k,p_k,h,k,z)\right)
$$

$$
q_{k+1} = q_k + dt \, p_{k+1}.
$$

Current big Push-T run:

- `control_depth=6`
- `time_emb_dim=64`
- `control_scale=1.0`

### Training Objective

Training uses a variational teacher-forced path loss.

For contact-Langevin models, the expert action chunk is converted to `q_seq` and `p_seq`. At each step, the model starts from the expert `q_k,p_k` and predicts `q_{k+1},p_{k+1}`.

Main terms:

- `loss_p`: normalized squared error on predicted next momentum/velocity-like state.
- `loss_q`: normalized squared error on predicted next position/action state.
- `path_kl`: control energy, usually `0.5 * dt * ||u||^2` when control is whitened.
- `latent_kl`: KL from posterior `q(z|h,future)` to prior `p(z|h)`.
- `latent_kl_loss`: latent KL after optional `free_nats` floor.
- `unroll_mse`: free-running chunk MSE from the same history, warmed up by `lambda_unroll`.
- `reference_reg`: small L2 penalty on learned `gamma` and `K`.
- `m_smooth_loss`: smoothness penalty on the learned attractor path `m(h,k)`.

For the continuous latent bridge:

$$
\mathcal{L}
= \mathbb{E}_{z \sim q(z|h,a_{future})}
[\text{loss}_p + \lambda_q \text{loss}_q + \beta_{KL}\text{path\_kl}]
+ \lambda_{unroll}\text{unroll\_mse}
+ \beta_z KL(q(z|h,a_{future})||p(z|h))
+ \lambda_{ref}\text{ref\_reg}
+ \lambda_m\text{m\_smooth}.
$$

Current big Push-T run:

- `lambda_unroll=1.0`, warmup `5000`
- `beta_z_start=0.001`
- `beta_z_end=0.01`
- `beta_z_warmup_steps=5000`
- `free_nats=0.05`
- `num_z_samples_train=1`
- `batch_size=512`
- `max_steps=300000`

### Inference And Closed Loop

At inference:

1. Encode current observation/action history.
2. Sample `z` from the prior.
3. Roll out a 16-step action chunk.
4. In receding-horizon mode, execute only the first `n_exec` actions, then replan.

Current Push-T settings:

- `inference.deterministic=true` for dynamics given `z`.
- `inference.latent_commitment=episode` in the base Push-T config.
- `inference.n_exec=8`.

In the simulator, each action is an absolute target position for the pusher. The environment follows that command through a PD controller, so commanded action points and realized pusher positions can differ.

### Current Big Push-T Instantiation

Command-level overrides used in the current rerun:

- `model.hidden_dim=2048`
- `model.h_emb_dim=2048`
- `model.encoder_depth=4`
- `model.control_depth=6`
- `model.time_emb_dim=64`
- `model.z_embed_dim=128`
- `model.z_dim=4`
- `optim.batch_size=512`

Approximate parameter count with Push-T state observations (`obs_dim=5`):

- Total parameters: `51,963,033`
- Trainable parameters: `51,963,031`
- History encoder: `12,619,776`
- Reference process: `799,751` total, `799,749` trainable
- Latent module: `17,156,240`
- Control network: `21,387,266`

### Diagnostics To Track

- Open-loop action MSE in raw pixels.
- Offline receding-horizon action MSE.
- Simulator `sim_success_rate`, `sim_max_reward`, `sim_final_reward`.
- `path_kl`, `control_energy`, `reference_reg`.
- `gamma_mean`, `k_diag_mean`, `k_diag_max`.
- Prior/posterior entropy and latent KL.
- `wrong_side_go_around_latent_chunks.png` for whether `z` controls left/right go-around behavior.
- `pusht_sim_contact_reference.png` for learned potential, reference path, planned chunk, and realized trajectory during simulator replanning.

### Open Questions

- Does the latent remain causally used after long training, or does the policy collapse to a deterministic controller?
- Are the learned attractor `m`, stiffness `K`, and damping `gamma` doing useful work, or is the control residual overriding the reference?
- Is the contact reference best expressed in absolute pusher target space, or should future variants use a more task-aware/object-centered coordinate?

## 2. JAX RLBench XYZ Contact Action Bridge

Query-only, state-conditioned RLBench policy. This is an independent JAX/Flax
implementation under `action_bridge/jax/`; it does not import from the ICIL
repository and contains no support demonstrations or meta-learning machinery.

### Inputs And Outputs

The cache supplies:

- RGB point-cloud history: `[B, T_obs, N, 3]` XYZ and optional RGB;
- low-dimensional gripper state history: `[B, T_obs, 8]`;
- executed absolute action history: `[B, T_act, 8]`;
- expert future chunk: `[B, H, 8]` during training only.

An RLBench action is the absolute target
`[x, y, z, qx, qy, qz, qw, gripper_open]`. The initial implementation uses
Euclidean contact dynamics only for normalized XYZ. Quaternion and gripper
channels use separate learned heads.

### Query Frontend

Implementation: `action_bridge/jax/models/rlbench_encoder.py`.

Each point-cloud frame is tokenized by either:

- `SupernodeFrameTokenizer`: deterministic center selection, soft spatial
  assignment, weighted point-feature pooling, and self-attention; or
- `PerceiverFrameTokenizer`: point/state tokens compressed by learned latents.

The tokenizer is adapted from the tested JAX RLBench frontend, but only the
query path is retained. Frame-position embeddings are added to visual tokens.
Absolute action-history vectors are projected into tokens with their own
position embeddings. Optional learned task and task-variation tokens are then
concatenated. A Perceiver compressor produces a bounded context-token set.

The decoder starts from `H` learned action-query vectors. Every layer applies
self-attention among chunk steps and cross-attention from those queries to the
context tokens. The output is a per-step context `c_k`. Thus chunk position is
represented by the learned action query itself; a bare time embedding is not
used as the cross-attention query.

### Continuous Chunk Latent

The bridge uses one latent for the complete chunk:

$$
p_\theta(z\mid h)=\mathcal{N}(\mu_p(h),\operatorname{diag}\sigma_p^2(h)),
$$

$$
q_\phi(z\mid h,a_{1:H})=
\mathcal{N}(\mu_q(h,a_{1:H}),\operatorname{diag}\sigma_q^2(h,a_{1:H})).
$$

Training samples from the posterior; deployment samples from the prior. The
default latent dimension is four and the same sampled `z` conditions every
step in the chunk.

### XYZ Contact Reference And Control

Raw XYZ is normalized with configured workspace center and scale. Initial
position comes from the most recent executed absolute action. Initial momentum
is the difference between the two most recent valid actions, divided by `dt`.

For each action query `c_k`, the reference head predicts:

- normalized attractor `m_k in [-1, 1]^3`;
- positive diagonal stiffness `K_k in [k_min, k_max]^3`;
- positive diagonal damping `gamma_k in [gamma_min, gamma_max]`.

The passive reference force is

$$
f_R(q_k,p_k,c_k)=-K_k(q_k-m_k)-\gamma_k p_k.
$$

The residual-control MLP sees `(c_k, q_k, p_k, z)` and predicts whitened
control `u_k in R^3`. The semi-implicit update is

$$
p_{k+1}=p_k+dt\,[f_R(q_k,p_k,c_k)+\sigma u_k],
$$

$$
q_{k+1}=q_k+dt\,p_{k+1}.
$$

The reference is history/state conditioned through `c_k`; the control also
receives the current rollout state and chunk latent directly.

### Quaternion And Gripper Heads

A shared auxiliary MLP receives `(c_k, q_{k+1}, p_{k+1}, z)` and emits four
quaternion values plus a gripper logit. The quaternion is normalized and its
sign is canonicalized relative to the current observed gripper quaternion.
The gripper logit is trained with binary cross-entropy and converted with a
sigmoid for execution.

### Training Objective

The contact model reports both teacher-forced one-step predictions and a free
chunk rollout. The configurable objective combines:

- normalized XYZ one-step error;
- momentum/velocity error;
- free-unroll XYZ error;
- sign-invariant quaternion loss `1 - <q_hat, q>^2`;
- gripper binary cross-entropy;
- whitened path-control energy `0.5 ||u_k||^2`;
- continuous-latent KL with warmup and free nats.

Validation logs posterior reconstruction and prior-conditioned deployment
metrics separately. Diagnostics plot the RGB point cloud, expert chunk, free
rollout, teacher-forced prediction, and learned attractor path in 3D.

### Direct Chunk Baseline

`DirectChunkBCPolicy` uses the identical history encoder and learned
action-query decoder. A feed-forward head directly predicts all eight action
channels for each query. It therefore controls for the expensive point-cloud
frontend while removing the latent, contact reference, and path-control loss.

With the default 256-wide frontend, four decoder layers, 512-wide dynamics
heads, 105 task IDs, and 393 task-variation IDs, the current implementations
contain approximately 13.1M parameters for the contact bridge and 11.6M for
direct chunk BC.

### Entry Points

- Contact bridge config: `rlbench_jax_contact_bridge`
- Direct baseline config: `rlbench_jax_direct_chunk_bc`
- Training: `python -m action_bridge.jax.training.train_rlbench`
- Cached-window evaluation: `python -m action_bridge.jax.eval.eval_rlbench`

JAX dependencies are optional: use `uv sync --extra jax-cpu` on macOS/CPU or
`uv sync --extra jax-cu13` on a CUDA 13 H200 node. Online RLBench/PyRep
closed-loop evaluation is not connected to this JAX policy yet.

## 2. Non-Contact Latent Action Bridge

Earlier toy and Push-T bridge variant using an autoregressive reference over raw actions rather than contact-Langevin state.

Implementation: same `ActionBridgePolicy`, but `reference.type` is typically `continuation`, `brownian`, `low_acceleration`, or `low_jerk`.

### Dynamics

The reference predicts a next-action mean:

$$
\mu_R(a_k,a_{k-1},h,k).
$$

The control network predicts:

$$
u_\theta(a_k,a_{k-1},h,k,z).
$$

The final conditional action mean is:

$$
\mu = \mu_R + u_\theta.
$$

Reference options:

- Brownian: `mu_R = a_k`.
- Continuation: `mu_R = a_k + alpha(a_k-a_{k-1})`.
- Low acceleration: continuation with learned/context-dependent `alpha(h,k)`.
- Low jerk: second-order continuation when an older action is available.

### Training

Teacher-forced over expert future actions. For each chunk step `k`, the previous actions are taken from `act_hist` and the expert future prefix. Loss terms:

- Gaussian NLL around `mu`.
- path KL/control energy `0.5 * ||u / sigma||^2`.
- latent KL between posterior and prior.
- optional unrolled/free-running MSE.
- optional tube noise on teacher-forced previous actions.

This model is useful as a simpler testbed for whether a reference path plus residual control helps before introducing second-order/contact dynamics.

## 3. Direct Chunk Behavior Cloning Baseline

Implementation: `DirectChunkBCPolicy` in `action_bridge/models/baselines.py`.

### Policy

Encode history with the same `HistoryEncoder`, then predict the entire action chunk in one MLP head:

$$
\hat{a}_{t:t+H-1}=g_\theta(h).
$$

No latent, no reference process, no path KL. This baseline tests whether the bridge objective helps beyond a high-capacity direct chunk regressor.

### Loss

Mean squared error to the expert future action chunk, with optional smoothing depending on config.

## 4. Autoregressive Behavior Cloning Baseline

Implementation: `AutoregressiveBCPolicy` in `action_bridge/models/baselines.py`.

### Policy

Encode history once, then predict each action step autoregressively:

$$
\hat{a}_k=g_\theta(\hat{a}_{k-1},\hat{a}_{k-2},h,k).
$$

Inputs per step:

- previous two actions
- history embedding
- sinusoidal time embedding

No latent, no reference process, no explicit path KL. This baseline tests whether sequential generation and previous-action conditioning are enough without minimum-control deformation.

### Loss

Mean squared error to the expert future action chunk under the BC loss.

## Update Log

- 2026-07-14: Initial architecture catalog. Added current latent contact-Langevin Push-T bridge, non-contact bridge, direct chunk BC, and autoregressive BC.
