# Implementation Prompt for Codex: Contact-Hamiltonian / Underdamped Langevin Reference Process for Action-Path Policies

## Goal

Implement a new reference-process option for the action-path policy:

```text
policy = dissipative reference process + learned control residual
```

The reference should be a discrete underdamped Langevin / noisy contact-Hamiltonian process over an internal coordinate `q` and its velocity `p`:

```math
dq_t = p_t dt
```

```math
dp_t = [-\nabla_q V_\eta(q_t,h,t) - \Gamma_\eta(h,t)p_t]dt + \sigma dW_t
```

The learned policy adds a control in the same channel as the noise:

```math
dp_t = [-\nabla_q V_\eta(q_t,h,t) - \Gamma_\eta(h,t)p_t + \sigma u_\theta(q_t,p_t,h,t)]dt + \sigma dW_t.
```

When reference and policy share the same diffusion `sigma`, the path-KL term is the simple squared-control cost:

```math
D_{KL}(P_{\eta,\theta} || R_\eta)
= \frac{1}{2}\mathbb E\sum_k \Delta t\,\|u_k\|^2.
```

The implementation should support both:

1. **Raw action coordinates**: define the reference directly in the action coordinates used by the dataset.
2. **Absolute-position coordinates**: internally convert delta actions into absolute target positions and define the reference over those absolute positions.

This is especially needed for the toy obstacle-avoidance experiments, where the dataset may use delta actions but we also want an option where actions are absolute target positions.

---

## Conceptual distinction

Do not assume that the dataset action tensor is always the physically meaningful coordinate.

Use the following terminology:

```text
raw action:     the action stored in the dataset or sent to the environment
q-coordinate:   the coordinate in which the reference process is defined
p-coordinate:   discrete velocity of q
```

For absolute-position actions:

```math
q_k = a_k
```

For delta actions, there are two valid choices.

### Choice A: raw-delta reference

Use:

```math
q_k = a_k = \Delta x_k.
```

Then the reference damps changes in the delta-command signal. This is mathematically valid, but it is command-space damping, not physical end-effector damping.

### Choice B: absolute-position reference from deltas

Use:

```math
q_0 = x_t
```

```math
q_{k+1} = q_k + a_{t+k}
```

where `a_{t+k}` is a delta action.

Then the reference is over the integrated end-effector target path. This is the preferred option when obstacles, contacts, or goals live in absolute/world coordinates.

At inference, generate `q_{k+1}` internally, then decode the raw delta action as:

```math
a_{t+k} = q_{k+1} - q_k.
```

---

## Required config fields

Add or extend a config block like this:

```yaml
reference:
  type: contact_langevin       # choices: brownian, continuation, contact_langevin

  # Coordinate convention.
  # raw_action: use dataset actions directly as q
  # absolute_action: dataset actions are absolute target positions
  # absolute_from_delta: dataset actions are deltas, but q is reconstructed as cumulative absolute position
  coordinate_mode: raw_action

  dt: 1.0

  # Langevin/contact reference.
  sigma: 0.05                  # scalar or per-action-dim diagonal std in q-velocity space
  control_is_whitened: true     # if true, acceleration correction is sigma * u

  # Damping.
  gamma_mode: constant          # constant | learned_scalar | learned_diag
  gamma_const: 0.2
  gamma_min: 0.0
  gamma_max: 0.95              # keep stable for dt=1.0

  # Potential.
  potential_type: none          # none | quadratic
  stiffness_mode: learned_diag  # only used when potential_type=quadratic
  k_min: 0.0
  k_max: 2.0
  attractor_mode: learned       # learned | previous_q | zero

  # Loss weights.
  beta_kl: 1.0
  lambda_q: 1.0
  lambda_ref_reg: 1e-4
  lambda_m_smooth: 1e-3

  # Inference.
  deterministic_inference: true
```

Recommended staged configs:

```yaml
# Stage 1: fixed damped continuation
reference:
  type: contact_langevin
  potential_type: none
  gamma_mode: constant
  gamma_const: 0.2
  beta_kl: 1.0
```

```yaml
# Stage 2: learned damping
reference:
  type: contact_langevin
  potential_type: none
  gamma_mode: learned_scalar
  gamma_min: 0.0
  gamma_max: 0.95
  beta_kl: 1.0
```

```yaml
# Stage 3: learned spring-damper reference
reference:
  type: contact_langevin
  potential_type: quadratic
  gamma_mode: learned_scalar
  stiffness_mode: learned_diag
  attractor_mode: learned
  beta_kl: 1.0
```

---

## Indexing convention

Use a pre-action coordinate `q_0` and generate `H` future raw actions.

For each training sample at environment time `t`:

```text
q_seq shape: [B, H+1, q_dim]
p_seq shape: [B, H+1, q_dim]
```

The transition at model step `k` predicts:

```text
(q_k, p_k) -> (q_{k+1}, p_{k+1})
```

for `k = 0, ..., H-1`.

The raw action at future step `k` is decoded from `(q_k, q_{k+1})`.

### Absolute action dataset

If the dataset action is an absolute target position:

```math
a_t = x^{target}_t
```

then define:

```math
q_0 = a_{t-1}
```

```math
q_{k+1} = a_{t+k}
```

The decoded raw action is:

```math
a_{t+k} = q_{k+1}.
```

The previous coordinate for velocity initialization is:

```math
q_{-1} = a_{t-2}.
```

Then:

```math
p_0 = \frac{q_0 - q_{-1}}{\Delta t}.
```

### Delta action dataset, raw-delta reference

If the dataset action is a delta:

```math
a_t = \Delta x_t
```

and `coordinate_mode=raw_action`, define:

```math
q_0 = a_{t-1}
```

```math
q_{k+1} = a_{t+k}.
```

The decoded raw action is:

```math
a_{t+k} = q_{k+1}.
```

This models smoothness of the delta-command sequence.

### Delta action dataset, absolute-position reference

If the dataset action is a delta and `coordinate_mode=absolute_from_delta`, define:

```math
q_0 = x_t
```

where `x_t` is the current absolute end-effector/agent position at the start of the chunk.

Then:

```math
q_{k+1} = q_k + a_{t+k}.
```

The decoded raw action is:

```math
a_{t+k} = q_{k+1} - q_k.
```

This is the preferred mode for obstacle-avoidance toy experiments when obstacles are defined in absolute space.

If the toy dataset does not store absolute positions but stores a trajectory of deltas, reconstruct chunk-local positions by setting:

```math
q_0 = 0
```

and cumulative summing deltas. This is valid for local path-shape experiments but not for absolute obstacle/contact potentials unless the global start position is also available.

---

## Implement `ActionCoordinateAdapter`

Create a module/class responsible for coordinate conversion.

Suggested interface:

```python
class ActionCoordinateAdapter:
    def __init__(self, coordinate_mode: str, dt: float, action_dim: int):
        self.coordinate_mode = coordinate_mode
        self.dt = dt
        self.action_dim = action_dim

    def build_q_sequence(self, batch) -> torch.Tensor:
        """Return q_seq with shape [B, H+1, A]."""
        raise NotImplementedError

    def build_p_sequence(self, q_seq: torch.Tensor, batch) -> torch.Tensor:
        """Return p_seq with shape [B, H+1, A]."""
        raise NotImplementedError

    def init_qp_from_history(self, batch):
        """Return q0, p0 for inference."""
        raise NotImplementedError

    def decode_raw_actions(self, q_seq: torch.Tensor) -> torch.Tensor:
        """Return raw action sequence with shape [B, H, A]."""
        raise NotImplementedError
```

Expected batch fields:

```python
batch["actions"]              # [B, H, A], future raw actions from dataset
batch["prev_actions"]         # [B, 2, A], raw actions at t-2 and t-1 when available
batch["current_position"]      # [B, A], required for absolute_from_delta if using physical coords
batch["previous_position"]     # [B, A], optional, for p0 if available
```

### Adapter behavior: `absolute_action`

```python
def build_q_sequence_absolute_action(actions, prev_actions):
    # actions: [B, H, A] = absolute target positions a_t ... a_{t+H-1}
    # prev_actions[:, -1] = a_{t-1}
    q0 = prev_actions[:, -1:]          # [B, 1, A]
    q_future = actions                 # [B, H, A]
    q_seq = torch.cat([q0, q_future], dim=1)
    return q_seq
```

For `p_seq`, use finite differences:

```python
def build_p_sequence_from_q(q_seq, q_minus_1, dt):
    # q_minus_1: [B, A]
    q_prev = torch.cat([q_minus_1[:, None], q_seq[:, :-1]], dim=1)
    p_seq = (q_seq - q_prev) / dt
    return p_seq
```

Decode:

```python
def decode_absolute_action(q_seq):
    return q_seq[:, 1:]                # [B, H, A]
```

### Adapter behavior: `raw_action`

```python
def build_q_sequence_raw_action(actions, prev_actions):
    q0 = prev_actions[:, -1:]          # previous raw action
    q_future = actions                 # future raw actions
    return torch.cat([q0, q_future], dim=1)
```

Decode:

```python
def decode_raw_action(q_seq):
    return q_seq[:, 1:]
```

### Adapter behavior: `absolute_from_delta`

```python
def build_q_sequence_absolute_from_delta(actions, current_position):
    # actions: [B, H, A] delta actions
    # current_position: [B, A]
    q0 = current_position[:, None]                 # [B, 1, A]
    q_future = q0 + torch.cumsum(actions, dim=1)   # [B, H, A]
    return torch.cat([q0, q_future], dim=1)
```

Decode:

```python
def decode_delta_from_absolute(q_seq):
    return q_seq[:, 1:] - q_seq[:, :-1]
```

For `p_seq`, either use finite differences of `q_seq`, or use previous position if available:

```python
q_minus_1 = batch.get("previous_position", None)
if q_minus_1 is None:
    # fallback: q_minus_1 = q0 - previous_delta
    previous_delta = batch["prev_actions"][:, -1]
    q_minus_1 = q_seq[:, 0] - previous_delta
p_seq = build_p_sequence_from_q(q_seq, q_minus_1, dt)
```

---

## Absolute-position actions in toy obstacle experiments

Add a toy-dataset option:

```yaml
toy:
  action_representation: delta        # existing default
  train_absolute_actions: false       # new option
  env_accepts_absolute_actions: false # if false, convert absolute target to delta before env.step
```

When `train_absolute_actions=true`, convert each trajectory of positions into absolute target actions.

Assume a trajectory has positions:

```python
positions: [T+1, A]     # x_0, ..., x_T
```

Existing delta actions are:

```python
deltas[t] = positions[t+1] - positions[t]
```

Absolute target actions should be:

```python
abs_actions[t] = positions[t+1]
```

For a chunk starting at `t`:

```python
q0 = positions[t]
q_future = positions[t+1:t+H+1]
a_abs = q_future
```

At inference:

```python
# policy generates absolute targets q_1, ..., q_H
if env_accepts_absolute_actions:
    env_action = q_1
else:
    env_action = q_1 - current_position
```

For multi-step execution, update the current position after each step and convert each absolute target into a delta:

```python
for q_next in q_seq[:, 1:]:
    delta = q_next - current_position
    obs, reward, done, info = env.step(delta)
    current_position = q_next  # or read it from obs if env dynamics are not identity
```

Prefer reading `current_position` from the environment observation after each step if the environment has dynamics/noise.

For obstacle potentials or mode labels, use `q_seq` in world coordinates, not the raw delta actions.

---

## Contact-Langevin reference module

Implement a module that returns reference force:

```math
f_{ref}(q,p,h,k) = -\nabla_q V_\eta(q,h,k) - \Gamma_\eta(h,k)p.
```

Start with a quadratic potential:

```math
V_\eta(q,h,k) = \frac{1}{2}(q-m_\eta(h,k))^T K_\eta(h,k)(q-m_\eta(h,k)).
```

Use diagonal positive stiffness:

```math
K_\eta(h,k) = \mathrm{diag}(k_\eta(h,k)), \quad k_\eta \ge 0.
```

Then:

```math
\nabla_q V_\eta(q,h,k) = K_\eta(h,k)(q-m_\eta(h,k)).
```

Implement bounded positive parameters:

```python
def bounded_positive(raw, min_val, max_val):
    return min_val + (max_val - min_val) * torch.sigmoid(raw)
```

Suggested implementation:

```python
class ContactLangevinReference(nn.Module):
    def __init__(self, h_dim, q_dim, cfg):
        super().__init__()
        self.q_dim = q_dim
        self.cfg = cfg

        self.use_potential = cfg.potential_type == "quadratic"
        self.gamma_mode = cfg.gamma_mode

        # Input is encoded history + time embedding.
        in_dim = h_dim + cfg.time_embed_dim
        hidden = cfg.hidden_dim

        if self.use_potential:
            self.m_net = MLP(in_dim, q_dim, hidden=hidden)
            self.k_net = MLP(in_dim, q_dim, hidden=hidden)

        if self.gamma_mode == "learned_scalar":
            self.gamma_net = MLP(in_dim, 1, hidden=hidden)
        elif self.gamma_mode == "learned_diag":
            self.gamma_net = MLP(in_dim, q_dim, hidden=hidden)

    def params(self, h, t_emb):
        hk = torch.cat([h, t_emb], dim=-1)

        if self.use_potential:
            m = self.m_net(hk)
            k_raw = self.k_net(hk)
            k_diag = bounded_positive(k_raw, self.cfg.k_min, self.cfg.k_max)
        else:
            m = None
            k_diag = None

        if self.gamma_mode == "constant":
            gamma = torch.full(
                (*h.shape[:-1], 1),
                self.cfg.gamma_const,
                device=h.device,
                dtype=h.dtype,
            )
        else:
            gamma_raw = self.gamma_net(hk)
            gamma = bounded_positive(gamma_raw, self.cfg.gamma_min, self.cfg.gamma_max)

        return m, k_diag, gamma

    def force(self, q, p, h, t_emb):
        m, k_diag, gamma = self.params(h, t_emb)

        if self.use_potential:
            grad_v = k_diag * (q - m)
        else:
            grad_v = torch.zeros_like(q)

        if gamma.shape[-1] == 1:
            damping = gamma * p
        else:
            damping = gamma * p

        f_ref = -grad_v - damping
        aux = {"m": m, "k_diag": k_diag, "gamma": gamma, "grad_v": grad_v}
        return f_ref, aux
```

Important: do not implement a fully arbitrary scalar `V(q,h,k)` in the first version. A scalar neural potential requires differentiating through `grad_q V`, which makes training heavier and introduces second derivatives when optimizing parameters. The quadratic potential is enough for the pilot.

---

## Control network

The control network outputs a whitened acceleration control `u`.

```python
class ControlResidual(nn.Module):
    def __init__(self, h_dim, q_dim, cfg):
        super().__init__()
        in_dim = h_dim + 2 * q_dim + cfg.time_embed_dim
        if cfg.use_latent_mode:
            in_dim += cfg.z_dim
        self.net = MLP(in_dim, q_dim, hidden=cfg.hidden_dim)

    def forward(self, q, p, h, t_emb, z=None):
        xs = [q, p, h, t_emb]
        if z is not None:
            xs.append(z)
        return self.net(torch.cat(xs, dim=-1))
```

If `control_is_whitened=true`, acceleration correction is:

```python
control_accel = sigma * u
```

and the KL term is:

```python
kl = 0.5 * dt * (u ** 2).sum(dim=-1)
```

If `control_is_whitened=false`, control output is acceleration `c` directly, and KL should be:

```python
kl = 0.5 * dt * (c / sigma).pow(2).sum(dim=-1)
```

Use whitened control by default.

---

## One-step transition

Use semi-implicit Euler. This is more stable than explicit Euler for damped second-order dynamics.

```python
def contact_langevin_step(q, p, h, t_emb, ref, ctrl, dt, sigma, z=None, noise=None):
    """
    q:     [B, A]
    p:     [B, A]
    h:     [B, Hdim]
    t_emb: [B, Tdim]
    sigma: scalar or [A]
    noise: None or [B, A]
    """
    f_ref, aux = ref.force(q, p, h, t_emb)
    u = ctrl(q, p, h, t_emb, z=z)

    control_accel = sigma * u

    if noise is None:
        noise_term = 0.0
    else:
        noise_term = (dt ** 0.5) * sigma * noise

    p_next = p + dt * (f_ref + control_accel) + noise_term
    q_next = q + dt * p_next

    return q_next, p_next, u, aux
```

During training with teacher forcing, input the ground-truth `(q_k, p_k)` and predict `(q_{k+1}, p_{k+1})`.

During rollout/inference, feed the predicted `(q_{k+1}, p_{k+1})` into the next step.

---

## Training loss

Given demonstration sequences `q_seq`, `p_seq`:

```text
q_seq[:, k]     = q_k
p_seq[:, k]     = p_k
q_seq[:, k+1]   = target q_{k+1}
p_seq[:, k+1]   = target p_{k+1}
```

Compute predicted means with teacher forcing:

```python
q_pred, p_pred, u, aux = contact_langevin_step(
    q=q_seq[:, k],
    p=p_seq[:, k],
    h=h,
    t_emb=t_emb_k,
    ref=ref,
    ctrl=ctrl,
    dt=dt,
    sigma=sigma,
    z=z,
    noise=None,
)
```

Use:

```python
loss_p = 0.5 * ((p_seq[:, k+1] - p_pred) / sigma_p).pow(2).sum(dim=-1).mean()
loss_q = 0.5 * ((q_seq[:, k+1] - q_pred) / sigma_q).pow(2).sum(dim=-1).mean()
loss_kl = 0.5 * dt * (u ** 2).sum(dim=-1).mean()
```

Total:

```python
loss = loss_p + lambda_q * loss_q + beta_kl * loss_kl + loss_ref_reg
```

Reference regularization:

```python
loss_ref_reg = 0.0
if aux["k_diag"] is not None:
    loss_ref_reg += lambda_ref_reg * aux["k_diag"].pow(2).mean()
if aux["gamma"] is not None:
    loss_ref_reg += lambda_ref_reg * aux["gamma"].pow(2).mean()
```

If using learned attractor `m(h,k)`, add a smoothness loss across future steps:

```python
m_values = []
for k in range(H):
    _, aux = ref.force(q_seq[:, k], p_seq[:, k], h, t_emb_k)
    if aux["m"] is not None:
        m_values.append(aux["m"])

if len(m_values) > 1:
    m_seq = torch.stack(m_values, dim=1)  # [B, H, A]
    loss_m_smooth = ((m_seq[:, 1:] - m_seq[:, :-1]) ** 2).mean()
    loss += lambda_m_smooth * loss_m_smooth
```

---

## Optional off-path tube training

After teacher-forced training works, add perturbations to the input state but keep the target unchanged.

```python
q_in = q_seq[:, k] + noise_q_std * torch.randn_like(q_seq[:, k])
p_in = p_seq[:, k] + noise_p_std * torch.randn_like(p_seq[:, k])
```

Then predict the same targets:

```python
q_target = q_seq[:, k+1]
p_target = p_seq[:, k+1]
```

This trains local recovery around the demonstration path.

Do not enable this until the deterministic teacher-forced version is stable.

---

## Inference rollout

```python
def rollout_contact_policy(batch, obs_encoder, adapter, ref, ctrl, H, cfg, z=None):
    h = obs_encoder(batch)
    q, p = adapter.init_qp_from_history(batch)

    q_list = [q]
    p_list = [p]
    u_list = []

    for k in range(H):
        t_emb = make_time_embedding(k, batch_size=q.shape[0], device=q.device)

        if cfg.reference.deterministic_inference:
            noise = None
        else:
            noise = torch.randn_like(q)

        q, p, u, aux = contact_langevin_step(
            q=q,
            p=p,
            h=h,
            t_emb=t_emb,
            ref=ref,
            ctrl=ctrl,
            dt=cfg.reference.dt,
            sigma=cfg.reference.sigma,
            z=z,
            noise=noise,
        )

        q_list.append(q)
        p_list.append(p)
        u_list.append(u)

    q_seq = torch.stack(q_list, dim=1)  # [B, H+1, A]
    raw_actions = adapter.decode_raw_actions(q_seq)
    return raw_actions, {"q_seq": q_seq, "p_seq": torch.stack(p_list, dim=1), "u_seq": torch.stack(u_list, dim=1)}
```

Initially use deterministic inference. Sampling can be added later for candidate generation.

---

## Relation to existing Brownian and continuation references

Keep the existing Brownian and continuation references as baselines.

For a first-order coordinate `q`:

```math
\text{Brownian:}\quad q_{k+1} \approx q_k
```

```math
\text{Continuation:}\quad q_{k+1} \approx q_k + \alpha(q_k-q_{k-1})
```

The contact-Langevin reference generalizes the continuation prior by making velocity an explicit state:

```math
p_{k+1} \approx (1-\Delta t\,\gamma)p_k
```

```math
q_{k+1}=q_k+\Delta t p_{k+1}.
```

With `V=0`, `sigma=0`, and constant `gamma`, this is just damped continuation. If `dt=1`, then:

```math
\alpha \approx 1 - \gamma.
```

So to match an existing continuation coefficient `alpha`, initialize:

```python
gamma_const = 1.0 - alpha
```

clamped to `[0, gamma_max]`.

---

## Normalization rules

Be careful with normalization.

If actions/positions are normalized, choose one of these approaches and use it everywhere:

### Recommended

Build `q_seq` in physical coordinates, then normalize `q_seq` with a `q_normalizer`. Compute `p_seq` in normalized coordinates as finite differences of normalized `q_seq`.

### Alternative

Normalize raw actions first, then build `q_seq` in normalized coordinates. This is acceptable for raw-action references, but less interpretable for absolute-position references.

Do not mix a normalized `q` with an unnormalized `p`.

Do not compute obstacle/contact potentials in normalized coordinates unless the obstacle positions are also transformed consistently.

---

## Required tests

Add unit tests for the adapter.

### Test 1: absolute action roundtrip

```python
q_seq = adapter.build_q_sequence(batch)
a_rec = adapter.decode_raw_actions(q_seq)
assert torch.allclose(a_rec, batch["actions"])
```

### Test 2: delta absolute-position roundtrip

```python
q_seq = adapter.build_q_sequence(batch)  # absolute_from_delta
raw_rec = adapter.decode_raw_actions(q_seq)
assert torch.allclose(raw_rec, batch["actions"])
```

### Test 3: damping decreases velocity when `u=0`, `V=0`

For constant `gamma > 0` and `dt=1`:

```python
p_next = p - gamma * p
assert torch.norm(p_next) <= torch.norm(p) + 1e-6
```

Use a stricter test when `0 <= gamma <= 1`.

### Test 4: KL is zero when control is zero

```python
u = torch.zeros(B, A)
kl = 0.5 * dt * (u ** 2).sum(-1).mean()
assert kl.item() == 0.0
```

### Test 5: output shape

```python
raw_actions, info = rollout_contact_policy(...)
assert raw_actions.shape == (B, H, action_dim)
assert info["q_seq"].shape == (B, H+1, action_dim)
```

---

## Required ablations

Add configs for these experiments:

1. `brownian_raw`: existing Brownian reference in raw action coordinates.
2. `continuation_raw`: existing continuation reference in raw action coordinates.
3. `contact_fixed_damping`: `V=0`, constant `gamma`, learned control.
4. `contact_learned_gamma`: `V=0`, learned positive `gamma`, learned control.
5. `contact_quadratic`: learned quadratic potential and learned damping.
6. `bc_plus_jerk`: standard BC with smoothness/jerk penalty.
7. Toy-only: `contact_absolute_actions`: train/evaluate toy policy with absolute target actions.
8. Toy-only: `contact_absolute_from_delta`: dataset stores deltas but reference is over integrated absolute positions.

---

## Metrics to log

Log task metrics plus path metrics.

```python
velocity_energy = ((q_seq[:, 1:] - q_seq[:, :-1]) ** 2).sum(-1).mean()
accel_energy = ((q_seq[:, 2:] - 2*q_seq[:, 1:-1] + q_seq[:, :-2]) ** 2).sum(-1).mean()
control_energy = 0.5 * dt * (u_seq ** 2).sum(-1).mean()
```

Also log:

```text
gamma_mean, gamma_min, gamma_max
k_diag_mean, k_diag_max
reference_only_loss
control_energy
chunk_boundary_discontinuity
mode_switch_rate for toy obstacle avoidance
```

For the toy obstacle experiment with absolute coordinates, compute mode labels from `q_seq`, not from raw deltas.

---

## Do not implement yet

Do not implement CuOT particle weights or mass-growth variables for this policy.

Do not implement full Sinkhorn/IPF training for this reference in the first version.

Do not implement arbitrary scalar neural potentials requiring `grad_q V` via autograd in the first version.

Do not change the diffusion covariance during policy generation unless the KL objective is updated accordingly.

---

## Acceptance criteria

The implementation is acceptable when:

1. Existing Brownian and continuation references still run unchanged.
2. `contact_langevin` runs with `potential_type=none` and constant `gamma`.
3. `contact_langevin` runs with learned `gamma`.
4. `contact_langevin` runs with quadratic potential.
5. The toy dataset can train with absolute target actions.
6. The toy dataset can train with delta actions while using `absolute_from_delta` internally.
7. The output raw action format matches the environment expectation.
8. Unit tests for adapter roundtrips pass.
9. KL/control energy is logged and nonzero only when the learned control residual is nonzero.
10. Deterministic inference works with `noise=None`.

