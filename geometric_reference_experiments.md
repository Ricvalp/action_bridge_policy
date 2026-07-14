# Push-T Geometric Reference Experiments

This note describes two implementation experiments:

1. **Hardcoded geometry reference-only controller**: no learned residual, no retraining. The goal is to test whether a hand-designed Push-T geometric potential can produce meaningful approach/contact/push/settle behavior.
2. **Fixed geometric reference + small residual**: retrain a low-capacity residual policy on top of the fixed reference. The goal is to test whether a structured reference can replace a large learned reference/control stack.

The two experiments are designed to plug into the existing contact-Langevin action bridge structure:

\[
q_{k+1}=q_k+\Delta t\,p_{k+1}
\]

\[
p_{k+1}=p_k+\Delta t\left[f_R(q_k,p_k,s,k)+\sigma u_\theta(q_k,p_k,h,k)\right].
\]

For experiment 1, set

\[
u_\theta=0.
\]

For experiment 2, freeze the geometric reference and learn only a small residual network.

---

## 0. Coordinate conventions

Use Push-T low-dimensional state:

\[
s = [x_p,y_p,x_T,y_T,\theta_T],
\]

where:

- \((x_p,y_p)\) is the pusher position;
- \((x_T,y_T)\) is the T-block center;
- \(\theta_T\) is the T-block orientation.

The policy action is the absolute 2D pusher target position:

\[
q=a\in\mathbb R^2.
\]

The velocity-like state is

\[
p_k=q_k-q_{k-1}.
\]

The geometric reference should operate in **denormalized pixel/world coordinates**, because the T geometry, target pose, pusher radius, and contact distances are all geometric quantities. If the training code stores normalized actions, add a coordinate adapter:

```python
class ActionNormalizer:
    def __init__(self, action_mean, action_std):
        self.mean = action_mean
        self.std = action_std

    def denorm(self, a_norm):
        return a_norm * self.std + self.mean

    def norm(self, a_px):
        return (a_px - self.mean) / self.std
```

Recommended implementation strategy:

1. Convert `q_norm`, `q_prev_norm` to pixel/world coordinates.
2. Compute `p_px = q_px - q_prev_px`.
3. Compute geometric reference acceleration `f_R_px`.
4. Integrate in pixel/world coordinates.
5. Convert predicted `q_next_px` back to normalized coordinates before comparing to normalized expert actions.

This avoids mixing normalized action units with pixel-based geometry.

---

# Experiment 1: Hardcoded geometric reference-only controller

## 1. Goal

Implement a non-learned reference controller that does the following:

1. Moves the pusher toward the T-block boundary.
2. Selects a contact point whose normal approximately pushes the T toward the goal pose.
3. Damps motion near contact.
4. Increases damping and decreases push aggressiveness as the T gets close to the goal.
5. Stops/settles near the T instead of oscillating or overpushing.

This controller is **not expected to solve Push-T perfectly**. It is successful if it produces meaningful trajectories:

\[
\text{approach T}\rightarrow\text{contact}\rightarrow\text{push toward target}\rightarrow\text{settle}.
\]

---

## 2. Required inputs

At every environment step, the reference needs:

```python
pusher_pos = np.array([x_p, y_p])
block_pos = np.array([x_T, y_T])
block_theta = theta_T
target_pos = np.array([x_target, y_target])
target_theta = theta_target
```

The target pose should be taken from the Push-T environment/task constants. Do not hardcode it by looking at one rollout unless the environment target is fixed across all episodes.

You also need the T-block geometry in its local frame:

```python
T_POLYGON_LOCAL = np.array([...])  # vertices in local block coordinates, CCW
```

Preferred options:

1. Query the polygon vertices from the simulator/body shape if available.
2. Otherwise, reconstruct the T polygon from the same dimensions used by the Push-T environment.
3. Validate by plotting transformed vertices over the environment frame.

The polygon should be oriented counter-clockwise. If not, outward normals will be flipped.

---

## 3. Transform the T polygon to world coordinates

```python
def rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def transform_polygon(poly_local, center, theta):
    R = rot2(theta)
    return center[None, :] + poly_local @ R.T
```

At each step:

```python
poly_world = transform_polygon(T_POLYGON_LOCAL, block_pos, block_theta)
```

---

## 4. Sample boundary points and outward normals

Use a dense set of candidate boundary contact points.

```python
def sample_polygon_boundary(poly, n_per_edge=12):
    """Return boundary points and outward normals for a CCW polygon."""
    pts = []
    normals = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        edge = b - a
        length = np.linalg.norm(edge) + 1e-8
        tangent = edge / length

        # For CCW polygon, outward normal is right normal.
        normal = np.array([tangent[1], -tangent[0]])

        for j in range(n_per_edge):
            t = (j + 0.5) / n_per_edge
            pts.append((1 - t) * a + t * b)
            normals.append(normal)

    return np.stack(pts), np.stack(normals)
```

Sanity check:

- Plot normals at several block poses.
- Outward normals should point away from the T interior.
- If normals point inward, flip the sign.

---

## 5. Compute desired object motion

The T should move from its current pose to the target pose.

```python
def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def desired_wrench(block_pos, block_theta, target_pos, target_theta,
                   k_trans=1.0, k_rot=0.4):
    e_pos = target_pos - block_pos
    e_theta = wrap_angle(target_theta - block_theta)
    return np.array([k_trans * e_pos[0],
                     k_trans * e_pos[1],
                     k_rot * e_theta])
```

This is not a real rigid-body wrench calculation. It is a heuristic scoring vector. The first two components describe desired translation; the third describes desired rotation.

Normalize robustly:

```python
def safe_unit(x, eps=1e-8):
    n = np.linalg.norm(x)
    if n < eps:
        return np.zeros_like(x)
    return x / n
```

---

## 6. Score contact candidates

For a boundary point \(b\) with outward normal \(n_b\), the pusher should push approximately into the object:

\[
f_b=-n_b.
\]

The approximate torque around the block center is

\[
\tau_b = (b-c_T)^\perp\cdot f_b.
\]

Then the candidate wrench is

\[
w_b = [f_{b,x}, f_{b,y}, \lambda_\tau \tau_b].
\]

Score:

\[
\mathrm{score}(b)
= \langle \hat w_b, \hat w_{des}\rangle
-\lambda_{travel}\|q_{pusher}-m_{pre}(b)\|^2
-\lambda_{back}\,\mathbb{1}[\text{bad approach side}].
\]

Implementation:

```python
def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def select_contact_point(boundary_pts, outward_normals, pusher_pos,
                         block_pos, block_theta, target_pos, target_theta,
                         pusher_radius=5.0,
                         delta_pre=6.0,
                         lambda_tau=0.4,
                         lambda_travel=1e-3,
                         k_trans=1.0,
                         k_rot=0.4,
                         temperature=0.05,
                         soft=False):
    w_des = desired_wrench(block_pos, block_theta, target_pos, target_theta,
                           k_trans=k_trans, k_rot=k_rot)
    w_des_u = safe_unit(w_des)

    scores = []
    m_pres = []

    for b, n_out in zip(boundary_pts, outward_normals):
        f = -n_out
        tau = cross2(b - block_pos, f)
        w = np.array([f[0], f[1], lambda_tau * tau])
        w_u = safe_unit(w)

        m_pre = b + (pusher_radius + delta_pre) * n_out
        travel = np.sum((pusher_pos - m_pre) ** 2)

        score = np.dot(w_u, w_des_u) - lambda_travel * travel
        scores.append(score)
        m_pres.append(m_pre)

    scores = np.asarray(scores)
    m_pres = np.asarray(m_pres)

    if soft:
        logits = scores / max(temperature, 1e-6)
        weights = np.exp(logits - logits.max())
        weights = weights / (weights.sum() + 1e-8)
        b_star = (weights[:, None] * boundary_pts).sum(axis=0)
        n_star = safe_unit((weights[:, None] * outward_normals).sum(axis=0))
        m_pre = (weights[:, None] * m_pres).sum(axis=0)
    else:
        idx = int(np.argmax(scores))
        b_star = boundary_pts[idx]
        n_star = outward_normals[idx]
        m_pre = m_pres[idx]

    return b_star, n_star, m_pre, scores
```

Start with `soft=False` for easier debugging. Later, use `soft=True` to avoid contact-point discontinuities.

---

## 7. Define contact and goal proximity

Distance from pusher to selected boundary point:

\[
d_{contact}=\|q-b^\star\|-r_{pusher}.
\]

Contact gate:

\[
\rho_{contact}=\sigma\left(\frac{d_0-d_{contact}}{\tau_d}\right).
\]

Goal proximity:

\[
\rho_{goal}=\exp\left(-\frac{\|c_T-c^\star\|^2+\lambda_\theta e_\theta^2}{\tau_{goal}^2}\right).
\]

Implementation:

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def proximity_terms(q, b_star, block_pos, block_theta, target_pos, target_theta,
                    pusher_radius=5.0,
                    d0=8.0,
                    tau_contact=4.0,
                    lambda_theta=25.0,
                    tau_goal=35.0):
    d_contact = np.linalg.norm(q - b_star) - pusher_radius
    rho_contact = sigmoid((d0 - d_contact) / tau_contact)

    e_pos = np.linalg.norm(block_pos - target_pos)
    e_theta = wrap_angle(block_theta - target_theta)
    goal_err = e_pos ** 2 + lambda_theta * e_theta ** 2
    rho_goal = np.exp(-goal_err / (tau_goal ** 2))

    return rho_contact, rho_goal, d_contact, goal_err
```

---

## 8. Define attractor, stiffness, and damping

Use two attractors:

Pre-contact standoff:

\[
m_{pre}=b^\star+(r_{pusher}+\delta_{pre})n^\star.
\]

Push attractor:

\[
m_{push}=b^\star-\delta_{push}n^\star.
\]

The push penetration should shrink near the target:

\[
\delta_{push}=\delta_{far}(1-\rho_{goal})+\delta_{near}\rho_{goal}.
\]

Interpolate:

\[
m_{geo}=(1-\rho_{contact})m_{pre}+\rho_{contact}m_{push}.
\]

Stiffness:

\[
K_{geo}=K_{free}(1-\rho_{contact})+K_{contact}\rho_{contact}.
\]

Damping:

\[
\gamma_{geo}=\gamma_{free}+\gamma_{contact}\rho_{contact}+\gamma_{goal}\rho_{goal}.
\]

A safer alternative is to set damping from critical damping:

\[
\gamma_{crit}=2\zeta\sqrt{K_{geo}},
\]

then clip:

\[
\gamma_{geo}=\mathrm{clip}(\gamma_{base}+\gamma_{crit},\gamma_{min},\gamma_{max}).
\]

Implementation:

```python
def geometric_params(q, p, obs_state, target_pose, T_POLYGON_LOCAL, cfg):
    pusher_pos = obs_state[:2]
    block_pos = obs_state[2:4]
    block_theta = obs_state[4]
    target_pos, target_theta = target_pose

    poly = transform_polygon(T_POLYGON_LOCAL, block_pos, block_theta)
    boundary_pts, normals = sample_polygon_boundary(poly, cfg.n_per_edge)

    b_star, n_star, m_pre, scores = select_contact_point(
        boundary_pts, normals, pusher_pos,
        block_pos, block_theta, target_pos, target_theta,
        pusher_radius=cfg.pusher_radius,
        delta_pre=cfg.delta_pre,
        lambda_tau=cfg.lambda_tau,
        lambda_travel=cfg.lambda_travel,
        k_trans=cfg.k_trans,
        k_rot=cfg.k_rot,
        soft=cfg.soft_contact_selection,
        temperature=cfg.contact_softmax_temp,
    )

    rho_contact, rho_goal, d_contact, goal_err = proximity_terms(
        q, b_star, block_pos, block_theta, target_pos, target_theta,
        pusher_radius=cfg.pusher_radius,
        d0=cfg.contact_gate_d0,
        tau_contact=cfg.tau_contact,
        lambda_theta=cfg.lambda_theta,
        tau_goal=cfg.tau_goal,
    )

    delta_push = cfg.delta_push_far * (1 - rho_goal) + cfg.delta_push_near * rho_goal
    m_push = b_star - delta_push * n_star
    m_geo = (1 - rho_contact) * m_pre + rho_contact * m_push

    K = cfg.K_free * (1 - rho_contact) + cfg.K_contact * rho_contact
    K = K * (1 + cfg.K_goal_gain * rho_goal)
    K = np.clip(K, cfg.K_min, cfg.K_max)

    gamma = cfg.gamma_free + cfg.gamma_contact * rho_contact + cfg.gamma_goal * rho_goal
    gamma = np.clip(gamma, cfg.gamma_min, cfg.gamma_max)

    debug = {
        "poly": poly,
        "boundary_pts": boundary_pts,
        "normals": normals,
        "b_star": b_star,
        "n_star": n_star,
        "m_pre": m_pre,
        "m_push": m_push,
        "m_geo": m_geo,
        "rho_contact": rho_contact,
        "rho_goal": rho_goal,
        "d_contact": d_contact,
        "goal_err": goal_err,
        "K": K,
        "gamma": gamma,
        "scores": scores,
    }
    return m_geo, K, gamma, debug
```

Recommended starting hyperparameters:

```yaml
geometric_reference:
  n_per_edge: 16
  pusher_radius: 5.0
  delta_pre: 8.0
  delta_push_far: 7.0
  delta_push_near: 2.0

  lambda_tau: 0.35
  lambda_travel: 0.0005
  k_trans: 1.0
  k_rot: 0.35

  soft_contact_selection: false
  contact_softmax_temp: 0.05

  contact_gate_d0: 8.0
  tau_contact: 4.0
  tau_goal: 35.0
  lambda_theta: 25.0

  K_free: 0.04
  K_contact: 0.12
  K_goal_gain: 1.5
  K_min: 0.0
  K_max: 0.35

  gamma_free: 0.08
  gamma_contact: 0.18
  gamma_goal: 0.35
  gamma_min: 0.02
  gamma_max: 0.85

  dt: 1.0
  max_step_norm: 12.0
  workspace_clip: [0.0, 512.0]
```

The values above assume pixel-scale actions. Tune them with visualization before full evaluation.

---

## 9. Reference force and rollout

The geometric potential is

\[
V_{geo}(q;s)=\frac12 K\|q-m_{geo}\|^2.
\]

Then

\[
-\nabla_q V_{geo}=-K(q-m_{geo}).
\]

The reference acceleration is

\[
f_R(q,p;s)= -K(q-m_{geo})-\gamma p.
\]

Implementation:

```python
def geometric_reference_accel(q, p, obs_state, target_pose, T_POLYGON_LOCAL, cfg):
    m, K, gamma, debug = geometric_params(q, p, obs_state, target_pose, T_POLYGON_LOCAL, cfg)
    f_R = -K * (q - m) - gamma * p
    debug["f_R"] = f_R
    return f_R, debug


def rollout_reference_only(q0, q_minus1, obs_state, target_pose, T_POLYGON_LOCAL, cfg, H=16):
    q = q0.copy()
    p = q0 - q_minus1
    actions = []
    debugs = []

    for k in range(H):
        f_R, debug = geometric_reference_accel(q, p, obs_state, target_pose, T_POLYGON_LOCAL, cfg)
        p = p + cfg.dt * f_R

        # Optional speed clamp.
        step_norm = np.linalg.norm(p)
        if step_norm > cfg.max_step_norm:
            p = p / (step_norm + 1e-8) * cfg.max_step_norm

        q = q + cfg.dt * p
        q = np.clip(q, cfg.workspace_clip[0], cfg.workspace_clip[1])

        actions.append(q.copy())
        debugs.append(debug)

    return np.stack(actions), debugs
```

Use a static `obs_state` over the 16-step open-loop chunk, matching the current policy setup. In closed-loop simulation, recompute the geometric reference after each replanning step using the latest observation.

---

## 10. Evaluation protocol

Run the hardcoded reference in the same closed-loop evaluation harness as the learned policy.

Recommended settings:

```yaml
inference:
  n_exec: 8
  H: 16
  deterministic: true
```

Metrics:

1. `sim_success_rate`
2. `sim_max_reward`
3. `sim_final_reward`
4. mean action speed
5. mean action acceleration
6. mean jerk
7. overpush rate near target
8. distance-to-T before first contact
9. distance-to-target after final contact
10. number of contact-loss events

Define overpush diagnostic:

```python
# Example: if max_reward crosses 0.90 and then later falls below 0.85.
overpush = (max_reward >= 0.90) and (final_reward <= 0.85)
```

This directly measures the failure mode where the T reaches high alignment and then gets pushed away.

---

## 11. Visualizations

Save one diagnostic figure per rollout:

- T polygon at several replanning times;
- selected contact point `b_star`;
- outward normal `n_star`;
- pre-contact attractor `m_pre`;
- push attractor `m_push`;
- actual executed pusher trajectory;
- commanded pusher targets;
- goal T pose;
- scalar traces of `rho_contact`, `rho_goal`, `K`, `gamma`.

Suggested filename:

```text
geom_ref_rollout_seed_XXXX.png
```

The most important plot is whether `gamma` increases near contact and near target, and whether `delta_push` shrinks near target.

---

# Experiment 2: Fixed geometric reference + small residual

## 1. Goal

Train a low-capacity residual policy on top of the fixed geometric reference:

\[
p_{k+1}=p_k+\Delta t\left[f_{geo}(q_k,p_k,s,k)+\sigma u_\theta(q_k,p_k,h,k)\right].
\]

The reference is frozen. The model learns only the residual.

This experiment tests whether the reference process can carry most of the manipulation structure, so the learned residual only makes precise task corrections.

---

## 2. Code-level integration

Add a new reference type:

```yaml
reference:
  type: geometric_pusht
  coordinate_mode: absolute_action
  learnable: false
```

Recommended file location:

```text
action_bridge/models/references.py
```

Add a class:

```python
class GeometricPushTReference(nn.Module):
    def __init__(self, cfg, normalizer, target_pose, T_POLYGON_LOCAL):
        super().__init__()
        self.cfg = cfg
        self.normalizer = normalizer
        self.target_pose = target_pose
        self.register_buffer("T_POLYGON_LOCAL", torch.as_tensor(T_POLYGON_LOCAL, dtype=torch.float32))

    def forward(self, q_norm, q_prev_norm, obs_state_norm, k=None):
        """
        q_norm: [B, 2]
        q_prev_norm: [B, 2]
        obs_state_norm: [B, 5], normalized lowdim observation
        returns:
            f_R_norm: [B, 2], acceleration-like reference term in normalized action units
            debug: dict
        """
        # 1. Denormalize q and q_prev to pixel coordinates.
        q_px = self.normalizer.denorm_action(q_norm)
        q_prev_px = self.normalizer.denorm_action(q_prev_norm)
        p_px = q_px - q_prev_px

        # 2. Denormalize observation to pixel/state coordinates.
        obs_px = self.normalizer.denorm_obs(obs_state_norm)

        # 3. Compute f_R in pixel coordinates using vectorized geometry.
        f_R_px, debug = self.compute_geometric_accel_batch(q_px, p_px, obs_px)

        # 4. Convert acceleration-like residual to normalized action units.
        # If q_norm = (q_px - mean) / std, then delta q_norm = delta q_px / std.
        f_R_norm = f_R_px / self.normalizer.action_std

        return f_R_norm, debug
```

If the existing reference API expects `q`, `p`, and `h` rather than raw observations, pass both:

- `h` for the learned control network;
- raw/denormalized observation state for the geometric reference.

Do not compute geometric quantities from the latent `h_emb`; use the actual block pose.

---

## 3. Vectorization notes

The hardcoded reference can start in NumPy for debugging. The training version should be PyTorch and batched.

Boundary sampling can be precomputed in the local frame:

```python
boundary_local, normals_local = sample_polygon_boundary(T_POLYGON_LOCAL, n_per_edge)
```

Then transform all candidates in batch:

\[
b_{B,N}=c_B+R(\theta_B)b_N^{local}
\]

\[
n_{B,N}=R(\theta_B)n_N^{local}.
\]

Torch-style pseudocode:

```python
def transform_boundary_batch(boundary_local, normals_local, block_pos, block_theta):
    # boundary_local: [N, 2]
    # normals_local: [N, 2]
    # block_pos: [B, 2]
    # block_theta: [B]
    c = torch.cos(block_theta)
    s = torch.sin(block_theta)
    R = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1),
    ], dim=-2)  # [B, 2, 2]

    b = torch.einsum("bij,nj->bni", R, boundary_local) + block_pos[:, None, :]
    n = torch.einsum("bij,nj->bni", R, normals_local)
    return b, n
```

Candidate scoring:

```python
def select_contact_batch(boundary_pts, normals, q_px, block_pos, block_theta, target_pos, target_theta, cfg):
    # boundary_pts: [B, N, 2]
    # normals: [B, N, 2]
    B, N, _ = boundary_pts.shape

    e_pos = target_pos[None, :] - block_pos
    e_theta = wrap_angle_torch(target_theta - block_theta)
    w_des = torch.cat([cfg.k_trans * e_pos, cfg.k_rot * e_theta[:, None]], dim=-1)  # [B, 3]
    w_des = safe_unit_torch(w_des)

    f = -normals  # [B, N, 2]
    r = boundary_pts - block_pos[:, None, :]
    tau = r[..., 0] * f[..., 1] - r[..., 1] * f[..., 0]
    w = torch.cat([f, cfg.lambda_tau * tau[..., None]], dim=-1)  # [B, N, 3]
    w = safe_unit_torch(w)

    m_pre = boundary_pts + (cfg.pusher_radius + cfg.delta_pre) * normals
    travel = ((q_px[:, None, :] - m_pre) ** 2).sum(dim=-1)

    scores = (w * w_des[:, None, :]).sum(dim=-1) - cfg.lambda_travel * travel

    if cfg.soft_contact_selection:
        weights = torch.softmax(scores / cfg.contact_softmax_temp, dim=-1)
        b_star = (weights[..., None] * boundary_pts).sum(dim=1)
        n_star = safe_unit_torch((weights[..., None] * normals).sum(dim=1))
        m_pre_star = (weights[..., None] * m_pre).sum(dim=1)
    else:
        idx = scores.argmax(dim=-1)
        b_star = boundary_pts[torch.arange(B), idx]
        n_star = normals[torch.arange(B), idx]
        m_pre_star = m_pre[torch.arange(B), idx]

    return b_star, n_star, m_pre_star, scores
```

For differentiability, use `soft_contact_selection=true` during training. Hard argmax is acceptable for reference-only evaluation, but soft selection is smoother for training.

---

## 4. Small residual architecture

Use a residual network whose expressive capacity is much smaller than the current 52M-parameter model and closer to the reference module scale.

Recommended small model:

```yaml
model:
  type: action_bridge
  hidden_dim: 256
  h_emb_dim: 256
  encoder_depth: 2
  control_depth: 3
  time_emb_dim: 32
  z_dim: 0
  z_embed_dim: 0

reference:
  type: geometric_pusht
  learnable: false

control:
  input: [q, p, h_emb, time_emb]
  output_dim: 2
  control_is_whitened: true
  sigma: 3.0
```

Approximate intended scale:

- history encoder: small MLP, roughly \(<0.3\)M parameters;
- control net: roughly \(<0.5\)M parameters;
- reference: 0 trainable parameters;
- total: around or below the scale of the previous learned reference module.

This is the important capacity-control experiment. If this model performs well, it supports the claim that the geometric reference carries real structure.

Avoid latent variables in the first run. Add latent only after the deterministic small residual establishes a baseline.

---

## 5. Training objective

Use the same contact-Langevin teacher-forced loss, but with frozen reference.

For expert normalized states \(q_k^*,p_k^*\):

\[
\hat p_{k+1}=p_k^*+\Delta t\left[f_{geo}(q_k^*,p_k^*,s,k)+\sigma u_\theta(q_k^*,p_k^*,h,k)\right]
\]

\[
\hat q_{k+1}=q_k^*+\Delta t\hat p_{k+1}.
\]

Loss:

\[
\mathcal L
=\text{loss}_p+\lambda_q\text{loss}_q+\beta_{KL}\text{path\_kl}+\lambda_{unroll}\text{unroll\_mse}.
\]

No latent KL. No learned-reference regularization. No learned attractor smoothness.

Path KL:

\[
\text{path\_kl}=\frac12\sum_k\|u_\theta(q_k,p_k,h,k)\|^2.
\]

Recommended start:

```yaml
loss:
  lambda_q: 1.0
  beta_kl: 0.003
  lambda_unroll: 1.0
  unroll_warmup_steps: 5000
  latent_kl: 0.0
  reference_reg: 0.0
  m_smooth: 0.0
```

Because the fixed reference is meaningful, try a stronger KL penalty than the previous big run.

Suggested sweep:

```yaml
beta_kl: [0.001, 0.003, 0.01]
sigma: [3.0, 5.0, 7.0]
```

Do not sweep too much initially. Start with `sigma=3.0` and `beta_kl=0.003`.

---

## 6. Training config template

```yaml
experiment: fixed_geometric_reference_small_residual_pusht

data:
  task: pusht_lowdim
  obs_dim: 5
  action_dim: 2
  obs_history: 2
  action_history: 2
  horizon: 16

model:
  class: ActionBridgePolicy
  hidden_dim: 256
  h_emb_dim: 256
  encoder_depth: 2
  control_depth: 3
  time_emb_dim: 32
  z_dim: 0
  z_embed_dim: 0

reference:
  type: geometric_pusht
  coordinate_mode: absolute_action
  learnable: false
  use_denormalized_geometry: true
  n_per_edge: 16
  pusher_radius: 5.0
  delta_pre: 8.0
  delta_push_far: 7.0
  delta_push_near: 2.0
  lambda_tau: 0.35
  lambda_travel: 0.0005
  k_trans: 1.0
  k_rot: 0.35
  soft_contact_selection: true
  contact_softmax_temp: 0.05
  contact_gate_d0: 8.0
  tau_contact: 4.0
  tau_goal: 35.0
  lambda_theta: 25.0
  K_free: 0.04
  K_contact: 0.12
  K_goal_gain: 1.5
  K_min: 0.0
  K_max: 0.35
  gamma_free: 0.08
  gamma_contact: 0.18
  gamma_goal: 0.35
  gamma_min: 0.02
  gamma_max: 0.85
  max_step_norm: 12.0

control:
  control_is_whitened: true
  sigma: 3.0
  control_scale: 1.0

loss:
  lambda_q: 1.0
  beta_kl: 0.003
  lambda_unroll: 1.0
  unroll_warmup_steps: 5000
  beta_z_start: 0.0
  beta_z_end: 0.0
  reference_reg: 0.0
  m_smooth: 0.0

optim:
  batch_size: 512
  lr: 1.0e-4
  max_steps: 300000
  grad_clip: 1.0

inference:
  deterministic: true
  n_exec: 8
```

---

## 7. Baselines for this experiment

Run at least these three models:

1. **Hardcoded geometric reference-only**
   - no training;
   - `u=0`.

2. **Fixed geometric reference + small residual**
   - frozen geometric reference;
   - small control net;
   - no latent.

3. **Small autoregressive BC baseline**
   - same history encoder size;
   - same control/residual net capacity;
   - no reference process.

The third baseline is essential. It tests whether success comes from the fixed reference or merely from having a small autoregressive network.

If possible, also compare to the current large checkpoint, but report it separately because the parameter budget is not comparable.

---

## 8. Evaluation metrics

Report:

```text
sim_success_rate
sim_max_reward
sim_final_reward
action_speed_mean
action_accel_mean
action_jerk_mean
boundary_discontinuity
path_kl
control_energy
reference_energy
residual_to_reference_ratio
overpush_rate
contact_loss_count
```

Define residual-to-reference ratio:

\[
R_{ctrl}=\frac{\|\sigma u_\theta\|}{\|f_{geo}\|+\|\sigma u_\theta\|+\epsilon}.
\]

A good result should have:

- competitive success;
- lower jerk than small AR BC;
- lower overpush rate;
- residual-to-reference ratio not close to 1 everywhere;
- meaningful reference-only behavior.

---

## 9. Success criteria

This experiment is successful if one of the following holds.

### Strong outcome

The fixed geometric reference + small residual reaches near the big model's performance, for example:

\[
\text{success} \ge 0.85
\]

with substantially lower capacity and better smoothness/overpush metrics.

### Medium outcome

The fixed geometric reference + small residual beats small AR BC by a clear margin, even if it does not match the big model.

### Weak but useful outcome

The reference-only controller produces meaningful approach/contact/push/settle behavior, and the residual improves it substantially. This still validates the reference as a useful prior.

### Negative outcome

Small AR BC matches or beats the fixed-reference model on success, jerk, and overpush. In that case, the hardcoded geometry prior is either too crude or not aligned with the dataset/action representation.

---

## 10. Debug checklist

Before training the residual, verify:

- transformed T polygon aligns with environment render;
- outward normals point outward;
- selected contact point changes sensibly with block pose;
- desired push direction points toward target;
- `rho_contact` increases near the T boundary;
- `rho_goal` increases near task completion;
- `gamma` increases near contact and near target;
- `delta_push` decreases near target;
- reference-only rollout does not leave workspace;
- action normalization/denormalization is correct.

If reference-only is unstable, reduce:

```yaml
K_contact
K_goal_gain
max_step_norm
delta_push_far
```

or increase:

```yaml
gamma_contact
gamma_goal
```

If reference-only never makes contact, increase:

```yaml
K_free
K_contact
delta_push_far
contact_gate_d0
```

If it overpushes near the target, decrease:

```yaml
delta_push_near
K_goal_gain
```

and increase:

```yaml
gamma_goal
```

---

# Implementation order

1. Implement NumPy hardcoded reference and visualize single-step geometry.
2. Run reference-only closed-loop evaluation.
3. Port reference to batched PyTorch.
4. Add `reference.type=geometric_pusht` to the existing bridge policy.
5. Train fixed geometric reference + small residual.
6. Train small AR BC with matched capacity.
7. Compare success, jerk, overpush, and residual/reference ratio.

---

# Expected interpretation

If the hardcoded reference-only controller moves meaningfully but fails, and the small residual makes it successful, that supports the central hypothesis:

\[
\text{geometric dissipative reference} + \text{small learned correction}
\]

can be a better inductive bias than learning the whole action generator from scratch.

The current learned checkpoint already suggests the reference is behaviorally meaningful because reference-only moves toward the T and control-only fails. These two experiments test whether that learned reference can be replaced by an interpretable geometric one.
