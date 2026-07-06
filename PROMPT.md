# Codex Prompt: Pilot Experiments for Latent Path-KL Action Bridge Policies

You are implementing a research pilot for a robotics-policy idea:

> Learn action chunks as **stochastic action-path laws** rather than as independent timestep marginals or direct chunk regressions. The policy is a learned minimum-control deformation of a robot-specific reference process.

This is not a full Schrödinger Bridge solver. Do **not** implement Sinkhorn, IPF, score matching, or diffusion noising. Implement an amortized **path-KL controlled action process** with latent mode commitment.

The pilot has two benchmarks:

1. A synthetic 2D obstacle-avoidance benchmark with coherent top/bottom or clockwise/counterclockwise path modes.
2. A low-dimensional Push-T benchmark used as a non-toy manipulation-policy viability test.

The goal is to test whether this objective trains stably, whether the reference process helps, and whether the latent variable produces coherent path-level mode commitment.

---

## 0. Main hypothesis

The policy should model:

```math
P_\theta(a_{0:H} \mid h)
```

not only per-timestep marginals:

```math
\rho_k(a_k \mid h).
```

Use a reference action process:

```math
R_h(a_{0:H}) = r_0(a_0 \mid h) \prod_{k=0}^{H-1} r_k(a_{k+1} \mid a_k, a_{k-1}, h).
```

The learned process is:

```math
p_\theta(a_{k+1} \mid a_k, a_{k-1}, h, z)
= \mathcal N\left(\mu_R(a_k,a_{k-1},h,k) + u_\theta(a_k,a_{k-1},h,k,z),\; \Sigma_R(k,h)\right).
```

Here:

- `mu_R` is the reference/default action dynamics.
- `u_theta` is the learned control residual.
- `z` is a latent variable sampled once per chunk or once per episode and then held fixed for coherent mode commitment.
- `Sigma_R` defines both stochastic sampling and the weighting of the path-KL/control-energy penalty.

The objective is:

```math
\mathcal L
= \mathcal L_\text{NLL/path}
+ \beta_R \mathcal L_\text{path-KL}
+ \beta_z D_\text{KL}\big(q_\phi(z \mid h,a^*_{0:H}) \Vert p_\psi(z \mid h)\big)
+ \text{optional auxiliary losses}.
```

For Gaussian transitions, use:

```math
\mathcal L_\text{path-KL}
= \sum_{k=0}^{H-1} \frac{1}{2}
 u_k^\top \Sigma_R^{-1} u_k.
```

This is the key: the policy is **reference process + learned control correction** during both training and inference.

---

## 1. Non-negotiable requirements

1. Do **not** implement Sinkhorn or IPF.
2. Do **not** implement a diffusion model.
3. Do **not** discard the reference process at inference. Inference must use:

   ```python
   a_next = mu_reference(a_k, a_k_minus_1, h, k) + u_theta(...)
   ```

4. Implement both latent choices:
   - categorical latent `z`, e.g. top/bottom or clockwise/counterclockwise;
   - continuous unconstrained latent `z in R^d`, trained VAE-style.
5. For categorical `z`, do not resample `z` at every step. Sample once and hold fixed across the generated chunk. For receding-horizon evaluation, support sticky mode persistence across chunks.
6. For continuous `z`, support both:
   - sample once per chunk and hold fixed;
   - sample once per episode and hold fixed for all chunks.
7. Implement reference-process ablations.
8. Implement baselines that can distinguish “path-KL process” from ordinary smoothing.
9. Implement metrics for path coherence, hybrid paths, collisions, smoothness, and goal error.
10. Produce saved plots and JSON/CSV metrics for every run.
11. Add basic unit tests and a small CPU smoke test.

---

## 2. Suggested repository structure

Create or adapt the repository to this structure:

```text
action_bridge/
  __init__.py
  data/
    __init__.py
    toy_obstacle.py
    toy_annular.py
    chunking.py
    pusht_adapter.py
  models/
    __init__.py
    encoders.py
    references.py
    latents.py
    action_bridge_policy.py
    baselines.py
  training/
    __init__.py
    losses.py
    train_toy.py
    train_pusht.py
    schedules.py
  eval/
    __init__.py
    metrics.py
    eval_toy.py
    eval_pusht.py
    rollout.py
    visualization.py
  configs/
    toy_delayed_categorical.yaml
    toy_delayed_continuous.yaml
    toy_annular_categorical.yaml
    toy_annular_continuous.yaml
    pusht_lowdim_categorical.yaml
    pusht_lowdim_continuous.yaml
  scripts/
    __init__.py
    generate_toy_delayed.py
    generate_toy_annular.py
    run_toy_sweep.py
    run_pusht_pilot.py
  tests/
    test_references.py
    test_latents.py
    test_toy_data.py
    test_metrics.py
README.md
```

Use PyTorch. Use Hydra/OmegaConf or plain YAML configs. Prefer simple code over elaborate abstractions.

---

## 3. Data interface

All datasets should yield batches with this common structure:

```python
batch = {
    "obs_hist": Tensor[B, L, obs_dim],
    "act_hist": Tensor[B, M, action_dim],
    "future_actions": Tensor[B, H, action_dim],
    "future_positions": Tensor[B, H + 1, 2],      # toy required, Push-T optional if available
    "mode_label": Tensor[B],                       # toy categorical label, optional for Push-T
    "true_mode_probs": Tensor[B, K],               # toy annular if known, optional
    "context": Dict[str, Tensor],                  # start, goal, obstacle params, etc.
}
```

Definitions:

- `h` is encoded from `obs_hist` and `act_hist`.
- `future_actions[:, 0]` is the first action to generate.
- `act_hist[:, -1]` and `act_hist[:, -2]` are used by the reference process.
- For toy data, `future_positions` are needed to compute collision, homotopy mode, and visualization metrics.

---

## 4. Toy benchmark A: delayed-branch top/bottom obstacle avoidance

Implement or adapt the existing dataset described below.

A point agent moves in the unit square. It starts near `(0.12, 0.5)`, aims for a goal near `(0.88, 0.5)`, and avoids a circular obstacle centered at `(0.5, 0.5)`. State is:

```text
[x_agent, y_agent, x_goal, y_goal]
```

Action is a 2D velocity command.

Demonstrations come in paired top/bottom modes. Mode `+1` goes above the obstacle. Mode `-1` goes below. Delayed-branch variants share the first few actions exactly, so the same pre-fork history can have two valid futures.

Implement:

```python
class DelayedBranchObstacleDataset(torch.utils.data.Dataset):
    ...
```

Config fields:

```yaml
num_contexts: 5000
paired_fraction: 0.5
trajectory_len: 64
chunk_horizon: 16
obs_history: 2
action_history: 2
obstacle_center: [0.5, 0.5]
obstacle_radius: 0.13
lane_margin: 0.08
start_mean: [0.12, 0.5]
start_jitter: [0.03, 0.08]
goal_mean: [0.88, 0.5]
goal_jitter: [0.03, 0.08]
shared_prefix_steps: 8
shared_prefix_target_x: 0.30
action_noise_std: 0.005
speed: 0.035
seed: 0
```

Generate trajectories through waypoints:

- shared prefix target around `x = shared_prefix_target_x`;
- lane waypoint near `x = 0.55` with `y = center_y +/- (radius + lane_margin)`;
- lane waypoint near `x = 0.72` with same lane height;
- final goal.

The delayed-branch setting is the cleanest diagnostic. It should be possible to ask the model for multiple samples from the same pre-fork history and observe either coherent top paths or coherent bottom paths, not hybrids.

---

## 5. Toy benchmark B: annular clockwise/counterclockwise obstacle avoidance

Implement a more general obstacle-avoidance dataset where start and goal are sampled around the obstacle.

### 5.1 Geometry

Obstacle:

```python
center = torch.tensor([0.5, 0.5])
r_obs = 0.15
r_clear = r_obs + margin
```

Sample start and goal in an annulus:

```math
r_\min < \|x - c\| < r_\max.
```

Suggested defaults:

```yaml
r_min: 0.28
r_max: 0.48
min_start_goal_distance: 0.35
require_interaction: true
interaction_distance_threshold: 0.18
```

If `require_interaction` is true, reject pairs whose straight line from start to goal stays far away from the obstacle. This prevents many trivial straight-line examples.

### 5.2 Modes

Modes:

```text
+1 = counterclockwise
-1 = clockwise
```

Compute angles:

```python
theta_s = atan2(start_y - c_y, start_x - c_x)
theta_g = atan2(goal_y - c_y, goal_x - c_x)
```

For each mode, compute approximate curve length:

```math
L_\text{ccw} = |r_s-r_\text{clear}| + r_\text{clear}\Delta\theta_\text{ccw} + |r_g-r_\text{clear}|
```

```math
L_\text{cw} = |r_s-r_\text{clear}| + r_\text{clear}\Delta\theta_\text{cw} + |r_g-r_\text{clear}|.
```

Use angular deltas in `[0, 2pi]`.

Mode probability should prefer the shorter path but never set the longer path to zero:

```math
p_\text{ccw}(s,g)
= p_\min + (1-2p_\min)
\frac{\exp(-L_\text{ccw}/\tau)}
{\exp(-L_\text{ccw}/\tau)+\exp(-L_\text{cw}/\tau)}.
```

```math
p_\text{cw}=1-p_\text{ccw}.
```

Suggested defaults:

```yaml
p_min: 0.08
temperature: 0.08
```

Do **not** make probability directly proportional to length unless a config flag intentionally tests the unnatural case where longer paths are more likely.

### 5.3 Path generation

For a given mode:

1. Move radially from start to the clearance circle.
2. Move along the clearance arc in the selected direction.
3. Move radially from the clearance circle to the goal.
4. Smooth with cubic interpolation or resample by arc length.
5. Convert positions to velocity actions:

```math
a_k = (x_{k+1} - x_k) / \Delta t.
```

Dataset fields should include:

```python
context = {
    "start": start,
    "goal": goal,
    "obstacle_center": center,
    "obstacle_radius": r_obs,
    "p_ccw_true": p_ccw,
    "p_cw_true": p_cw,
    "length_ccw": L_ccw,
    "length_cw": L_cw,
}
```

### 5.4 Paired and single-sample contexts

Generate both:

- paired contexts: both modes are present for the same start/goal;
- single-sample contexts: one mode sampled according to `p(mode | start, goal)`.

Suggested defaults:

```yaml
paired_fraction: 0.3
single_sample_fraction: 0.7
```

The paired subset is for clean analysis. The single-sample subset mimics realistic demonstration data where exact duplicate contexts are rare.

---

## 6. Reference processes

Implement reference processes in `models/references.py`.

All references expose:

```python
class ReferenceProcess(nn.Module):
    def forward(self, a_k, a_k_minus_1, h_emb, k, extra=None):
        """Return mu_R, Sigma_R or log_sigma_R."""
```

### 6.1 Brownian/raw action reference

Weak baseline:

```math
\mu_R = a_k.
```

```python
a_next = a_k + noise
```

### 6.2 Continuation reference

Primary reference:

```math
\mu_R = a_k + \alpha(a_k - a_{k-1}).
```

`alpha` can be fixed or learned scalar constrained to `[0, 1]`.

### 6.3 Low-acceleration reference

Equivalent to continuation with stronger damping. Optionally parameterize with velocity:

```math
v_k = a_k-a_{k-1}
```

```math
\mu_R = a_k + \alpha_k v_k.
```

Allow `alpha_k` to be phase/time dependent through an MLP:

```python
alpha_k = sigmoid(alpha_net(h_emb, time_emb))
```

### 6.4 Low-jerk reference

Optional second-order reference using the last three actions:

```math
\mu_R = a_k + (a_k-a_{k-1}) + \rho\big[(a_k-a_{k-1})-(a_{k-1}-a_{k-2})\big].
```

Implement only after the simple references work.

### 6.5 Obstacle/contact-aware covariance

For toy obstacle data, optionally reduce variance near the obstacle:

```math
\sigma_R(x) = \sigma_\text{far} - (\sigma_\text{far}-\sigma_\text{near})\exp(-d(x,\text{obstacle})^2 / \ell^2).
```

For Push-T, optionally reduce variance near contact using a contact proxy if the low-dimensional state exposes pusher/object positions. This is an ablation only. The main method should work with the continuation reference.

---

## 7. Latent variables

Implement in `models/latents.py`.

### 7.1 Categorical latent

Use:

```math
z \in \{1,\dots,K\}.
```

For toy top/bottom or clockwise/counterclockwise, default `K=2`.

Networks:

```python
p_psi(z | h)                         # prior logits from history
q_phi(z | h, future_actions)          # posterior logits from history + future path
```

Training:

- Prefer exact enumeration over `K` when `K` is small.
- For each category, compute reconstruction/path loss conditioned on the category embedding.
- Weight by posterior probability `q_phi`.
- Add analytic categorical KL:

```math
D_\text{KL}(q \Vert p) = \sum_z q_z (\log q_z - \log p_z).
```

Inference:

- Sample `z ~ p_psi(z | h)` or take argmax.
- Hold `z` fixed for the generated chunk.
- For receding horizon, support sticky transition:

```math
p(z_t | h_t, z_{t-1}) \propto p_\psi(z_t | h_t) \exp(-\kappa 1[z_t \ne z_{t-1}]).
```

### 7.2 Continuous unconstrained latent

Implement this too.

Use:

```math
z \in \mathbb R^{d_z}.
```

Default:

```yaml
latent_type: continuous
z_dim: 4
```

Posterior:

```math
q_\phi(z | h, a^*_{0:H}) = \mathcal N(\mu_q, \operatorname{diag}(\sigma_q^2)).
```

Prior options:

1. Standard normal:

   ```math
   p(z|h) = \mathcal N(0,I)
   ```

2. Learnable conditional Gaussian:

   ```math
   p_\psi(z|h) = \mathcal N(\mu_p(h), \operatorname{diag}(\sigma_p^2(h))).
   ```

Implement both. Default to learnable conditional Gaussian.

Use reparameterization:

```python
z = mu_q + exp(0.5 * logvar_q) * eps
```

KL:

```math
D_\text{KL}\big(\mathcal N(\mu_q,\sigma_q^2) \Vert \mathcal N(\mu_p,\sigma_p^2)\big)
= \frac12 \sum_j \left[
\log\frac{\sigma_{p,j}^2}{\sigma_{q,j}^2}
+ \frac{\sigma_{q,j}^2 + (\mu_{q,j}-\mu_{p,j})^2}{\sigma_{p,j}^2}
- 1
\right].
```

Inference:

- sample `z ~ p_psi(z | h)` once per chunk or once per episode;
- hold it fixed during generation;
- for receding horizon, support continuous sticky latent:

```math
z_t = \rho_z z_{t-1} + \sqrt{1-\rho_z^2}\, \tilde z_t,
\quad \tilde z_t \sim p_\psi(z|h_t).
```

Also support `z_t = z_{t-1}` for full episode-level commitment.

Important: the continuous latent is unconstrained. Do not discretize it. For analysis, classify generated paths into modes after rollout and then study how regions of latent space correspond to modes.

### 7.3 Avoid latent collapse

Implement:

- KL annealing schedule for `beta_z`;
- free-bits/free-nats threshold;
- optional mutual-information-style auxiliary mode loss on toy data;
- logging of posterior entropy and prior entropy;
- logging of how often different modes are sampled.

Suggested defaults:

```yaml
beta_z_start: 0.0
beta_z_end: 0.01
beta_z_warmup_steps: 10000
free_nats: 0.1
```

For categorical toy data, optionally add:

```math
\mathcal L_\text{mode-ce} = CE(\hat z_\text{posterior}, z_\text{true})
```

but keep it disabled by default unless latent collapse occurs. The main VAE objective should work without using labels.

---

## 8. Policy model

Implement in `models/action_bridge_policy.py`.

### 8.1 Encoder

History encoder:

```python
h_emb = HistoryEncoder(obs_hist, act_hist)
```

Default architecture:

- flatten `obs_hist` and `act_hist`;
- MLP width 256, depth 3;
- LayerNorm or not configurable;
- output dimension 256.

### 8.2 Future encoder for posterior

For posterior `q_phi(z | h, future_actions)`, encode future action path:

```python
future_emb = FutureActionEncoder(future_actions)
```

Default:

- flatten future actions;
- MLP width 256, depth 2;
- concatenate with `h_emb`.

Optionally include future positions for toy posterior, but default should use future actions only.

### 8.3 Control residual network

```python
u = ControlNet(a_k, a_k_minus_1, h_emb, time_emb, z_emb)
```

Inputs:

- current action `a_k`;
- previous action `a_{k-1}`;
- history embedding `h_emb`;
- time embedding for `k`;
- latent embedding `z_emb`.

Categorical latent:

- use learned embedding table `Embedding(K, z_embed_dim)`.

Continuous latent:

- pass `z` through an MLP to `z_embed_dim`.

Default control net:

- MLP width 256;
- depth 4;
- output dimension `action_dim`.

Optional:

- output log-variance correction, but default should use fixed/reference variance.

---

## 9. Training loss

Implement in `training/losses.py`.

Given expert actions:

```python
future_actions: [B, H, action_dim]
act_hist: [B, M, action_dim]
```

Initialize:

```python
a_minus_1 = act_hist[:, -2]
a_0_prev = act_hist[:, -1]
```

During teacher forcing, use this indexing convention:

- model predicts `future_actions[:, k]` from the previous two actions;
- for `k=0`, previous two actions are `act_hist[:, -2]` and `act_hist[:, -1]`;
- for `k=1`, previous two actions are `act_hist[:, -1]` and `future_actions[:, 0]`;
- for `k>=2`, previous two actions are `future_actions[:, k-2]` and `future_actions[:, k-1]`.

Implement with a helper:

```python
def teacher_forced_prev_actions(act_hist, future_actions, k):
    if k == 0:
        return act_hist[:, -2], act_hist[:, -1]
    if k == 1:
        return act_hist[:, -1], future_actions[:, 0]
    return future_actions[:, k-2], future_actions[:, k-1]
```

For each step:

```python
mu_R, log_sigma_R = reference(a_prev, a_prevprev, h_emb, k)
u = control_net(a_prev, a_prevprev, h_emb, k, z)
mu = mu_R + u
target = future_actions[:, k]
```

NLL/MSE:

```math
\mathcal L_\text{NLL}
= \sum_k \frac12 \left\|\frac{a^*_k - (\mu_R + u_k)}{\sigma_R}\right\|^2
+ \sum_k \log \sigma_R.
```

Path-KL/control energy:

```math
\mathcal L_\text{path-KL}
= \sum_k \frac12 \left\|\frac{u_k}{\sigma_R}\right\|^2.
```

Total:

```math
\mathcal L
= \mathcal L_\text{NLL}
+ \beta_R \mathcal L_\text{path-KL}
+ \beta_z \mathcal L_\text{latent-KL}
+ \beta_\text{aux}\mathcal L_\text{aux}.
```

### 9.1 Categorical latent loss

If `K` is small, enumerate categories:

```python
for z_id in range(K):
    loss_z = path_loss_conditioned_on_z(z_id)
expected_path_loss = (q_probs * loss_z_per_category).sum(dim=-1).mean()
latent_kl = categorical_kl(q_logits, p_logits).mean()
```

This is more stable than Gumbel sampling.

### 9.2 Continuous latent loss

Use one or more posterior samples:

```python
z = reparameterize(mu_q, logvar_q)
path_loss = path_loss_conditioned_on_z(z)
latent_kl = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
```

Support config:

```yaml
num_z_samples_train: 1
```

### 9.3 Tube perturbation training

Implement optional structured perturbation around expert intermediate actions.

For each teacher-forced previous action, add noise:

```python
a_prev_tilde = a_prev + eps_prev
a_prevprev_tilde = a_prevprev + eps_prevprev
```

Use the perturbed actions as inputs but keep the expert target:

```python
mu_R = reference(a_prev_tilde, a_prevprev_tilde, h_emb, k)
u = control_net(a_prev_tilde, a_prevprev_tilde, h_emb, k, z)
target = future_actions[:, k]
```

Noise schedule:

```yaml
tube_training: true
tube_noise_std_start: 0.0
tube_noise_std_end: 0.03
tube_noise_warmup_steps: 5000
```

This is meant to train local recovery around the demonstration path.

---

## 10. Inference and rollout

Implement in `eval/rollout.py`.

### 10.1 Open-loop chunk generation

```python
def generate_chunk(policy, obs_hist, act_hist, mode="sample", deterministic=True):
    h_emb = policy.encode_history(obs_hist, act_hist)
    z = policy.sample_prior_z(h_emb, mode=mode)

    a_prevprev = act_hist[:, -2]
    a_prev = act_hist[:, -1]
    actions = []

    for k in range(H):
        mu_R, log_sigma_R = policy.reference(a_prev, a_prevprev, h_emb, k)
        u = policy.control(a_prev, a_prevprev, h_emb, k, z)
        mu = mu_R + u
        if deterministic:
            a = mu
        else:
            a = mu + exp(log_sigma_R) * torch.randn_like(mu)
        actions.append(a)
        a_prevprev, a_prev = a_prev, a

    return torch.stack(actions, dim=1), z
```

### 10.2 Receding-horizon generation

Support:

- execute first `n_exec` actions;
- replan;
- keep/stick latent across chunks.

Categorical sticky mode:

```python
z_t = sticky_categorical_prior(p_logits, z_prev, kappa)
```

Continuous sticky mode:

```python
z_t = z_prev
```

or:

```python
z_t = rho_z * z_prev + sqrt(1-rho_z**2) * sample_prior(h_t)
```

Default for mode-commitment evaluation:

```yaml
latent_commitment: episode
```

meaning sample one latent at the beginning of the rollout and hold it fixed.

---

## 11. Baselines

Implement these in `models/baselines.py` and training scripts.

### 11.1 Reference-only

No learned control:

```python
a_next = mu_R(a_prev, a_prevprev, h, k)
```

### 11.2 Direct chunk BC

Predict all future actions directly:

```python
future_actions = MLP(h_emb)
```

Loss: MSE.

### 11.3 Autoregressive BC without reference

```python
a_next = f_theta(a_prev, a_prevprev, h, k)
```

No `mu_R`, no path-KL.

### 11.4 BC + smoothness penalty

Direct or autoregressive BC plus:

```math
\lambda_\text{acc}\sum_k \|a_{k+1}-2a_k+a_{k-1}\|^2
```

and optionally jerk penalty:

```math
\lambda_\text{jerk}\sum_k \|a_{k+1}-3a_k+3a_{k-1}-a_{k-2}\|^2.
```

This is the critical baseline. It tests whether path-KL is more than ordinary smoothing.

### 11.5 Path-KL without latent

Use reference + control residual, but no `z`.

### 11.6 Path-KL + categorical latent

Main toy version.

### 11.7 Path-KL + continuous latent

Main continuous-latent version.

### 11.8 Path-KL + latent + tube training

Most complete version.

---

## 12. Toy evaluation metrics

Implement in `eval/metrics.py`.

### 12.1 Basic metrics

- action MSE;
- final goal error;
- path length;
- collision rate;
- minimum obstacle clearance;
- acceleration energy:

```math
\sum_t \|a_t-a_{t-1}\|^2
```

- jerk energy:

```math
\sum_t \|a_t-3a_{t-1}+3a_{t-2}-a_{t-3}\|^2
```

### 12.2 Mode metrics for delayed top/bottom

Define obstacle region:

```python
in_obstacle_band = (x > center_x - band) & (x < center_x + band)
```

Classify local mode by mean y in the obstacle band:

```python
mode = +1 if mean_y_above_center else -1
```

Hybrid path:

- path crosses from above to below inside obstacle interaction region;
- or has both substantial above and below segments around the obstacle;
- or collides with obstacle while switching.

Metrics:

```text
mode_switch_rate
hybrid_rate
valid_top_rate
valid_bottom_rate
collision_rate
```

### 12.3 Mode metrics for annular benchmark

Compute angle around obstacle:

```python
theta_t = unwrap(atan2(y_t - c_y, x_t - c_x))
total_delta = theta_last - theta_first
```

Classify:

```python
ccw if total_delta > 0
cw if total_delta < 0
```

Hybrid/switching if angular velocity changes sign repeatedly or if the path cuts through the obstacle.

Metrics:

```text
cw_rate
ccw_rate
mode_switch_count
hybrid_rate
collision_rate
```

### 12.4 Prior calibration

For toy annular data, true mode probability is known.

Categorical latent:

```math
\text{calibration error} = |p_\psi(\text{ccw}|h) - p_\text{true}(\text{ccw}|h)|
```

Continuous latent:

- sample `N` latents from prior;
- generate paths;
- classify each path into cw/ccw;
- estimate empirical `p_hat(ccw|h)`;
- compare to true `p_true(ccw|h)`.

### 12.5 Path-level distribution metrics

Implement at least one:

1. MMD over flattened trajectories;
2. MMD over feature vectors:

```python
features = [final_error, min_clearance, path_length, mean_y_or_angle, mode, acceleration, jerk]
```

Signature distance is optional. Do MMD first.

---

## 13. Toy visualizations

Save plots to `outputs/<run_id>/figures/`.

Required figures:

1. Dataset samples with obstacle.
2. Generated samples for the same ambiguous history.
3. Categorical prior probability vs true mode probability for annular benchmark.
4. Continuous latent scatter plot:
   - sample `z` from prior;
   - generate trajectories;
   - classify mode;
   - plot first two latent dimensions colored by generated mode.
5. Histograms of acceleration, jerk, and path-KL energy.
6. Failure cases: collisions and hybrid paths.

---

## 14. Push-T benchmark

Push-T is the non-toy benchmark. Use it to test whether the objective can train a real manipulation policy and whether references help smoothness/continuity.

Start with low-dimensional Push-T. Do not start with image Push-T. The research variable is action-path generation, not visual representation learning.

Implement `data/pusht_adapter.py` with flexible backends:

1. If a local Diffusion Policy Push-T zarr dataset is available, load it.
2. If LeRobot Push-T dataset is available, load it.
3. If no dataset is available, raise a clear error with setup instructions.

Do not vendor external repositories into this project.

### 14.1 Push-T data interface

Return the same batch format:

```python
batch = {
    "obs_hist": ...,
    "act_hist": ...,
    "future_actions": ...,
    "future_positions": optional,
    "mode_label": None,
    "true_mode_probs": None,
    "context": {...}
}
```

### 14.2 Push-T references

Start with:

```math
\mu_R = a_k + \alpha(a_k-a_{k-1}).
```

Then test:

- Brownian/raw action reference;
- continuation reference;
- learned alpha reference;
- optional contact-aware covariance if lowdim state exposes pusher/object geometry.

### 14.3 Push-T training variants

Train:

1. Direct BC.
2. Autoregressive BC.
3. BC + smoothness penalty.
4. Path-KL without latent.
5. Path-KL + continuous latent.
6. Path-KL + categorical latent if you define unsupervised `K=4`; do not expect clean categories.
7. Path-KL + continuous latent + tube training.

For Push-T, continuous latent may be more natural than categorical latent. Categorical `z` is still useful as an ablation.

### 14.4 Push-T metrics

If closed-loop environment is available:

- max target coverage/overlap;
- final target coverage/overlap;
- success rate using the environment's success definition;
- episode length;
- action acceleration;
- action jerk;
- chunk-boundary discontinuity;
- path-KL energy.

If only offline dataset is available:

- action NLL/MSE;
- rollout-in-dataset action error;
- acceleration/jerk of generated chunks;
- chunk-boundary discontinuity under simulated receding-horizon generation from logged histories.

Closed-loop evaluation is preferred.

---

## 15. Training scripts

Implement:

```bash
python -m action_bridge.training.train_toy --config-name toy_delayed_categorical
python -m action_bridge.training.train_toy --config-name toy_delayed_continuous
python -m action_bridge.training.train_toy --config-name toy_annular_categorical
python -m action_bridge.training.train_toy --config-name toy_annular_continuous
python -m action_bridge.training.train_pusht --config-name pusht_lowdim_continuous
```

Every training run should save:

```text
outputs/<run_id>/
  config.yaml
  checkpoints/latest.pt
  checkpoints/best.pt
  metrics/train_metrics.csv
  metrics/val_metrics.csv
  metrics/test_metrics.json
  figures/
```

Use deterministic seeds.

---

## 16. Config defaults

### 16.1 Toy default

```yaml
seed: 0
device: cuda
benchmark: toy_delayed
trajectory_len: 64
chunk_horizon: 16
obs_history: 2
action_history: 2
action_dim: 2
obs_dim: 4

model:
  hidden_dim: 256
  h_emb_dim: 256
  time_emb_dim: 32
  z_embed_dim: 32
  latent_type: categorical
  num_categories: 2
  z_dim: 4

reference:
  type: continuation
  alpha: 0.8
  sigma: 0.05
  learn_alpha: false
  learn_sigma: false

loss:
  beta_R: 0.01
  beta_z_start: 0.0
  beta_z_end: 0.01
  beta_z_warmup_steps: 10000
  free_nats: 0.1
  tube_training: false
  tube_noise_std_start: 0.0
  tube_noise_std_end: 0.02
  tube_noise_warmup_steps: 5000

optim:
  lr: 0.0003
  batch_size: 256
  max_steps: 100000
  grad_clip: 1.0

inference:
  deterministic: true
  num_samples: 32
  latent_commitment: chunk
```

### 16.2 Push-T default

```yaml
seed: 0
device: cuda
benchmark: pusht_lowdim
chunk_horizon: 16
obs_history: 2
action_history: 2
action_dim: 2

model:
  hidden_dim: 512
  h_emb_dim: 512
  time_emb_dim: 32
  z_embed_dim: 64
  latent_type: continuous
  z_dim: 8
  continuous_prior: learned_conditional_gaussian

reference:
  type: continuation
  alpha: 0.8
  sigma: 0.05
  learn_alpha: true
  learn_sigma: false

loss:
  beta_R: 0.005
  beta_z_start: 0.0
  beta_z_end: 0.001
  beta_z_warmup_steps: 20000
  free_nats: 0.05
  tube_training: true
  tube_noise_std_start: 0.0
  tube_noise_std_end: 0.01
  tube_noise_warmup_steps: 10000

optim:
  lr: 0.0002
  batch_size: 256
  max_steps: 200000
  grad_clip: 1.0

inference:
  deterministic: true
  num_samples: 8
  latent_commitment: episode
  n_exec: 8
```

---

## 17. Acceptance criteria

### 17.1 Code-level acceptance

The following must run on CPU in small mode:

```bash
python -m action_bridge.scripts.generate_toy_delayed --num-contexts 32 --out /tmp/toy_delayed_small.pt
python -m action_bridge.training.train_toy --config-name toy_delayed_categorical optim.max_steps=50 device=cpu
python -m action_bridge.eval.eval_toy --checkpoint outputs/.../checkpoints/latest.pt device=cpu
pytest tests -q
```

### 17.2 Toy scientific acceptance

For the delayed-branch toy benchmark, produce a table comparing:

- direct BC;
- autoregressive BC;
- BC + smoothness;
- path-KL no latent;
- path-KL categorical latent;
- path-KL continuous latent;
- path-KL latent + tube training.

Report:

```text
goal_error
collision_rate
hybrid_rate
mode_switch_rate
acceleration_energy
jerk_energy
path_KL_energy
```

Expected qualitative outcome:

- latent path-KL models should generate coherent top or bottom paths from ambiguous pre-fork histories;
- direct BC and no-latent models may average, collide, or switch;
- BC + smoothness may be smooth but should not solve mode commitment as cleanly.

### 17.3 Annular scientific acceptance

Report:

```text
mode_probability_calibration
cw/ccw empirical sample rates
hybrid_rate
collision_rate
goal_error
```

For continuous `z`, include latent-space visualizations showing whether sampled regions of `z` correspond to coherent clockwise/counterclockwise generated paths.

### 17.4 Push-T acceptance

At minimum, produce offline metrics for Push-T.

Preferred: closed-loop Push-T evaluation with:

```text
max_overlap/final_overlap/success_rate
action_acceleration
action_jerk
chunk_boundary_discontinuity
path_KL_energy
```

The pilot is positive if path-KL models are competitive with BC-style baselines on task performance while improving smoothness, receding-horizon continuity, or robustness.

---

## 18. What not to overclaim

In code comments, README, and logs, do not call this an exact Schrödinger Bridge solver.

Use names like:

```text
Path-KL Action Bridge
Latent Controlled Action Bridge
SB-inspired Action Bridge Policy
```

The implemented objective is an amortized path-space KL/control-energy objective relative to a reference action process. It is inspired by the Schrödinger Bridge/path-space optimal-control interpretation but does not solve the full SB problem.

---

## 19. README content to generate

Update `README.md` with:

1. One-paragraph project description.
2. Installation instructions.
3. Toy dataset generation commands.
4. Training commands.
5. Evaluation commands.
6. Description of categorical and continuous latent variants.
7. Explanation of reference processes.
8. Explanation that no Sinkhorn/IPF is used.
9. Description of metrics and expected figures.

---

## 20. Implementation priority

Implement in this order:

1. Toy delayed-branch dataset loader/generator.
2. Chunking utilities.
3. Reference processes.
4. Path-KL policy without latent.
5. Categorical latent VAE version.
6. Continuous latent VAE version.
7. Toy evaluation metrics and visualizations.
8. Baselines.
9. Annular dataset.
10. Tube perturbation training.
11. Push-T adapter.
12. Push-T training/evaluation.

Do not start with Push-T. First make the delayed-branch toy result work and produce figures.

---

## 21. Minimal expected deliverable

The first complete deliverable should include:

```text
- working delayed-branch toy generator
- working path-KL policy
- categorical and continuous latent variants
- at least three baselines
- evaluation script with hybrid/mode-switch/collision metrics
- plots showing multiple samples from the same ambiguous history
- README with commands
```

Only after this should Push-T integration be considered complete.
