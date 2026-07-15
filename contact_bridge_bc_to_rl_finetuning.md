# BC-to-RL Fine-Tuning for Latent Contact-Langevin Action Bridge Policy

## Scope

This document gives implementation instructions for fine-tuning a behavior-cloned Schrödinger Bridge / action-bridge policy with a dissipative contact action-path reference on Push-T.

The assumed pretrained actor is the current **Latent Contact-Langevin Action Bridge**:

- Policy class: `ActionBridgePolicy`.
- Reference: `contact_langevin` in absolute action coordinates.
- Action: absolute 2D pusher target position.
- State observation: low-dimensional Push-T state, usually `[pusher_x, pusher_y, block_x, block_y, block_theta]`.
- Chunk horizon: `H = 16`.
- Receding-horizon execution: currently `n_exec = 8`.
- Dynamics:

\[
p_{k+1} = p_k + dt\left(f_R(q_k,p_k,h,k)+\sigma u_\theta(q_k,p_k,h,z,k)\right)
\]

\[
q_{k+1}=q_k+dt\,p_{k+1}.
\]

The RL fine-tuning algorithm below keeps the same interpretation:

\[
\text{policy path} = \text{dissipative reference path} + \text{learned control deformation}.
\]

The central change from ordinary SAC is that the entropy regularizer is replaced by a **path-space KL / control-energy penalty** relative to the dissipative contact reference.

---

## Recommendation

Start with **Path-AWAC** or **conservative Path-SAC**, not aggressive PPO-style fine-tuning.

The current BC policy already reaches about `0.8` Push-T success. The goal of RL should not be unrestricted exploration. The goal should be:

\[
\text{increase return while preserving dissipative action-path structure}.
\]

The first RL objective should therefore be conservative:

\[
\max_\theta\;\mathbb E\left[Q_\phi(h,A)-\alpha c_{\mathrm{ref}}(\Xi,h)-\lambda c_{\mathrm{BC}}(\Xi,h)\right].
\]

Here:

- \(\Xi=(z,q_{0:H},p_{0:H},u_{0:H})\) is the internal generated path.
- \(A\) is the executed action sequence, usually the first `n_exec` actions.
- \(c_{\mathrm{ref}}\) is the path-KL/control-energy cost to the dissipative reference.
- \(c_{\mathrm{BC}}\) keeps the fine-tuned actor close to the pretrained BC actor early in training.

---

## Algorithm Name

Use one of these names in code and notes:

- `ContactBridgeSAC`
- `PathKLSAC`
- `DissipativeBridgeSAC`
- `ContactBridgeAWAC`

For the first implementation, use:

```text
ContactBridgeSAC with BC regularization
```

or, if you want the safer variant:

```text
ContactBridgeAWAC
```

---

## Key Objects

### 1. Reference process

The pretrained actor has a learned contact-Langevin reference:

\[
f_R(q,p,h,k)=-\nabla_q V_\eta(q,h,k)-\gamma_\eta(h,k)p.
\]

The policy applies a residual control:

\[
f_\theta(q,p,h,z,k)=f_R(q,p,h,k)+\sigma u_\theta(q,p,h,z,k).
\]

For RL fine-tuning, initially **freeze the reference**:

```python
for p in policy.reference.parameters():
    p.requires_grad_(False)
```

Reason: the reference is the robot prior. If it is updated by reward gradients immediately, it may stop being a passive/dissipative reference and become just another actor network.

Recommended first trainable actor modules:

```text
trainable:
  - latent prior network p(z | h)
  - latent embedding network
  - control residual network u_theta
  - optionally final layers of history encoder

frozen initially:
  - contact reference m(h,k), K(h,k), gamma(h,k)
  - action normalization statistics
  - observation normalization statistics
  - a frozen copy of the original BC actor
```

After stable RL improvement, try unfreezing the last reference layers with a very small learning rate.

---

### 2. Path-KL / reference cost

The current architecture uses `control_is_whitened=true`. In that case, the applied residual acceleration is:

\[
\sigma u_\theta.
\]

The control-energy/path-KL term is approximately:

\[
c_{\mathrm{ref}}(\Xi,h)
=
\frac12\sum_{k=0}^{K-1}\|u_{\theta,k}\|^2.
\]

Use either:

\[
K=n_{\mathrm{exec}}
\]

for the executed part of the chunk, or

\[
K=H
\]

for the full planned chunk.

Recommended first choice:

\[
K=n_{\mathrm{exec}}=8.
\]

Reason: the critic target evaluates the executed transition, so the regularizer should primarily penalize the part of the plan that actually affects the environment.

Code helper:

```python
def compute_ref_cost(info, upto):
    # info["u_seq"] shape: [B, H, action_dim]
    u = info["u_seq"][:, :upto]
    return 0.5 * (u ** 2).sum(dim=-1).sum(dim=-1)
```

Normalized version, often easier to tune:

```python
def compute_ref_cost_mean(info, upto):
    u = info["u_seq"][:, :upto]
    return 0.5 * (u ** 2).sum(dim=-1).mean(dim=-1)
```

Use the normalized version for alpha tuning. Log both.

---

### 3. BC regularization cost

Keep a frozen copy of the pretrained policy:

```python
bc_policy = deepcopy(policy).eval()
for p in bc_policy.parameters():
    p.requires_grad_(False)
```

Use a simple initial BC regularizer:

\[
c_{\mathrm{BC}}
=
\frac{1}{H d_a}\|A_\theta-A_{\mathrm{BC}}\|^2.
\]

Code:

```python
with torch.no_grad():
    bc_actions, bc_info = bc_policy.forward_rl(
        obs_hist, act_hist,
        deterministic=True,
        return_info=True,
    )

c_bc = ((actions - bc_actions) ** 2).mean(dim=(1, 2))
```

This is not a true path KL to the BC policy, but it is stable and good enough for the first fine-tuning run.

A more structural alternative is a control-space BC penalty:

\[
c_{\mathrm{BC,u}}
=
\frac12\sum_k\|u_\theta(x_k,h,z,k)-u_{\mathrm{BC}}(x_k,h,z_{BC},k)\|^2.
\]

Use this later. It is more faithful to the bridge formulation but more sensitive to latent mismatch.

---

## Required Actor API Additions

Add an RL-oriented forward method to `ActionBridgePolicy`.

```python
def forward_rl(
    self,
    obs_hist,
    act_hist,
    deterministic: bool = False,
    sample_latent: bool = True,
    sample_dynamics_noise: bool = False,
    z_override=None,
    return_info: bool = True,
):
    """
    Returns:
        actions: [B, H, action_dim]
        info: dict with:
            z: [B, z_dim]
            prior_mean: [B, z_dim]
            prior_log_std: [B, z_dim]
            q_seq: [B, H+1, action_dim]
            p_seq: [B, H+1, action_dim]
            u_seq: [B, H, action_dim]
            ref_accel_seq: [B, H, action_dim]
            path_kl_seq: [B, H]
            path_kl: [B]
            gamma_seq, k_diag_seq, attractor_seq if available
    """
```

Important details:

1. Reparameterize latent samples during actor updates:

```python
z = prior_mean + prior_std * torch.randn_like(prior_mean)
```

2. For evaluation, use either prior mean or fixed episode latent:

```python
z = prior_mean
```

3. Initially use deterministic dynamics given `z`:

```python
sample_dynamics_noise = False
```

4. Use stochasticity through `z`, not raw action noise, as the primary exploration mechanism.

5. Return `u_seq` before multiplying by `sigma` if `control_is_whitened=true`, because the KL cost is `0.5 ||u||^2`.

---

## Replay Buffer

Use a chunk-level replay buffer.

Each decision step executes `m = n_exec` low-level actions, then replans.

Store:

```python
Transition = {
    "obs_hist": obs_hist_t,             # [obs_history, obs_dim]
    "act_hist": act_hist_t,             # [action_history, action_dim]
    "exec_actions": actions[:m],        # [m, action_dim]
    "planned_actions": actions,         # [H, action_dim], optional
    "reward_m": discounted_chunk_reward,
    "next_obs_hist": obs_hist_tp,
    "next_act_hist": act_hist_tp,
    "done": done,
    "discount_m": gamma ** m,
    "success": success_flag,
    "coverage_t": coverage_before,
    "coverage_tp": coverage_after,
    "path_kl": path_kl_executed,
    "bc_cost": bc_cost,
}
```

### Reward for Push-T

Use one of two reward definitions.

#### Option 1: chunk-summed environment reward

\[
R_t^{(m)}=\sum_{i=0}^{m-1}\gamma^i r_{t+i}.
\]

This rewards reaching good coverage quickly.

#### Option 2: coverage progress reward

\[
R_t^{(m)}=\mathrm{coverage}_{t+m}-\mathrm{coverage}_t.
\]

Add terminal success bonus:

\[
R_t^{(m)} \leftarrow R_t^{(m)} + b_{success}\mathbf 1[\mathrm{success}].
\]

Recommended first choice:

```text
chunk-summed environment reward
```

because it is closer to the Push-T evaluation metric and easier to debug.

---

## Critic

Use double Q networks.

Input:

```text
obs_hist flattened
act_hist flattened
exec_actions flattened
```

Output:

```text
Q1, Q2 scalar
```

Recommended architecture:

```python
class ChunkQNetwork(nn.Module):
    def __init__(self, obs_history, obs_dim, action_history, action_dim, n_exec, hidden_dim=512):
        super().__init__()
        in_dim = obs_history * obs_dim + action_history * action_dim + n_exec * action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_hist, act_hist, exec_actions):
        x = torch.cat([
            obs_hist.flatten(1),
            act_hist.flatten(1),
            exec_actions.flatten(1),
        ], dim=-1)
        return self.net(x).squeeze(-1)
```

Use target networks:

```python
q1_target = deepcopy(q1)
q2_target = deepcopy(q2)
```

Soft update:

```python
for p, p_targ in zip(q.parameters(), q_target.parameters()):
    p_targ.data.mul_(1 - tau).add_(tau * p.data)
```

Recommended:

```yaml
tau: 0.005
critic_lr: 3.0e-4
critic_hidden_dim: 512
critic_batch_size: 256
```

If Q overestimation is severe, increase critic width or use REDQ-style ensembles later. Do not start there.

---

## Contact-Bridge SAC Objective

### Critic target

Sample next actions from the current or target actor:

\[
\Xi'\sim P_\theta(\cdot\mid h')
\]

\[
A'=G(\Xi')_{0:m-1}.
\]

Compute:

\[
c'_{\mathrm{ref}}=\frac12\sum_{k=0}^{m-1}\|u'_k\|^2.
\]

Compute optional BC cost:

\[
c'_{\mathrm{BC}}=\frac{1}{Hd_a}\|A' - A'_{BC}\|^2.
\]

Target:

\[
y
=
R_t^{(m)}
+
\gamma^m(1-d)
\left[
\min_j Q_{\bar\phi_j}(h',A')
-
\alpha c'_{\mathrm{ref}}
-
\lambda c'_{\mathrm{BC}}
\right].
\]

Critic loss:

\[
\mathcal L_Q
=
\mathbb E\left[(Q_{\phi_1}(h,A)-y)^2+(Q_{\phi_2}(h,A)-y)^2\right].
\]

### Actor loss

Sample from the actor using reparameterization:

\[
\Xi\sim P_\theta(\cdot\mid h).
\]

Execute part:

\[
A=G(\Xi)_{0:m-1}.
\]

Actor loss:

\[
\mathcal L_\pi
=
\mathbb E\left[
\alpha c_{\mathrm{ref}}(\Xi,h)
+
\lambda c_{\mathrm{BC}}(\Xi,h)
-
\min_j Q_{\phi_j}(h,A)
\right].
\]

This is the central algorithm.

It is SAC-like, but instead of maximizing raw action entropy, it penalizes deviation from the dissipative contact reference.

---

## Alpha Tuning

In SAC, alpha targets entropy. Here alpha should target a path-control budget.

Let:

\[
\bar c_{\mathrm{ref}} = \mathbb E[c_{\mathrm{ref}}].
\]

Set target budget:

\[
\kappa = \text{mean BC path-KL on successful validation rollouts}.
\]

Practical starting point:

```text
kappa = median path_kl_executed from pretrained BC rollouts
```

Alpha loss:

\[
\mathcal L_\alpha
=
\log \alpha\;\left(\kappa - c_{\mathrm{ref}}\right)_{\mathrm{detach}}.
\]

Use:

```python
alpha = log_alpha.exp()
alpha_loss = (log_alpha * (target_ref_cost - ref_cost.detach())).mean()
```

Check sign carefully:

- If `ref_cost > target_ref_cost`, alpha should increase.
- If `ref_cost < target_ref_cost`, alpha may decrease.

A robust implementation:

```python
alpha_loss = -(log_alpha * (ref_cost.detach() - target_ref_cost)).mean()
```

Then:

```python
# ref_cost too high -> gradient pushes log_alpha up
```

Recommended initialization:

```yaml
alpha_init: 0.05
alpha_lr: 1.0e-4
target_ref_cost: bc_validation_ref_cost_executed
alpha_min: 1.0e-4
alpha_max: 10.0
```

Clamp alpha for early experiments.

---

## BC Regularization Schedule

Start with BC regularization strong enough to prevent collapse.

Example:

```yaml
lambda_bc_start: 10.0
lambda_bc_end: 0.5
lambda_bc_anneal_steps: 100000
```

If reward improves but action smoothness degrades, keep `lambda_bc` higher.

If reward does not improve and actor barely changes, lower `lambda_bc` faster.

---

## Training Schedule

### Stage 0: Evaluate pretrained BC actor

Run at least:

```text
100 evaluation episodes
```

Record:

```text
success_rate
mean_max_reward
mean_final_reward
mean_path_kl_executed
mean_control_energy
mean_jerk
mean_boundary_discontinuity
latent_entropy
wrong_side_go_around diagnostic
```

This becomes the safety baseline.

---

### Stage 1: Build offline replay

Use a mixture of:

1. Demonstration chunks, if rewards can be reconstructed.
2. Rollouts from the pretrained BC actor.
3. Slightly perturbed BC rollouts.

Recommended minimum:

```text
200-500 BC policy episodes
```

Because Push-T episodes are short, this is feasible.

For perturbations:

```text
sample z from prior instead of deterministic prior mean
small action-space Gaussian noise only if needed
no large raw action noise near contact
```

Store chunk transitions with `m=n_exec=8`.

---

### Stage 2: Critic pretraining

Freeze actor.

Train Q networks on the offline replay buffer.

Use target:

\[
y=R_t^{(m)}+\gamma^m(1-d)\min_j Q_{\bar\phi_j}(h', A'_{BC}).
\]

Here `A'_BC` comes from the frozen BC actor.

Train for:

```yaml
critic_pretrain_steps: 20000 to 100000
```

Stop when Q predictions correlate with empirical returns.

Diagnostics:

```text
Q(h, BC actions) vs realized return
Q scale
Bellman loss
Q overestimation on random bad chunks
```

---

### Stage 3: Conservative online RL

Start online environment interaction.

At each iteration:

1. Collect one or more episodes using the current actor.
2. Add chunk transitions to replay.
3. Run gradient updates.
4. Evaluate every fixed number of environment steps.

Recommended settings:

```yaml
n_exec: 8
actor_lr: 1.0e-5 initially
critic_lr: 3.0e-4
batch_size: 256
updates_per_env_step: 1
online_fraction_in_batch: 0.25 to 0.5
gamma: 0.99
polyak_tau: 0.005
actor_update_delay: 2
max_grad_norm: 1.0
```

Use a low actor LR. The actor is already competent.

---

### Stage 4: Gradual unfreezing

Initial trainable modules:

```text
control_net last layers
latent prior
latent embedding
```

After improvement is stable:

```text
unfreeze all control_net
unfreeze history encoder last layer
```

Only after that, optionally:

```text
unfreeze reference attractor m(h,k)
keep K and gamma strongly regularized
```

Do not unfreeze all reference parameters early.

---

## Collection Policy

For online collection, use stochastic latent exploration:

```python
actions, info = policy.forward_rl(
    obs_hist,
    act_hist,
    deterministic=False,
    sample_latent=True,
    sample_dynamics_noise=False,
    return_info=True,
)
```

For evaluation:

```python
actions, info = policy.forward_rl(
    obs_hist,
    act_hist,
    deterministic=True,
    sample_latent=False,
    sample_dynamics_noise=False,
    return_info=True,
)
```

If exploration is insufficient, add small residual-control noise rather than raw action noise:

\[
u_k \leftarrow u_k + \epsilon_k,
\qquad
\epsilon_k\sim\mathcal N(0,\sigma_u^2I).
\]

Recommended:

```yaml
u_noise_std: 0.05 initially
u_noise_decay: true
```

Avoid large action-space noise near contact. It destroys the point of the dissipative reference.

---

## Main Training Loop Pseudocode

```python
# pretrained policy loaded
policy = load_pretrained_action_bridge(checkpoint)
bc_policy = deepcopy(policy).eval()
freeze(bc_policy)

# freeze reference initially
freeze(policy.reference)
partial_unfreeze(policy.control_net, last_n_layers=2)
unfreeze(policy.latent)

q1, q2 = ChunkQNetwork(...), ChunkQNetwork(...)
q1_targ, q2_targ = deepcopy(q1), deepcopy(q2)

replay = ChunkReplayBuffer(...)

# Stage 1: fill replay
for ep in range(num_bc_rollout_episodes):
    traj = rollout_env(policy=bc_policy, stochastic_latent=True)
    replay.add_chunked(traj, n_exec=8)

# Stage 2: critic pretraining
for step in range(critic_pretrain_steps):
    batch = replay.sample(batch_size)
    with torch.no_grad():
        next_actions, next_info = bc_policy.forward_rl(
            batch.next_obs_hist, batch.next_act_hist,
            deterministic=True,
            return_info=True,
        )
        next_exec = next_actions[:, :n_exec]
        target_q = torch.min(
            q1_targ(batch.next_obs_hist, batch.next_act_hist, next_exec),
            q2_targ(batch.next_obs_hist, batch.next_act_hist, next_exec),
        )
        y = batch.reward_m + batch.discount_m * (1 - batch.done) * target_q

    q_loss = mse(q1(batch.obs_hist, batch.act_hist, batch.exec_actions), y) \
           + mse(q2(batch.obs_hist, batch.act_hist, batch.exec_actions), y)
    update(q_loss, q_optim)
    soft_update(q1, q1_targ)
    soft_update(q2, q2_targ)

# Stage 3: online RL
for env_step in range(total_env_steps):
    traj = collect_one_episode(policy, stochastic_latent=True)
    replay.add_chunked(traj, n_exec=n_exec)

    for update_idx in range(updates_per_episode):
        batch = replay.sample_mixed(batch_size, online_fraction=0.5)

        # critic update
        with torch.no_grad():
            next_actions, next_info = policy.forward_rl(
                batch.next_obs_hist,
                batch.next_act_hist,
                deterministic=False,
                sample_latent=True,
                return_info=True,
            )
            next_exec = next_actions[:, :n_exec]
            next_ref_cost = compute_ref_cost_mean(next_info, upto=n_exec)

            bc_next_actions, _ = bc_policy.forward_rl(
                batch.next_obs_hist,
                batch.next_act_hist,
                deterministic=True,
                return_info=True,
            )
            next_bc_cost = ((next_actions - bc_next_actions) ** 2).mean(dim=(1, 2))

            next_q = torch.min(
                q1_targ(batch.next_obs_hist, batch.next_act_hist, next_exec),
                q2_targ(batch.next_obs_hist, batch.next_act_hist, next_exec),
            )

            y = batch.reward_m + batch.discount_m * (1 - batch.done) * (
                next_q - alpha * next_ref_cost - lambda_bc * next_bc_cost
            )

        q_loss = mse(q1(batch.obs_hist, batch.act_hist, batch.exec_actions), y) \
               + mse(q2(batch.obs_hist, batch.act_hist, batch.exec_actions), y)
        update(q_loss, q_optim)

        # actor update, delayed
        if update_idx % actor_update_delay == 0:
            actions, info = policy.forward_rl(
                batch.obs_hist,
                batch.act_hist,
                deterministic=False,
                sample_latent=True,
                return_info=True,
            )
            exec_actions = actions[:, :n_exec]
            ref_cost = compute_ref_cost_mean(info, upto=n_exec)

            with torch.no_grad():
                bc_actions, _ = bc_policy.forward_rl(
                    batch.obs_hist,
                    batch.act_hist,
                    deterministic=True,
                    return_info=True,
                )
            bc_cost = ((actions - bc_actions) ** 2).mean(dim=(1, 2))

            q_pi = torch.min(
                q1(batch.obs_hist, batch.act_hist, exec_actions),
                q2(batch.obs_hist, batch.act_hist, exec_actions),
            )

            actor_loss = (alpha * ref_cost + lambda_bc * bc_cost - q_pi).mean()
            update(actor_loss, actor_optim)

            # alpha update
            alpha_loss = -(log_alpha * (ref_cost.detach() - target_ref_cost)).mean()
            update(alpha_loss, alpha_optim)
            alpha = log_alpha.exp().clamp(alpha_min, alpha_max)

        soft_update(q1, q1_targ, tau)
        soft_update(q2, q2_targ, tau)
```

---

## Safer Alternative: Contact-Bridge AWAC

If SAC is unstable, use AWAC-style actor updates.

### Critic

Train Q with ordinary TD targets using data actions:

\[
y=R_t^{(m)}+\gamma^m(1-d)V(h').
\]

Use:

\[
V(h)=\mathbb E_{A\sim P_{BC}}[Q(h,A)]
\]

or simply:

\[
V(h)=Q(h,A_{BC}).
\]

### Advantage

\[
A(h,A)=Q(h,A)-V(h).
\]

### Actor update

For replay actions \(A\), maximize weighted path likelihood:

\[
\mathcal L_\pi
=
-\mathbb E\left[\exp\left(\frac{A(h,A)}{\lambda_A}\right)\log P_\theta(A\mid h)\right]
+\alpha c_{\mathrm{ref}}.
\]

Because exact full likelihood may be awkward, use the supervised path loss already implemented in `losses.py`, weighted by advantage:

```python
weights = torch.exp(advantage / awac_lambda).clamp(max=max_weight).detach()

bc_path_loss = policy.teacher_forced_path_loss(
    obs_hist=batch.obs_hist,
    act_hist=batch.act_hist,
    future_actions=batch.exec_actions_or_full_chunk,
    weights=weights,
)

loss = bc_path_loss + alpha * ref_cost
```

Recommended AWAC settings:

```yaml
awac_lambda: 0.3 to 1.0
max_weight: 20.0
actor_lr: 3.0e-5
critic_lr: 3.0e-4
```

This is less ambitious than SAC but much more stable.

---

## Which Variant To Try First

Recommended order:

1. **Critic-only evaluation**: train Q and verify it ranks successful BC chunks above failed chunks.
2. **Contact-Bridge AWAC**: safest actor improvement.
3. **Contact-Bridge SAC with frozen reference**: main algorithm.
4. **Contact-Bridge SAC with partially unfrozen reference**: only after stable improvement.

Do not begin with full actor/reference fine-tuning.

---

## Diagnostics

Track these every evaluation interval.

### Task metrics

```text
success_rate
mean_max_reward
mean_final_reward
area_coverage_auc
```

### Path metrics

```text
mean_ref_cost_executed
mean_ref_cost_full
mean_control_energy
mean_action_velocity
mean_action_acceleration
mean_action_jerk
chunk_boundary_discontinuity
```

### Reference usage

```text
gamma_mean
gamma_min/gamma_max
k_diag_mean
k_diag_max
attractor_path_length
ratio_control_to_reference = ||sigma * u|| / (||f_R|| + eps)
```

### Latent diagnostics

```text
prior_entropy
z_std_mean
latent_kl_to_BC_prior
wrong_side_go_around_latent_chunks.png
```

### RL diagnostics

```text
Q_mean_data
Q_mean_policy
Q_target_mean
Bellman_loss
actor_loss
alpha
lambda_bc
bc_cost
TD_error_percentiles
```

### Failure indicators

Stop or reduce actor LR if:

```text
success drops by > 0.1 absolute for two consecutive evals
mean jerk > 2x BC baseline
ref_cost_executed > 3x BC baseline
Q_policy >> realized_return without success improvement
boundary discontinuity > 2x BC baseline
```

---

## Ablations Needed For A Paper

If RL helps, run these ablations:

1. BC only.
2. BC + ordinary action-space Gaussian SAC fine-tuning.
3. BC + AWAC/IQL-style fine-tuning with direct action policy.
4. Contact Bridge BC only.
5. Contact Bridge + RL without path-KL cost.
6. Contact Bridge + RL with path-KL cost but no BC regularizer.
7. Contact Bridge + RL with path-KL and BC regularizer.
8. Optional: reference frozen vs reference partially unfrozen.

The important comparison is:

```text
higher reward at the same or lower jerk / boundary discontinuity / contact instability
```

not just final success rate.

---

## Hyperparameter Starting Point

```yaml
algorithm: contact_bridge_sac
n_exec: 8
chunk_horizon: 16
gamma_rl: 0.99
batch_size: 256
replay_size: 1000000
updates_per_env_step: 1
actor_update_delay: 2

critic:
  hidden_dim: 512
  depth: 3
  lr: 3.0e-4
  target_tau: 0.005
  grad_clip: 1.0

actor:
  lr: 1.0e-5
  train_modules:
    - latent
    - control_net_last_2_layers
  freeze_reference: true
  grad_clip: 1.0

regularization:
  alpha_init: 0.05
  alpha_lr: 1.0e-4
  alpha_min: 1.0e-4
  alpha_max: 10.0
  target_ref_cost: bc_validation_ref_cost_executed
  lambda_bc_start: 10.0
  lambda_bc_end: 0.5
  lambda_bc_anneal_steps: 100000

exploration:
  stochastic_latent: true
  deterministic_dynamics_given_z: true
  u_noise_std: 0.0
  optional_u_noise_std_if_needed: 0.05

data:
  prefill_bc_episodes: 200
  critic_pretrain_steps: 50000
  online_fraction_in_batch: 0.25

evaluation:
  eval_every_env_steps: 5000
  eval_episodes: 50
  deterministic_eval: true
```

---

## Expected Outcomes

A positive result is:

```text
BC success: ~0.80
RL success: >0.85
jerk: similar or lower than BC
ref_cost: controlled, not exploding
boundary discontinuity: similar or lower than BC
```

A weak but still useful result is:

```text
success unchanged
but smoother / lower ref-cost / better contact stability
```

A negative result is:

```text
success improves only by exploiting high-jerk or high-control-energy commands
```

That would mean the RL objective is overpowering the dissipative reference.

---

## Implementation Checklist

### Actor

- [ ] Add `forward_rl` returning `actions` and internal path info.
- [ ] Return `u_seq` before sigma scaling.
- [ ] Return `q_seq`, `p_seq`, `ref_accel_seq`, `path_kl_seq`.
- [ ] Support deterministic and stochastic latent sampling.
- [ ] Support deterministic dynamics given latent.

### Frozen BC copy

- [ ] Load pretrained checkpoint.
- [ ] Deepcopy to `bc_policy`.
- [ ] Freeze all parameters.
- [ ] Use for BC regularization and critic pretraining targets.

### Replay

- [ ] Store chunk-level transitions.
- [ ] Store executed action sequence `[n_exec, action_dim]`.
- [ ] Store chunk reward and `gamma ** n_exec`.
- [ ] Store diagnostics.

### Critic

- [ ] Implement double chunk Q networks.
- [ ] Implement target networks.
- [ ] Pretrain critic before actor updates.
- [ ] Add Q-vs-return diagnostic.

### RL losses

- [ ] Critic TD target includes `- alpha * ref_cost`.
- [ ] Actor loss includes `alpha * ref_cost + lambda_bc * bc_cost - Q`.
- [ ] Alpha targets path-control budget, not entropy.
- [ ] Lambda BC anneals slowly.

### Evaluation

- [ ] Deterministic evaluation with prior mean or fixed episode latent.
- [ ] Log all BC baseline metrics and RL metrics.
- [ ] Stop training when action smoothness/path-KL degrades sharply.

---

## Short Interpretation

The RL fine-tuning algorithm is not ordinary maximum-entropy RL. It is maximum-return learning under a dissipative path-prior constraint:

\[
\max_\theta
\mathbb E[R]
-
\alpha D_{\mathrm{KL}}(P_\theta(\Xi\mid h)\Vert R_h(\Xi))
-
\lambda D_{\mathrm{BC}}(P_\theta,P_{BC}).
\]

Because the bridge policy has a tractable path-control energy,

\[
D_{\mathrm{KL}}(P_\theta\Vert R_h)
\approx
\frac12\mathbb E\sum_k\|u_k\|^2,
\]

the actor can be fine-tuned with SAC-like gradients while retaining the physical interpretation:

```text
RL reward improvement must be paid for by explicit control energy away from passive dissipative robot behavior.
```

That is the non-obvious part worth testing.
