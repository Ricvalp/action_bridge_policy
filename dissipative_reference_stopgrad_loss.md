# Stop-Gradient Reference Loss for a Dissipative Contact-Bridge Policy

This note describes a modified training loss for a dissipative action-bridge policy. The goal is to prevent the learned reference process from collapsing into a full behavior-cloning policy, while still allowing the reference to adapt to demonstration-derived passive structure.

## 1. Notation

For each demonstration chunk, define the action phase-space state

\[
x_k = (q_k,p_k),
\]

where typically

\[
q_k = a_k, \qquad p_k = a_k-a_{k-1}.
\]

Let

\[
y_k = x^*_{k+1}
\]

be the demonstrated next phase-space state.

The policy transition is decomposed as

\[
\mu_{\theta,\eta}(x_k,h,k)
=
f_\eta(x_k,h,k)+B u_\theta(x_k,h,k),
\]

where:

- \(f_\eta\) is the learned dissipative/contact reference transition;
- \(u_\theta\) is the learned task/control residual;
- \(B\) maps the residual control into the transition space;
- \(Q\) is the reference transition covariance / path-KL metric;
- \(h\) is the observation-action history.

The intended interpretation is

\[
\text{policy} = \text{dissipative reference} + \text{minimal task correction}.
\]

The issue with a simple end-to-end likelihood loss is that \(f_\eta\) and \(u_\theta\) are not identifiable: if \(f_\eta\) is expressive, it can learn the full demonstrated transition and force \(u_\theta\approx 0\). This gives good BC loss but destroys the path-KL/control interpretation.

## 2. EMA reference and stop-gradient control loss

Maintain an exponential-moving-average reference network

\[
\bar\eta \leftarrow \rho \bar\eta + (1-\rho)\eta,
\]

with \(\rho\in[0.99,0.999]\). Let

\[
f_{\bar\eta}
\]

be the EMA reference.

When training the control residual, detach the reference:

\[
\mathcal L_u
=
\frac12
\left\|
y_k - \operatorname{sg}(f_{\bar\eta}(x_k,h,k)) - B u_\theta(x_k,h,k)
\right\|^2_{Q^{-1}}
+
\frac{\beta}{2}
\left\|B u_\theta(x_k,h,k)\right\|^2_{Q^{-1}}.
\]

The first term says: given the current passive reference, learn the residual needed to match the demonstration. The second term is the path-KL/control-energy penalty.

This is closer to the SB/control interpretation than fully end-to-end training, because during the control update the reference is treated as fixed.

## 3. Reference loss

The reference should not learn the full demonstrated transition \(y_k\). Instead, it should learn a demonstration-derived passive target

\[
\bar y_k.
\]

Use

\[
\mathcal L_{\mathrm{ref}}
=
\frac12
\left\|\bar y_k - f_\eta(x_k,h,k)\right\|^2
+
\lambda_{\mathrm{diss}}\mathcal L_{\mathrm{diss}}
+
\lambda_{\mathrm{slow}}
\left\|f_\eta(x_k,h,k)-f_{\bar\eta}(x_k,h,k)\right\|^2.
\]

The total loss is

\[
\mathcal L
=
\mathcal L_u
+
\lambda_{\mathrm{ref}}\mathcal L_{\mathrm{ref}}.
\]

Typical first values:

\[
\beta\in[10^{-3},10^{-1}],\qquad
\lambda_{\mathrm{ref}}\in[0.1,1],
\]

\[
\lambda_{\mathrm{diss}}\in[10^{-4},10^{-2}],\qquad
\lambda_{\mathrm{slow}}\in[10^{-3},10^{-1}].
\]

Tune \(\beta\) and \(\lambda_{\mathrm{ref}}\) first. Keep \(\lambda_{\mathrm{diss}}\) small unless the learned reference visibly violates damping/passivity.

## 4. What \(\bar y\) should represent

\(\bar y_k\) should be the part of the demonstrated transition that can plausibly be explained by passive dissipative/contact dynamics.

It should not be the full expert transition. If

\[
\bar y_k = y_k,
\]

then \(f_\eta\) is again encouraged to become a behavior-cloning policy.

Instead, \(\bar y_k\) should be a filtered, projected, or phase-conditioned version of the demonstration transition. It should satisfy the rough principle

\[
\bar y_k \approx \text{passive component of the demo},
\]

while

\[
y_k-\bar y_k \approx \text{active task correction}.
\]

## 5. Three Push-T choices for \(\bar y\)

All three options below derive \(\bar y\) from demonstrations.

### Choice 1: Damped-continuation projection

This is the cleanest first option.

Let

\[
p_k = a_k-a_{k-1},
\qquad
p^*_{k+1}=a^*_{k+1}-a^*_k.
\]

Project the demonstrated next velocity onto the damped-continuation family

\[
\mathcal C_k = \{\alpha p_k : 0\le \alpha \le \alpha_{\max}\}.
\]

Compute

\[
\alpha_k^*
=
\operatorname{clip}
\left(
\frac{\langle p^*_{k+1},p_k\rangle}{\|p_k\|^2+\epsilon},
0,
\alpha_{\max}
\right).
\]

Then set

\[
\bar p_{k+1}=\alpha_k^*p_k,
\]

\[
\bar q_{k+1}=q_k+\bar p_{k+1},
\]

and

\[
\bar y_k=(\bar q_{k+1},\bar p_{k+1}).
\]

Interpretation: the reference learns the part of the demonstration that is explainable as damped continuation of the current action trend. Any rotation, contact maneuver, or aggressive task correction must be learned by \(u_\theta\).

Recommended starting values:

\[
\alpha_{\max}=1.0
\]

or, if the policy oscillates,

\[
\alpha_{\max}=0.8.
\]

If \(\|p_k\|\) is very small, use \(\bar p_{k+1}=0\) or fall back to Choice 2.

### Choice 2: Smoothed demonstration path target

Smooth the demonstrated action path offline using an EMA or Savitzky-Golay filter:

\[
\tilde a_{0:H}=\operatorname{Smooth}(a^*_{0:H}).
\]

Then define

\[
\bar q_{k+1}=\tilde a_{k+1},
\]

\[
\bar p_{k+1}=\tilde a_{k+1}-\tilde a_k,
\]

and

\[
\bar y_k=(\bar q_{k+1},\bar p_{k+1}).
\]

Interpretation: the reference learns the low-frequency, smooth component of the demonstration. The residual learns the higher-frequency task corrections that the smooth passive component cannot explain.

This option is simple and robust, but it is less theoretically clean than Choice 1 because the smoothed target may still contain too much task-specific behavior. Use stronger smoothing if \(u_\theta\) collapses toward zero.

Suggested settings:

- EMA smoothing factor: \(0.7\) to \(0.95\);
- Savitzky-Golay window: 5 to 11 steps;
- use a larger smoothing window near contact/stabilization if phase labels are available.

### Choice 3: Contact-direction projection

This option is more Push-T specific.

Use demonstration observations to estimate a contact/pushing direction \(e_k\). Examples:

\[
e_k = \frac{o^*_{k+1}-o^*_k}{\|o^*_{k+1}-o^*_k\|+\epsilon},
\]

where \(o_k\) is the T-block center, or

\[
e_k = \frac{o_{\mathrm{goal}}-o_k}{\|o_{\mathrm{goal}}-o_k\|+\epsilon}.
\]

Decompose the demonstrated action velocity:

\[
p^*_{k+1,\parallel}
= \langle p^*_{k+1},e_k\rangle e_k,
\]

\[
p^*_{k+1,\perp}
= p^*_{k+1}-p^*_{k+1,\parallel}.
\]

Let \(c_k\in[0,1]\) be a contact score, for example based on pusher-block distance. Near contact, \(c_k\approx 1\); far away, \(c_k\approx 0\).

Define

\[
\bar p_{k+1}
=
(1-c_k)p^*_{k+1}
+
c_k\left(\lambda_{\parallel}p^*_{k+1,\parallel}+\lambda_{\perp}p^*_{k+1,\perp}\right),
\]

with

\[
0\le \lambda_{\perp} < \lambda_{\parallel}\le 1.
\]

Then

\[
\bar q_{k+1}=q_k+\bar p_{k+1},
\qquad
\bar y_k=(\bar q_{k+1},\bar p_{k+1}).
\]

Interpretation: near contact, the reference keeps the part of the demonstrated action that is aligned with stable pushing and damps lateral components that may cause chatter, slip, or overshoot. The residual learns the remaining task-specific corrections.

Suggested first values:

\[
\lambda_{\parallel}=0.8,
\qquad
\lambda_{\perp}=0.2.
\]

This choice is best if you have reliable object pose/keypoints or a good contact proxy.

## 6. Recommended order of experiments

Start with Choice 1.

It is the closest to a literal projection onto a dissipative reference class and should make the role of \(u_\theta\) clear.

Then test Choice 2 as a robust practical baseline.

Use Choice 3 if Push-T failures involve contact chatter, overshoot, or lateral instability near the block.

For each option, monitor:

\[
\|u_\theta\|^2,
\qquad
\mathcal L_{\mathrm{KL}},
\qquad
\text{success/coverage},
\]

\[
\text{action acceleration},
\qquad
\text{jerk},
\qquad
\text{chunk-boundary discontinuity}.
\]

If \(u_\theta\to 0\) and success remains high, the reference is probably absorbing the policy. Increase smoothing/damping, lower reference capacity, increase \(\lambda_{\mathrm{diss}}\), or reduce \(\lambda_{\mathrm{ref}}\).

If \(u_\theta\) is very large and success is poor, the reference target is too passive or too far from the demonstrations. Relax the damping/projection.

## 7. Minimal training pseudocode

```python
for batch in dataloader:
    h, x, y, y_bar = batch

    # EMA reference used as fixed mirror/reference for control update
    with torch.no_grad():
        f_ema = ref_ema(x, h, k)

    u = control_net(x, h, k)
    Bu = B(u)

    # Control residual loss: reference is detached
    residual = y - f_ema - Bu
    loss_u = 0.5 * mahalanobis(residual, Q_inv)
    loss_kl = 0.5 * beta * mahalanobis(Bu, Q_inv)

    # Current reference update: target is passive demo-derived y_bar, not y
    f = ref_net(x, h, k)
    loss_ref = 0.5 * mse(f, y_bar)
    loss_ref += lambda_diss * dissipation_loss(ref_net, x, h, k)
    loss_ref += lambda_slow * mse(f, f_ema.detach())

    loss = loss_u + loss_kl + lambda_ref * loss_ref

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_ema(ref_ema, ref_net, rho=0.995)
```

## 8. Summary

The stop-gradient/EMA version is useful because it preserves the intended decomposition:

\[
\text{reference} = \text{passive dissipative component},
\]

\[
\text{control} = \text{active task correction}.
\]

The key is that \(f_\eta\) must not be trained directly toward the full demonstrated next transition. It should be trained toward \(\bar y\), a demonstration-derived passive/contact-damped target.

The most theoretically clean first choice for Push-T is the damped-continuation projection. The most practical baseline is the smoothed demonstration target. The most contact-specific option is the contact-direction projection.
