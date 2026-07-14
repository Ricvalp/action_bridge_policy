# Training and Retraining Ablation Plan

Keep the number of new training experiments small. Each experiment should be run with at least 3 seeds if compute allows. Use the same Push-T evaluation protocol as the current checkpoint.

Primary metrics:

- success rate
- max/final reward
- jerk and acceleration
- chunk-boundary jump
- control energy / path KL
- reference/control norm ratio
- parameter count and inference time

---

## 1. Limited-Capacity Contact Bridge vs Limited-Capacity Baselines

Purpose: test whether the contact-Langevin reference provides useful inductive bias when expressive capacity is limited.

Train a small contact bridge with total capacity comparable to the learned reference module, not the full 52M-parameter model.

Suggested target:

```text
total parameters: ~0.8M to 1.5M
```

This is comparable to the current learned reference process scale.

Variants:

1. **Small Contact-Langevin Bridge**
   - small history encoder
   - small control network
   - no latent or `z_dim <= 2`
   - learned `m`, `K`, `gamma`

2. **Small Autoregressive BC**
   - matched parameter count
   - same history/action inputs
   - no reference, no path KL

3. **Small Direct Chunk BC**
   - matched parameter count
   - same history inputs
   - predicts the full chunk directly

Positive result:

- The small contact bridge matches or beats small AR/direct BC, especially on smoothness, boundary continuity, and robustness.

Negative result:

- Small AR BC performs the same or better. Then the reference may not be adding much beyond autoregression.

---

## 2. Reference-Structure Ablation

Purpose: determine which part of the contact reference matters.

Train matched models with the same encoder/control capacity and only change the reference structure.

Variants:

1. **Full learned contact reference**
   - learned `m(h,k)`, learned diagonal `K(h,k)`, learned scalar `gamma(h,k)`.

2. **No damping**
   - force `gamma = 0` during training and inference.

3. **No potential**
   - force `K = 0`; keep damping.

4. **Fixed damped continuation**
   - no learned `m`, `K`, or `gamma`; use a fixed low-acceleration/damped continuation reference.

Positive result:

- Full learned contact reference gives better success/smoothness/path-KL Pareto behavior than no-damping and no-potential variants.

---

## 3. Path-KL Strength and Noise Calibration

Purpose: test whether the model is truly learning a minimum-control deformation or whether the path-KL is too weak.

Train with a small sweep:

```text
beta_kl in {0, 1e-4, 1e-3, 1e-2}
```

Optionally pair with:

```text
sigma in {3, 7}
```

Do not expand this into a large grid unless results are ambiguous.

Positive result:

- Increasing `beta_kl` reduces control energy, jerk, or boundary jumps without collapsing success.
- A clear Pareto frontier appears between task performance and control energy.

Negative result:

- Only `beta_kl = 0` works, or any meaningful KL penalty destroys success. Then the learned reference is not strong enough.

---

## 4. Coordinate-System Ablation

Purpose: test whether the contact reference should live in absolute action space or in a task-aware coordinate system.

Train:

1. **Absolute action coordinates**
   - current setup: `q` is absolute pusher target position.

2. **Object-centered coordinates**
   - `q = action - block_position`.

3. **Block-frame coordinates**
   - rotate into the T-block frame using block orientation.

Keep the policy class and capacity fixed.

Positive result:

- Object-centered or block-frame reference improves contact approach, smoothness near contact, and generalization to initial block poses.

Interpretation:

- If object-centered coordinates help, the reference process is more naturally a contact/manipulation prior than an absolute-pixel prior.
