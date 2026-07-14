# Checkpoint-Only Experiments

These experiments use the existing trained checkpoint only. They require no training or retraining. Run all variants with the same evaluation seeds, same number of Push-T rollouts, and the same normalization/denormalization path.

Report for every variant:

- `sim_success_rate`, `sim_max_reward`, `sim_final_reward`
- action velocity, acceleration, jerk
- chunk-boundary jump at replanning times
- mean `path_kl` / control energy
- mean and distribution of `gamma`, `K`, and reference/control norm ratio

---

## 1. Causal Inference-Time Reference Interventions

Purpose: determine whether the learned contact-Langevin reference is doing useful work or whether the control network alone solves the task.

Run the same checkpoint with these inference-time variants:

1. **Full policy**
   - Use learned reference plus learned control.

2. **Reference-only**
   - Set `u_theta = 0`.
   - Roll out only the learned reference dynamics.
   - This tests whether `m(h,k)`, `K(h,k)`, and `gamma(h,k)` encode a plausible passive policy.

3. **Control-only**
   - Set `f_R = 0`, keep the learned control network unchanged.
   - Dynamics become controlled inertial autoregression.
   - If success remains close to the full model, the reference is probably not essential.

4. **No damping**
   - Set `gamma = 0` at inference.
   - Keep the learned potential and control.
   - This isolates the contribution of damping.

5. **No potential**
   - Set `K = 0`, so `-grad V = 0`.
   - Keep damping and control.
   - This isolates the contribution of the learned attractor/potential.

Interpretation:

- If full policy clearly outperforms control-only while also being smoother, the reference is useful.
- If reference-only produces smooth but task-incomplete behavior, that is acceptable.
- If reference-only is nonsensical and control-only keeps high success, the contact reference is probably not carrying much structure.

---

## 2. Receding-Horizon Stability Stress Test

Purpose: test whether the dissipative reference improves chunk-to-chunk stability.

Run the same checkpoint with:

```text
n_exec in {1, 2, 4, 8, 16}
```

Keep everything else fixed.

Metrics:

- success rate
- boundary discontinuity
- jerk
- action acceleration
- control energy per planned chunk
- realized pusher/action mismatch

Interpretation:

- A useful dissipative reference should degrade gracefully as replanning becomes more frequent.
- If performance collapses at `n_exec=1` or `n_exec=2`, the policy may rely on long open-loop chunks rather than stable receding-horizon dynamics.

---

## 3. Latent Causal-Use Test

Purpose: determine whether `z` is actually used for behavior selection or has collapsed.

Run these inference modes:

1. **Prior mean**
   - Use `z = mean[p(z|h)]`.

2. **Episode-sticky sampled latent**
   - Sample `z` once per episode and keep it fixed.

3. **Replan-resampled latent**
   - Resample `z` at every replanning step.

4. **Latent sweep**
   - Fix the same history and evaluate chunks for several manually chosen `z` values along principal prior directions.

Metrics:

- success rate and variance across latents
- action-path diversity
- wrong-side / go-around diagnostic
- contact side consistency across chunks

Interpretation:

- If changing `z` barely changes planned chunks, the latent is not causally active.
- If resampling `z` increases mode switches or jerk, sticky latent commitment is important.

---

## 4. Learned Reference Field Diagnostics

Purpose: inspect whether the learned potential and damping are geometrically meaningful.

No simulator changes are required. Dump the following during closed-loop rollouts and on held-out expert chunks:

```text
m(h,k)
K(h,k)
gamma(h,k)
f_R(q,p,h,k)
sigma * u_theta(q,p,h,z,k)
||f_R|| / (||f_R|| + ||sigma u_theta||)
p^T gamma p
||q - m(h,k)||
```

Plot each quantity against:

- chunk step `k`
- distance between pusher and block
- distance to goal
- contact / no-contact if available
- success versus failure rollouts

Interpretation:

- Good `gamma`: small in free motion, larger near contact or stabilization.
- Good `K`: not always zero, not always saturated, and higher when the attractor is meaningful.
- Good `m`: smooth, task-related, and not merely noisy future-action memorization.
- Good reference/control split: the residual should not dominate the reference everywhere.
