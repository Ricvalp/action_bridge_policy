# What this experiment measures

The question is whether a learned contact-aligned reference improves imitation
and recovery, and which component causes any improvement. Negative results are
valid. A short smoke run does not establish a generator advantage.

## Physical and command coordinates

The surface has unit outward normal `n`, tangent `(n_y, -n_x)`, and offset `c`.
`n·x-c >= 0` is free space; negative distance is physical penetration. A fixed
PD controller tracks the absolute target `q`. It is not the physical position
`x`. A unilateral spring/damper never attracts the point; smooth Coulomb friction
opposes tangential motion. The nominal physics step is 2 ms, controller period
40 ms, and episode duration 6 s. All parameters are saved with each context.

Every policy observes physical position/velocity, normal/offset, signed distance,
tangential goal, normal force, elapsed time, and the same previous-target history.
Randomized mass, friction, contact stiffness/damping and controller gains are
hidden. The independent expert can use those parameters; its task logic and
force compensation never enter a learner loss.

Training normals are -20°, 0°, 20° from vertical; held-out orientation sweeps
use ±45° and ±65°. Training/validation/test seed blocks are disjoint. Low-data
normalization is fitted only on the selected training prefix. A single scalar
action scale preserves Euclidean rotation geometry.

## Learned reference and losses

Internal `q` is the normalized absolute target; `p` is its finite difference
per command index (`dt=1`), not physical velocity. The inherited semi-implicit
Action Bridge dynamics use a general, unprojected 2D whitened residual `u`:

```text
f_ref = -k_N (n·q - c_normalized - d_star/scale) n
        -gamma_N (n·p) n - gamma_T (t·p) t
p_next = p + f_ref + sigma*u
q_next = q + p_next
```

History/chunk-phase networks learn bounded `gamma_N`, `gamma_T`, `k_N`, and
`d_star`. There is no damping ordering and no tangential spring/goal attractor.
The goal may condition the learned scalar coefficients via the shared history,
but cannot directly produce tangential reference attraction. Both damping
directions start equally weak; learned differences are measured, not assumed.
Positive damping dissipates command velocity at fixed coefficients. A changing
potential/offset can inject energy; the bounds and fixed-reference test are not
a proof of passivity for the complete history-conditioned policy.

Bridge training reuses the existing teacher-forced q/p path losses and optional
free-running chunk MSE. Path sums are averaged over horizon × action dimensions.
With `sigma=1, dt=1, lambda_q=1`, the combined q/p fitting term equals normalized
teacher-forced target MSE. The objective adds `beta_kl * mean(0.5*u²)` and
`lambda_unroll * generated_chunk_MSE`, with no latent or tube supervision.
CLI defaults are explicit in `config.py`; `--config` can override reference
bounds, loss weights and model settings for validation-only searches.

World and contact-frame diffusion reuse the repository's conditional temporal
U-Net and Diffusers DDIM (100 cosine noise levels, epsilon prediction, eta=0).
They learn ordinary noise-prediction BC, with EMA and the same action chunks.
Contact-frame diffusion changes coordinates, not available information.
An optional `prediction_type=sample` uses standard clean-target (x0) denoising
MSE instead. Both corruption and DDIM sampling still use Diffusers; this choice
adds no contact-specific information or training penalty.

Residual diffusion freezes the selected full bridge's reference and history
encoder. Exact inverse command dynamics turn expert targets into whitened
control sequences; DDIM learns those sequences and the frozen dynamics decode
them. Residual statistics are fitted on training windows only. This comparison
has extra reference-training cost and frozen parameters, both recorded; do not
present it as equal total compute without accounting for that cost.

## Reading the comparisons

- Full vs no-KL tests the control-energy term, keeping the architecture fixed.
- Full vs isotropic tests learned anisotropy, retaining the normal-only spring.
- Reference-only removes the residual from the exact fitted model. It is not an
  independently optimized competing policy; it need not reach the tangent goal.
- World vs contact-frame diffusion tests whether changing coordinates suffices.
- Full vs residual diffusion tests generator choice with the same reference.
  If they match, the evidence favors reference structure, not Action Bridge
  specifically. The same applies if simpler coordinate structure closes the gap.

For a defensible comparison use the same demo subsets, seeds, horizon, execution
schedule, optimizer/update budget, held-out contexts and candidate count.
Minibatch draws use a separate seeded generator, independent of model/noise RNG.
Tune capacity, diffusion steps, and loss weights on validation only. Default primary
sizes are approximately matched: full bridge 303,056 parameters, isotropic
302,839, and world/frame diffusion 304,482 (within 1%). Counts are saved;
capacity sweeps remain necessary before claiming a structural advantage. OOD results must never
choose checkpoints or hyperparameters.

## Metrics and limitations

Task success requires terminal tangent error <=2.5 cm and speed <=0.08 m/s.
Contact success requires contact in >=90% of the final 0.5 s. Clean success
requires both, penetration <=2 cm, peak force <=30 N and at most one contact
loss. Thresholds are recorded; they are not fitted from evaluation outcomes.

Peak force/penetration include physics substeps. RMS force and velocity metrics
are controller-rate sampled, as are contact losses and chatter (25 Hz resolution).
Contact-loss duration starts only after first
engagement; never-contact episodes must therefore be read with contact success,
not mistaken for perfectly stable contact. Chatter excludes tiny velocity flips.
Recovery requires pre-impulse contact and sustained return near the previous
normal distance with low normal speed. Missing recovery time is `null`, with
eligibility/success counts; an unrecovered rollout is not assigned zero seconds.

Reference “forces” are normalized command accelerations, not Newtons. Their
energies and KL sum only executed portions of chunks; physical quantities are
reported separately. Replan discontinuity compares replan-boundary target jumps
with within-chunk jumps. No timed tangent profile is supplied, so tracking RMSE
is `null`; endpoint error and completion time measure task performance instead.

Parameter/energy time series and phase averages show whether anisotropy emerges
and whether the residual overrides the reference. An average residual/reference
norm ratio can be large when reference force is near zero; interpret it together
with both absolute energy measurements, never alone.
