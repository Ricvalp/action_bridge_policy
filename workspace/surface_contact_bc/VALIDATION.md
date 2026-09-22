# Engineering checks (2026-09-15)

These are implementation checks, not results establishing an advantage for any
generator. The full matched, multi-seed/low-data/OOD comparison has **not** run.

- 100/100 seeded nominal expert episodes passed the collection quality checks.
- Both generators passed actual tiny-demo overfit tests, including standard
  epsilon-DDIM and optional clean-target DDIM.
- 69 new tests passed, including the three opt-in overfit checks. All method
  adapters trained briefly on CPU, reloaded checkpoints and executed
  real simulator steps. Residual diffusion's frozen-reference checkpoint is
  self-contained. Geometry, data splits, no privileged losses, metrics, plots
  and reference-only overlays have focused tests.
- 69 relevant existing core/toy/Push-T/dissipative tests passed. The complete
  existing `tests/` suite stops during collection with 12 unrelated missing or
  mismatched IsaacLab/RLBench backend API imports. No backend/core source changed.

Exploratory seed-0 runs used 128 demos, horizon 8 and two executed targets.
The approximately 303k-parameter bridge, trained for 10k updates at lr=0.001,
had validation target MSE 2.53e-7 m² and completed four ID contexts cleanly.
The approximately 304k-parameter epsilon-DDIM had validation MSE 6.99e-6 m²;
clean-target DDIM reached 4.91e-6 m². **Both diffusion runs failed the terminal
task criterion on those four contexts**, despite passing the tiny overfit test.
Small chunk errors accumulate in feedback; held-out chunk MSE is not success.
No indexing/coordinate mismatch was found in the shared execution pipeline.

Thus, do not interpret the bridge/diffusion difference as evidence for the
reference hypothesis yet. The diffusion baseline needs further ID convergence
checks, with tuning on validation only, followed by all ablations, matched seeds
and data budgets. No large OOD sweep was started. Keep final reporting contexts
separate from these repeatedly inspected engineering contexts.
The final trainer also isolates minibatch RNG from model/noise RNG; the exploratory
runs above preceded that bookkeeping change and are not the final sweep.

Local artifacts (ignored by Git) are under `runs/`: `dev-ab-matched-lr1e3`,
`dev-dp-lr1e3`, and `dev-dp-sample-lr1e3`; each has checkpoint, config, validation
metrics, ID metrics, traces, plots and a GIF. The generated collection is
`datasets/development`. Earlier short diagnostic runs are retained separately.
