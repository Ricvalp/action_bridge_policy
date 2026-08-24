# Instructions for coding agents

## Repository boundary

This repository owns policy models, training, checkpoint handling, and thin
adapters to simulator backends. The `phi-*` projects under `workspace/` are
independent backend repositories.

- Treat `phi-*` packages and sibling checkouts as immutable dependencies during
  policy work.
- Do not edit, vendor, copy, monkey-patch, or commit backend source unless the
  user explicitly requests a backend API or semantic change.
- Import only public paths documented by the pinned backend release.
- Keep model, optimizer, framework conversion, checkpoint, and policy-adapter
  code in this repository.
- If a required capability is missing, report the public API gap instead of
  importing an undocumented private module.

## Data and native runtimes

- Treat production datasets, simulator installations, and released
  checkpoints as immutable.
- Keep caches, logs, temporary files, videos, and run outputs below explicit
  workspace or caller-provided roots.
- Never install policy dependencies into a shared simulator runtime or use a
  broad process-kill command on a multi-user workstation.

## Reproducibility

- Pin every backend to an immutable release or full Git commit and commit the
  resolved lock for shared experiments.
- Record policy/backend commits, lock digests, dataset manifests, checkpoint
  hashes, and observation/action profile identifiers.
