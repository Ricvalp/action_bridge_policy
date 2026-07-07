"""Sequential overnight experiment launcher.

The script intentionally shells out to the training entrypoints instead of
calling train() directly, so each run starts with a fresh process and a clean
WandB lifecycle.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Experiment:
    run_id: str
    module: str
    config_name: str
    overrides: tuple[str, ...]
    optional: bool = False


PUSHT_DATASET = "data/pusht/pusht_cchi_v7_replay.zarr"

PUSHT_COMMON = (
    "data.backend=auto",
    "optim.batch_size=256",
    "optim.max_steps=100000",
    "logging.log_every_steps=100",
    "logging.eval_every_steps=1000",
    "logging.full_eval_every_steps=5000",
    "logging.full_eval_split=val",
    "logging.full_eval_max_batches=4",
    "logging.full_eval_closed_loop_episodes=128",
    "logging.full_eval_num_samples=64",
)

PUSHT_BRIDGE_LARGE = (
    "model.hidden_dim=2048",
    "model.h_emb_dim=2048",
    "model.encoder_depth=4",
    "model.control_depth=6",
    "model.time_emb_dim=64",
    "model.z_embed_dim=128",
    "model.z_dim=4",
)

PUSHT_RAW_SCALE_BRIDGE = (
    "model.control_scale=100.0",
    "reference.sigma=7.0",
    "reference.learn_sigma=true",
    "loss.beta_R=0.001",
    "loss.beta_z_start=0.001",
    "loss.beta_z_end=0.01",
    "loss.beta_z_warmup_steps=5000",
)

PUSHT_AR_46M = (
    "model.policy_type=autoregressive_bc",
    "model.latent_type=none",
    "model.hidden_dim=2560",
    "model.h_emb_dim=2560",
    "model.depth=6",
    "model.time_emb_dim=64",
)

PUSHT_DIRECT_46M = (
    "model.policy_type=direct_bc",
    "model.latent_type=none",
    "model.hidden_dim=2560",
    "model.h_emb_dim=2560",
    "model.depth=6",
)

TOY_COMMON = (
    "optim.max_steps=100000",
    "logging.log_every_steps=100",
    "logging.eval_every_steps=1000",
    "logging.full_eval_every_steps=5000",
    "logging.full_eval_split=val",
    "logging.full_eval_max_batches=4",
    "logging.full_eval_closed_loop_episodes=256",
    "logging.full_eval_num_samples=128",
)


def pusht_experiment(run_id: str, *overrides: str, optional: bool = False) -> Experiment:
    return Experiment(
        run_id=run_id,
        module="action_bridge.training.train_pusht",
        config_name="pusht_lowdim_continuous",
        overrides=tuple(PUSHT_COMMON + overrides),
        optional=optional,
    )


def toy_experiment(
    run_id: str,
    config_name: str,
    *overrides: str,
    optional: bool = False,
) -> Experiment:
    return Experiment(
        run_id=run_id,
        module="action_bridge.training.train_toy",
        config_name=config_name,
        overrides=tuple(TOY_COMMON + overrides),
        optional=optional,
    )


EXPERIMENTS = (
    pusht_experiment(
        "pusht_bridge_ref_continuation_large_z4",
        *PUSHT_BRIDGE_LARGE,
        *PUSHT_RAW_SCALE_BRIDGE,
        "reference.type=continuation",
    ),
    pusht_experiment(
        "pusht_bridge_ref_brownian_large_z4",
        *PUSHT_BRIDGE_LARGE,
        *PUSHT_RAW_SCALE_BRIDGE,
        "reference.type=brownian",
    ),
    pusht_experiment(
        "pusht_bridge_ref_lowaccel_large_z4",
        *PUSHT_BRIDGE_LARGE,
        *PUSHT_RAW_SCALE_BRIDGE,
        "reference.type=low_acceleration",
        "reference.time_emb_dim=64",
        "reference.hidden_dim=512",
    ),
    pusht_experiment(
        "pusht_bridge_ref_lowjerk_large_z4",
        *PUSHT_BRIDGE_LARGE,
        *PUSHT_RAW_SCALE_BRIDGE,
        "reference.type=low_jerk",
        "reference.rho=0.2",
    ),
    pusht_experiment(
        "pusht_bridge_continuation_betaR0_large_z4",
        *PUSHT_BRIDGE_LARGE,
        *PUSHT_RAW_SCALE_BRIDGE,
        "reference.type=continuation",
        "loss.beta_R=0.0",
    ),
    pusht_experiment(
        "pusht_bridge_continuation_nolatent_large",
        *PUSHT_BRIDGE_LARGE,
        *PUSHT_RAW_SCALE_BRIDGE,
        "reference.type=continuation",
        "model.latent_type=none",
    ),
    pusht_experiment(
        "pusht_ar_bc_46m",
        *PUSHT_AR_46M,
    ),
    pusht_experiment(
        "pusht_direct_bc_smooth_46m",
        *PUSHT_DIRECT_46M,
        "loss.lambda_acc=0.001",
        "loss.lambda_jerk=0.0001",
    ),
    toy_experiment(
        "toy_delayed_bridge_ref_continuation_cat",
        "toy_delayed_categorical",
        "reference.type=continuation",
    ),
    toy_experiment(
        "toy_delayed_bridge_ref_brownian_cat",
        "toy_delayed_categorical",
        "reference.type=brownian",
    ),
    toy_experiment(
        "toy_delayed_bridge_betaR0_cat",
        "toy_delayed_categorical",
        "reference.type=continuation",
        "loss.beta_R=0.0",
    ),
    toy_experiment(
        "toy_delayed_ar_bc",
        "toy_delayed_categorical",
        "model.policy_type=autoregressive_bc",
        "model.latent_type=none",
    ),
    pusht_experiment(
        "pusht_sb_continuation_large_z4_300k",
        *PUSHT_BRIDGE_LARGE,
        *PUSHT_RAW_SCALE_BRIDGE,
        "reference.type=continuation",
        "optim.max_steps=300000",
        "logging.full_eval_every_steps=10000",
    ),
    pusht_experiment(
        "pusht_direct_bc_46m",
        *PUSHT_DIRECT_46M,
        optional=True,
    ),
    toy_experiment(
        "toy_delayed_bridge_ref_lowaccel_cat",
        "toy_delayed_categorical",
        "reference.type=low_acceleration",
        optional=True,
    ),
    toy_experiment(
        "toy_annular_bridge_ref_continuation_cat",
        "toy_annular_categorical",
        "reference.type=continuation",
        optional=True,
    ),
    toy_experiment(
        "toy_annular_bridge_betaR0_cat",
        "toy_annular_categorical",
        "reference.type=continuation",
        "loss.beta_R=0.0",
        optional=True,
    ),
)


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    experiments = [exp for exp in EXPERIMENTS if args.include_optional or not exp.optional]
    if args.only:
        requested = set(args.only)
        experiments = [exp for exp in experiments if exp.run_id in requested]
    if args.skip:
        skipped = set(args.skip)
        experiments = [exp for exp in experiments if exp.run_id not in skipped]
    if args.start_at:
        run_ids = [exp.run_id for exp in experiments]
        if args.start_at not in run_ids:
            raise ValueError(f"--start-at {args.start_at!r} is not in the selected suite.")
        experiments = experiments[run_ids.index(args.start_at) :]
    return experiments


def command_for_experiment(exp: Experiment, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        exp.module,
        "--config-name",
        exp.config_name,
        f"run_id={args.run_prefix}{exp.run_id}",
        f"seed={args.seed}",
        f"device={args.device}",
        "logging.wandb.enabled=true",
        f"logging.wandb.project={args.wandb_project}",
    ]
    if exp.module.endswith("train_pusht"):
        command.append(f"data.dataset_path={args.pusht_dataset}")
    command.extend(exp.overrides)
    command.extend(args.extra_override)
    return command


def print_suite(experiments: Iterable[Experiment], args: argparse.Namespace) -> None:
    print("Selected experiments:")
    for idx, exp in enumerate(experiments, start=1):
        cmd = command_for_experiment(exp, args)
        optional = " optional" if exp.optional else ""
        print(f"\n[{idx:02d}] {exp.run_id}{optional}")
        print(shell_join(cmd))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pusht-dataset", default=PUSHT_DATASET)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="action-bridge-policy-experiment-batch")
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--start-at", default=None, help="Run id to start from inside the selected suite.")
    parser.add_argument("--only", nargs="*", default=(), help="Run only these run ids.")
    parser.add_argument("--skip", nargs="*", default=(), help="Skip these run ids.")
    parser.add_argument(
        "extra_override",
        nargs="*",
        help="Extra config overrides appended to every command, for example logging.wandb.entity=my-team.",
    )
    args = parser.parse_args()

    experiments = selected_experiments(args)
    if not experiments:
        raise SystemExit("No experiments selected.")

    print_suite(experiments, args)
    if args.dry_run:
        return

    failures = []
    start_time = time.time()
    for idx, exp in enumerate(experiments, start=1):
        command = command_for_experiment(exp, args)
        print(f"\n=== [{idx}/{len(experiments)}] Starting {exp.run_id} ===")
        print(shell_join(command))
        exp_start = time.time()
        result = subprocess.run(command, check=False)
        elapsed = time.time() - exp_start
        if result.returncode == 0:
            print(f"=== Finished {exp.run_id} in {elapsed / 60.0:.1f} min ===")
            continue
        failures.append((exp.run_id, result.returncode))
        print(f"=== FAILED {exp.run_id} after {elapsed / 60.0:.1f} min: exit {result.returncode} ===")
        if not args.continue_on_error:
            break

    total_elapsed = time.time() - start_time
    print(f"\nTotal launcher time: {total_elapsed / 3600.0:.2f} h")
    if failures:
        print("Failures:")
        for run_id, returncode in failures:
            print(f"  {run_id}: exit {returncode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
