"""Evaluate a trusted Action Bridge checkpoint through phi-mujoco."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path

from phi_mujoco.evaluation import EvaluationConfig, EvaluationRunner

from action_bridge.training.mujoco_provenance import training_provenance

from .torch_backend import load_torch_policy_adapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trusted-checkpoint", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--max-steps", type=int, help="default: integration episode horizon"
    )
    parser.add_argument(
        "--actions-per-plan", type=int, help="default: checkpoint setting"
    )
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--quiet", action="store_true", help="save simulator output to a log"
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-success", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stage = "checkpoint_load"
    try:
        adapter = load_torch_policy_adapter(
            args.checkpoint,
            trusted_checkpoint=args.trusted_checkpoint,
            device=args.device,
            actions_per_plan=args.actions_per_plan,
        )
        config = EvaluationConfig(
            episodes=args.episodes,
            base_seed=args.seed,
            max_steps=args.max_steps,
            # Chunk execution belongs to our adapter so it sees every observation.
            actions_per_plan=1,
            run_directory=args.run_dir,
            record_video=args.record_video,
            render_width=args.render_width,
            render_height=args.render_height,
        )
        stage = "native_evaluation"
        if args.run_dir.exists():
            raise FileExistsError(f"run directory already exists: {args.run_dir}")
        args.run_dir.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            progress = None
            if args.progress:
                from tqdm.auto import tqdm

                bar = stack.enter_context(
                    tqdm(total=args.episodes, desc="Evaluating", unit="episode")
                )

                def progress(completed: int, total: int) -> None:
                    del total
                    bar.update(completed - bar.n)

            if args.quiet:
                log = stack.enter_context(
                    args.run_dir.with_suffix(".log").open("x", encoding="utf-8")
                )
                stack.enter_context(redirect_stdout(log))
                stack.enter_context(redirect_stderr(log))
            result = EvaluationRunner(
                adapter.integration, adapter, config, progress_callback=progress
            ).run()
        record = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_identifier": adapter.checkpoint_identifier,
            "actions_per_plan": adapter.actions_per_plan,
            "online_evaluation": adapter.metadata.to_json_dict(),
            "training_provenance": adapter.provenance,
            "evaluation_provenance": training_provenance(),
        }
        (args.run_dir / "policy_metadata.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary = result.as_dict()
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"Run: {result.run_directory}")
            print(f"Success: {result.successful_episodes}/{result.attempted_episodes}")
        return int(args.require_success and result.success_rate != 1.0)
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001 - CLI reports simulator and policy failures as JSON.
        print(
            json.dumps(
                {
                    "stage": stage,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 2 if stage == "checkpoint_load" else 3
