"""One background, CPU-worker MuJoCo evaluation alongside GPU training.

The trainer owns W&B logging. Children only write artifacts, so late evaluation
results can be logged against their checkpoint step without rewinding W&B's step.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from uuid import uuid4

from action_bridge.training.common import append_csv, save_json
from action_bridge.training.train_toy import save_checkpoint


@dataclass
class _Evaluation:
    process: subprocess.Popen
    step: int
    checkpoint: Path
    output_dir: Path
    log_path: Path


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"logging.{name} must be an integer >= {minimum}")
    return int(value)


class AsyncMujocoEvaluator:
    """Keep at most one evaluation in flight; skip requests while it is busy."""

    def __init__(self, config, run_dir: Path, wandb_run):
        self.run_dir = Path(run_dir).resolve()
        self.wandb_run = wandb_run
        self.pending: _Evaluation | None = None
        logging = config.get("logging", {})
        evaluation = config.get("eval", {})
        self.episodes = _integer(
            "sim_eval_episodes", logging.get("sim_eval_episodes", 40)
        )
        self.num_workers = _integer(
            "sim_eval_num_workers", logging.get("sim_eval_num_workers", 8)
        )
        self.worker_threads = _integer(
            "sim_eval_worker_threads", logging.get("sim_eval_worker_threads", 1)
        )
        seed = logging.get("sim_eval_seed", 2_000_000)
        if seed is None:
            seed = evaluation.get("online_seed", 2_000_000)
        self.seed = _integer("sim_eval_seed", seed, 0)
        n_exec = logging.get("sim_eval_n_exec")
        if n_exec is None:
            n_exec = evaluation.get("actions_per_plan", 1)
        self.n_exec = _integer("sim_eval_n_exec", n_exec)
        if self.n_exec > int(config.get("chunk_horizon", 1)):
            raise ValueError("logging.sim_eval_n_exec must not exceed chunk_horizon")
        self.max_steps = logging.get("sim_eval_max_steps")
        if self.max_steps is not None:
            self.max_steps = _integer("sim_eval_max_steps", self.max_steps)
        self.success_videos = _integer(
            "sim_eval_success_videos", logging.get("sim_eval_success_videos", 0), 0
        )
        self.failure_videos = _integer(
            "sim_eval_failure_videos", logging.get("sim_eval_failure_videos", 0), 0
        )
        self.video_backend = str(logging.get("sim_eval_video_backend", "egl"))
        if self.video_backend not in {"egl", "osmesa"}:
            raise ValueError("logging.sim_eval_video_backend must be 'egl' or 'osmesa'")
        self.render_width = _integer(
            "sim_eval_render_width", logging.get("sim_eval_render_width", 640)
        )
        self.render_height = _integer(
            "sim_eval_render_height", logging.get("sim_eval_render_height", 480)
        )
        self.keep_checkpoints = bool(logging.get("sim_eval_keep_checkpoints", False))
        online = config.get("online_evaluation", {})
        self.protocol = {
            "episodes": self.episodes,
            "seed": self.seed,
            "actions_per_plan": self.n_exec,
            "max_steps": self.max_steps,
            "integration": config.get("data", {}).get("integration"),
            "deterministic_latent": online.get("deterministic_latent"),
            "latent_commitment": online.get("latent_commitment"),
            "clip_actions": online.get("clip_actions"),
        }
        self.best_success = -math.inf
        self.best_metadata_path = self.run_dir / "metrics" / "best_success.json"
        if self.best_metadata_path.exists():
            previous = json.loads(self.best_metadata_path.read_text())
            if previous["protocol"] != self.protocol:
                raise ValueError(
                    "closed-loop evaluation protocol differs from best_success.json; "
                    "use a new run_id when changing evaluation seeds, episodes, or horizon"
                )
            if not (self.run_dir / "checkpoints" / "best_success.pt").is_file():
                raise FileNotFoundError(
                    "best_success.json exists but best_success.pt is missing"
                )
            self.best_success = float(previous["success_rate"])
            if not 0 <= self.best_success <= 1:
                raise ValueError("best_success.json has an invalid success rate")
        if wandb_run is not None:
            wandb_run.define_metric("sim_eval/checkpoint_step")
            wandb_run.define_metric("sim_eval/*", step_metric="sim_eval/checkpoint_step")

    def _error(self, step, message, log_path):
        path = self.run_dir / "metrics" / "sim_eval_errors.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            entry = {"step": step, "error": str(message), "log_path": str(log_path)}
            stream.write(json.dumps(entry) + "\n")
        print(
            f"MuJoCo evaluation at step {step}: {message}; see {log_path}", flush=True
        )

    def _cleanup(self, checkpoint):
        if not self.keep_checkpoints:
            checkpoint.unlink(missing_ok=True)

    def submit(self, model, optimizer, config, step: int, best_mse: float) -> bool:
        """Snapshot synchronously, then return without waiting for simulations."""

        self.poll()
        if self.pending is not None:
            print(
                f"Skipping MuJoCo evaluation at step {step}: "
                "previous evaluation is still running.",
                flush=True,
            )
            return False
        # Distinct paths also allow resuming an earlier checkpoint in this run.
        name = f"sim_step_{step:06d}_{uuid4().hex[:8]}"
        eval_root = self.run_dir / "eval"
        eval_root.mkdir(parents=True, exist_ok=True)
        checkpoint = eval_root / f"{name}_checkpoint.pt"
        output_dir = eval_root / name
        log_path = eval_root / f"{name}.log"
        command = [
            sys.executable, "-m", "action_bridge.eval.mujoco_online.parallel",
            "--checkpoint", str(checkpoint), "--trusted-checkpoint",
            "--run-dir", str(output_dir),
            "--episodes", str(self.episodes), "--num-workers", str(self.num_workers),
            "--worker-threads", str(self.worker_threads), "--seed", str(self.seed),
            "--actions-per-plan", str(self.n_exec),
            "--success-videos", str(self.success_videos),
            "--failure-videos", str(self.failure_videos),
            "--video-backend", self.video_backend,
            "--render-width", str(self.render_width), "--render-height", str(self.render_height),
        ]
        if self.max_steps is not None:
            command.extend(["--max-steps", str(self.max_steps)])
        environment = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = str(self.worker_threads)
        # Keep Slurm's GPU visibility for the optional renderer. Score workers
        # explicitly disable CUDA and rendering in their own child environments.
        try:
            save_checkpoint(checkpoint, model, optimizer, config, step, best_mse)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except Exception as exc:
            self._error(step, f"could not launch evaluation: {exc}", log_path)
            self._cleanup(checkpoint)
            return False
        self.pending = _Evaluation(process, step, checkpoint, output_dir, log_path)
        print(f"Started MuJoCo evaluation of step {step}: {output_dir}", flush=True)
        return True

    def _record_result(self, job):
        summary = json.loads((job.output_dir / "summary.json").read_text())
        attempts = summary["attempted_episodes"]
        successes = summary["successful_episodes"]
        rate = float(summary["success_rate"])
        if (
            type(attempts) is not int or attempts != self.episodes
            or type(successes) is not int or not 0 <= successes <= attempts
            or not math.isfinite(rate) or not math.isclose(rate, successes / attempts)
            or summary["actions_per_plan"] != self.n_exec
        ):
            raise ValueError("incomplete or inconsistent evaluation summary")
        append_csv(self.run_dir / "metrics" / "sim_eval_metrics.csv", {
            "step": job.step, "success_rate": rate,
            "successful_episodes": successes, "attempted_episodes": attempts,
            "actions_per_plan": self.n_exec,
        })
        if rate > self.best_success:
            # Copy the exact snapshot, not the trainer's now-newer parameters.
            best_path = self.run_dir / "checkpoints" / "best_success.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = best_path.with_suffix(".pt.tmp")
            shutil.copy2(job.checkpoint, temporary)
            temporary.replace(best_path)
            save_json(self.best_metadata_path, {
                "step": job.step, "success_rate": rate, "protocol": self.protocol,
                "checkpoint": str(best_path), "evaluation_directory": str(job.output_dir),
                "checkpoint_identifier": summary.get("checkpoint_identifier"),
            })
            self.best_success = rate
        print(
            f"MuJoCo success at checkpoint step {job.step}: "
            f"{successes}/{attempts} ({rate:.1%})",
            flush=True,
        )
        for error in summary.get("video_errors", []):
            self._error(job.step, f"video: {error}", job.log_path)
        if self.wandb_run is not None:
            self._log_wandb(job, summary)

    def _log_wandb(self, job, summary):
        payload = {
            "sim_eval/checkpoint_step": job.step,
            "sim_eval/success_rate": summary["success_rate"],
        }
        try:
            # Score logging must still succeed when a selected clip is missing.
            counts = {"success": 0, "failure": 0}
            for video in summary.get("selected_videos", []):
                try:
                    import wandb

                    path = job.output_dir / video["path"]
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    outcome = "success" if video["success"] else "failure"
                    counts[outcome] += 1
                    payload[f"sim_eval/{outcome}_{counts[outcome]}"] = wandb.Video(
                        str(path), format="mp4",
                        caption=f"Step {job.step}; seed {video['seed']}; {outcome}",
                    )
                except Exception as exc:
                    self._error(job.step, f"could not log video: {exc}", job.log_path)
            # No explicit step= here: evaluation can finish after newer training logs.
            self.wandb_run.log(payload)
        except Exception as exc:
            self._error(job.step, f"could not log to W&B: {exc}", job.log_path)

    def poll(self) -> None:
        """Collect a finished evaluation without waiting for a running child."""

        job = self.pending
        if job is None:
            return
        return_code = job.process.poll()
        if return_code is None:
            return
        try:
            if return_code != 0:
                self._stop_process_group(job)
                raise RuntimeError(f"evaluation process exited with status {return_code}")
            self._record_result(job)
        except Exception as exc:
            self._error(job.step, exc, job.log_path)
        finally:
            self._cleanup(job.checkpoint)
            self.pending = None

    def finish(self) -> None:
        """At normal training completion, wait for the last batch and its videos."""

        if self.pending is not None:
            print(
                "Waiting for the last MuJoCo evaluation and selected videos...",
                flush=True,
            )
            self.pending.process.wait()
            self.poll()

    @staticmethod
    def _stop_process_group(job) -> None:
        # start_new_session makes the child's original PID our owned process
        # group ID. Address that group, even if its coordinator has already died:
        # remaining workers keep the group alive. Never signal a PID directly.
        try:
            os.killpg(job.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            job.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Workers may outlive the coordinator or ignore SIGTERM.
        try:
            os.killpg(job.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        job.process.wait(timeout=5)

    def close(self) -> None:
        """On interruption, terminate only this manager's own subprocess group."""

        job = self.pending
        if job is None:
            return
        self._stop_process_group(job)
        self._cleanup(job.checkpoint)
        self.pending = None
