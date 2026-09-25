"""One independent CPU evaluation at a time; optimization never waits for it.

Only checkpoint serialization happens in the training process. A fresh Python
process runs the same evaluator as the standalone commands, keeping simulator
state, inference, video encoding, and their random generators out of training.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

from action_bridge.configs.sb_pusht import METHODS
from action_bridge.plan_revision import checkpoints


@dataclass
class _Job:
    process: subprocess.Popen
    directory: Path
    checkpoint: Path
    step: int


class AsyncEvaluation:
    """Evaluate immutable EMA snapshots and report results to the training owner.

    ``on_result(step, metrics, checkpoint_path)`` runs in the parent process,
    before the transient checkpoint is removed. It can copy the exact evaluated
    checkpoint to ``best.pt``. A busy interval is skipped, not queued. ``finish``
    waits for the last submitted evaluation after optimization has ended.
    """

    def __init__(self, output, config, on_result, *, device="cpu", threads=2,
                 save_videos=True, episodes=None, source_gap_windows=None,
                 source_gap_episodes=8, source_gap_replans=16):
        self.output = Path(output) / "sim_eval"
        self.method = config["method"]
        if self.method not in METHODS:
            raise ValueError(f"No closed-loop evaluator for {self.method}")
        self.seeds = list(config["validation_seeds"])
        if not self.seeds or any(not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("validation_seeds must be a nonempty list of nonnegative integers")
        if episodes is not None:
            if not isinstance(episodes, int) or episodes < 1:
                raise ValueError("Evaluation episodes must be a positive integer")
            # Evaluation-only override: leave training/source-cache identities
            # intact when resuming with a larger validation panel.
            self.seeds = list(range(self.seeds[0], self.seeds[0] + episodes))
        if threads < 1:
            raise ValueError("Evaluation threads must be positive")
        self.on_result = on_result
        self.device = str(device)
        self.threads = int(threads)
        self.save_videos = bool(save_videos)
        self.source_gap_windows = (Path(source_gap_windows).resolve()
                                   if source_gap_windows is not None else None)
        if (not isinstance(source_gap_episodes, int) or source_gap_episodes < 1
                or not isinstance(source_gap_replans, int) or source_gap_replans < 2):
            raise ValueError("Source-gap episodes must be a positive integer and replans at least 2")
        self.source_gap_episodes = source_gap_episodes
        self.source_gap_replans = source_gap_replans
        self.last_submitted_step = None
        self._job = None

    @property
    def busy(self):
        return self._job is not None

    def submit(self, payload):
        """Write a snapshot and launch evaluation, or return False while busy."""
        self.poll()
        if self.busy:
            return False
        if payload["config"]["method"] != self.method:
            raise ValueError("Evaluation snapshot method differs from the training method")
        step = int(payload["step"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = self.output / f"step_{step:09d}-{stamp}"
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = directory / "checkpoint.pt"
        checkpoints.save(checkpoint, payload)
        command = [sys.executable, "-m", f"action_bridge.scripts.eval_pusht_{self.method}",
                   "--checkpoint", str(checkpoint.resolve()),
                   "--output-dir", str((directory / "results").resolve()),
                   "--device", self.device, "--threads", str(self.threads),
                   "--no-progress"]
        if self.source_gap_windows is not None:
            command.extend([
                "--source-gap-windows", str(self.source_gap_windows),
                "--source-gap-episodes", str(self.source_gap_episodes),
                "--source-gap-replans", str(self.source_gap_replans),
            ])
        command.extend(["--seeds", *map(str, self.seeds)])
        if not self.save_videos:
            command.append("--no-save-videos")
        environment = dict(os.environ, OMP_NUM_THREADS=str(self.threads),
                           MKL_NUM_THREADS=str(self.threads), PYTHONUNBUFFERED="1",
                           WANDB_MODE="disabled")
        if self.device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        try:
            with (directory / "worker.log").open("w") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           env=environment)
        except OSError as error:
            self._failure(directory, step, f"Cannot start evaluation: {error}")
            checkpoint.unlink()
            return False
        self._job = _Job(process, directory, checkpoint, step)
        self.last_submitted_step = step
        return True

    def _failure(self, directory, step, message):
        row = {"step": step, "directory": str(directory), "error": message}
        with (self.output / "errors.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        warnings.warn(f"Closed-loop evaluation at step {step} failed: {message}. "
                      f"See {directory / 'worker.log'}", RuntimeWarning, stacklevel=2)

    def poll(self):
        """Collect a completed worker without waiting for a running worker."""
        job = self._job
        if job is None:
            return
        code = job.process.poll()
        if code is None:
            return
        self._job = None
        if code != 0:
            self._failure(job.directory, job.step, f"Worker exited with status {code}")
            job.checkpoint.unlink()
            return
        try:
            metrics = json.loads((job.directory / "results" / "metrics.json").read_text())
            success = metrics["success_rate"]
            if not isinstance(success, (int, float)) or not math.isfinite(success) or not 0 <= success <= 1:
                raise ValueError("success_rate must be a finite number between zero and one")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self._failure(job.directory, job.step, f"Invalid evaluation result: {error}")
            job.checkpoint.unlink()
            return
        # Parent callback errors must not be mislabeled as simulation failures.
        # Keep the snapshot if promotion/logging fails so it can be recovered.
        self.on_result(job.step, metrics, job.checkpoint)
        job.checkpoint.unlink()

    def finish(self):
        """Drain the outstanding evaluation only after optimization finishes."""
        while self._job is not None:
            try:
                self._job.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                continue
            self.poll()

    def close(self):
        """On interruption, terminate only the child owned by this instance."""
        job = self._job
        if job is None:
            return
        self._job = None
        if job.process.poll() is None:
            job.process.terminate()
            try:
                job.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                job.process.kill()
                job.process.wait(timeout=5)
        job.checkpoint.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is None:
            try:
                self.finish()
            finally:
                self.close()
        else:
            self.close()
