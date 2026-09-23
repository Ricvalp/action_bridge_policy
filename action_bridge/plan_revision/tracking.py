"""Optional experiment tracking, kept separate from the training configuration."""
from __future__ import annotations

import importlib
import json
import random
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@contextmanager
def preserve_rng():
    """Plotting and logging must not change the next training batch or noise draw."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    # Do not initialize a GPU just to log a CPU experiment.
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@dataclass(frozen=True)
class TrackingOptions:
    enabled: bool = False
    project: str = "action-bridge-policy"
    entity: str | None = None
    mode: str = "online"
    images_every: int = 5000
    image_count: int = 3

    def __post_init__(self):
        if self.mode not in {"online", "offline"}:
            raise ValueError("W&B mode must be 'online' or 'offline'")
        if self.image_count < 0 or self.images_every < 1:
            raise ValueError("image_count must be nonnegative and images_every must be positive")


class Tracker:
    """One lazy W&B run per training stage; disabled tracking is a no-op.

    Online restarts reuse the stage's run ID. Offline invocations produce separate
    local runs (W&B does not resume offline runs).
    """

    def __init__(self, output, config, metadata, options, *, group, name):
        self.output = Path(output)
        self.config = config
        self.metadata = metadata
        self.options = options
        self.group = group
        self.name = name
        self.run = None
        self._wandb = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish(exit_code=0 if exc_type is None else 1)

    def _start(self):
        if self.run is not None:
            return
        try:
            self._wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError("W&B logging requires wandb in the policy environment") from exc
        self.output.mkdir(parents=True, exist_ok=True)
        options = self.options
        identity = {"project": options.project, "entity": options.entity}
        run_id = uuid.uuid4().hex[:8]
        record_path = self.output / "wandb_run.json"
        if options.mode == "online" and record_path.exists():
            previous = json.loads(record_path.read_text())
            if all(previous.get(key) == value for key, value in identity.items()):
                run_id = previous["id"]
        init_kwargs = dict(project=options.project, entity=options.entity,
                           mode=options.mode, group=self.group, name=self.name,
                           dir=str(self.output), id=run_id,
                           config={**self.config, "metadata": self.metadata})
        if options.mode == "online":
            init_kwargs["resume"] = "allow"
        self.run = self._wandb.init(**init_kwargs)
        self.run.define_metric("train_step")
        self.run.define_metric("*", step_metric="train_step")
        self.run.define_metric("sim_eval/checkpoint_step")
        self.run.define_metric("sim_eval/*", step_metric="sim_eval/checkpoint_step")
        if options.mode == "online":
            record_path.write_text(json.dumps({**identity, "id": self.run.id}, indent=2) + "\n")

    def log(self, step, metrics):
        if not self.options.enabled or not metrics:
            return
        with preserve_rng():
            self._start()
            # Let W&B advance its internal row counter: several rows may describe
            # the same optimizer step (e.g. training, validation, and images).
            self.run.log({**metrics, "train_step": int(step)})

    def images(self, step, paths):
        if not self.options.enabled or not self.options.image_count:
            return
        paths = list(paths)[:self.options.image_count]
        if not paths:
            return
        with preserve_rng():
            self._start()
            images = [self._wandb.Image(str(path), caption=Path(path).stem) for path in paths]
            self.run.log({"train_step": int(step), "examples/action_chunks": images})

    def log_evaluation(self, step, metrics):
        """Late results use their checkpoint's step, never rewind training's axis."""
        if not self.options.enabled:
            return
        with preserve_rng():
            self._start()
            self.run.log({"sim_eval/checkpoint_step": int(step),
                          **{f"sim_eval/{key}": value for key, value in metrics.items()}})

    def images_due(self, step, final=False):
        return bool(self.options.enabled and self.options.image_count
                    and (final or step % self.options.images_every == 0))

    def finish(self, exit_code=0):
        if self.run is not None:
            with preserve_rng():
                self.run.finish(exit_code=exit_code)
            self.run = None
