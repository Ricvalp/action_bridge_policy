"""Train the context-conditioned diffusion policy baseline."""

from __future__ import annotations

from pathlib import Path

from absl import app
from ml_collections import config_flags

from .train import train_from_config


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "diffusion_delayed_modes.py"
_CONFIG = config_flags.DEFINE_config_file(
    "config",
    str(DEFAULT_CONFIG),
    "Path to a diffusion policy config file.",
)


def main(argv) -> None:
    if len(argv) > 1:
        raise app.UsageError(f"Unknown arguments: {argv[1:]}")
    train_from_config(_CONFIG.value)


if __name__ == "__main__":
    app.run(main)
