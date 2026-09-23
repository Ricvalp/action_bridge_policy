"""Check node CUDA, a real asynchronous Push-T rollout, and headless MP4 output.

The policy is tiny and untrained: this checks infrastructure, not performance.
No dataset, EGL, display server, or simulator GPU is required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from action_bridge.configs.sb_pusht import get_config
from action_bridge.eval.revision_pusht_async import AsyncEvaluation
from action_bridge.plan_revision.models import build_policy


def check_h200():
    """Check the CUDA-visible allocation, not a host-wide GPU ordinal."""
    if not torch.cuda.is_available():
        raise RuntimeError("The job has no usable CUDA GPU; request one H200.")
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    if "H200" not in name.upper():
        raise RuntimeError(f"Expected an H200 allocation, but CUDA sees {name!r}.")
    matrix = torch.ones((32, 32), device=f"cuda:{device}")
    product = matrix @ matrix
    torch.cuda.synchronize(device)
    if not bool((product == 32).all()):
        raise RuntimeError("The CUDA matrix multiplication check failed.")
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A new directory for the worker log, rollout, and smoke.json")
    parser.add_argument("--require-h200", action="store_true",
                        help="Also require an H200 and test a small CUDA matrix multiplication")
    args = parser.parse_args(argv)
    # Set before the child imports pygame. The CPU smoke never claims a GPU.
    os.environ.update(SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", MPLBACKEND="Agg")
    if not args.require_h200:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu = check_h200() if args.require_h200 else None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    config = get_config("ddim") | {
        "horizon": 8, "execute": 4, "channels": [8, 16],
        "history_dim": 16, "hidden_dim": 16, "time_dim": 8,
        "num_inference_steps": 2, "max_episode_steps": 8,
        "validation_seeds": [500_000],
    }
    metadata = {
        "purpose": "infrastructure smoke; synthetic normalization, untrained policy",
        "observation_profile": "pusht-state-5d",
        "action_profile": "pusht-absolute-target-2d",
        "normalization": {"obs_mean": [256., 256., 256., 256., 0.],
                          "obs_std": [256., 256., 256., 256., np.pi]},
        "codec": {"mean": [256., 256.], "std": [256., 256.],
                  "low": [0., 0.], "high": [512., 512.], "units": "pixels"},
    }
    policy = build_policy(config).eval()
    payload = dict(config=config, metadata=metadata, dependencies={},
                   ema=policy.state_dict(), step=0, direction="forward")
    results = []

    def completed(step, metrics, checkpoint):
        results.append({"step": step, "metrics": metrics,
                        "results_directory": str(checkpoint.parent / "results")})

    # This is the same subprocess path used by training, including video encoding.
    with AsyncEvaluation(output, config, completed, device="cpu", threads=2) as evaluator:
        if not evaluator.submit(payload):
            raise RuntimeError(f"Could not start the evaluation worker; inspect {output / 'sim_eval'}")
    if len(results) != 1:
        raise RuntimeError(f"The evaluation worker failed; inspect {output / 'sim_eval'} for worker.log")
    videos = sorted(Path(results[0]["results_directory"]).glob("*.mp4"))
    if not videos:
        raise RuntimeError("The worker completed without saving a rollout MP4.")
    import imageio.v2 as imageio

    with imageio.get_reader(videos[0], format="FFMPEG") as reader:
        frame = reader.get_data(0)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.std() == 0:
            raise RuntimeError("The recorded MP4 has an invalid or blank frame.")
    report = {"passed": True, "gpu": gpu, "evaluation_device": "cpu",
              "video": str(videos[0]), "frame_shape": list(frame.shape), **results[0]}
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {'H200 CUDA, ' if gpu else ''}asynchronous CPU Push-T simulation and headless MP4.")
    print(f"Artifacts: {output}")
    print("The policy is untrained; its success rate is not an experiment result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
