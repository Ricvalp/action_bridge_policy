"""Shared helpers for training and evaluation scripts."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch

from action_bridge.config import to_plain_dict
from action_bridge.data.pusht_adapter import PushTLowDimDataset
from action_bridge.data.toy_annular import AnnularObstacleDataset
from action_bridge.data.toy_obstacle import DelayedBranchObstacleDataset
from action_bridge.models.action_bridge_policy import ActionBridgePolicy
from action_bridge.models.baselines import AutoregressiveBCPolicy, DirectChunkBCPolicy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    return value


def slice_batch(batch: Dict[str, Any], item_slice) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value[item_slice]
        elif isinstance(value, dict):
            out[key] = slice_batch(value, item_slice)
        else:
            out[key] = value
    return out


def build_dataset(config: Dict, split: str):
    benchmark = config.get("benchmark", "toy_delayed")
    base = {
        "seed": int(config.get("seed", 0)),
        "trajectory_len": int(config.get("trajectory_len", 64)),
        "chunk_horizon": int(config.get("chunk_horizon", 16)),
        "obs_history": int(config.get("obs_history", 2)),
        "action_history": int(config.get("action_history", 2)),
    }
    data_cfg = dict(config.get("data", {}))
    data_cfg.update(base)
    if benchmark == "toy_delayed":
        return DelayedBranchObstacleDataset(data_cfg, split=split)
    if benchmark == "toy_annular":
        return AnnularObstacleDataset(data_cfg, split=split)
    if benchmark == "pusht_lowdim":
        return PushTLowDimDataset(
            dataset_path=data_cfg.get("dataset_path"),
            backend=str(data_cfg.get("backend", "auto")),
            split=split,
            obs_history=int(config.get("obs_history", 2)),
            action_history=int(config.get("action_history", 2)),
            chunk_horizon=int(config.get("chunk_horizon", 16)),
            train_fraction=float(data_cfg.get("train_fraction", 0.8)),
            val_fraction=float(data_cfg.get("val_fraction", 0.1)),
            obs_key=data_cfg.get("obs_key"),
            action_key=data_cfg.get("action_key"),
            episode_ends_key=data_cfg.get("episode_ends_key"),
            max_episodes=data_cfg.get("max_episodes"),
            normalize=bool(data_cfg.get("normalize", False)),
            normalization_stats=data_cfg.get("normalization_stats"),
            normalization_eps=float(data_cfg.get("normalization_eps", 1e-6)),
            pad_episode_starts=bool(data_cfg.get("pad_episode_starts", False)),
        )
    raise ValueError(f"Unsupported benchmark {benchmark!r} for train_toy.")


def build_model(config: Dict):
    model_cfg = dict(config.get("model", {}))
    policy_type = model_cfg.get("policy_type", "action_bridge")
    args = dict(
        obs_dim=int(config.get("obs_dim", 4)),
        action_dim=int(config.get("action_dim", 2)),
        obs_history=int(config.get("obs_history", 2)),
        action_history=int(config.get("action_history", 2)),
        chunk_horizon=int(config.get("chunk_horizon", 16)),
        model_config=model_cfg,
    )
    if policy_type == "action_bridge":
        return ActionBridgePolicy(reference_config=dict(config.get("reference", {})), **args)
    if policy_type in {"direct_bc", "bc_smooth"}:
        return DirectChunkBCPolicy(**args)
    if policy_type in {"autoregressive_bc", "ar_bc"}:
        return AutoregressiveBCPolicy(**args)
    raise ValueError(f"Unknown model.policy_type {policy_type!r}.")


def make_run_dir(config: Dict) -> Path:
    run_id = config.get("run_id")
    if not run_id:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{config.get('config_name', 'run')}_{config.get('model', {}).get('policy_type', 'action_bridge')}"
    root = Path(config.get("output_dir", "outputs"))
    path = root / run_id
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not bool(config.get("resume", False)):
        raise FileExistsError(
            f"Output directory already exists: {path}. Use a new run_id, remove the directory, "
            "or pass resume=true to allow reusing it."
        )
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (path / "metrics").mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_plain_dict(data), f, indent=2, sort_keys=True)


def tensor_metrics_to_float(metrics: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (float, int)):
            out[key] = float(value)
    return out


def cycle(loader: Iterable):
    while True:
        for batch in loader:
            yield batch
