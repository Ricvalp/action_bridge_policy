"""Low-dimensional Push-T adapter.

This pilot intentionally does not vendor external Push-T code or datasets. The
adapter exposes the common batch schema when a local dataset is supplied and
otherwise fails with setup instructions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


class PushTLowDimDataset(Dataset):
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        backend: str = "auto",
        split: str = "train",
        obs_history: int = 2,
        action_history: int = 2,
        chunk_horizon: int = 16,
    ):
        del split, obs_history, action_history, chunk_horizon
        if dataset_path is None:
            raise RuntimeError(
                "Push-T lowdim data was not found. Provide data.dataset_path "
                "pointing to a local Diffusion Policy zarr dataset or a locally "
                "materialized LeRobot Push-T dataset. This project does not vendor "
                "external repositories or datasets."
            )
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Push-T dataset path does not exist: {path}")
        raise NotImplementedError(
            f"Push-T backend {backend!r} at {path} is not loaded by the minimal toy-first pilot yet. "
            "Add a small backend-specific loader here that emits obs_hist, act_hist, "
            "future_actions, optional future_positions, mode_label=None, "
            "true_mode_probs=None, and context."
        )

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError(idx)
