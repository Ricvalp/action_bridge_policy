"""NumPy batch sampling and background prefetch for JAX RLBench training."""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from action_bridge.data.rlbench_numpy_dataset import NumpyRLBenchDataset


def dataset_kwargs(config, split: str) -> Dict[str, Any]:
    data = config.data
    return {
        "cache_root": str(data.cache_root),
        "split": str(split),
        "tasks": list(data.tasks) or None,
        "exclude_tasks": list(data.exclude_tasks),
        "variation_ids": list(data.variation_ids) or None,
        "obs_history": int(data.obs_history),
        "action_history": int(data.action_history),
        "chunk_horizon": int(data.chunk_horizon),
        "obs_stride": int(data.obs_stride),
        "action_stride": int(data.action_stride),
        "action_offset": int(data.action_offset),
        "action_representation": str(data.action_representation),
        "train_fraction": float(data.train_fraction),
        "val_fraction": float(data.val_fraction),
        "split_seed": int(data.split_seed),
        "pad_episode_starts": bool(data.pad_episode_starts),
        "pad_episode_ends": bool(data.pad_episode_ends),
        "include_rgb": bool(data.include_rgb),
        "include_mask_id": bool(data.include_mask_id),
        "point_count": int(data.point_count),
        "point_sampling": str(data.point_sampling),
        "point_sampling_seed": int(config.seed),
        "max_episodes_per_variation": data.max_episodes_per_variation,
        "keep_h5_open": True,
        "preload_to_memory": bool(data.preload_to_memory),
    }


class BatchSource:
    def __init__(
        self,
        dataset: NumpyRLBenchDataset,
        *,
        batch_size: int,
        sampling_strategy: str,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(int(seed))
        self.batch_index = 0
        weights = dataset.sampling_weights(str(sampling_strategy))
        self.probabilities = weights / weights.sum()

    def __call__(self) -> Dict[str, np.ndarray]:
        self.dataset.set_epoch(self.batch_index)
        self.batch_index += 1
        indices = self.rng.choice(
            len(self.dataset),
            size=self.batch_size,
            replace=True,
            p=self.probabilities,
        )
        items = [self.dataset[int(index)] for index in indices]
        return {
            key: np.stack([np.asarray(item[key]) for item in items], axis=0)
            for key in items[0]
        }


@dataclass(frozen=True)
class _WorkerError:
    exception: BaseException
    traceback_text: str


class BackgroundBatchPrefetcher:
    """Build batches in background threads, each with process-local HDF5 handles."""

    def __init__(
        self,
        make_source: Callable[[int], BatchSource],
        *,
        num_workers: int,
        max_prefetch: int,
    ):
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(max_prefetch)))
        self._stop = threading.Event()
        self._threads = []
        for worker_id in range(max(1, int(num_workers))):
            thread = threading.Thread(
                target=self._worker,
                args=(make_source(worker_id),),
                name=f"rlbench-jax-prefetch-{worker_id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _put(self, value: Any) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def _worker(self, source: BatchSource) -> None:
        while not self._stop.is_set():
            try:
                self._put(source())
            except BaseException as exception:
                self._put(_WorkerError(exception, traceback.format_exc()))
                return

    def get(self) -> Dict[str, np.ndarray]:
        value = self._queue.get()
        if isinstance(value, _WorkerError):
            self.close()
            raise RuntimeError(
                f"RLBench batch prefetch worker failed:\n{value.traceback_text}"
            ) from value.exception
        return value

    def qsize(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
