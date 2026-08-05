"""Datasets for latent action bridge pilots.

Imports are lazy so cache-only and JAX workflows do not require PyTorch.
"""

from importlib import import_module

__all__ = [
    "AnnularObstacleDataset",
    "DelayedBranchObstacleDataset",
    "RLBenchCacheStore",
    "RLBenchDataset",
    "NumpyRLBenchDataset",
]


_EXPORTS = {
    "AnnularObstacleDataset": ("action_bridge.data.toy_annular", "AnnularObstacleDataset"),
    "DelayedBranchObstacleDataset": (
        "action_bridge.data.toy_obstacle",
        "DelayedBranchObstacleDataset",
    ),
    "RLBenchCacheStore": ("action_bridge.data.rlbench_cache", "RLBenchCacheStore"),
    "RLBenchDataset": ("action_bridge.data.rlbench_dataset", "RLBenchDataset"),
    "NumpyRLBenchDataset": (
        "action_bridge.data.rlbench_numpy_dataset",
        "NumpyRLBenchDataset",
    ),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
