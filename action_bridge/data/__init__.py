"""Datasets for latent action bridge pilots.

Imports are lazy so cache-only and JAX workflows do not require PyTorch.
"""

from importlib import import_module

__all__ = [
    "AnnularObstacleDataset",
    "DelayedBranchObstacleDataset",
    "NumpyRLBenchDataset",
    "RLBenchCacheStore",
    "RLBenchDataset",
]


_EXPORTS = {
    "AnnularObstacleDataset": ("action_bridge.data.toy_annular", "AnnularObstacleDataset"),
    "DelayedBranchObstacleDataset": (
        "action_bridge.data.toy_obstacle",
        "DelayedBranchObstacleDataset",
    ),
    "RLBenchCacheStore": ("phi_rlbench.data.cache", "RLBenchCacheStore"),
    "RLBenchDataset": ("phi_rlbench.data.torch_dataset", "RLBenchDataset"),
    "NumpyRLBenchDataset": (
        "phi_rlbench.data.numpy_dataset",
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
