"""Compatibility coverage for the deprecated Action Bridge RLBench imports."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest
from phi_rlbench.data.actions import decode_action_chunk
from phi_rlbench.data.builder import convert_rlbench_dataset
from phi_rlbench.data.cache import RLBenchCacheStore
from phi_rlbench.data.numpy_dataset import NumpyRLBenchDataset
from phi_rlbench.data.torch_dataset import RLBenchDataset


@pytest.mark.parametrize(
    ("module_name", "replacement_name", "replacement"),
    [
        (
            "action_bridge.data.rlbench_cache",
            "RLBenchCacheStore",
            RLBenchCacheStore,
        ),
        (
            "action_bridge.data.rlbench_cache_builder",
            "convert_rlbench_dataset",
            convert_rlbench_dataset,
        ),
        (
            "action_bridge.data.rlbench_dataset",
            "RLBenchDataset",
            RLBenchDataset,
        ),
        (
            "action_bridge.data.rlbench_numpy_dataset",
            "NumpyRLBenchDataset",
            NumpyRLBenchDataset,
        ),
    ],
)
def test_deprecated_data_shim_warns_once_and_reexports_replacement(
    module_name,
    replacement_name,
    replacement,
):
    sys.modules.pop(module_name, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(module_name)

    future_warnings = [item for item in caught if item.category is FutureWarning]
    assert len(future_warnings) == 1
    assert "0.2.0" in str(future_warnings[0].message)
    assert getattr(module, replacement_name) is replacement


def test_torch_shim_preserves_action_helper_export():
    module = importlib.import_module("action_bridge.data.rlbench_dataset")
    assert module.decode_action_chunk is decode_action_chunk


def test_action_bridge_data_lazy_exports_point_directly_to_phi_rlbench():
    package = importlib.import_module("action_bridge.data")
    for name in ("RLBenchCacheStore", "RLBenchDataset", "NumpyRLBenchDataset"):
        package.__dict__.pop(name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert package.RLBenchCacheStore is RLBenchCacheStore
        assert package.RLBenchDataset is RLBenchDataset
        assert package.NumpyRLBenchDataset is NumpyRLBenchDataset

    assert not [item for item in caught if item.category is FutureWarning]
