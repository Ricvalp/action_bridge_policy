from __future__ import annotations

import subprocess
import sys


def test_online_adapter_import_does_not_eagerly_import_jax() -> None:
    code = """
import sys
assert 'jax' not in sys.modules
import action_bridge.jax.eval.rlbench_online
import action_bridge.jax.eval.rlbench_online.jax_backend
assert 'jax' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
