from __future__ import annotations

import subprocess
import sys


def test_offline_training_and_checkpoint_modules_do_not_import_native_isaac() -> None:
    source = """
import sys
from action_bridge.config import load_config
from action_bridge.training import common, train_isaaclab
from action_bridge.eval import eval_isaaclab
from action_bridge.eval import isaaclab_online
from phi_isaaclab import windows
load_config('isaaclab_franka_cube_lift_no_latent')
forbidden = [
    name for name in sys.modules
    if name == 'isaaclab' or name.startswith('isaaclab.')
    or name == 'isaacsim' or name.startswith('isaacsim.')
    or name == 'omni' or name.startswith('omni.')
    or name == 'carb' or name.startswith('carb.')
    or name == 'warp' or name.startswith('warp.')
]
if forbidden:
    raise SystemExit('native imports leaked into offline process: ' + repr(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
