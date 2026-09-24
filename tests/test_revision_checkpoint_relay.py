"""Exercise the laptop relay without connecting or transferring any files."""
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "hpc/rsync_sb_pusht_checkpoints.sh"
STUBS = """
mkdir() { printf 'mkdir <%s>\\n' "$@"; }
ssh() { printf 'ssh <%s>\\n' "$@"; }
rsync() {
    printf 'rsync <%s>\\n' "$@"
    return "$relay_test_status"
}
export -f mkdir ssh rsync
export relay_test_status="$1"
shift
bash "$@"
"""


def run_script(*arguments, rsync_status=0):
    return subprocess.run(["bash", "-c", STUBS, "test-relay", str(rsync_status),
                           str(SCRIPT), *arguments], capture_output=True, text=True)


def test_relay_paths_filters_and_order():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = run_script("self-source-hpc-test", "lab-workstation")
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "peano:/hpc/home/phi/rvalperga/action_bridge_policy/workspace/sb_pusht/self-source-hpc-test/" in output
    assert f"{Path.home()}/Downloads/sb-pusht/self-source-hpc-test/" in output
    assert "lab-workstation:/home/rvalperga/action_bridge_policy/workspace/sb_pusht/from-hpc/self-source-hpc-test/" in output
    for pattern in ("*/", "best.pt", "latest.pt", "best_eval.json", "manifest.json"):
        assert output.count(f"rsync <--include={pattern}>") == 2
    assert output.count("rsync <--exclude=*>") == 2
    assert "--delete" not in output
    assert output.index("peano:") < output.index("ssh <") < output.index("lab-workstation:")


def test_failed_download_does_not_start_workstation_transfer():
    result = run_script("test-run", "workstation", rsync_status=23)
    assert result.returncode == 23
    assert "ssh <" not in result.stdout
    assert "Done." not in result.stdout


@pytest.mark.parametrize("arguments", [(), ("run",), ("../run", "workstation"),
                                      ("/absolute/path", "workstation"),
                                      ("run", "-bad-option")])
def test_invalid_arguments_fail_before_file_or_network_operations(arguments):
    result = run_script(*arguments)
    assert result.returncode == 2
    assert not result.stdout
