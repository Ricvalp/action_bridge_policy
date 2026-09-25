"""Exercise the ablation launchers without Slurm, CUDA, or training jobs."""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from action_bridge.configs.sb_pusht import METHODS, get_config


JOBS = Path(__file__).resolve().parents[1] / "hpc/sb_pusht_ablations"
VARIANTS = {
    "baseline_seed0", "baseline_seed1", "baseline_seed2", "h16_k4", "h16_k16",
    "h32_k8", "h32_k16", "h64_k8", "h64_k32", "wide18m", "wide42m",
    "long600k", "wide18m_long600k", "deep18m", "temperature001", "temperature020",
    "damping05", "damping8", "self_sources_all",
    "scarcity50", "scarcity25", "scarcity10", "direct_mlp", "brownian", "isotropic_ou",
}


def option(arguments, name):
    """Read either '--flag value' or '--flag=value' from a shell invocation."""
    for index, argument in enumerate(arguments):
        if argument == name:
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    raise AssertionError(f"Missing {name} in {arguments}")


@pytest.fixture
def launch(tmp_path):
    checkout = tmp_path / "checkout"
    shutil.copytree(JOBS, checkout / "hpc/sb_pusht_ablations")
    (checkout / "hpc/logs").mkdir()
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    records = tmp_path / "commands.jsonl"
    stub = f"""#!{sys.executable}
import json, os, pathlib, sys
target = pathlib.Path(os.environ['ABLATION_TEST_RECORDS'])
rows = target.read_text().splitlines() if target.exists() else []
kind = pathlib.Path(sys.argv[0]).name
row = dict(kind=kind, args=sys.argv[1:], cwd=os.getcwd(),
           cuda=os.environ.get('CUDA_VISIBLE_DEVICES'),
           project=os.environ.get('WANDB_PROJECT'))
with target.open('a') as stream:
    stream.write(json.dumps(row) + '\\n')
if kind == 'sbatch':
    count = 1 + sum(json.loads(line)['kind'] == 'sbatch' for line in rows)
    if str(count) == os.environ.get('ABLATION_TEST_SBATCH_FAIL_AT'):
        raise SystemExit(1)
    print(1000 + count)
elif kind == 'python' and len(sys.argv) == 3 and sys.argv[1] == '-':
    names = ['repeat', 'fixed_damped', 'learned_dissipative', 'direct_mlp']
    modes = os.environ.get('ABLATION_TEST_COMPLETION_MODES', '0,1,2').split(',')
    print('\\n'.join(names[int(mode)] for mode in modes))
"""
    for target in (executable_dir / "sbatch", checkout / ".venv-sb-pusht/bin/python"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(stub)
        target.chmod(0o755)
    campaign = tmp_path / "campaign"
    dataset = tmp_path / "pusht.zarr"
    dataset.mkdir()
    environment = dict(os.environ, PATH=f"{executable_dir}:{os.environ['PATH']}",
                       SLURM_SUBMIT_DIR=str(checkout),
                       SB_PUSHT_CAMPAIGN_ROOT=str(campaign), PUSHT_DATASET=str(dataset),
                       ABLATION_TEST_RECORDS=str(records), CUDA_VISIBLE_DEVICES="7",
                       WANDB_MODE="offline")

    class Launcher:
        def __init__(self):
            self.checkout, self.campaign, self.dataset = checkout, campaign, dataset
            self.env = environment

        def run(self, script, *arguments):
            return subprocess.run(
                ["bash", str(checkout / "hpc/sb_pusht_ablations" / script), *arguments],
                cwd=checkout, env=self.env, text=True, capture_output=True, timeout=20)

        def rows(self, kind=None):
            rows = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
            return rows if kind is None else [row for row in rows if row["kind"] == kind]

        def prepared(self, variant="baseline_seed0"):
            root = campaign / variant
            (root / "reference").mkdir(parents=True)
            (root / "windows.pt").touch()
            (root / "reference/latest.pt").touch()
            return root

    return Launcher()


def test_all_shell_scripts_parse_and_training_allocation_is_h200_partition():
    scripts = sorted(JOBS.glob("*.sh")) + sorted(JOBS.glob("*.sbatch"))
    assert scripts
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    train = (JOBS / "train.sbatch").read_text()
    for directive in ("--partition=gpuq", "--gres=gpu:1", "--cpus-per-task=8",
                      "--mem=64G", "--time=10:00:00"):
        assert f"#SBATCH {directive}" in train
    assert "B200" not in train


def test_configurations_cover_ready_experiments_and_valid_training_budgets():
    configs = {path.stem: json.loads(path.read_text()) for path in (JOBS / "configs").glob("*.json")}
    assert set(configs) == VARIANTS
    base = get_config()
    for variant, overrides in configs.items():
        assert not set(overrides).difference(base), variant
        config = base | overrides
        assert config["updates"] == 2 * config["rounds"] * config["phase_updates"]
        assert config["updates"] % config["training_blocks"] == 0
        assert config["training_blocks"] == config["rounds"]
        assert len(config["source_self_probabilities"]) == config["training_blocks"]
        assert all(0 <= probability <= 1 for probability in config["source_self_probabilities"])
        assert 1 <= config["execute"] <= config["horizon"]
        assert config["protocol"] == "self_source_v1"
        assert not config["external_bootstrap"]
        if variant.startswith("h"):
            horizon, execute = variant.split("_")
            assert (config["horizon"], config["execute"]) == (int(horizon[1:]), int(execute[1:]))
    for seed in range(3):
        assert (base | configs[f"baseline_seed{seed}"])["seed"] == seed
    assert configs["wide18m"]["channels"] == [160, 320, 640]
    assert configs["wide42m"]["channels"] == [256, 512, 1024]
    for variant in ("long600k", "wide18m_long600k"):
        assert configs[variant]["updates"] == 600000
    assert configs["temperature001"]["temperature"] == .01
    assert configs["temperature020"]["temperature"] == .20
    assert configs["damping05"]["revision_gamma"] == .5
    assert configs["damping8"]["revision_gamma"] == 8
    assert configs["self_sources_all"]["source_self_probabilities"] == [1, 1, 1, 1]
    for variant, fraction in [("scarcity50", .5), ("scarcity25", .25), ("scarcity10", .1)]:
        assert configs[variant] == {"train_episode_fraction": fraction, "subset_seed": 0}
    assert configs["direct_mlp"] == {
        "training_completion_modes": [0, 1, 3], "completion_id": 3, "direct_tail_updates": 20000}
    assert configs["brownian"] == {"reference_kind": "brownian"}
    assert configs["isotropic_ou"] == {"reference_kind": "isotropic_ou"}


@pytest.mark.parametrize("method", METHODS)
def test_training_keeps_cuda_visibility_and_uses_named_config_project_and_async_eval(launch, method):
    root = launch.prepared("h32_k8")
    result = launch.run("train.sbatch", "h32_k8", method)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = launch.rows("python")
    assert calls and calls[0]["args"] == ["-"]  # Real scripts check H200 before training.
    command = calls[-1]["args"]
    assert command[:3] == ["-m", "action_bridge.scripts.sb_pusht", method]
    assert option(command, "--run-root") == str(root)
    assert option(command, "--dataset") == str(launch.dataset)
    assert option(command, "--config") == str(launch.checkout / "hpc/sb_pusht_ablations/configs/h32_k8.json")
    assert option(command, "--wandb-project") == "sb-pusht-ablations"
    assert option(command, "--wandb-mode") == "offline"
    assert option(command, "--threads") == "4"
    assert option(command, "--device") == "cuda"
    assert option(command, "--eval-every") == "10000"
    assert option(command, "--eval-episodes") == "20"
    assert option(command, "--eval-device") == "cpu"
    assert option(command, "--eval-threads") == "2"
    assert {"--wandb", "--sim-eval", "--eval-videos"}.issubset(command)
    assert all(call["cuda"] == "7" for call in calls)


@pytest.mark.parametrize("stage", ["prepare", "reference"])
def test_prepare_and_reference_share_variant_configuration(launch, stage):
    root = launch.prepared("h64_k8")
    result = launch.run(stage + ".sbatch", "h64_k8")
    assert result.returncode == 0, result.stdout + result.stderr
    command = launch.rows("python")[-1]["args"]
    assert command[:3] == ["-m", "action_bridge.scripts.sb_pusht", stage]
    assert option(command, "--run-root") == str(root)
    assert option(command, "--config").endswith("/configs/h64_k8.json")
    if stage == "reference":
        assert option(command, "--wandb-project") == "sb-pusht-ablations"
        assert "--wandb" in command


def test_submission_snapshots_config_and_assigns_only_required_dependencies(launch):
    result = launch.run("submit.sh", "baseline_seed1")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = launch.rows("sbatch")
    assert len(calls) == 6
    prepare = next(row for row in calls if any(arg.endswith("prepare.sbatch") for arg in row["args"]))
    prepare_id = 1001 + calls.index(prepare)
    reference = next(row for row in calls if any(arg.endswith("reference.sbatch") for arg in row["args"]))
    reference_id = 1001 + calls.index(reference)
    assert option(reference["args"], "--dependency") == f"afterok:{prepare_id}"
    train = [row for row in calls if any(arg.endswith("train.sbatch") for arg in row["args"])]
    assert {row["args"][-1] for row in train} == {"ddim", "fm_paired", "sb_ou", "sb_kinetic"}
    for row in train:
        dependency = prepare_id if row["args"][-1] == "ddim" else reference_id
        assert option(row["args"], "--dependency") == f"afterok:{dependency}"
    snapshot = launch.campaign / "baseline_seed1/config.json"
    assert json.loads(snapshot.read_text()) == json.loads((JOBS / "configs/baseline_seed1.json").read_text())
    duplicate = launch.run("submit.sh", "baseline_seed1")
    assert duplicate.returncode != 0
    assert launch.rows("sbatch") == calls


def test_manual_training_uses_submission_snapshot_if_available(launch):
    root = launch.prepared()
    snapshot = root / "config.json"
    snapshot.write_text('{"seed": 123}\n')
    result = launch.run("train.sbatch", "baseline_seed0", "ddim")
    assert result.returncode == 0, result.stdout + result.stderr
    assert option(launch.rows("python")[-1]["args"], "--config") == str(snapshot)


def test_ddim_only_submission_does_not_require_reference_job(launch):
    result = launch.run("submit.sh", "baseline_seed0", "ddim")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = launch.rows("sbatch")
    assert len(calls) == 2
    assert not any(arg.endswith("reference.sbatch") for row in calls for arg in row["args"])


def test_failed_preparation_submission_does_not_schedule_dependent_jobs(launch):
    launch.env["ABLATION_TEST_SBATCH_FAIL_AT"] = "1"
    result = launch.run("submit.sh", "baseline_seed0")
    assert result.returncode != 0
    assert len(launch.rows("sbatch")) == 1


def test_relative_campaign_root_fails_before_submission(launch):
    launch.env["SB_PUSHT_CAMPAIGN_ROOT"] = "relative-campaign"
    result = launch.run("submit.sh", "baseline_seed0")
    assert result.returncode != 0
    assert not launch.rows()


def test_first_batch_submits_twenty_policies_in_six_comparisons(launch):
    result = launch.run("submit_first_batch.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = launch.rows("sbatch")
    train = [row for row in calls if any(arg.endswith("train.sbatch") for arg in row["args"])]
    assert len(train) == 20
    assert Counter(row["args"][-1] for row in train) == {
        "ddim": 6, "fm_paired": 6, "sb_kinetic": 6, "sb_ou": 2}
    assert Counter(row["args"][-2] for row in train) == {
        "baseline_seed1": 4, "baseline_seed2": 4,
        "h32_k8": 3, "h64_k8": 3, "wide18m": 3, "wide42m": 3}
    assert len(calls) == 32  # Six preparation jobs and six independent reference fits.


def test_first_batch_checks_all_existing_roots_before_submitting_anything(launch):
    (launch.campaign / "wide42m").mkdir(parents=True)
    result = launch.run("submit_first_batch.sh")
    assert result.returncode != 0
    assert not launch.rows("sbatch")


def test_second_batch_submits_seventeen_policies_in_six_comparisons(launch):
    result = launch.run("submit_second_batch.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = launch.rows("sbatch")
    train = [row for row in calls if any(arg.endswith("train.sbatch") for arg in row["args"])]
    assert len(train) == 17
    assert Counter(row["args"][-1] for row in train) == {
        "ddim": 3, "fm_paired": 4, "sb_kinetic": 4, "sb_ou": 6}
    assert Counter(row["args"][-2] for row in train) == {
        "scarcity50": 4, "scarcity25": 4, "scarcity10": 4,
        "direct_mlp": 3, "brownian": 1, "isotropic_ou": 1}
    assert len(calls) == 29
    for variant in ("scarcity50", "scarcity25", "scarcity10", "direct_mlp", "brownian", "isotropic_ou"):
        preparation = next((index, row) for index, row in enumerate(calls)
                           if row["args"][-2:] == ["hpc/sb_pusht_ablations/prepare.sbatch", variant])
        reference = next((index, row) for index, row in enumerate(calls)
                         if row["args"][-2:] == ["hpc/sb_pusht_ablations/reference.sbatch", variant])
        assert option(reference[1]["args"], "--dependency") == f"afterok:{1001 + preparation[0]}"
        for row in train:
            if row["args"][-2] == variant:
                required = preparation[0] if row["args"][-1] == "ddim" else reference[0]
                assert option(row["args"], "--dependency") == f"afterok:{1001 + required}"


def test_second_batch_checks_all_roots_before_submission(launch):
    (launch.campaign / "isotropic_ou").mkdir(parents=True)
    result = launch.run("submit_second_batch.sh")
    assert result.returncode != 0
    assert not launch.rows("sbatch")


@pytest.mark.parametrize("variant,methods", [
    ("brownian", {"sb_ou"}), ("isotropic_ou", {"sb_ou"}),
    ("direct_mlp", {"fm_paired", "sb_ou", "sb_kinetic"}),
])
def test_new_mechanism_variants_choose_only_relevant_default_methods(launch, variant, methods):
    result = launch.run("submit.sh", variant)
    assert result.returncode == 0, result.stdout + result.stderr
    train = [row for row in launch.rows("sbatch") if any(arg.endswith("train.sbatch") for arg in row["args"])]
    assert {row["args"][-1] for row in train} == methods


@pytest.mark.parametrize("script,arguments", [
    ("train.sbatch", ("../outside", "ddim")),
    ("train.sbatch", ("baseline_seed0", "invalid_method")),
    ("submit.sh", ("not_a_variant",)),
    ("submit.sh", ("baseline_seed0", "ddim", "ddim")),
    ("submit.sh", ("brownian", "sb_kinetic")),
    ("submit.sh", ("isotropic_ou", "fm_paired")),
    ("submit.sh", ("direct_mlp", "ddim")),
    ("train.sbatch", ("brownian", "fm_paired")),
    ("train.sbatch", ("isotropic_ou", "sb_kinetic")),
    ("train.sbatch", ("direct_mlp", "ddim")),
])
def test_bad_variant_and_method_fail_before_any_job_or_python_invocation(launch, script, arguments):
    result = launch.run(script, *arguments)
    assert result.returncode != 0
    assert not launch.rows()


def test_completion_comparison_uses_same_checkpoint_and_seed_panel(launch):
    checkpoint_root = launch.prepared()
    (checkpoint_root / "sb_kinetic").mkdir()
    checkpoint = checkpoint_root / "sb_kinetic/best.pt"
    checkpoint.touch()
    result = launch.run("completion_eval.sbatch", str(checkpoint_root), "sb_kinetic")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [row["args"] for row in launch.rows("python") if "--checkpoint" in row["args"]]
    assert len(commands) == 3
    assert {option(command, "--completion") for command in commands} == {
        "repeat", "fixed_damped", "learned_dissipative"}
    assert {option(command, "--checkpoint") for command in commands} == {str(checkpoint)}
    for command in commands:
        assert option(command, "--device") == "cpu"
        assert option(command, "--workers") == "4"
        assert option(command, "--threads") == "1"
        assert option(command, "--episodes") == "200"
        assert option(command, "--seed") == "1000000"
    assert len({option(command, "--output-dir") for command in commands}) == 3


def test_completion_evaluation_accepts_latest_and_custom_common_seed_panel(launch):
    root = launch.prepared()
    (root / "fm_paired").mkdir()
    checkpoint = root / "fm_paired/latest.pt"
    checkpoint.touch()
    launch.env.update(EVAL_EPISODES="30", EVAL_SEED="2000000")
    result = launch.run("completion_eval.sbatch", str(root), "fm_paired", "latest.pt")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [row["args"] for row in launch.rows("python") if "--checkpoint" in row["args"]]
    assert len(commands) == 3
    for command in commands:
        assert option(command, "--checkpoint") == str(checkpoint)
        assert option(command, "--episodes") == "30"
        assert option(command, "--seed") == "2000000"


def test_direct_mlp_completion_comparison_uses_only_checkpoint_trained_modes(launch):
    root = launch.prepared("direct_mlp")
    (root / "sb_kinetic").mkdir()
    (root / "sb_kinetic/best.pt").touch()
    launch.env["ABLATION_TEST_COMPLETION_MODES"] = "0,1,3"
    result = launch.run("completion_eval.sbatch", str(root), "sb_kinetic")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [row["args"] for row in launch.rows("python") if "--checkpoint" in row["args"]]
    assert {option(command, "--completion") for command in commands} == {
        "repeat", "fixed_damped", "direct_mlp"}


@pytest.mark.parametrize("modes,expected", [
    ([0, 1, 2], ["repeat", "fixed_damped", "learned_dissipative"]),
    ([0, 1, 3], ["repeat", "fixed_damped", "direct_mlp"]),
    ([3], ["direct_mlp"]),
])
def test_checkpoint_mode_reader_loads_actual_trusted_checkpoint(tmp_path, modes, expected):
    script = (JOBS / "completion_eval.sbatch").read_text().split("<<'PY_MODES'\n", 1)[1].split("\nPY_MODES", 1)[0]
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": {"training_completion_modes": modes}}, checkpoint)
    result = subprocess.run([sys.executable, "-", str(checkpoint)], input=script,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected


@pytest.mark.parametrize("modes", [[], [0, 0], [False], [4]])
def test_checkpoint_mode_reader_rejects_invalid_modes(tmp_path, modes):
    script = (JOBS / "completion_eval.sbatch").read_text().split("<<'PY_MODES'\n", 1)[1].split("\nPY_MODES", 1)[0]
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": {"training_completion_modes": modes}}, checkpoint)
    result = subprocess.run([sys.executable, "-", str(checkpoint)], input=script,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "invalid training_completion_modes" in result.stderr
