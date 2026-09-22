"""Fast shape/conditioning/sampling checks; these are not learning experiments."""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest
import torch
from torch import nn

from action_bridge.plan_revision.models import FieldNet, build_policy


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(method, horizon=9, action_dim=6):
    return dict(method=method, horizon=horizon, action_dim=action_dim,
                obs_dim=7, obs_history=3, action_history=2, channels=[8, 16],
                history_dim=16, hidden_dim=16, time_dim=8, num_inference_steps=4)


def batch(cfg):
    return {"obs_hist": torch.randn(2, cfg["obs_history"], cfg["obs_dim"]),
            "act_hist": torch.randn(2, cfg["action_history"], cfg["action_dim"]),
            "future_actions": torch.randn(2, cfg["horizon"], cfg["action_dim"]),
            "source_actions": torch.randn(2, cfg["horizon"], cfg["action_dim"]),
            "valid_mask": torch.ones(2, cfg["horizon"], dtype=torch.bool),
            "completion_id": torch.tensor([0, 2])}


@pytest.mark.parametrize("method", ["ddim", "fm_paired", "fm_local_ot"])
@pytest.mark.parametrize("shape", [(16, 2), (9, 6)])
def test_loss_backward_sampling_and_reload(method, shape):
    cfg = config(method, *shape)
    policy = build_policy(cfg)
    data = batch(cfg)
    loss = policy.loss(data, generator=torch.Generator().manual_seed(11))["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in policy.parameters()]
    assert all(x is not None and torch.isfinite(x).all() for x in gradients)
    policy.eval()
    generated, diagnostics = policy.sample(
        data["obs_hist"], data["act_hist"], data["completion_id"],
        source_actions=data["source_actions"], generator=torch.Generator().manual_seed(12),
    )
    assert generated.shape == data["future_actions"].shape
    assert torch.isfinite(generated).all()
    assert diagnostics["nfe"] == cfg["num_inference_steps"]
    checkpoint = {"config": cfg, "model": copy.deepcopy(policy.state_dict())}
    restored = build_policy(checkpoint["config"]).eval()
    restored.load_state_dict(checkpoint["model"])
    replay, _ = restored.sample(
        data["obs_hist"], data["act_hist"], data["completion_id"],
        source_actions=data["source_actions"], generator=torch.Generator().manual_seed(12),
    )
    torch.testing.assert_close(replay, generated, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["ddim", "fm_paired", "sb_ou", "sb_kinetic"])
def test_partial_endpoint_masks_are_rejected(method):
    cfg = config(method)
    data = batch(cfg)
    data["valid_mask"][0, -1] = False
    policy = build_policy(cfg)
    kwargs = {"reference": None} if method.startswith("sb_") else {}
    with pytest.raises(ValueError, match="fully valid"):
        policy.loss(data, **kwargs)


def test_ddim_library_scheduler_metadata_and_independent_noise():
    cfg = config("ddim")
    policy = build_policy(cfg).eval()
    metadata = policy.scheduler_metadata()
    assert metadata["config"]["num_train_timesteps"] == 100
    assert metadata["config"]["clip_sample"] is False
    assert metadata["config"]["beta_schedule"] == "squaredcos_cap_v2"
    assert metadata["config"]["prediction_type"] == "epsilon"
    assert metadata["diffusers_version"]
    data = batch(cfg)
    first, _ = policy.sample(data["obs_hist"], data["act_hist"], generator=torch.Generator().manual_seed(1))
    second, _ = policy.sample(data["obs_hist"], data["act_hist"], generator=torch.Generator().manual_seed(2))
    assert not torch.equal(first, second)


def test_midpoint_counts_two_evaluations_per_step(monkeypatch):
    cfg = config("fm_paired")
    policy = build_policy(cfg)
    data = batch(cfg)
    times = []

    def constant_velocity(state, time, conditioning):
        times.append(float(time))
        return torch.ones_like(state)

    monkeypatch.setattr(policy.field, "conditioned", constant_velocity)
    actions, info = policy.sample(data["obs_hist"], data["act_hist"], data["completion_id"],
                                  source_actions=data["source_actions"])
    torch.testing.assert_close(actions, data["source_actions"] + 1)
    assert times == [0., .25, .5, .75]
    assert info["nfe"] == len(times)


class MappingEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Linear(cfg["obs_dim"] + cfg["action_dim"], cfg["history_dim"])

    def forward(self, obs, actions):
        return self.net(torch.cat((obs["state"][:, -1], actions[:, -1]), dim=-1))


@pytest.mark.parametrize("method", ["ddim", "fm_paired", "sb_kinetic"])
def test_encoder_is_injected_and_accepts_dictionary_observations(method):
    cfg = config(method)
    policy = build_policy(cfg, encoder=MappingEncoder(cfg))
    data = batch(cfg)
    data["obs_hist"] = {"state": data["obs_hist"]}
    if method.startswith("sb_"):
        state = torch.randn(2, 2 * cfg["horizon"] * cfg["action_dim"])
        result = policy.forward_field(state, torch.tensor([.3, .7]), data["obs_hist"], data["act_hist"], data["completion_id"])
        assert result.shape == (2, cfg["horizon"] * cfg["action_dim"])
        assert policy.forward_field.history_encoder is not policy.reverse_field.history_encoder
    else:
        loss = policy.loss(data)["loss"]
        loss.backward()
        assert torch.isfinite(loss)
        result, _ = policy.sample(data["obs_hist"], data["act_hist"], data["completion_id"],
                                  source_actions=data["source_actions"])
        assert result.shape == data["future_actions"].shape


def test_conditioning_uses_completion_id_and_no_teacher_endpoint():
    cfg = config("fm_paired")
    data = batch(cfg)
    field = FieldNet(cfg)
    first = field.conditioning(data["obs_hist"], data["act_hist"], torch.zeros(2, dtype=torch.long))
    second = field.conditioning(data["obs_hist"], data["act_hist"], torch.ones(2, dtype=torch.long))
    assert not torch.equal(first, second)
    data["future_actions"].fill_(1e9)
    repeat = field.conditioning(data["obs_hist"], data["act_hist"], torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(first, repeat)


def test_unknown_encoder_spec_requires_explicit_factory():
    cfg = config("fm_paired") | {"encoder_spec": {"kind": "MyImageEncoder", "observations": "mapping"}}
    with pytest.raises(ValueError, match="caller-provided factory"):
        build_policy(cfg)
    policy = build_policy(cfg, encoder=MappingEncoder(cfg))
    assert isinstance(policy.field.history_encoder, MappingEncoder)


def test_generic_policy_import_has_no_simulator_dependency():
    # Use an isolated import: other tests may legitimately import the simulator.
    result = subprocess.run([sys.executable, "-c", """
import builtins
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'gym_pusht', 'pymunk', 'gymnasium'}:
        raise AssertionError('generic policy imported a simulator: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from action_bridge.plan_revision.models import build_policy
policy = build_policy(dict(method='fm_paired', horizon=9, action_dim=6,
    obs_dim=7, obs_history=3, action_history=2, channels=[8,16],
    history_dim=16, hidden_dim=16, time_dim=8, num_inference_steps=4))
"""], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("method", ["sb_ou", "sb_kinetic"])
@pytest.mark.parametrize("shape", [(16, 2), (9, 6)])
def test_bridge_backward_sampling_and_reload(method, shape):
    from action_bridge.plan_revision.gaussian import GaussianReference
    cfg = config(method, *shape)
    data = batch(cfg)
    n = cfg["horizon"] * cfg["action_dim"]
    reference = GaussianReference(torch.eye(n).expand(2, n, n), torch.zeros(2, n),
                                  kind="kinetic" if method == "sb_kinetic" else "ou")
    policy = build_policy(cfg)
    for direction in ("forward", "reverse"):
        loss = policy.loss(data, reference, direction=direction, generator=torch.Generator().manual_seed(3))["loss"]
        assert torch.isfinite(loss)
        loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
    sampled, metrics = policy.sample(data["obs_hist"], data["act_hist"], data["completion_id"],
                                     source_actions=data["source_actions"], reference=reference,
                                     generator=torch.Generator().manual_seed(4))
    assert sampled.shape == data["source_actions"].shape
    assert torch.isfinite(sampled).all()
    assert metrics["nfe"] == cfg["num_inference_steps"]
    assert (metrics["control_energy"] >= 0).all()
    restored = build_policy(cfg)
    restored.load_state_dict(copy.deepcopy(policy.state_dict()))
    repeated, _ = restored.sample(data["obs_hist"], data["act_hist"], data["completion_id"],
                                  source_actions=data["source_actions"], reference=reference,
                                  generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(repeated, sampled, rtol=0, atol=0)
