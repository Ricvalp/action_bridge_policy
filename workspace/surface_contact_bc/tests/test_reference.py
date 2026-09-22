"""Geometry, command-reference semantics, and shared policy adapter contracts."""

import copy
import math
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from action_bridge.models.references import ContactLangevinReference
from workspace.surface_contact_bc.policies import METHODS, build_policy
from workspace.surface_contact_bc.reference import ContactFrameReference, contact_projectors


def stats():
    return {"obs_mean": [0.] * 11, "obs_scale": [1.] * 11, "action_mean": [0., 0.], "action_scale": 1.}


def config(method="action_bridge_contact_frame_full"):
    return {"method": method, "hidden_dim": 16, "h_emb_dim": 16,
            "time_emb_dim": 8, "horizon": 4, "unet_channels": [8, 16],
            "num_train_timesteps": 20, "num_inference_steps": 4}


def batch():
    torch.manual_seed(31)
    obs = torch.randn(3, 2, 11)
    obs[..., 4:6] = torch.tensor([0., 1.])
    return {"obs_hist": obs, "act_hist": torch.randn(3, 2, 2), "future_actions": torch.randn(3, 4, 2)}


def test_projectors_and_rotating_damping():
    n = torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    pn, pt = contact_projectors(n)
    eye = torch.eye(2, dtype=n.dtype)[None]
    torch.testing.assert_close(pn + pt, eye)
    for projector in (pn, pt):
        torch.testing.assert_close(projector, projector.transpose(-2, -1))
        torch.testing.assert_close(projector @ projector, projector)
    torch.testing.assert_close(pn @ pt, torch.zeros_like(pn), atol=1e-12, rtol=0)
    angle = 0.73
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=n.dtype)
    pn_rotated, pt_rotated = contact_projectors(n @ rotation.T)
    damping = 0.2 * pn + 0.6 * pt
    torch.testing.assert_close(0.2 * pn_rotated + 0.6 * pt_rotated, rotation @ damping @ rotation.T)


def test_parameter_bounds_and_no_forced_damping_order():
    reference = ContactFrameReference(config(), stats())
    with torch.no_grad():
        reference.parameter_net[-1].bias[:2] = torch.tensor([-4., 4.])
    values = reference.parameters_at(torch.randn(12, 16), 2)
    assert (values["gamma_normal"] < values["gamma_tangent"]).all()
    for name in ("gamma_normal", "gamma_tangent"):
        assert (values[name] >= reference.gamma_min).all()
        assert (values[name] <= reference.gamma_max).all()
    assert (values["stiffness"] >= reference.k_min).all()
    assert (values["stiffness"] <= reference.k_max).all()
    assert (values["desired_offset"].abs() <= reference.offset_max).all()


def test_normal_spring_has_no_tangential_force_or_goal_attractor():
    reference = ContactFrameReference(config(), stats())
    q, p, h = torch.tensor([[3., 0.2]]), torch.zeros(1, 2), torch.zeros(1, 16)
    obs = torch.zeros(1, 11)
    obs[:, 5] = 1.
    force, aux = reference.force(q, p, h, 0, obs)
    assert force[0, 0] == 0
    assert force[0, 1] < 0
    obs[:, 8] = 10000  # Goal cannot directly enter the reference force.
    changed, _ = reference.force(q, p, h, 0, obs)
    torch.testing.assert_close(force, changed)
    torch.testing.assert_close(aux["grad_v"][:, 0], torch.zeros(1))


def test_geometry_uses_physical_surface_with_shifted_action_normalization():
    normalizer = stats()
    normalizer.update(action_mean=[0.2, -0.3], action_scale=0.4)
    normalizer["obs_mean"] = [0.1] * 11
    normalizer["obs_scale"] = [0.2] * 11
    reference = ContactFrameReference(config(), normalizer)
    raw = torch.zeros(1, 11)
    raw[:, 4:6] = torch.tensor([0.6, 0.8])
    raw[:, 6] = 0.12
    normalized_obs = (raw - reference.obs_mean) / reference.obs_scale
    world_q = torch.tensor([[0.072, 0.096]])  # n.q = c: precisely on the line.
    q = (world_q - reference.action_mean) / reference.action_scale
    force, _ = reference.force(q, torch.zeros_like(q), torch.zeros(1, 16), 0, normalized_obs)
    torch.testing.assert_close(force, torch.zeros_like(force), atol=1e-8, rtol=0)


def test_reference_damps_normal_target_energy():
    reference = ContactFrameReference(config(), stats())
    q, p, h = torch.tensor([[0., 0.3]]), torch.tensor([[0., 0.05]]), torch.zeros(1, 16)
    obs = torch.zeros(1, 11)
    obs[:, 5] = 1
    stiffness = reference.parameters_at(h, 0)["stiffness"].item()
    energy_before = 0.5 * (p[:, 1].square() + stiffness * q[:, 1].square())
    for k in range(100):
        q, p, _ = reference.reference_step(q, p, h, k, obs)
    energy_after = 0.5 * (p[:, 1].square() + stiffness * q[:, 1].square())
    assert (energy_after < energy_before * 0.05).all()


def test_isotropic_damping_matches_existing_reference():
    reference = ContactFrameReference(config("action_bridge_isotropic"), stats())
    with torch.no_grad():
        reference.parameter_net[-1].bias[1] = -100  # Zero spring, isolate damping.
    h = torch.randn(3, 16)
    gamma = reference.parameters_at(h, 0)["gamma_normal"][0].item()
    existing = ContactLangevinReference(2, 16, gamma_const=gamma, potential_type="none")
    data = batch()
    q, p = data["act_hist"][:, -1], data["act_hist"][:, -2]
    force, _ = reference.force(q, p, h, 0, data["obs_hist"][:, -1])
    old_force, _ = existing.force(q, p, h, 0)
    torch.testing.assert_close(force, old_force)


def test_reference_only_loads_full_checkpoint_and_has_zero_kl():
    policy = build_policy(config(), stats())
    reference_only = build_policy(config("reference_only"), stats())
    reference_only.load_state_dict(policy.state_dict(), strict=True)
    data = batch()
    actions, diagnostics = reference_only.generate(data["obs_hist"], data["act_hist"], diagnostics=True)
    assert actions.shape == (3, 4, 2)
    torch.testing.assert_close(diagnostics["path_kl"], torch.zeros(3, 4))
    torch.testing.assert_close(diagnostics["residual_normal"], torch.zeros(3, 4))


def test_zero_control_has_zero_training_path_kl():
    policy = build_policy(config(), stats())
    with torch.no_grad():
        for parameter in policy.model.control_net.parameters():
            parameter.zero_()
    assert policy.loss(batch())["path_kl"] == 0


def test_diffusion_frame_roundtrip_preserves_world_action_mse():
    policy = build_policy(config("diffusion_contact_frame"), stats())
    data = batch()
    data["obs_hist"][..., 4:6] = torch.tensor([0.6, 0.8])
    _, _, local = policy.encode_frame(data["obs_hist"], data["act_hist"], data["future_actions"])
    world = policy.decode_frame(local, data["obs_hist"])
    torch.testing.assert_close(world, data["future_actions"], atol=5e-7, rtol=1e-5)
    perturbation = torch.randn_like(local)
    world_delta = policy.decode_frame(local + perturbation, data["obs_hist"]) - world
    torch.testing.assert_close(world_delta.square().sum(-1), perturbation.square().sum(-1))


def test_residual_diffusion_inverse_dynamics_and_self_contained_reload():
    bridge = build_policy(config(), stats())
    policy = build_policy(config("diffusion_residual"), stats(), reference_policy=bridge)
    data = batch()
    controls = policy.decoder.inverse_controls(data)
    decoded, _ = policy.decoder.decode(controls, data["obs_hist"], data["act_hist"], False)
    torch.testing.assert_close(decoded, data["future_actions"], atol=5e-6, rtol=1e-5)
    policy.fit_residual_normalization(data)
    state = copy.deepcopy(policy.state_dict())
    reloaded = build_policy(policy.config, stats())
    reloaded.load_state_dict(state, strict=True)
    assert not any(p.requires_grad for p in reloaded.decoder.parameters())
    policy.eval()
    reloaded.eval()
    outputs = []
    for model in (policy, reloaded):
        outputs.append(model.generate(data["obs_hist"], data["act_hist"], generator=torch.Generator().manual_seed(91))[0])
    torch.testing.assert_close(*outputs)


def test_diffusion_sample_loss_targets_clean_actions_and_labels_metrics():
    policy = build_policy(config("diffusion_world") | {"prediction_type": "sample"}, stats())
    data = batch()
    # Perfect clean-action prediction must have zero denoising loss, regardless
    # of which random corruption/timestep was drawn.
    with patch.object(policy.model, "predict_noise", return_value=data["future_actions"]):
        losses = policy.loss(data)
        generated, _ = policy.generate(data["obs_hist"], data["act_hist"],
                                        generator=torch.Generator().manual_seed(3))
    assert losses["loss"] == 0
    torch.testing.assert_close(generated, data["future_actions"])
    assert "action_denoising_mse" in losses
    assert "noise_mse" not in losses
    assert policy.model.noise_scheduler.config.prediction_type == "sample"


@pytest.mark.parametrize("method", ["diffusion_world", "diffusion_contact_frame", "diffusion_residual"])
def test_sample_prediction_all_diffusion_variants_reload_and_generate(method):
    settings = config(method) | {"prediction_type": "sample"}
    reference = build_policy(config(), stats()) if method == "diffusion_residual" else None
    policy = build_policy(settings, stats(), reference_policy=reference)
    data = batch()
    if method == "diffusion_residual":
        policy.fit_residual_normalization(data)
    losses = policy.loss(data)
    losses["loss"].backward()
    assert losses["loss"].isfinite()
    assert "noise_mse" not in losses
    restored = build_policy(policy.config, stats())
    restored.load_state_dict(policy.state_dict(), strict=True)
    outputs = [model.generate(data["obs_hist"], data["act_hist"],
                              generator=torch.Generator().manual_seed(4))[0]
               for model in (policy.eval(), restored.eval())]
    torch.testing.assert_close(*outputs)
    assert outputs[0].isfinite().all()


def test_diffusion_prediction_type_validation_and_default_epsilon():
    policy = build_policy(config("diffusion_world"), stats())
    assert policy.prediction_type == "epsilon"
    losses = policy.loss(batch())
    assert "noise_mse" in losses and "action_denoising_mse" not in losses
    with pytest.raises(ValueError, match="prediction_type"):
        build_policy(config("diffusion_world") | {"prediction_type": "invalid"}, stats())


@pytest.mark.parametrize("method", [method for method in METHODS if method not in {"reference_only", "diffusion_residual"}])
def test_each_primary_policy_loss_backprop_and_generation(method):
    policy = build_policy(config(method), stats())
    data = batch()
    loss = policy.loss(data)["loss"]
    assert loss.isfinite()
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    actions, _ = policy.generate(data["obs_hist"], data["act_hist"], generator=torch.Generator().manual_seed(4))
    assert actions.shape == data["future_actions"].shape
    assert actions.isfinite().all()


@pytest.mark.skipif(os.environ.get("CONTACT_BC_SLOW_TESTS") != "1",
                    reason="Set CONTACT_BC_SLOW_TESTS=1 for the CPU tiny-data overfit check")
@pytest.mark.parametrize("method,prediction_type,updates", [
    ("action_bridge_contact_frame_full", "epsilon", 500),
    ("diffusion_world", "epsilon", 2500),
    ("diffusion_world", "sample", 1000),
])
def test_overfit_actual_expert_windows(tmp_path, method, prediction_type, updates):
    from workspace.surface_contact_bc.data import EpisodeWindows, make_dataset
    from workspace.surface_contact_bc.train import update_ema

    make_dataset(tmp_path / "demos", train_episodes=1, val_episodes=0, test_episodes=0)
    windows = EpisodeWindows(tmp_path / "demos", horizon=4)
    indices = np.tile(np.linspace(0, len(windows) - 1, 8, dtype=int), 16)
    data = {key: torch.tensor(np.stack([windows[int(index)][key] for index in indices]))
            for key in ("obs_hist", "act_hist", "future_actions")}
    torch.manual_seed(1)
    settings = config(method) | {"hidden_dim": 128, "h_emb_dim": 128,
                                 "time_emb_dim": 32, "unet_channels": [32, 64],
                                 "num_train_timesteps": 100, "num_inference_steps": 20,
                                 "prediction_type": prediction_type}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        policy = build_policy(settings, windows.normalizer.to_dict())
        optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001)
        ema = copy.deepcopy(policy).eval()
        ema.requires_grad_(False)

        def generated_mse():
            prediction, _ = ema.generate(data["obs_hist"], data["act_hist"],
                                         generator=torch.Generator().manual_seed(0))
            return (prediction - data["future_actions"]).square().mean().item()

        before = generated_mse()
        for step in range(updates):
            optimizer.param_groups[0]["lr"] = 0.001 * (0.1 + 0.9 * (1 + math.cos(math.pi * step / updates)) / 2)
            optimizer.zero_grad(set_to_none=True)
            policy.loss(data)["loss"].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            update_ema(ema, policy, min(0.995, (step + 2) / (step + 11)))
        after = generated_mse()
        assert after < before * 0.05
        assert after * windows.normalizer.action_scale ** 2 < 1e-4  # m², sampled chunks, no best-of-N.
    finally:
        torch.set_num_threads(previous_threads)
