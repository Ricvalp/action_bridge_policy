from __future__ import annotations

import builtins

import pytest
import torch
import torch.nn.functional as F
from torch import nn

DDIMScheduler = pytest.importorskip("diffusers").DDIMScheduler

from action_bridge.models.diffusion_policy import DiffusionPolicy


def make_policy(horizon=8, **overrides):
    config = {
        "unet_channels": [16, 32, 64],
        "h_emb_dim": 32,
        "hidden_dim": 32,
        "time_emb_dim": 16,
        "num_train_timesteps": 100,
        "num_inference_steps": 4,
        **overrides,
    }
    return DiffusionPolicy(23, 7, 2, 2, horizon, config)


def make_batch(horizon=8, batch_size=2):
    return {
        "obs_hist": torch.randn(batch_size, 2, 23),
        "act_hist": torch.randn(batch_size, 2, 7),
        "future_actions": torch.randn(batch_size, horizon, 7),
    }


def test_default_scheduler_uses_100_cosine_levels_and_unclipped_epsilon_ddim():
    policy = make_policy(num_inference_steps=20)
    scheduler = policy.noise_scheduler
    assert isinstance(scheduler, DDIMScheduler)
    assert scheduler.config.num_train_timesteps == 100
    assert scheduler.config.beta_schedule == "squaredcos_cap_v2"
    assert scheduler.config.prediction_type == "epsilon"
    assert scheduler.config.clip_sample is False
    assert scheduler.config.timestep_spacing == "leading"
    assert policy.num_inference_steps == 20
    scheduler.set_timesteps(policy.num_inference_steps)
    # Avoid near-zero terminal SNR: epsilon-to-x0 divides by sqrt(alpha_cumprod).
    assert scheduler.alphas_cumprod[scheduler.timesteps[0]] > 0.001
    assert any(isinstance(layer, nn.GroupNorm) for layer in policy.modules())
    assert not any(isinstance(layer, nn.BatchNorm1d) for layer in policy.modules())


def test_noise_loss_uses_library_add_noise_and_epsilon_target(monkeypatch):
    policy = make_policy()
    batch = make_batch()
    captured = {}
    original_add_noise = policy.noise_scheduler.add_noise
    original_predict_noise = policy.predict_noise

    def add_noise(actions, noise, timesteps):
        captured.update(actions=actions, noise=noise, timesteps=timesteps)
        return original_add_noise(actions, noise, timesteps)

    def predict_noise(actions, timesteps, history):
        prediction = original_predict_noise(actions, timesteps, history)
        captured["prediction"] = prediction
        return prediction

    monkeypatch.setattr(policy.noise_scheduler, "add_noise", add_noise)
    monkeypatch.setattr(policy, "predict_noise", predict_noise)
    metrics = policy.diffusion_loss(batch)
    assert set(metrics) == {"loss", "noise_mse"}
    assert captured["actions"] is batch["future_actions"]
    assert captured["timesteps"].shape == (2,)
    assert captured["timesteps"].dtype == torch.long
    assert (captured["timesteps"] >= 0).all()
    assert (captured["timesteps"] < 100).all()
    expected = F.mse_loss(captured["prediction"], captured["noise"])
    torch.testing.assert_close(metrics["loss"], expected)
    torch.testing.assert_close(metrics["noise_mse"], expected.detach())
    assert not metrics["noise_mse"].requires_grad


def test_noise_loss_trains_history_encoder_and_unet():
    policy = make_policy().train()
    metrics = policy.diffusion_loss(make_batch())
    assert metrics["loss"].ndim == 0
    assert torch.isfinite(metrics["loss"])
    metrics["loss"].backward()
    for component in (policy.history_encoder, policy.time_embedding, policy.unet):
        gradients = [parameter.grad for parameter in component.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
        assert sum(gradient.abs().sum() for gradient in gradients).item() > 0


@pytest.mark.parametrize("horizon", [1, 4, 5, 8, 9])
def test_full_chunk_shape_and_finite_loss_for_short_or_odd_horizons(horizon):
    policy = make_policy(horizon).eval()
    batch = make_batch(horizon, batch_size=1)
    actions = policy.generate(
        batch["obs_hist"],
        batch["act_hist"],
        generator=torch.Generator().manual_seed(10),
    )
    assert actions.shape == (1, horizon, 7)
    assert torch.isfinite(actions).all()
    assert not actions.requires_grad
    assert torch.isfinite(policy.diffusion_loss(batch)["loss"])


def test_explicit_generator_repeats_sampling_without_consuming_global_rng():
    policy = make_policy().eval()
    batch = make_batch()
    global_state = torch.random.get_rng_state().clone()
    first = policy.generate(
        batch["obs_hist"],
        batch["act_hist"],
        generator=torch.Generator().manual_seed(321),
    )
    second = policy.generate(
        batch["obs_hist"],
        batch["act_hist"],
        generator=torch.Generator().manual_seed(321),
    )
    different = policy.generate(
        batch["obs_hist"],
        batch["act_hist"],
        generator=torch.Generator().manual_seed(322),
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.equal(global_state, torch.random.get_rng_state())


def test_sampling_uses_20_library_ddim_steps_with_eta_zero(monkeypatch):
    policy = make_policy(num_inference_steps=20).eval()
    batch = make_batch()
    generator = torch.Generator().manual_seed(123)
    original_step = policy.noise_scheduler.step
    recorded = []

    def step(noise, timestep, actions, *, eta, generator):
        recorded.append((int(timestep), eta, generator))
        return original_step(noise, timestep, actions, eta=eta, generator=generator)

    monkeypatch.setattr(policy.noise_scheduler, "step", step)
    policy.generate(batch["obs_hist"], batch["act_hist"], generator=generator)
    assert [timestep for timestep, _, _ in recorded] == list(range(95, -1, -5))
    assert all(eta == 0.0 for _, eta, _ in recorded)
    assert all(actual_generator is generator for _, _, actual_generator in recorded)


def test_forward_delegates_to_chunk_generation(monkeypatch):
    policy = make_policy()
    batch = make_batch()
    expected = torch.ones(2, 8, 7)

    def generate(obs_hist, act_hist):
        assert obs_hist is batch["obs_hist"]
        assert act_hist is batch["act_hist"]
        return expected

    monkeypatch.setattr(policy, "generate", generate)
    assert policy(batch["obs_hist"], batch["act_hist"]) is expected


def test_default_square_policy_is_about_five_million_parameters():
    policy = DiffusionPolicy(23, 7, 2, 2, 8, {})
    assert sum(parameter.numel() for parameter in policy.parameters()) == 5_258_455
    assert all(parameter.requires_grad for parameter in policy.parameters())


def test_history_encoder_respects_configured_hidden_width():
    policy = make_policy(hidden_dim=48)
    assert policy.history_encoder.net[0].out_features == 48
    assert policy.history_encoder.net[-1].out_features == 32


def test_missing_optional_dependency_has_setup_hint(monkeypatch):
    original_import = builtins.__import__

    def import_without_diffusers(name, *args, **kwargs):
        if name == "diffusers":
            raise ModuleNotFoundError("No module named 'diffusers'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_diffusers)
    with pytest.raises(RuntimeError, match="--extra diffusion"):
        make_policy()


@pytest.mark.parametrize(
    "overrides",
    [
        {"unet_channels": []},
        {"unet_channels": [15, 32, 64]},
        {"num_train_timesteps": 0},
        {"num_inference_steps": 0},
        {"num_inference_steps": 101},
    ],
)
def test_invalid_diffusion_settings_fail_clearly(overrides):
    with pytest.raises(ValueError):
        make_policy(**overrides)
