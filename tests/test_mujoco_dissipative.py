from __future__ import annotations

import pytest
import torch

from action_bridge.config import apply_overrides, load_config
from action_bridge.eval.rollout import generate_chunk, predict_actions
from action_bridge.models.action_bridge_policy import ContactControlNet, ControlNet
from action_bridge.models.references import ContactLangevinReference
from action_bridge.training.common import build_model
from action_bridge.training.losses import (
    contact_path_losses_conditioned,
    contact_stopgrad_path_losses_conditioned,
    model_loss,
)
from action_bridge.training.train_toy import maybe_update_reference_ema


JOINT = "mujoco_robomimic_square_dissipative"
STOPGRAD = "mujoco_robomimic_square_dissipative_stopgrad"


def _tiny_config(name, *, latent_type="none"):
    return apply_overrides(load_config(name), [
        "device=cpu",
        "model.hidden_dim=16",
        "model.h_emb_dim=16",
        "model.time_emb_dim=8",
        "model.z_embed_dim=8",
        "model.z_dim=4",
        f'model.latent_type="{latent_type}"',
        "reference.hidden_dim=16",
        "reference.time_emb_dim=8",
    ])


def _batch(config):
    generator = torch.Generator().manual_seed(23)
    # Nonzero action differences exercise both damping and potential gradients.
    return {
        "obs_hist": torch.randn(3, 2, 23, generator=generator) * 0.2,
        "act_hist": torch.randn(3, 2, 7, generator=generator) * 0.2,
        "future_actions": torch.randn(3, config.chunk_horizon, 7, generator=generator) * 0.2,
    }


def _assert_has_gradient(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def _assert_no_gradient(module):
    assert all(p.grad is None for p in module.parameters())


@pytest.mark.parametrize("name", [JOINT, STOPGRAD])
def test_dissipative_configs_construct_the_intended_five_million_parameter_model(name):
    config = load_config(name)
    assert config.benchmark == "mujoco"
    assert config.data.integration == "robomimic_square"
    assert (config.obs_dim, config.action_dim) == (23, 7)
    assert config.data.normalize is True
    assert config.chunk_horizon == 8
    assert config.eval.actions_per_plan == config.inference.n_exec == 4
    assert config.model.latent_type in {None, "none"}
    assert config.reference.type == "contact_langevin"
    assert config.reference.coordinate_mode == "raw_action"
    assert config.reference.potential_type == "quadratic"
    assert config.reference.attractor_mode == "learned"
    assert config.reference.stiffness_mode == "learned_diag"
    assert config.reference.gamma_mode == "learned_scalar"
    assert config.reference.control_is_whitened is True
    model = build_model(config)
    assert isinstance(model.reference_process, ContactLangevinReference)
    assert isinstance(model.control_net, ContactControlNet)
    assert not isinstance(model.control_net, ControlNet)
    assert model.latent is None
    assert model.z_embed_dim == 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 4_500_000 <= trainable <= 5_500_000
    assert hasattr(model, "reference_process_ema") == (name == STOPGRAD)
    if name == STOPGRAD:
        assert config.loss.contact_objective == "stopgrad_reference"
        assert config.loss.passive_target == "damped_continuation"
        assert all(not p.requires_grad for p in model.reference_process_ema.parameters())
    else:
        assert config.loss.contact_objective == "standard"


@pytest.mark.parametrize("name", [JOINT, STOPGRAD])
def test_phase_coordinates_are_normalized_commands_and_control_is_not_tanh_bounded(name):
    config = _tiny_config(name)
    model = build_model(config)
    batch = _batch(config)
    adapter = model.coordinate_adapter
    q, p = adapter.init_qp_from_history(batch)
    assert torch.equal(q, batch["act_hist"][:, -1])
    assert torch.equal(p, (batch["act_hist"][:, -1] - batch["act_hist"][:, -2]) / adapter.dt)
    sequence = adapter.build_q_sequence(batch)
    assert torch.equal(adapter.decode_raw_actions(sequence), batch["future_actions"])
    # These q values are commands, not integrated XYZ/axis-angle physical poses.
    assert torch.equal(sequence[:, 1:], batch["future_actions"])
    with torch.no_grad():
        for parameter in model.control_net.parameters():
            parameter.zero_()
        final_linear = [layer for layer in model.control_net.modules() if isinstance(layer, torch.nn.Linear)][-1]
        final_linear.bias.fill_(5.0 / model.control_net.control_scale)
        history = model.encode_history(batch["obs_hist"], batch["act_hist"])
        control = model.contact_control(q, p, history, 0)
    torch.testing.assert_close(control, torch.full_like(control, 5.0))


@pytest.mark.parametrize("name", [JOINT, STOPGRAD])
def test_task_gradient_routing_distinguishes_joint_from_stopgrad_reference(name):
    torch.manual_seed(17)
    config = _tiny_config(name)
    model = build_model(config)
    batch = _batch(config)
    history = model.encode_history(batch["obs_hist"], batch["act_hist"])
    if name == STOPGRAD:
        metrics = contact_stopgrad_path_losses_conditioned(model, history, batch, None, config.loss)
    else:
        metrics = contact_path_losses_conditioned(model, history, batch, None)
    task_loss = (metrics["nll"] + model.reference_process.beta_kl * metrics["path_kl"]).mean()
    task_loss.backward()
    _assert_has_gradient(model.control_net)
    _assert_has_gradient(model.history_encoder)
    for head in (model.reference_process.m_net, model.reference_process.k_net, model.reference_process.gamma_net):
        if name == STOPGRAD:
            _assert_no_gradient(head)
        else:
            _assert_has_gradient(head)
    if name == STOPGRAD:
        _assert_no_gradient(model.reference_process_ema)


def test_stopgrad_passive_loss_trains_reference_not_control():
    torch.manual_seed(17)
    config = _tiny_config(STOPGRAD)
    model = build_model(config)
    batch = _batch(config)
    history = model.encode_history(batch["obs_hist"], batch["act_hist"])
    metrics = contact_stopgrad_path_losses_conditioned(model, history, batch, None, config.loss)
    passive_loss = (
        metrics["reference_target_loss"]
        + config.loss.lambda_slow * metrics["reference_slow_loss"]
        + config.loss.lambda_diss * metrics["reference_dissipation_loss"]
    ).mean()
    passive_loss.backward()
    for head in (model.reference_process.m_net, model.reference_process.k_net, model.reference_process.gamma_net):
        _assert_has_gradient(head)
    _assert_has_gradient(model.history_encoder)
    _assert_no_gradient(model.control_net)
    _assert_no_gradient(model.reference_process_ema)


@pytest.mark.parametrize("name", [JOINT, STOPGRAD])
@pytest.mark.parametrize("latent_type", ["none", "continuous"])
def test_complete_dissipative_loss_is_finite_and_updates_model_and_ema(name, latent_type):
    torch.manual_seed(17)
    config = _tiny_config(name, latent_type=latent_type)
    model = build_model(config)
    assert (model.latent is None) == (latent_type == "none")
    batch = _batch(config)
    before = {key: value.clone() for key, value in model.reference_process.state_dict().items()}
    output = model_loss(model, batch, config.loss, global_step=10_000)
    assert output["loss"].ndim == 0
    assert all(torch.isfinite(value).all() for value in output.values() if torch.is_tensor(value))
    if latent_type == "none":
        assert output["latent_kl"].item() == 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output["loss"].backward()
    _assert_has_gradient(model.control_net)
    for head in (model.reference_process.m_net, model.reference_process.k_net, model.reference_process.gamma_net):
        _assert_has_gradient(head)
    optimizer.step()
    assert any(not torch.equal(before[key], value) for key, value in model.reference_process.state_dict().items())
    maybe_update_reference_ema(model, config)
    if name == STOPGRAD:
        decay = config.loss.ema_decay
        live_state = model.reference_process.state_dict()
        for key, value in model.reference_process_ema.state_dict().items():
            torch.testing.assert_close(value, decay * before[key] + (1 - decay) * live_state[key])
        _assert_no_gradient(model.reference_process_ema)
    else:
        assert not hasattr(model, "reference_process_ema")


@pytest.mark.parametrize("name", [JOINT, STOPGRAD])
def test_dissipative_inference_uses_ema_only_for_stopgrad_models(name):
    torch.manual_seed(17)
    config = _tiny_config(name)
    model = build_model(config).eval()
    batch = _batch(config)
    inputs = {key: batch[key] for key in ("obs_hist", "act_hist")}
    before = predict_actions(model, inputs)["actions"]
    assert before.shape == (3, 8, 7)
    assert torch.isfinite(before).all()
    with torch.no_grad():
        model.reference_process.m_net[-1].bias.add_(0.3)
    after_live_change = generate_chunk(model, inputs["obs_hist"], inputs["act_hist"])["actions"]
    if name == STOPGRAD:
        torch.testing.assert_close(after_live_change, before, rtol=0, atol=0)
        with torch.no_grad():
            model.reference_process_ema.m_net[-1].bias.add_(0.3)
        after_ema_change = predict_actions(model, inputs)["actions"]
        assert not torch.allclose(after_ema_change, before)
    else:
        assert not torch.allclose(after_live_change, before)
