"""Small checks that the comparison uses shared information and physical units."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from workspace.surface_contact_bc.env import SurfaceContactEnv, sample_context
from workspace.surface_contact_bc.policies import build_policy
from workspace.surface_contact_bc.train import validation_mse


@pytest.mark.parametrize("suite,value", [("mass", 2.), ("stiffness", 3.),
    ("damping", 2.), ("friction", 3.), ("gains", .5)])
def test_hidden_physics_changes_do_not_change_initial_observation(suite, value):
    nominal = sample_context(17)
    shifted = sample_context(17, suite, value)
    np.testing.assert_array_equal(SurfaceContactEnv(nominal).observe(), SurfaceContactEnv(shifted).observe())
    assert nominal.physics != shifted.physics
    assert nominal.goal == shifted.goal


def test_rotating_context_and_targets_rotates_physical_rollout():
    context = sample_context(5)
    angle = .63
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = replace(context, normal=rotation @ context.normal,
                      initial_position=rotation @ context.initial_position,
                      initial_velocity=rotation @ context.initial_velocity)
    first, second = SurfaceContactEnv(context), SurfaceContactEnv(rotated)
    for step in range(40):
        target = context.normal * (context.offset - .06) + context.tangent * (context.goal + step * .001)
        first.step(target)
        second.step(rotation @ target)
        np.testing.assert_allclose(second.x, rotation @ first.x, atol=1e-12)
        np.testing.assert_allclose(second.v, rotation @ first.v, atol=1e-12)
        assert second.last_info["normal_force"] == pytest.approx(first.last_info["normal_force"])


@pytest.mark.parametrize("method", ["action_bridge_contact_frame_full", "action_bridge_isotropic",
    "action_bridge_contact_frame_no_kl", "diffusion_world", "diffusion_contact_frame"])
def test_learning_losses_ignore_evaluation_and_hidden_physics_fields(method):
    config = dict(method=method, hidden_dim=16, h_emb_dim=16, time_emb_dim=8,
                  horizon=4, unet_channels=[8, 16], num_train_timesteps=20, num_inference_steps=4)
    stats = dict(obs_mean=[0.] * 11, obs_scale=[1.] * 11, action_mean=[0., 0.], action_scale=1.)
    policy = build_policy(config, stats)
    obs = torch.randn(2, 2, 11)
    obs[..., 4:6] = torch.tensor([0., 1.])
    batch = dict(obs_hist=obs, act_hist=torch.randn(2, 2, 2), future_actions=torch.randn(2, 4, 2))
    torch.manual_seed(44)
    ordinary = policy.loss(batch)["loss"]
    # Evaluation-only information may exist in stored episodes, but it is not
    # used by the learner. NaNs make accidental dependence fail conspicuously.
    enriched = dict(batch, task_cost=torch.tensor(float("nan")),
                    success=torch.tensor(float("nan")), mass=torch.tensor(float("nan")),
                    penetration_penalty=torch.tensor(float("nan")))
    torch.manual_seed(44)
    augmented = policy.loss(enriched)["loss"]
    torch.testing.assert_close(ordinary, augmented)
    assert augmented.isfinite()


def test_validation_mse_is_world_coordinate_chunk_mean_not_teacher_forcing():
    class ConstantPolicy(torch.nn.Module):
        def generate(self, obs_hist, act_hist, generator=None):
            # Future demonstration actions are deliberately not arguments.
            return torch.zeros(len(obs_hist), 4, 2), {}

    batch = dict(obs_hist=torch.zeros(3, 2, 11), act_hist=torch.zeros(3, 2, 2),
                 future_actions=torch.full((3, 4, 2), 2.))
    assert validation_mse(ConstantPolicy(), batch, action_scale=.3) == pytest.approx(.36)
