import torch

from action_bridge.models.latents import CategoricalLatent, ContinuousLatent, categorical_kl, gaussian_kl


def test_categorical_latent_shapes_and_kl():
    latent = CategoricalLatent(h_emb_dim=8, chunk_horizon=4, action_dim=2, num_categories=2, z_embed_dim=5, hidden_dim=16)
    h = torch.randn(3, 8)
    future = torch.randn(3, 4, 2)
    p = latent.prior_logits(h)
    q = latent.posterior_logits(h, future)
    assert p.shape == (3, 2)
    assert q.shape == (3, 2)
    assert latent.embed_ids(torch.tensor([0, 1, 0])).shape == (3, 5)
    assert categorical_kl(q, p).shape == (3,)


def test_continuous_latent_shapes_and_kl():
    latent = ContinuousLatent(h_emb_dim=8, chunk_horizon=4, action_dim=2, z_dim=3, z_embed_dim=5, hidden_dim=16)
    h = torch.randn(3, 8)
    future = torch.randn(3, 4, 2)
    mu_p, logvar_p = latent.prior_params(h)
    mu_q, logvar_q = latent.posterior_params(h, future)
    z = latent.reparameterize(mu_q, logvar_q)
    assert z.shape == (3, 3)
    assert latent.embed(z).shape == (3, 5)
    assert gaussian_kl(mu_q, logvar_q, mu_p, logvar_p).shape == (3,)
