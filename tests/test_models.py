import torch

from vae_lab.models import (
    ConditionalVAE,
    GaussianVAE,
    VQVAE,
    iwae_loss,
    kl_standard_normal,
    vae_loss,
)


def test_standard_normal_kl_is_zero() -> None:
    mu = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)
    assert torch.allclose(kl_standard_normal(mu, logvar), torch.zeros(4))


def test_gaussian_vae_shapes_and_gradients() -> None:
    model = GaussianVAE(latent_dim=5, hidden_dim=32)
    x = torch.rand(4, 1, 28, 28)
    logits, mu, logvar, z = model(x)
    assert logits.shape == x.shape
    assert mu.shape == logvar.shape == z.shape == (4, 5)
    objective = vae_loss(logits, x, mu, logvar)
    objective.loss.backward()
    assert model.mu_head.weight.grad is not None
    assert torch.isfinite(objective.loss)


def test_conditional_vae_shapes() -> None:
    model = ConditionalVAE(latent_dim=4, hidden_dim=32)
    x = torch.rand(3, 1, 28, 28)
    labels = torch.tensor([0, 4, 9])
    logits, mu, logvar, z = model(x, labels)
    assert logits.shape == x.shape
    assert mu.shape == logvar.shape == z.shape == (3, 4)
    assert model.sample(labels).shape == x.shape


def test_iwae_bound_is_finite_and_differentiable() -> None:
    model = GaussianVAE(latent_dim=3, hidden_dim=32)
    x = torch.rand(2, 1, 28, 28)
    loss = iwae_loss(model, x, importance_samples=3)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.logvar_head.weight.grad is not None


def test_vqvae_shapes_and_gradients() -> None:
    model = VQVAE(embedding_dim=8, num_embeddings=16)
    x = torch.rand(2, 1, 28, 28)
    logits, vq_loss, indices = model(x)
    assert logits.shape == x.shape
    assert indices.shape == (2, 7, 7)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, x) + vq_loss
    loss.backward()
    assert model.encoder[0].weight.grad is not None
    assert model.quantizer.embedding.weight.grad is not None

