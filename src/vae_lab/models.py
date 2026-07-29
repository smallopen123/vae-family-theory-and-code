"""VAE-family models with formulas mirrored in the accompanying documentation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Sample z = mu + exp(0.5 log(sigma^2)) * epsilon."""
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL[N(mu, diag(exp(logvar))) || N(0, I)], one value per batch item."""
    return -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=-1)


def bernoulli_nll(logits: Tensor, target: Tensor) -> Tensor:
    """Negative Bernoulli log-likelihood, summed except for the first dimension."""
    elementwise = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return elementwise.flatten(start_dim=1).sum(dim=1)


@dataclass
class VAELoss:
    loss: Tensor
    reconstruction: Tensor
    kl: Tensor


def vae_loss(
    logits: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
) -> VAELoss:
    """Negative beta-ELBO, averaged over the batch."""
    reconstruction = bernoulli_nll(logits, target)
    kl = kl_standard_normal(mu, logvar)
    return VAELoss(
        loss=(reconstruction + beta * kl).mean(),
        reconstruction=reconstruction.mean(),
        kl=kl.mean(),
    )


class GaussianVAE(nn.Module):
    """Small MLP VAE for 28x28 single-channel images."""

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(x)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden).clamp(-20.0, 10.0)
        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z).reshape(*z.shape[:-1], 1, 28, 28)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    @torch.no_grad()
    def sample(self, count: int, device: torch.device | str = "cpu") -> Tensor:
        z = torch.randn(count, self.latent_dim, device=device)
        return torch.sigmoid(self.decode(z))


class ConditionalVAE(nn.Module):
    """Class-conditional VAE using one-hot labels in both encoder and decoder."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 + num_classes, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
        )

    def _condition(self, labels: Tensor) -> Tensor:
        return F.one_hot(labels, self.num_classes).to(dtype=torch.float32)

    def encode(self, x: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        inputs = torch.cat([x.flatten(start_dim=1), self._condition(labels)], dim=1)
        hidden = self.encoder(inputs)
        return self.mu_head(hidden), self.logvar_head(hidden).clamp(-20.0, 10.0)

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        inputs = torch.cat([z, self._condition(labels)], dim=-1)
        return self.decoder(inputs).reshape(-1, 1, 28, 28)

    def forward(self, x: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x, labels)
        z = reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar, z

    @torch.no_grad()
    def sample(self, labels: Tensor) -> Tensor:
        z = torch.randn(labels.shape[0], self.latent_dim, device=labels.device)
        return torch.sigmoid(self.decode(z, labels))


def _log_standard_normal(z: Tensor) -> Tensor:
    return (-0.5 * (math.log(2.0 * math.pi) + z.square())).sum(dim=-1)


def _log_diag_normal(z: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    return (
        -0.5
        * (math.log(2.0 * math.pi) + logvar + (z - mu).square() / logvar.exp())
    ).sum(dim=-1)


def iwae_loss(model: GaussianVAE, x: Tensor, importance_samples: int = 5) -> Tensor:
    """Negative Monte Carlo IWAE bound with K importance samples."""
    if importance_samples < 1:
        raise ValueError("importance_samples must be at least 1")

    mu, logvar = model.encode(x)
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn(
        importance_samples, *std.shape, device=std.device, dtype=std.dtype
    )
    z = mu.unsqueeze(0) + std.unsqueeze(0) * epsilon  # [K, B, D]
    logits = model.decode(z)  # [K, B, 1, 28, 28]
    targets = x.unsqueeze(0).expand_as(logits)

    # First two dimensions are [importance sample, batch], so flatten only pixels.
    log_px_z = -F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).flatten(start_dim=2).sum(dim=2)
    log_pz = _log_standard_normal(z)
    log_qz_x = _log_diag_normal(z, mu.unsqueeze(0), logvar.unsqueeze(0))
    log_weights = log_px_z + log_pz - log_qz_x

    log_mean_weight = torch.logsumexp(log_weights, dim=0) - math.log(
        importance_samples
    )
    return -log_mean_weight.mean()


class VectorQuantizer(nn.Module):
    """Nearest-neighbour codebook with straight-through gradients."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / num_embeddings,
            1.0 / num_embeddings,
        )

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # BCHW -> BHWC -> (BHW)C so each spatial vector is quantized.
        z_bhwc = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z = z_bhwc.reshape(-1, self.embedding_dim)
        codebook = self.embedding.weight

        distances = (
            flat_z.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1)
            - 2.0 * flat_z @ codebook.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(z_bhwc)

        codebook_loss = F.mse_loss(quantized, z_bhwc.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z_bhwc)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized_st = z_bhwc + (quantized - z_bhwc).detach()
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()
        index_map = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        return quantized_st, vq_loss, index_map


class VQVAE(nn.Module):
    """Compact convolutional VQ-VAE for MNIST-sized images."""

    def __init__(
        self,
        embedding_dim: int = 32,
        num_embeddings: int = 128,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embedding_dim, 4, stride=2, padding=1),
        )
        self.quantizer = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z_e)
        return self.decoder(z_q), vq_loss, indices

