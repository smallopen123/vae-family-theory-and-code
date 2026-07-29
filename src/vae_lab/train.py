"""Minimal MNIST trainer for all implemented VAE variants."""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import ConditionalVAE, GaussianVAE, VQVAE, iwae_loss, vae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["vae", "beta-vae", "cvae", "iwae", "vq-vae"],
        default="vae",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--importance-samples", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        datasets.MNIST(
            args.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    if args.model == "cvae":
        model = ConditionalVAE(latent_dim=args.latent_dim).to(device)
    elif args.model == "vq-vae":
        model = VQVAE(embedding_dim=args.latent_dim).to(device)
    else:
        model = GaussianVAE(latent_dim=args.latent_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()

            if args.model == "cvae":
                logits, mu, logvar, _ = model(x, labels)
                loss = vae_loss(logits, x, mu, logvar).loss
            elif args.model == "iwae":
                loss = iwae_loss(model, x, args.importance_samples)
            elif args.model == "vq-vae":
                logits, vq_loss, _ = model(x)
                reconstruction = F.binary_cross_entropy_with_logits(logits, x)
                loss = reconstruction + vq_loss
            else:
                logits, mu, logvar, _ = model(x)
                beta = args.beta if args.model == "beta-vae" else 1.0
                loss = vae_loss(logits, x, mu, logvar, beta=beta).loss

            loss.backward()
            optimizer.step()
            total += loss.item() * x.shape[0]

        print(
            f"epoch={epoch:03d} model={args.model} "
            f"loss={total / len(loader.dataset):.4f} device={device}"
        )


if __name__ == "__main__":
    main()

