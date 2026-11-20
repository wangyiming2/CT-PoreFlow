"""Real-data one-step verification for both inverse-design baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import PorousInverseDesignDataset
from .models import ConditionalDiffusionUNet, ConditionalPatchDiscriminator, ConditionalPoreGenerator
from .physics import connectivity_loss, porosity_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-root", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PorousInverseDesignDataset(args.steady_root, "train", max_samples=8)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    real = batch["pore"].to(device)
    condition = batch["condition"].to(device)
    raw = batch["condition_raw"].to(device)

    generator = ConditionalPoreGenerator(latent_dim=32, base=8).to(device)
    discriminator = ConditionalPatchDiscriminator(base=8).to(device)
    fake = generator(torch.randn(2, 32, device=device), condition)
    gan_loss = (
        F.binary_cross_entropy_with_logits(discriminator(fake, condition), torch.ones_like(discriminator(fake, condition)))
        + 10.0 * porosity_loss(fake, raw)
        + 0.1 * connectivity_loss(fake)
    )
    gan_loss.backward()

    diffusion = ConditionalDiffusionUNet(base=8).to(device)
    prediction = diffusion(torch.randn_like(real), torch.rand(2, device=device), condition)
    diffusion_loss = F.mse_loss(prediction, torch.randn_like(prediction))
    diffusion_loss.backward()
    print(
        f"cGAN real-data smoke: loss={float(gan_loss.detach()):.6g}, shape={tuple(fake.shape)}; "
        f"diffusion real-data smoke: loss={float(diffusion_loss.detach()):.6g}, "
        f"shape={tuple(prediction.shape)}, device={device}"
    )


if __name__ == "__main__":
    main()
