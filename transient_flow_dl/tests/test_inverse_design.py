from __future__ import annotations

import torch

from inverse_design.models import (
    ConditionalDiffusionUNet,
    ConditionalPatchDiscriminator,
    ConditionalPoreGenerator,
)
from inverse_design.physics import connectivity_loss, porosity_loss


def test_inverse_design_forward_backward() -> None:
    batch = 2
    condition = torch.randn(batch, 5)
    raw = torch.tensor([[0.45, 0.40, 0.1, 0.5, 1.2], [0.60, 0.55, 0.08, 0.8, 1.3]])
    generator = ConditionalPoreGenerator(latent_dim=16, base=8)
    discriminator = ConditionalPatchDiscriminator(base=8)
    fake = generator(torch.randn(batch, 16), condition)
    score = discriminator(fake, condition)
    loss = score.mean() + porosity_loss(fake, raw) + 0.1 * connectivity_loss(fake)
    loss.backward()
    assert fake.shape == (batch, 1, 100, 100)
    assert score.ndim == 4
    diffusion = ConditionalDiffusionUNet(base=8)
    prediction = diffusion(torch.randn_like(fake), torch.rand(batch), condition)
    prediction.square().mean().backward()
    assert prediction.shape == fake.shape


if __name__ == "__main__":
    test_inverse_design_forward_backward()
    print("inverse_design_forward_backward_ok")
