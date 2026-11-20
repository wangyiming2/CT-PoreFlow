from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ScalarEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            torch.linspace(0, -math.log(10000.0), half, device=value.device, dtype=value.dtype)
        )
        angles = value[:, None] * frequencies[None] * 2.0 * math.pi
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class ConditionalPoreGenerator(nn.Module):
    def __init__(self, condition_dim: int = 5, latent_dim: int = 128, base: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition = nn.Sequential(nn.Linear(condition_dim, 64), nn.SiLU(), nn.Linear(64, 64))
        self.project = nn.Linear(latent_dim + 64, base * 8 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.ReLU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.ConvTranspose2d(base, base // 2, 4, 2, 1), nn.BatchNorm2d(base // 2), nn.ReLU(),
            nn.Conv2d(base // 2, 1, 3, padding=1),
        )

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        embedded = self.condition(condition)
        hidden = self.project(torch.cat([noise, embedded], dim=1)).view(noise.shape[0], -1, 7, 7)
        logits = self.decoder(hidden)
        logits = F.interpolate(logits, size=(100, 100), mode="bilinear", align_corners=False)
        return logits.tanh()


class ConditionalPatchDiscriminator(nn.Module):
    def __init__(self, condition_dim: int = 5, base: int = 32) -> None:
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(condition_dim, 32), nn.LeakyReLU(0.2), nn.Linear(32, 16))
        self.network = nn.Sequential(
            nn.Conv2d(17, base, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.InstanceNorm2d(base * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.InstanceNorm2d(base * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(base * 4, 1, 3, padding=1),
        )

    def forward(self, pore: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        embedded = self.condition(condition)[:, :, None, None].expand(-1, -1, *pore.shape[-2:])
        return self.network(torch.cat([pore, embedded], dim=1))


class ConditionedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.affine = nn.Linear(embedding_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(inputs)))
        scale, shift = self.affine(embedding).chunk(2, dim=1)
        hidden = self.norm2(self.conv2(hidden))
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return F.silu(hidden + self.skip(inputs))


class ConditionalDiffusionUNet(nn.Module):
    def __init__(self, condition_dim: int = 5, base: int = 32) -> None:
        super().__init__()
        embedding_dim = base * 4
        self.time = nn.Sequential(
            ScalarEmbedding(embedding_dim), nn.Linear(embedding_dim, embedding_dim), nn.SiLU()
        )
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.enc1 = ConditionedBlock(1, base, embedding_dim)
        self.enc2 = ConditionedBlock(base, base * 2, embedding_dim)
        self.enc3 = ConditionedBlock(base * 2, base * 4, embedding_dim)
        self.mid = ConditionedBlock(base * 4, base * 4, embedding_dim)
        self.dec2 = ConditionedBlock(base * 4 + base * 2, base * 2, embedding_dim)
        self.dec1 = ConditionedBlock(base * 2 + base, base, embedding_dim)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, noisy_pore: torch.Tensor, diffusion_time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        embedding = self.time(diffusion_time) + self.condition(condition)
        e1 = self.enc1(noisy_pore, embedding)
        e2 = self.enc2(F.avg_pool2d(e1, 2), embedding)
        e3 = self.enc3(F.avg_pool2d(e2, 2), embedding)
        hidden = self.mid(e3, embedding)
        hidden = F.interpolate(hidden, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec2(torch.cat([hidden, e2], dim=1), embedding)
        hidden = F.interpolate(hidden, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec1(torch.cat([hidden, e1], dim=1), embedding)
        return self.output(hidden)
