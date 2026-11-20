from __future__ import annotations

import torch
from torch.nn import functional as F


def pore_probability(generated: torch.Tensor) -> torch.Tensor:
    return ((generated + 1.0) * 0.5).clamp(0.0, 1.0)


def porosity_loss(generated: torch.Tensor, raw_condition: torch.Tensor) -> torch.Tensor:
    predicted = pore_probability(generated).mean(dim=(-2, -1)).squeeze(1)
    return F.l1_loss(predicted, raw_condition[:, 0])


def soft_outlet_reachability(generated: torch.Tensor, propagation_steps: int = 100) -> torch.Tensor:
    """Differentiable approximation of inlet-to-outlet pore connectivity.

    Flow is along image axis 0, matching the LBM dataset convention.  Repeated
    max-pool propagation is gated by the pore probability.
    """

    pore = pore_probability(generated)
    reached = torch.zeros_like(pore)
    reached[:, :, 0, :] = pore[:, :, 0, :]
    for _ in range(propagation_steps):
        reached = torch.maximum(reached, F.max_pool2d(reached, 3, stride=1, padding=1) * pore)
    return reached[:, :, -1, :].amax(dim=-1).squeeze(1)


def connectivity_loss(generated: torch.Tensor) -> torch.Tensor:
    return (1.0 - soft_outlet_reachability(generated)).mean()


def cosine_beta_schedule(steps: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, device=device)
    cumulative = torch.cos(((x / steps) + 0.008) / 1.008 * torch.pi * 0.5).square()
    cumulative = cumulative / cumulative[0]
    return (1.0 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999)
