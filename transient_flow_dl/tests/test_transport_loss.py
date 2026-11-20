from __future__ import annotations

import torch

from src.losses import transient_loss


def _loss(
    prediction: torch.Tensor, target: torch.Tensor, time: float | torch.Tensor
) -> tuple[torch.Tensor, dict]:
    pore = torch.ones(prediction.shape[0], 1, 6, 5)
    query_time = torch.as_tensor(time, dtype=torch.float32).reshape(-1)
    return transient_loss(
        prediction,
        target,
        pore,
        torch.zeros(prediction.shape[0]),
        gradient_weight=0.0,
        steady_fill_weight=1.0,
        query_time=query_time,
        transport_weight=0.01,
        transport_time_power=4.0,
    )


def test_transport_loss_is_zero_for_matching_boundary_flux() -> None:
    target = torch.zeros(1, 3, 6, 5)
    target[:, 0] = 0.2
    total, components = _loss(target.clone(), target, 1.0)
    assert torch.isclose(total, torch.zeros_like(total))
    assert torch.isclose(components["transport_loss"], torch.zeros_like(total))


def test_transport_loss_penalizes_late_boundary_flux_more_than_early_flux() -> None:
    target = torch.zeros(2, 3, 6, 5)
    target[:, 0] = 0.2
    prediction = target.clone()
    prediction[0, 0, 0] = 0.1
    prediction[0, 0, -1] = 0.1
    _, early = _loss(prediction, target, torch.tensor([0.25, 1.0]))
    _, late = _loss(prediction, target, torch.tensor([1.0, 0.25]))
    assert late["transport_loss"] > early["transport_loss"]
