from __future__ import annotations

import torch
from torch.nn import functional as F


def spatial_gradients(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = field[..., 1:, :] - field[..., :-1, :]
    dy = field[..., :, 1:] - field[..., :, :-1]
    return dx, dy


def transient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pore: torch.Tensor,
    steady_fill: torch.Tensor,
    gradient_weight: float,
    steady_fill_weight: float,
    query_time: torch.Tensor | None = None,
    transport_weight: float = 0.0,
    transport_time_power: float = 4.0,
    transport_flux_floor: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = pore.expand_as(target)
    sample_weight = torch.where(
        steady_fill > 0.5,
        torch.full_like(steady_fill, steady_fill_weight),
        torch.ones_like(steady_fill),
    )[:, None, None, None]
    point_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    data_loss = (point_loss * mask * sample_weight).sum() / (mask * sample_weight).sum().clamp_min(1.0)

    pred_dx, pred_dy = spatial_gradients(prediction)
    target_dx, target_dy = spatial_gradients(target)
    mask_dx = mask[..., 1:, :] * mask[..., :-1, :]
    mask_dy = mask[..., :, 1:] * mask[..., :, :-1]
    gradient_loss = (
        ((pred_dx - target_dx).abs() * mask_dx).sum() / mask_dx.sum().clamp_min(1.0)
        + ((pred_dy - target_dy).abs() * mask_dy).sum() / mask_dy.sum().clamp_min(1.0)
    ) * 0.5
    # Match the physical observable used for permeability evaluation.  The
    # dataset flows along tensor axis 0, so the first and last rows are the
    # inlet and outlet.  Normalising by the target mean flux makes the term
    # comparable across geometries; the floor keeps initially quiescent frames
    # numerically well behaved.  A late-time weight focuses this auxiliary
    # constraint on developed flow without weakening the full-field loss.
    transport_loss = prediction.new_zeros(())
    if transport_weight > 0.0:
        pred_ux = prediction[:, 0]
        target_ux = target[:, 0]
        pore_2d = pore[:, 0]
        pred_boundary_flux = torch.stack(
            [
                (pred_ux[:, 0] * pore_2d[:, 0]).sum(dim=1),
                (pred_ux[:, -1] * pore_2d[:, -1]).sum(dim=1),
            ],
            dim=1,
        )
        target_boundary_flux = torch.stack(
            [
                (target_ux[:, 0] * pore_2d[:, 0]).sum(dim=1),
                (target_ux[:, -1] * pore_2d[:, -1]).sum(dim=1),
            ],
            dim=1,
        )
        target_mean_flux = target_boundary_flux.mean(dim=1).abs().detach()
        flux_scale = target_mean_flux.clamp_min(float(transport_flux_floor))[:, None]
        relative_boundary_error = (pred_boundary_flux - target_boundary_flux) / flux_scale
        per_sample = F.smooth_l1_loss(
            relative_boundary_error,
            torch.zeros_like(relative_boundary_error),
            beta=0.1,
            reduction="none",
        ).mean(dim=1)
        if query_time is None:
            time_weight = torch.ones_like(per_sample)
        else:
            time_weight = query_time.clamp(0.0, 1.0).pow(float(transport_time_power))
        transport_loss = (per_sample * time_weight).sum() / time_weight.sum().clamp_min(1e-6)

    total = data_loss + gradient_weight * gradient_loss + transport_weight * transport_loss
    return total, {
        "data_loss": data_loss.detach(),
        "gradient_loss": gradient_loss.detach(),
        "transport_loss": transport_loss.detach(),
    }
