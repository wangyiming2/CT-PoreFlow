from __future__ import annotations

import torch


@torch.no_grad()
def batch_metrics(prediction: torch.Tensor, target: torch.Tensor, pore: torch.Tensor) -> dict[str, float]:
    mask = pore.expand_as(target)
    error = (prediction - target) * mask
    denominator = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
    mae = error.abs().sum(dim=(0, 2, 3)) / denominator
    rmse = ((error.square().sum(dim=(0, 2, 3)) / denominator).sqrt())
    target_norm = torch.linalg.vector_norm((target * mask).flatten(1), dim=1)
    nonzero = target_norm > 1e-6
    if nonzero.any():
        rel = (
            torch.linalg.vector_norm(error.flatten(1), dim=1)[nonzero]
            / target_norm[nonzero]
        ).mean()
    else:
        rel = torch.zeros((), device=target.device)
    return {
        "mae_ux": float(mae[0]),
        "mae_uy": float(mae[1]),
        "mae_p": float(mae[2]),
        "rmse_ux": float(rmse[0]),
        "rmse_uy": float(rmse[1]),
        "rmse_p": float(rmse[2]),
        "relative_l2_nonzero_targets": float(rel),
    }
