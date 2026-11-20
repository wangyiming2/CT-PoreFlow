"""One real-data optimization step for the ConvLSTM and Transformer additions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import TransientFlowDataset
from src.losses import transient_loss
from src.models import build_model
from src.utils import load_json


def run_one(config: dict, model_config: dict, velocity_scale: float, device: torch.device) -> None:
    dataset = TransientFlowDataset(
        config["steady_root"],
        config["transient_root"],
        "train",
        velocity_scale,
        config["rho_in"],
        config["rho_out"],
        config["max_iteration"],
        mode="random_time",
        time_indices=list(range(30)),
        max_samples=4,
        seed=2026,
        topology_root=model_config.get("topology_root"),
        time_transform="log",
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    geometry = batch["geometry"].to(device)
    time = batch["time"].to(device)
    target = batch["target"].to(device)
    pore = batch["pore"].to(device)
    steady_fill = batch["steady_fill"].to(device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(geometry, time)
    loss, _ = transient_loss(prediction, target, pore, steady_fill, 0.05, 0.5)
    loss.backward()
    optimizer.step()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert prediction.shape == target.shape
    assert torch.isfinite(loss)
    assert torch.count_nonzero(prediction * (1.0 - pore)) == 0
    print(
        f"{model_config['model']}: parameters={parameters:,} loss={float(loss.detach()):.6g} "
        f"shape={tuple(prediction.shape)} device={device}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.base_config)
    normalization = load_json(config["normalization_file"])
    velocity_scale = float(normalization["velocity_scale"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_one(
        config,
        {
            "model": "query_time_convlstm",
            "input_channels": 3,
            "convlstm_base_channels": 8,
            "convlstm_hidden_channels": 16,
            "convlstm_recurrent_steps": 3,
        },
        velocity_scale,
        device,
    )
    run_one(
        config,
        {
            "model": "geometry_time_transformer",
            "input_channels": 5,
            "topology_root": config.get("topology_root"),
            "transformer_d_model": 32,
            "transformer_heads": 4,
            "transformer_layers": 2,
            "transformer_patch_size": 10,
        },
        velocity_scale,
        device,
    )


if __name__ == "__main__":
    main()
