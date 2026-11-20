from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import TransientFlowDataset
from src.models import build_model
from src.utils import load_json, save_json, set_seed


def _finite_ratio(numerator: float, denominator: float, threshold: float = 1e-12) -> float:
    return numerator / denominator if abs(denominator) > threshold else float("nan")


def _flux_and_transport_metrics(
    velocity: torch.Tensor,
    pressure: torch.Tensor,
    pore: torch.Tensor,
    viscosity: float,
    imposed_pressure_drop: float,
) -> dict[str, float]:
    """Physical metrics for one [2,H,W] lattice-velocity field.

    The dataset's x-flow direction is axis 0, so inlet/outlet are rows 0/-1.
    """
    ux = velocity[0]
    uy = velocity[1]
    height, width = ux.shape
    inlet_flux = float(ux[0].sum())
    outlet_flux = float(ux[-1].sum())
    mean_flux = 0.5 * (inlet_flux + outlet_flux)
    flux_scale = 0.5 * (abs(inlet_flux) + abs(outlet_flux))
    mass_imbalance = _finite_ratio(abs(inlet_flux - outlet_flux), flux_scale)
    darcy_velocity = mean_flux / float(width)
    permeability = darcy_velocity * viscosity * float(height) / imposed_pressure_drop

    speed = torch.sqrt(ux.square() + uy.square())
    pore_speed = speed[pore]
    pore_ux = ux[pore]
    tortuosity = _finite_ratio(float(pore_speed.sum()), abs(float(pore_ux.sum())))

    inlet_pressure = float(pressure[0][pore[0]].mean()) if pore[0].any() else float("nan")
    outlet_pressure = float(pressure[-1][pore[-1]].mean()) if pore[-1].any() else float("nan")
    pressure_drop = inlet_pressure - outlet_pressure

    interior = pore[1:-1, 1:-1].clone()
    interior &= pore[2:, 1:-1] & pore[:-2, 1:-1] & pore[1:-1, 2:] & pore[1:-1, :-2]
    divergence = 0.5 * (
        ux[2:, 1:-1] - ux[:-2, 1:-1] + uy[1:-1, 2:] - uy[1:-1, :-2]
    )
    divergence_l1 = float(divergence[interior].abs().mean()) if interior.any() else float("nan")
    return {
        "inlet_flux_lattice": inlet_flux,
        "outlet_flux_lattice": outlet_flux,
        "mean_flux_lattice": mean_flux,
        "mass_balance_error": mass_imbalance,
        "darcy_velocity_lattice": darcy_velocity,
        "permeability_lattice": permeability,
        "hydraulic_tortuosity": tortuosity,
        "pressure_drop_normalized": pressure_drop,
        "divergence_l1_lattice": divergence_l1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--time-indices",
        type=int,
        nargs="+",
        help="Stored time indices to evaluate; defaults to all 30.",
    )
    parser.add_argument(
        "--evaluation-name",
        help="Output subdirectory under evaluation/; defaults to the split name.",
    )
    args = parser.parse_args()
    config = load_json(args.run_dir / "config.json")
    set_seed(int(config["seed"]))
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    velocity_scale = float(checkpoint["velocity_scale"])
    dataset = TransientFlowDataset(
        config["steady_root"],
        config["transient_root"],
        args.split,
        velocity_scale,
        config["rho_in"],
        config["rho_out"],
        config["max_iteration"],
        mode="fixed_times",
        time_indices=args.time_indices or list(range(30)),
        max_samples=args.max_samples,
        seed=config["seed"],
        topology_root=config.get("topology_root"),
        time_transform=config.get("time_encoding", "log"),
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows: list[dict[str, object]] = []
    totals: defaultdict[str, float] = defaultdict(float)
    metric_counts: defaultdict[str, int] = defaultdict(int)
    start = time.perf_counter()
    model_seconds = 0.0
    viscosity = (float(config.get("tau", 1.0)) - 0.5) / 3.0
    imposed_pressure_drop = (float(config["rho_in"]) - float(config["rho_out"])) / 3.0
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            geometry = batch["geometry"].to(device)
            query_time = batch["time"].to(device)
            target = batch["target"].to(device)
            pore = batch["pore"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(geometry, query_time)
            if device.type == "cuda":
                torch.cuda.synchronize()
            model_seconds += time.perf_counter() - model_start
            mask = pore.expand_as(target)
            error = (prediction - target) * mask
            for index in range(geometry.shape[0]):
                valid = mask[index] > 0.5
                channel_mae = []
                channel_rmse = []
                channel_rel = []
                for channel in range(3):
                    channel_valid = valid[channel]
                    e = error[index, channel][channel_valid]
                    y = target[index, channel][channel_valid]
                    channel_mae.append(float(e.abs().mean()))
                    channel_rmse.append(float(e.square().mean().sqrt()))
                    target_norm = torch.linalg.vector_norm(y)
                    channel_rel.append(
                        float(torch.linalg.vector_norm(e) / target_norm)
                        if float(target_norm) > 1e-6
                        else float("nan")
                    )
                velocity_error = error[index, :2][valid[:2]]
                velocity_target = target[index, :2][valid[:2]]
                velocity_target_norm = torch.linalg.vector_norm(velocity_target)
                relative_velocity = (
                    float(torch.linalg.vector_norm(velocity_error) / velocity_target_norm)
                    if float(velocity_target_norm) > 1e-6
                    else float("nan")
                )
                pore_2d = pore[index, 0] > 0.5
                pred_speed = torch.sqrt(prediction[index, 0].square() + prediction[index, 1].square())
                true_speed = torch.sqrt(target[index, 0].square() + target[index, 1].square())
                speed_mae_lattice = float((pred_speed[pore_2d] - true_speed[pore_2d]).abs().mean()) * velocity_scale
                pred_velocity_lattice = prediction[index, :2].float() * velocity_scale
                true_velocity_lattice = target[index, :2].float() * velocity_scale
                pred_transport = _flux_and_transport_metrics(
                    pred_velocity_lattice,
                    prediction[index, 2].float(),
                    pore_2d,
                    viscosity,
                    imposed_pressure_drop,
                )
                true_transport = _flux_and_transport_metrics(
                    true_velocity_lattice,
                    target[index, 2].float(),
                    pore_2d,
                    viscosity,
                    imposed_pressure_drop,
                )
                relative_flux = _finite_ratio(
                    abs(pred_transport["mean_flux_lattice"] - true_transport["mean_flux_lattice"]),
                    abs(true_transport["mean_flux_lattice"]),
                )
                relative_tortuosity = _finite_ratio(
                    abs(pred_transport["hydraulic_tortuosity"] - true_transport["hydraulic_tortuosity"]),
                    abs(true_transport["hydraulic_tortuosity"]),
                )
                pressure_drop_mae = abs(
                    pred_transport["pressure_drop_normalized"] - true_transport["pressure_drop_normalized"]
                )
                is_final_time = int(batch["iteration"][index]) == int(config["max_iteration"])
                relative_permeability = (
                    _finite_ratio(
                        abs(pred_transport["permeability_lattice"] - true_transport["permeability_lattice"]),
                        abs(true_transport["permeability_lattice"]),
                    )
                    if is_final_time
                    else float("nan")
                )
                relative_flux_final = relative_flux if is_final_time else float("nan")
                relative_tortuosity_final = relative_tortuosity if is_final_time else float("nan")
                pred_mass_balance_final = (
                    pred_transport["mass_balance_error"] if is_final_time else float("nan")
                )
                true_mass_balance_final = (
                    true_transport["mass_balance_error"] if is_final_time else float("nan")
                )
                solid_2d = ~pore_2d
                solid_speed = torch.sqrt(
                    pred_velocity_lattice[0].square() + pred_velocity_lattice[1].square()
                )[solid_2d]
                row = {
                    "sample_id": batch["sample_id"][index],
                    "iteration": int(batch["iteration"][index]),
                    "source_iteration": int(batch["source_iteration"][index]),
                    "steady_fill": int(batch["steady_fill"][index]),
                    "mae_ux_normalized": channel_mae[0],
                    "mae_uy_normalized": channel_mae[1],
                    "mae_p_normalized": channel_mae[2],
                    "rmse_ux_normalized": channel_rmse[0],
                    "rmse_uy_normalized": channel_rmse[1],
                    "rmse_p_normalized": channel_rmse[2],
                    "relative_l2_ux": channel_rel[0],
                    "relative_l2_uy": channel_rel[1],
                    "relative_l2_p": channel_rel[2],
                    "relative_l2_velocity": relative_velocity,
                    "mae_speed_lattice": speed_mae_lattice,
                    "relative_error_mean_flux": relative_flux,
                    "relative_error_mean_flux_final": relative_flux_final,
                    "relative_error_permeability_final": relative_permeability,
                    "relative_error_hydraulic_tortuosity": relative_tortuosity,
                    "relative_error_hydraulic_tortuosity_final": relative_tortuosity_final,
                    "pressure_drop_mae_normalized": pressure_drop_mae,
                    "pred_mass_balance_error": pred_transport["mass_balance_error"],
                    "true_mass_balance_error": true_transport["mass_balance_error"],
                    "pred_mass_balance_error_final": pred_mass_balance_final,
                    "true_mass_balance_error_final": true_mass_balance_final,
                    "pred_divergence_l1_lattice": pred_transport["divergence_l1_lattice"],
                    "true_divergence_l1_lattice": true_transport["divergence_l1_lattice"],
                    "pred_mean_flux_lattice": pred_transport["mean_flux_lattice"],
                    "true_mean_flux_lattice": true_transport["mean_flux_lattice"],
                    "pred_permeability_lattice": pred_transport["permeability_lattice"] if is_final_time else float("nan"),
                    "true_permeability_lattice": true_transport["permeability_lattice"] if is_final_time else float("nan"),
                    "pred_hydraulic_tortuosity": pred_transport["hydraulic_tortuosity"],
                    "true_hydraulic_tortuosity": true_transport["hydraulic_tortuosity"],
                    "max_solid_speed_lattice": float(solid_speed.max()) if solid_speed.numel() else 0.0,
                }
                rows.append(row)
                for key, value in row.items():
                    if isinstance(value, float) and np.isfinite(value):
                        totals[key] += value
                        metric_counts[key] += 1
            if batch_number % 100 == 0 or batch_number == len(loader):
                processed = min(batch_number * int(config["batch_size"]), len(dataset))
                print(
                    f"evaluation {processed}/{len(dataset)} frames "
                    f"elapsed={time.perf_counter() - start:.1f}s",
                    flush=True,
                )
    elapsed = time.perf_counter() - start
    output_dir = args.run_dir / "evaluation" / (args.evaluation_name or args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_sample_time_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: value / metric_counts[key] for key, value in totals.items()}
    summary.update(
        {
            "split": args.split,
            "evaluation_name": args.evaluation_name or args.split,
            "time_indices": args.time_indices or list(range(30)),
            "training_time_indices": config.get("training_time_indices", list(range(30))),
            "records": len(rows),
            "geometries": len(dataset.pairs),
            "seconds": elapsed,
            "milliseconds_per_frame": 1000.0 * elapsed / len(rows),
            "end_to_end_milliseconds_per_frame": 1000.0 * elapsed / len(rows),
            "model_forward_seconds": model_seconds,
            "model_milliseconds_per_frame": 1000.0 * model_seconds / len(rows),
            "model_frames_per_second": len(rows) / model_seconds,
            "velocity_scale": velocity_scale,
            "viscosity_lattice": viscosity,
            "imposed_pressure_drop_lattice": imposed_pressure_drop,
            "checkpoint": str(args.run_dir / "best.pt"),
        }
    )
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
