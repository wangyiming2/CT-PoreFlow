"""Run several trained surrogates on the same representative trajectories.

This script is designed to run on the GPU workstation where both the transient
dataset and checkpoints live.  It exports one compressed bundle per sample with
all 30 LBM frames and predictions from every requested model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import TransientFlowDataset
from src.models import build_model
from src.utils import load_json


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--sample-indices", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sample_indices = [int(value) for value in args.sample_indices.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions: dict[int, dict[str, np.ndarray]] = {index: {} for index in sample_indices}
    common: dict[int, dict[str, np.ndarray | str | float]] = {}

    for label, run_dir in args.run:
        config = load_json(run_dir / "config.json")
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
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
            time_indices=list(range(30)),
            max_samples=0,
            seed=config["seed"],
            topology_root=config.get("topology_root"),
            time_transform=config.get("time_encoding", "log"),
        )
        model = build_model(config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        print(f"{label}: device={device} samples={sample_indices}", flush=True)
        with torch.inference_mode():
            for sample_index in sample_indices:
                predictions: list[np.ndarray] = []
                truths: list[np.ndarray] = []
                iterations: list[int] = []
                sample_id = ""
                pore = connected = edt = None
                for time_index in range(30):
                    item = dataset[sample_index * 30 + time_index]
                    geometry = item["geometry"][None].to(device)
                    query_time = item["time"][None].to(device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        prediction = model(geometry, query_time)[0].float().cpu().numpy()
                    scale = np.asarray([velocity_scale, velocity_scale, 1.0])[:, None, None]
                    predictions.append((prediction * scale).astype(np.float32))
                    truths.append((item["target"].numpy() * scale).astype(np.float32))
                    iterations.append(int(item["iteration"]))
                    sample_id = str(item["sample_id"])
                    pore = item["pore"].numpy()[0].astype(np.uint8)
                    connected = item["geometry"].numpy()[1].astype(np.uint8)
                    edt = item["geometry"].numpy()[2].astype(np.float32)
                all_predictions[sample_index][label] = np.stack(predictions)
                if sample_index not in common:
                    common[sample_index] = {
                        "sample_id": sample_id,
                        "iterations": np.asarray(iterations, dtype=np.int64),
                        "truth": np.stack(truths),
                        "pore": np.asarray(pore),
                        "connected": np.asarray(connected),
                        "edt": np.asarray(edt),
                        "velocity_scale": velocity_scale,
                    }
                print(f"  exported sample {sample_index}: {sample_id}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    labels = [label for label, _ in args.run]
    for sample_index in sample_indices:
        item = common[sample_index]
        payload: dict[str, np.ndarray] = {
            "sample_id": np.asarray(item["sample_id"]),
            "sample_index": np.asarray(sample_index, dtype=np.int64),
            "iterations": np.asarray(item["iterations"]),
            "pore_mask": np.asarray(item["pore"]),
            "connected_pore_mask": np.asarray(item["connected"]),
            "edt_normalized": np.asarray(item["edt"]),
            "true_fields": np.asarray(item["truth"]),
            "fixed_speed_color_scale_lattice": np.asarray(item["velocity_scale"]),
            "model_labels": np.asarray(labels),
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "split": args.split,
                        "sample_index": sample_index,
                        "field_channels": ["ux_lattice", "uy_lattice", "pressure_normalized"],
                        "all_models_share_training_q995_speed_scale": True,
                    }
                )
            ),
        }
        for label in labels:
            payload[f"predicted_fields__{label}"] = all_predictions[sample_index][label]
        output = args.output / f"sample_{sample_index:04d}_multimodel_trajectory.npz"
        np.savez_compressed(output, **payload)
        print(output, flush=True)


if __name__ == "__main__":
    main()
