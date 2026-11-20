from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.data import TransientFlowDataset
from src.models import build_model
from src.utils import load_json
from figure_style import add_panel_label, apply_nature_style


def log_iterations(count: int, maximum: int) -> np.ndarray:
    values = np.expm1(np.linspace(0.0, math.log1p(maximum), count))
    values = np.rint(values).astype(np.int64)
    values[0] = 0
    values[-1] = maximum
    # Rounding can duplicate the first few logarithmic values. Preserve the
    # requested frame count with a monotone real-valued query grid in the NPZ,
    # while integer labels remain convenient for the animation title.
    return values


def render_prediction(
    pore: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    pressure: np.ndarray,
    iteration: int,
    speed_vmax: float,
) -> Image.Image:
    apply_nature_style()
    speed = np.sqrt(ux * ux + uy * uy) * pore
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    panels = [
        (speed, "Predicted speed (lattice)", "viridis", 0.0, speed_vmax),
        (pressure * pore, "Predicted normalized pressure", "coolwarm", 0.0, 1.0),
    ]
    solid = pore < 0.5
    for panel_index, (axis, (field, title, cmap, lower, upper)) in enumerate(zip(axes, panels)):
        image = axis.imshow(np.ma.masked_where(solid, field), cmap=cmap, vmin=lower, vmax=upper)
        axis.imshow(solid, cmap="gray_r", vmin=0, vmax=1, alpha=solid.astype(float))
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        add_panel_label(axis, chr(ord("a") + panel_index), x=-0.03, y=1.03)
    fig.suptitle(f"Continuous-time CT-PoreFlow query | LBM iteration {iteration:,}")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=135)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a dense continuous-time numerical trajectory and fixed-scale GIF."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--duration-ms", type=int, default=110)
    args = parser.parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")

    config = load_json(args.run_dir / "config.json")
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
        time_indices=[0],
        max_samples=0,
        seed=config["seed"],
        topology_root=config.get("topology_root"),
        time_transform=config.get("time_encoding", "log"),
    )
    item = dataset[args.sample_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    maximum = int(config["max_iteration"])
    iterations_float = np.expm1(np.linspace(0.0, math.log1p(maximum), args.frames))
    iteration_labels = np.rint(iterations_float).astype(np.int64)
    if config.get("time_encoding", "log") == "linear":
        normalized_time = iterations_float / float(maximum)
    else:
        normalized_time = np.log1p(iterations_float) / math.log1p(maximum)
    geometry = item["geometry"][None].to(device)
    pore = item["pore"].numpy()[0].astype(np.uint8)
    predictions: list[np.ndarray] = []
    frames: list[Image.Image] = []
    with torch.inference_mode():
        for iteration, time_value in zip(iteration_labels, normalized_time):
            query = torch.tensor([time_value], dtype=torch.float32, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model(geometry, query)[0].float().cpu().numpy()
            physical = prediction.copy()
            physical[0:2] *= velocity_scale
            predictions.append(physical.astype(np.float32))
            frames.append(
                render_prediction(
                    pore,
                    physical[0],
                    physical[1],
                    physical[2],
                    int(iteration),
                    velocity_scale,
                )
            )

    output = args.run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"sample_{args.sample_index:04d}_continuous_{args.frames:03d}"
    gif_path = output / f"{prefix}.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    array = np.stack(predictions)
    delta_rho = float(config["rho_in"]) - float(config["rho_out"])
    trajectory_path = output / f"{prefix}.npz"
    np.savez_compressed(
        trajectory_path,
        sample_id=np.asarray(str(item["sample_id"])),
        query_iterations_float=iterations_float.astype(np.float64),
        query_iteration_labels=iteration_labels,
        normalized_query_time=normalized_time.astype(np.float64),
        pore_mask=pore,
        predicted_velocity_x_raw_lattice=array[:, 0],
        predicted_velocity_y_raw_lattice=array[:, 1],
        predicted_speed_raw_lattice=np.sqrt(array[:, 0] ** 2 + array[:, 1] ** 2).astype(
            np.float32
        ),
        predicted_pressure_normalized=array[:, 2],
        predicted_density_raw_lattice=(
            float(config["rho_out"]) + array[:, 2] * delta_rho
        ).astype(np.float32),
        fixed_speed_color_scale_lattice=np.asarray(velocity_scale, dtype=np.float64),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "model_run": args.run_dir.name,
                    "continuous_query": True,
                    "stored_LBM_ground_truth_available_only_at_original_30_times": True,
                    "visualization_scale_source": "training speed q99.5",
                    "RGB_is_not_a_training_target": True,
                }
            )
        ),
    )
    print(gif_path)
    print(trajectory_path)


if __name__ == "__main__":
    main()
