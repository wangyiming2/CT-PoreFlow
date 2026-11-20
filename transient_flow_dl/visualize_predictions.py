from __future__ import annotations

import argparse
import io
import json
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
from figure_style import add_panel_label, apply_nature_style, save_publication_figure


def render_frame(
    pore: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    iteration: int,
    vmax: float,
) -> Image.Image:
    apply_nature_style()
    truth_speed = np.sqrt(truth[0] ** 2 + truth[1] ** 2) * pore
    pred_speed = np.sqrt(prediction[0] ** 2 + prediction[1] ** 2) * pore
    error = np.abs(pred_speed - truth_speed) * pore
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    fields = [truth_speed, pred_speed, error]
    titles = ["LBM ground truth", "Neural prediction", "Absolute error"]
    cmaps = ["viridis", "viridis", "magma"]
    maxima = [vmax, vmax, vmax * 0.25]
    for panel_index, (axis, field, title, cmap, upper) in enumerate(
        zip(axes, fields, titles, cmaps, maxima)
    ):
        image = axis.imshow(np.ma.masked_where(pore < 0.5, field), cmap=cmap, vmin=0.0, vmax=upper)
        axis.imshow(pore < 0.5, cmap="gray_r", vmin=0, vmax=1, alpha=(pore < 0.5).astype(float))
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        add_panel_label(axis, chr(ord("a") + panel_index), x=-0.04, y=1.03)
    fig.suptitle(f"LBM iteration {iteration} | fixed training-set speed scale")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_six_time_figure(
    pore: np.ndarray,
    truths: np.ndarray,
    predictions: np.ndarray,
    iterations: np.ndarray,
    selected_indices: list[int],
    vmax: float,
    output_path: Path,
) -> list[Path]:
    """Export a compact fixed-scale truth/prediction/error trajectory panel."""

    apply_nature_style()
    fig, axes = plt.subplots(3, len(selected_indices), figsize=(7.2, 4.55), constrained_layout=True)
    solid = pore < 0.5
    last_images = [None, None, None]
    for column, time_index in enumerate(selected_indices):
        truth_speed = np.hypot(truths[time_index, 0], truths[time_index, 1]) * pore
        predicted_speed = np.hypot(
            predictions[time_index, 0], predictions[time_index, 1]
        ) * pore
        fields = (truth_speed, predicted_speed, np.abs(predicted_speed - truth_speed) * pore)
        for row, field in enumerate(fields):
            cmap = "viridis" if row < 2 else "magma"
            upper = vmax if row < 2 else vmax * 0.25
            last_images[row] = axes[row, column].imshow(
                np.ma.masked_where(solid, field), cmap=cmap, vmin=0.0, vmax=upper
            )
            axes[row, column].imshow(
                solid, cmap="gray_r", vmin=0, vmax=1, alpha=solid.astype(float)
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_visible(False)
        axes[0, column].set_title(f"Iter. {int(iterations[time_index]):,}", pad=3)

    row_labels = ("LBM truth", "Neural prediction", "Absolute error")
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, labelpad=8, fontweight="bold")
        add_panel_label(axes[row, 0], chr(ord("a") + row), x=-0.34, y=1.03)

    for row, image in enumerate(last_images):
        colorbar = fig.colorbar(image, ax=axes[row, :], fraction=0.012, pad=0.012)
        colorbar.set_label("Speed (lattice units)" if row < 2 else "Absolute error")
    fig.suptitle("Transient pore-scale prediction with one fixed training-set colour scale", y=1.02)
    outputs = save_publication_figure(fig, output_path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    config = load_json(args.run_dir / "config.json")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    velocity_scale = float(checkpoint["velocity_scale"])
    dataset = TransientFlowDataset(
        config["steady_root"], config["transient_root"], args.split,
        velocity_scale, config["rho_in"], config["rho_out"], config["max_iteration"],
        mode="fixed_times", time_indices=list(range(30)), max_samples=0, seed=config["seed"],
        topology_root=config.get("topology_root"),
        time_transform=config.get("time_encoding", "log"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    frames: list[Image.Image] = []
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    iterations: list[int] = []
    source_iterations: list[int] = []
    sample_id = ""
    pore_for_export: np.ndarray | None = None
    base_index = args.sample_index * 30
    with torch.inference_mode():
        for time_index in range(30):
            item = dataset[base_index + time_index]
            geometry = item["geometry"][None].to(device)
            query_time = item["time"][None].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(geometry, query_time)[0].float().cpu().numpy()
            truth = item["target"].numpy()
            pore = item["pore"].numpy()[0]
            physical_scale = np.array([velocity_scale, velocity_scale, 1.0])[:, None, None]
            truth_physical = truth * physical_scale
            prediction_physical = pred * physical_scale
            frame = render_frame(
                pore,
                truth_physical,
                prediction_physical,
                int(item["iteration"]),
                velocity_scale,
            )
            frames.append(frame)
            predictions.append(prediction_physical.astype(np.float32))
            truths.append(truth_physical.astype(np.float32))
            iterations.append(int(item["iteration"]))
            source_iterations.append(int(item["source_iteration"]))
            sample_id = str(item["sample_id"])
            pore_for_export = pore.astype(np.uint8)
    output = args.run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    gif_path = output / f"sample_{args.sample_index:04d}_prediction.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=220, loop=0, optimize=True)
    contact_path = output / f"sample_{args.sample_index:04d}_six_times.png"
    prediction_array = np.stack(predictions)
    truth_array = np.stack(truths)
    save_six_time_figure(
        np.asarray(pore_for_export),
        truth_array,
        prediction_array,
        np.asarray(iterations),
        [0, 5, 10, 15, 20, 29],
        velocity_scale,
        contact_path,
    )
    delta_rho = float(config["rho_in"]) - float(config["rho_out"])
    trajectory_path = output / f"sample_{args.sample_index:04d}_trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        sample_id=np.asarray(sample_id),
        iterations=np.asarray(iterations, dtype=np.int64),
        source_iterations=np.asarray(source_iterations, dtype=np.int64),
        pore_mask=np.asarray(pore_for_export, dtype=np.uint8),
        predicted_velocity_x_raw_lattice=prediction_array[:, 0],
        predicted_velocity_y_raw_lattice=prediction_array[:, 1],
        predicted_speed_raw_lattice=np.sqrt(
            prediction_array[:, 0] ** 2 + prediction_array[:, 1] ** 2
        ).astype(np.float32),
        predicted_pressure_normalized=prediction_array[:, 2],
        predicted_density_raw_lattice=(
            float(config["rho_out"]) + prediction_array[:, 2] * delta_rho
        ).astype(np.float32),
        true_velocity_x_raw_lattice=truth_array[:, 0],
        true_velocity_y_raw_lattice=truth_array[:, 1],
        true_speed_raw_lattice=np.sqrt(truth_array[:, 0] ** 2 + truth_array[:, 1] ** 2).astype(
            np.float32
        ),
        true_pressure_normalized=truth_array[:, 2],
        fixed_speed_color_scale_lattice=np.asarray(velocity_scale, dtype=np.float64),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "model_run": args.run_dir.name,
                    "split": args.split,
                    "sample_index": args.sample_index,
                    "time_encoding": config.get("time_encoding", "log"),
                    "solid_values_are_hard_zero": True,
                    "visualization_scale_source": "training speed q99.5",
                }
            )
        ),
    )
    print(gif_path)
    print(contact_path)
    print(trajectory_path)


if __name__ == "__main__":
    main()
