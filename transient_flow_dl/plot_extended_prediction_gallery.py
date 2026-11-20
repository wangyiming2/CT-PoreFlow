"""Create fixed-scale terminal and transient galleries for 16 objective test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from figure_style import apply_nature_style, save_publication_figure


DISPLAY = {
    **{f"porosity_bin_{i:02d}": f"Porosity bin {i + 1}" for i in range(8)},
    "high_tortuosity": "High tortuosity",
    "high_dead_pore": "High dead-pore ratio",
    "low_permeability": "Low permeability",
    "high_permeability": "High permeability",
    "easy_case": "Low-error case",
    "hard_case": "Hard case",
    "ct_advantage": "CT advantage",
    "unet_advantage": "U-Net advantage",
}


def load(selection_path: Path, bundle_dir: Path) -> list[dict]:
    selections = json.loads(selection_path.read_text(encoding="utf-8"))
    bundles: list[dict] = []
    for record in selections:
        path = bundle_dir / f"sample_{int(record['sample_index']):04d}_multimodel_trajectory.npz"
        with np.load(path) as source:
            bundle = {key: np.asarray(source[key]) for key in source.files}
        bundle["selection"] = record
        bundles.append(bundle)
    if len(bundles) != 16:
        raise RuntimeError(f"Expected 16 bundles, found {len(bundles)}")
    return bundles


def velocity_magnitude(fields: np.ndarray) -> np.ndarray:
    return np.hypot(fields[..., 0, :, :], fields[..., 1, :, :])


def vector_error(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.hypot(predicted[..., 0, :, :] - truth[..., 0, :, :], predicted[..., 1, :, :] - truth[..., 1, :, :])


def geometry_rgb(pore: np.ndarray, connected: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*pore.shape, 3), dtype=float)
    rgb[pore > 0.5] = (0.78, 0.79, 0.82)
    rgb[connected > 0.5] = (0.12, 0.48, 0.72)
    return rgb


def clean(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.35)
        spine.set_color("#555555")


def label(record: dict) -> str:
    return (
        f"{DISPLAY.get(record['selection_label'], record['selection_label'])}\n"
        f"$\\phi$={record['porosity']:.3f}, $k$={record['permeability']:.2f}, "
        f"$\\tau_h$={record['tortuosity']:.2f}\n"
        f"$L^2_r$: U={record['unet_mean_velocity_l2']:.2f}, "
        f"CT={record['ct_mean_velocity_l2']:.2f}"
    )


def terminal_error_limit(bundles: list[dict]) -> float:
    values = []
    for bundle in bundles:
        truth = bundle["true_fields"][-1]
        pore = bundle["pore_mask"].astype(bool)
        for model in ("Time-U-Net", "CT-PoreFlow"):
            values.append(vector_error(bundle[f"predicted_fields__{model}"][-1], truth)[pore])
    return float(np.quantile(np.concatenate(values), 0.99))


def terminal_gallery(
    bundles: list[dict], output: Path, speed_map, error_map, speed_limit: float, error_limit: float, group: str
) -> None:
    fig, axes = plt.subplots(len(bundles), 6, figsize=(8.65, 10.25), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.018, hspace=0.018)
    speed_image = error_image = None
    for row, bundle in enumerate(bundles):
        pore = bundle["pore_mask"].astype(bool)
        connected = bundle["connected_pore_mask"].astype(bool)
        truth = bundle["true_fields"][-1]
        unet = bundle["predicted_fields__Time-U-Net"][-1]
        ct = bundle["predicted_fields__CT-PoreFlow"][-1]
        axes[row, 0].imshow(geometry_rgb(pore, connected), interpolation="nearest")
        speed_image = axes[row, 1].imshow(
            np.ma.masked_where(~pore, velocity_magnitude(truth)), cmap=speed_map, vmin=0, vmax=speed_limit
        )
        axes[row, 2].imshow(
            np.ma.masked_where(~pore, velocity_magnitude(unet)), cmap=speed_map, vmin=0, vmax=speed_limit
        )
        error_image = axes[row, 3].imshow(
            np.ma.masked_where(~pore, vector_error(unet, truth)), cmap=error_map, vmin=0, vmax=error_limit
        )
        axes[row, 4].imshow(
            np.ma.masked_where(~pore, velocity_magnitude(ct)), cmap=speed_map, vmin=0, vmax=speed_limit
        )
        axes[row, 5].imshow(
            np.ma.masked_where(~pore, vector_error(ct, truth)), cmap=error_map, vmin=0, vmax=error_limit
        )
        for axis in axes[row]:
            clean(axis)
        axes[row, 0].set_ylabel(
            label(bundle["selection"]), rotation=0, ha="right", va="center", fontweight="bold", labelpad=11
        )
        axes[row, 0].text(-0.10, 1.01, chr(ord("a") + row), transform=axes[row, 0].transAxes, fontweight="bold")

    headings = ["Geometry", "LBM speed", "Time-U-Net", "U-Net vector error", "CT-PoreFlow", "CT vector error"]
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading, pad=4)
    speed_bar = fig.colorbar(speed_image, ax=axes[:, 1:3], fraction=0.011, pad=0.010)
    speed_bar.set_label(f"Speed (lattice units), fixed 0--{speed_limit:.3e}")
    error_bar = fig.colorbar(error_image, ax=axes[:, 3:6], fraction=0.011, pad=0.010)
    error_bar.set_label(f"Vector error (lattice units), shared gallery Q99={error_limit:.2e}")
    fig.suptitle(f"Terminal prediction gallery {group}: eight objectively selected test geometries", y=1.032)
    save_publication_figure(fig, output, dpi=400)
    plt.close(fig)


def transient_gallery(
    bundles: list[dict],
    output: Path,
    speed_map,
    speed_limit: float,
    time_indices: list[int],
    title: str,
    *,
    local_triplet_scale: bool = False,
    show_colorbar: bool = True,
) -> None:
    models = ("LBM", "Time-U-Net", "CT-PoreFlow")
    block_rows = 2
    block_columns = 2
    rows = block_rows * len(models)
    columns = block_columns * len(time_indices)
    figure_width = 10.9 if show_colorbar else 10.45
    fig, axes = plt.subplots(rows, columns, figsize=(figure_width, 7.35), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.015, hspace=0.018)
    image = None
    for sample_no, bundle in enumerate(bundles):
        pore = bundle["pore_mask"].astype(bool)
        display_pore = np.rot90(pore, k=1)
        record = bundle["selection"]
        block_row = sample_no // block_columns
        block_column = sample_no % block_columns
        row_start = block_row * len(models)
        column_start = block_column * len(time_indices)
        scales: dict[int, float] = {}
        if local_triplet_scale:
            model_fields = (
                bundle["true_fields"],
                bundle["predicted_fields__Time-U-Net"],
                bundle["predicted_fields__CT-PoreFlow"],
            )
            for time_index in time_indices:
                pooled = np.concatenate(
                    [velocity_magnitude(fields)[time_index][pore] for fields in model_fields]
                )
                scales[time_index] = max(float(np.max(pooled)), np.finfo(float).eps)
        for model_no, model in enumerate(models):
            row = row_start + model_no
            fields = bundle["true_fields"] if model == "LBM" else bundle[f"predicted_fields__{model}"]
            speeds = velocity_magnitude(fields)
            for column, time_index in enumerate(time_indices):
                axis = axes[row, column_start + column]
                display_speed = np.rot90(speeds[time_index], k=1)
                image = axis.imshow(
                    np.ma.masked_where(~display_pore, display_speed),
                    cmap=speed_map,
                    vmin=0,
                    vmax=scales.get(time_index, speed_limit),
                )
                clean(axis)
                if model_no == 0:
                    heading = f"Iteration {int(bundle['iterations'][time_index]):,}"
                    if column == 0:
                        case_name = DISPLAY.get(record["selection_label"], record["selection_label"])
                        heading = f"{chr(ord('a') + sample_no)}  {case_name}\n{heading}"
                    axis.set_title(heading, pad=3)
            axes[row, column_start].set_ylabel(
                model,
                rotation=90,
                ha="center",
                va="bottom",
                fontweight="bold" if model_no == 0 else "normal",
                labelpad=2,
            )
    if show_colorbar:
        bar = fig.colorbar(image, ax=axes, fraction=0.014, pad=0.012)
        bar.set_label(f"Speed (lattice units), fixed 0--{speed_limit:.3e}")
    fig.suptitle(title, y=1.038)
    save_publication_figure(fig, output, dpi=400)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path, default=Path("output/extended_prediction_gallery/selected_samples_16.json")
    )
    parser.add_argument("--output", type=Path, default=Path("output/extended_prediction_gallery/figures"))
    args = parser.parse_args()
    bundles = load(args.selection, args.bundle_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    apply_nature_style(font_size=8.3)
    speed_map = plt.colormaps["viridis"].copy()
    error_map = plt.colormaps["magma"].copy()
    speed_map.set_bad("#050505")
    error_map.set_bad("#050505")
    speed_limit = float(bundles[0]["fixed_speed_color_scale_lattice"])
    error_limit = terminal_error_limit(bundles)
    terminal_gallery(bundles[:8], args.output / "terminal_gallery_porosity_bins", speed_map, error_map, speed_limit, error_limit, "A")
    terminal_gallery(bundles[8:], args.output / "terminal_gallery_extremes_errors", speed_map, error_map, speed_limit, error_limit, "B")

    by_label = {bundle["selection"]["selection_label"]: bundle for bundle in bundles}
    transient_gallery(
        [by_label[key] for key in ("porosity_bin_00", "porosity_bin_03", "porosity_bin_05", "porosity_bin_07")],
        args.output / "transient_gallery_porosity_range",
        speed_map,
        speed_limit,
        [5, 12, 20, 29],
        "Transient predictions across the porosity range",
    )
    transient_gallery(
        [by_label[key] for key in ("high_tortuosity", "hard_case", "ct_advantage", "unet_advantage")],
        args.output / "transient_gallery_challenging_cases",
        speed_map,
        speed_limit,
        [5, 12, 20, 29],
        "Transient predictions for challenging and model-disagreement cases",
        local_triplet_scale=True,
        show_colorbar=False,
    )
    manifest = {
        "selected_samples": [bundle["selection"] for bundle in bundles],
        "fixed_speed_scale_lattice": speed_limit,
        "shared_terminal_error_q99_lattice": error_limit,
        "time_indices": [5, 12, 20, 29],
        "models_displayed": ["Time-U-Net", "CT-PoreFlow"],
        "challenging_gallery_scale": "shared by LBM, Time-U-Net, and CT-PoreFlow within each case-time triplet; no global colorbar",
    }
    (args.output / "gallery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figures": 4, "samples": 16, "speed_limit": speed_limit, "error_limit": error_limit}))


if __name__ == "__main__":
    main()
