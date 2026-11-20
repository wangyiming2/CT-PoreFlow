"""Create publication-style multi-sample, multi-model transient comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from figure_style import add_panel_label, apply_nature_style, save_publication_figure


DISPLAY_NAMES = {
    "low_porosity": "Relatively low porosity",
    "median_case": "Median case",
    "high_tortuosity": "High tortuosity",
    "high_permeability": "High permeability",
}


def speed(fields: np.ndarray, pore: np.ndarray) -> np.ndarray:
    return np.hypot(fields[..., 0, :, :], fields[..., 1, :, :]) * pore


def geometry_rgb(pore: np.ndarray, connected: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*pore.shape, 3), dtype=float)
    rgb[pore > 0.5] = (0.74, 0.76, 0.79)  # dead/non-through pore
    rgb[connected > 0.5] = (0.16, 0.55, 0.78)  # through-connected pore
    return rgb


def clean_axis(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def show_speed(axis, field: np.ndarray, pore: np.ndarray, vmax: float, cmap) -> object:
    image = axis.imshow(
        np.ma.masked_where(pore < 0.5, field),
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    clean_axis(axis)
    return image


def load_bundles(bundle_dir: Path, selected: list[dict]) -> list[dict]:
    bundles = []
    for record in selected:
        path = bundle_dir / f"sample_{int(record['sample_index']):04d}_multimodel_trajectory.npz"
        with np.load(path) as source:
            bundle = {key: np.asarray(source[key]) for key in source.files}
        bundle["selection"] = record
        bundles.append(bundle)
    return bundles


def terminal_comparison(bundles: list[dict], output: Path, cmap) -> None:
    labels = [str(value) for value in bundles[0]["model_labels"]]
    vmax = float(bundles[0]["fixed_speed_color_scale_lattice"])
    fig, axes = plt.subplots(len(bundles), len(labels) + 2, figsize=(10.6, 7.2), constrained_layout=True)
    final = -1
    last_image = None
    for row, bundle in enumerate(bundles):
        pore = bundle["pore_mask"]
        connected = bundle["connected_pore_mask"]
        axes[row, 0].imshow(geometry_rgb(pore, connected), interpolation="nearest")
        clean_axis(axes[row, 0])
        truth_speed = speed(bundle["true_fields"], pore)[final]
        last_image = show_speed(axes[row, 1], truth_speed, pore, vmax, cmap)
        for column, label in enumerate(labels, start=2):
            prediction = bundle[f"predicted_fields__{label}"]
            predicted_speed = speed(prediction, pore)[final]
            show_speed(axes[row, column], predicted_speed, pore, vmax, cmap)
            relative_l2 = np.linalg.norm(predicted_speed - truth_speed) / max(
                np.linalg.norm(truth_speed), 1e-12
            )
            axes[row, column].text(
                0.5,
                -0.055,
                f"$L^2_r$={relative_l2:.3f}",
                transform=axes[row, column].transAxes,
                ha="center",
                va="top",
                fontsize=6.2,
            )
        record = bundle["selection"]
        axes[row, 0].set_ylabel(
            f"{DISPLAY_NAMES.get(record['selection_label'], record['selection_label'])}\n"
            f"$\\phi$={record['porosity']:.3f}, $k$={record['permeability']:.2f}, "
            f"$\\tau_h$={record['tortuosity']:.2f}",
            fontweight="bold",
            labelpad=9,
        )
        add_panel_label(axes[row, 0], chr(ord("a") + row), x=-0.61, y=1.02)
    titles = ["Geometry", "LBM"] + labels
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, pad=4)
    colorbar = fig.colorbar(last_image, ax=axes[:, 1:], fraction=0.012, pad=0.012)
    colorbar.set_label("Speed (lattice units); one fixed train-set scale")
    fig.suptitle("Representative steady-state comparison at iteration 50,000", y=1.015)
    save_publication_figure(fig, output, dpi=400)
    plt.close(fig)


def one_case_temporal(bundle: dict, output: Path, time_indices: list[int], cmap) -> None:
    labels = ["LBM"] + [str(value) for value in bundle["model_labels"]]
    vmax = float(bundle["fixed_speed_color_scale_lattice"])
    pore = bundle["pore_mask"]
    fig, axes = plt.subplots(len(labels), len(time_indices), figsize=(9.0, 7.8), constrained_layout=True)
    last_image = None
    for row, label in enumerate(labels):
        fields = bundle["true_fields"] if label == "LBM" else bundle[f"predicted_fields__{label}"]
        speeds = speed(fields, pore)
        for column, time_index in enumerate(time_indices):
            last_image = show_speed(axes[row, column], speeds[time_index], pore, vmax, cmap)
            if row == 0:
                axes[row, column].set_title(
                    f"Iter. {int(bundle['iterations'][time_index]):,}", pad=4
                )
        axes[row, 0].set_ylabel(label, fontweight="bold", labelpad=7)
        add_panel_label(axes[row, 0], chr(ord("a") + row), x=-0.28, y=1.02)
    colorbar = fig.colorbar(last_image, ax=axes, fraction=0.012, pad=0.012)
    colorbar.set_label("Speed (lattice units); one fixed train-set scale")
    record = bundle["selection"]
    fig.suptitle(
        f"Temporal evolution across models — {DISPLAY_NAMES.get(record['selection_label'], record['selection_label'])}",
        y=1.015,
    )
    save_publication_figure(fig, output, dpi=400)
    plt.close(fig)


def grand_mosaic(bundles: list[dict], output: Path, time_indices: list[int], cmap) -> None:
    labels = ["LBM"] + [str(value) for value in bundles[0]["model_labels"]]
    vmax = float(bundles[0]["fixed_speed_color_scale_lattice"])
    rows = len(bundles) * len(labels)
    fig, axes = plt.subplots(rows, len(time_indices), figsize=(8.8, 23.5), constrained_layout=True)
    last_image = None
    panel_index = 0
    for sample_no, bundle in enumerate(bundles):
        pore = bundle["pore_mask"]
        record = bundle["selection"]
        for model_no, label in enumerate(labels):
            row = sample_no * len(labels) + model_no
            fields = bundle["true_fields"] if label == "LBM" else bundle[f"predicted_fields__{label}"]
            speeds = speed(fields, pore)
            for column, time_index in enumerate(time_indices):
                last_image = show_speed(axes[row, column], speeds[time_index], pore, vmax, cmap)
                if sample_no == 0 and model_no == 0:
                    axes[row, column].set_title(
                        f"Iter. {int(bundle['iterations'][time_index]):,}", pad=3
                    )
            sample_label = DISPLAY_NAMES.get(record["selection_label"], record["selection_label"])
            axes[row, 0].set_ylabel(
                f"{sample_label}\n{label}" if model_no == 0 else label,
                fontweight="bold" if model_no == 0 else "normal",
                labelpad=7,
            )
            if model_no == 0:
                add_panel_label(axes[row, 0], chr(ord("a") + sample_no), x=-0.43, y=1.02)
            panel_index += 1
        if sample_no < len(bundles) - 1:
            divider_row = (sample_no + 1) * len(labels) - 1
            for axis in axes[divider_row, :]:
                axis.plot([0, 1], [-0.08, -0.08], transform=axis.transAxes, color="#6B7280", lw=0.8, clip_on=False)
    colorbar = fig.colorbar(last_image, ax=axes, fraction=0.008, pad=0.012)
    colorbar.set_label("Speed (lattice units); one fixed train-set scale")
    fig.suptitle("Four test cases × five methods × four physical times", y=1.004)
    save_publication_figure(fig, output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument(
        "--selection", type=Path, default=Path("output/representative_samples/selected_samples.json")
    )
    parser.add_argument("--output", type=Path, default=Path("output/multimodel_mosaics"))
    parser.add_argument("--time-indices", default="5,12,20,29")
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    bundles = load_bundles(args.bundle_dir, selected)
    args.output.mkdir(parents=True, exist_ok=True)
    apply_nature_style(font_size=7.2)
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#050505")
    time_indices = [int(value) for value in args.time_indices.split(",")]
    terminal_comparison(bundles, args.output / "terminal_four_samples_four_models", cmap)
    one_case_temporal(
        bundles[1], args.output / "median_sample_five_methods_five_times", [5, 12, 20, 25, 29], cmap
    )
    grand_mosaic(bundles, args.output / "grand_four_samples_five_methods_four_times", time_indices, cmap)
    print(args.output)


if __name__ == "__main__":
    main()
