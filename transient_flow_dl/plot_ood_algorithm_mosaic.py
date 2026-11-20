from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import apply_nature_style, save_publication_figure


FAMILIES = ("circle_array", "snowflakes", "trees", "staggered_baffles")
FAMILY_LABELS = {
    "circle_array": "Circular array",
    "snowflakes": "Snowflakes",
    "trees": "Tree obstacles",
    "staggered_baffles": "Staggered baffles",
}
MODELS = (
    ("unet_time_seed2026", "Time-U-Net"),
    ("fno_time_seed2026", "Time-FNO"),
    ("deeponet_seed2026", "DeepONet"),
    ("ct_poreflow_seed2026", "CT-PoreFlow"),
)
SPEED_SCALE = 5.024437996326014e-06


def speed(fields: np.ndarray) -> np.ndarray:
    return np.hypot(fields[0], fields[1])


def vector_error(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.hypot(predicted[0] - truth[0], predicted[1] - truth[1])


def geometry_rgb(pore: np.ndarray, connected: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*pore.shape, 3), dtype=float)
    rgb[pore] = (0.78, 0.79, 0.82)
    rgb[connected] = (0.12, 0.48, 0.72)
    return rgb


def clean(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.35)
        spine.set_color("#555555")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, path, dpi=400)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def make_mosaic(output_root: Path, time_index: int = -1) -> Path:
    apply_nature_style(font_size=7.1)
    truth_root = output_root / "lbm_truth"
    prediction_root = output_root / "selected_predictions"
    records = []
    for family in FAMILIES:
        sample_id = f"{family}_phi68_r00"
        with np.load(truth_root / f"{sample_id}.npz") as truth:
            solid = np.asarray(truth["solid"], dtype=bool)
            pore = np.asarray(truth["pore"], dtype=bool)
            connected = np.asarray(truth["through_connected_pore"], dtype=bool)
            truth_uv = np.stack(
                [np.asarray(truth["transient_velocity_x_raw_lattice"])[time_index],
                 np.asarray(truth["transient_velocity_y_raw_lattice"])[time_index]]
            )
        predictions = {}
        for model_name, _ in MODELS:
            with np.load(prediction_root / f"{model_name}__{sample_id}.npz") as predicted:
                normalized = np.asarray(predicted["prediction_normalized"])[time_index]
                scale = float(np.asarray(predicted["velocity_scale"]))
            predictions[model_name] = normalized[:2] * scale
        records.append((family, solid, pore, connected, truth_uv, predictions))

    error_values = []
    for _, _, pore, _, truth_uv, predictions in records:
        for model_name in ("fno_time_seed2026", "ct_poreflow_seed2026"):
            error_values.append(vector_error(predictions[model_name], truth_uv)[pore])
    error_limit = float(np.quantile(np.concatenate(error_values), 0.99))
    speed_map = plt.colormaps["viridis"].copy()
    error_map = plt.colormaps["magma"].copy()
    speed_map.set_bad("#050505")
    error_map.set_bad("#050505")

    fig, axes = plt.subplots(len(records), 8, figsize=(15.5, 8.8), squeeze=False)
    speed_image = error_image = None
    titles = ["Geometry", "LBM speed", "Time-U-Net", "Time-FNO", "DeepONet", "CT-PoreFlow", "FNO vector error", "CT vector error"]
    for row, (family, solid, pore, connected, truth_uv, predictions) in enumerate(records):
        axes[row, 0].imshow(
            np.transpose(geometry_rgb(pore, connected), (1, 0, 2)),
            interpolation="nearest",
            origin="lower",
        )
        truth_speed = np.ma.masked_where(~pore, speed(truth_uv))
        speed_image = axes[row, 1].imshow(truth_speed.T, origin="lower", interpolation="nearest", cmap=speed_map, vmin=0, vmax=SPEED_SCALE)
        for col, (model_name, _) in enumerate(MODELS, start=2):
            display = np.ma.masked_where(~pore, speed(predictions[model_name])).T
            axes[row, col].imshow(display, origin="lower", interpolation="nearest", cmap=speed_map, vmin=0, vmax=SPEED_SCALE)
        fno_err = np.ma.masked_where(~pore, vector_error(predictions["fno_time_seed2026"], truth_uv)).T
        ct_err = np.ma.masked_where(~pore, vector_error(predictions["ct_poreflow_seed2026"], truth_uv)).T
        error_image = axes[row, 6].imshow(fno_err, origin="lower", interpolation="nearest", cmap=error_map, vmin=0, vmax=error_limit)
        axes[row, 7].imshow(ct_err, origin="lower", interpolation="nearest", cmap=error_map, vmin=0, vmax=error_limit)
        for axis in axes[row]:
            clean(axis)
        axes[row, 0].set_ylabel(FAMILY_LABELS[family], rotation=90, labelpad=8, fontweight="bold")
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, pad=4)
    speed_axis = fig.add_axes([0.19, 0.052, 0.45, 0.014])
    speed_bar = fig.colorbar(speed_image, cax=speed_axis, orientation="horizontal")
    speed_bar.set_label(f"Speed (lattice units), fixed 0--{SPEED_SCALE:.3e}", labelpad=2)
    error_axis = fig.add_axes([0.72, 0.052, 0.22, 0.014])
    error_bar = fig.colorbar(error_image, cax=error_axis, orientation="horizontal")
    error_bar.set_label(f"Vector error (lattice units), shared OOD Q99={error_limit:.2e}", labelpad=2)
    fig.text(0.5, 0.008, "Terminal stored state (time index 29); encodings follow the paper-wide prediction-gallery contract.", ha="center", fontsize=8.2)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.94, bottom=0.11, wspace=0.055, hspace=0.13)
    destination = output_root / "figures" / "rule_ood_algorithm_mosaic"
    _save(fig, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output" / "ood_rule_validation")
    parser.add_argument("--time-index", type=int, default=-1)
    args = parser.parse_args()
    make_mosaic(args.output_root.resolve(), args.time_index)


if __name__ == "__main__":
    main()
