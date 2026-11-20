from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import (
    add_panel_label,
    apply_nature_style,
    close_figure,
    method_style,
    save_publication_figure,
    soften_axis,
)


def _load_final(run: Path) -> dict[str, dict[str, float]]:
    source = run / "evaluation" / "test" / "per_sample_time_metrics.csv"
    result: dict[str, dict[str, float]] = {}
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not np.isfinite(float(row["true_permeability_lattice"])):
                continue
            iteration = int(row["iteration"])
            previous = result.get(row["sample_id"])
            if previous is not None and int(previous["iteration"]) >= iteration:
                continue
            result[row["sample_id"]] = {
                "iteration": iteration,
                "true_permeability": float(row["true_permeability_lattice"]),
                "pred_permeability": float(row["pred_permeability_lattice"]),
                "true_flux": float(row["true_mean_flux_lattice"]),
                "pred_flux": float(row["pred_mean_flux_lattice"]),
            }
    return result


def _summary(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - true
    total = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / max(total, 1e-30)
    correlation = float(np.corrcoef(true, predicted)[0, 1]) if true.size > 1 else float("nan")
    return {
        "n": int(true.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_relative_absolute_error": float(
            np.mean(np.abs(residual) / np.maximum(np.abs(true), 1e-30))
        ),
        "r2": r2,
        "pearson_r": correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != len(args.label):
        raise ValueError("Each --run requires one --label")
    loaded = {label: _load_final(run) for run, label in zip(args.run, args.label)}
    common = sorted(set.intersection(*(set(value) for value in loaded.values())))
    if not common:
        raise RuntimeError("No shared final-time records")

    apply_nature_style(font_size=7.0)
    rows: list[dict[str, float | int | str]] = []
    fig, axes = plt.subplots(
        2,
        len(args.label),
        figsize=(7.2, 4.25),
        squeeze=False,
        constrained_layout=True,
    )
    # Flux is O(1e-5). Explicit scaled units avoid Matplotlib's offset text
    # colliding with the long x-axis label in the multi-panel figure.
    definitions = [
        ("permeability", "Permeability (lattice units)", 1.0),
        ("flux", r"Mean flux ($\times 10^{-5}$ lattice units)", 1.0e5),
    ]
    limits: dict[str, tuple[float, float]] = {}
    for metric, _, display_scale in definitions:
        values: list[np.ndarray] = []
        for label in args.label:
            values.extend(
                [
                    np.asarray([loaded[label][sample][f"true_{metric}"] for sample in common]),
                    np.asarray([loaded[label][sample][f"pred_{metric}"] for sample in common]),
                ]
            )
        merged = np.concatenate(values) * display_scale
        lower = float(np.nanmin(merged))
        upper = float(np.nanmax(merged))
        pad = max((upper - lower) * 0.025, 1e-12)
        limits[metric] = (lower - pad, upper + pad)
    for column, label in enumerate(args.label):
        for row_index, (metric, axis_label, display_scale) in enumerate(definitions):
            true = np.asarray([loaded[label][sample][f"true_{metric}"] for sample in common])
            predicted = np.asarray([loaded[label][sample][f"pred_{metric}"] for sample in common])
            stats = _summary(true, predicted)
            rows.append({"model": label, "metric": metric, **stats})
            true_display = true * display_scale
            predicted_display = predicted * display_scale
            axis = axes[row_index, column]
            axis.scatter(
                true_display,
                predicted_display,
                s=4.5,
                alpha=0.30,
                color=method_style(label, column)["color"],
                edgecolors="none",
                rasterized=True,
            )
            lower, upper = limits[metric]
            axis.plot([lower, upper], [lower, upper], color="#4B5563", linewidth=0.9, linestyle="--")
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_aspect("equal", adjustable="box")
            axis.text(
                0.04,
                0.95,
                f"$R^2$={stats['r2']:.3f}\n$r$={stats['pearson_r']:.3f}\n$n$={stats['n']}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=5.8,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2},
            )
            if row_index == 0:
                axis.set_title(label, weight="bold")
            if column == 0:
                axis.set_ylabel(f"Predicted {axis_label}")
            # The row label already names the physical quantity and units.
            # A compact common x label stays legible in four-column layouts.
            axis.set_xlabel("LBM reference")
            soften_axis(axis)
            add_panel_label(axis, chr(ord("a") + row_index * len(args.label) + column), x=-0.17, y=1.03)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "physical_calibration.png"
    save_publication_figure(fig, figure_path)
    close_figure(fig)
    table_path = args.output_dir / "physical_calibration_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(table_path)
    print(figure_path)


if __name__ == "__main__":
    main()
