"""Rebuild paper result figures with a traceable Ref_code-inspired visual grammar.

The script never changes or synthesizes numerical results.  It reads the formal
evaluation CSV/JSON artifacts, adds distribution/density/uncertainty encodings,
and exports 600-dpi PNG plus editable PDF/SVG and a source-data CSV per figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from figure_style import (
    add_panel_label,
    apply_nature_style,
    close_figure,
    method_style,
    save_publication_figure,
    soften_axis,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "output" / "formal_results"
DEFAULT_OUT = ROOT / "output" / "paper" / "figures_v2"

MODELS = [
    ("Time-U-Net", FORMAL / "unet_v2_transport_matched_seed2026"),
    ("Time-FNO", FORMAL / "fno_time_seed2026"),
    ("DeepONet", FORMAL / "deeponet_seed2026"),
    ("CT-PoreFlow", FORMAL / "ct_poreflow_v2_transport_seed2026"),
]
HOLDOUT_MODELS = [
    ("Time-U-Net", FORMAL / "unet_temporal_holdout_seed2026"),
    ("CT-PoreFlow", FORMAL / "ct_poreflow_v2_temporal_holdout_seed2026"),
]

TIME_TICKS = [0, 10, 100, 1_000, 50_000]
TIME_TICK_LABELS = ["0", "10", r"$10^2$", r"$10^3$", r"$5\times10^4$"]
STAGES = [
    (0, 22, "early", "#4DBBD5"),
    (32, 1_049, "transition", "#F39B7F"),
    (1_544, 50_000, "late", "#8491B4"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty source-data table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: str | float | int) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def normal_mean_ci(mean: float, std: float, n: int) -> tuple[float, float]:
    if n <= 1 or not np.isfinite(std):
        return mean, mean
    half = 1.959963984540054 * std / np.sqrt(n)
    return mean - half, mean + half


def add_stage_background(axis) -> None:
    for left, right, label, color in STAGES:
        axis.axvspan(left, right, color=color, alpha=0.045, linewidth=0, zorder=0)
    axis.text(0.16, 0.985, "early", transform=axis.transAxes, ha="center", va="top", fontsize=7.4, color="#4B5563")
    axis.text(0.53, 0.985, "transition", transform=axis.transAxes, ha="center", va="top", fontsize=7.4, color="#4B5563")
    axis.text(0.88, 0.985, "late", transform=axis.transAxes, ha="center", va="top", fontsize=7.4, color="#4B5563")


def finish_time_axis(axis, ylabel: str) -> None:
    axis.set_xscale("symlog", linthresh=1.0)
    axis.set_xticks(TIME_TICKS, TIME_TICK_LABELS)
    axis.set_xlabel("LBM iteration")
    axis.set_ylabel(ylabel)
    axis.margins(y=0.09)
    soften_axis(axis)


def temporal_field_figure(output: Path) -> dict[str, object]:
    metrics = [
        ("relative_l2_velocity", r"Velocity relative $L_2$"),
        ("mae_p_normalized", "Normalized pressure MAE"),
    ]
    source_rows: list[dict[str, object]] = []
    apply_nature_style(font_size=8.6)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    for axis in axes:
        add_stage_background(axis)
    for model_index, (label, run) in enumerate(MODELS):
        rows = read_rows(run / "analysis" / "per_time_metrics.csv")
        style = method_style(label, model_index)
        for axis, (metric, _) in zip(axes, metrics):
            x: list[int] = []
            y: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for row in rows:
                mean = finite_float(row[f"{metric}_mean"])
                std = finite_float(row[f"{metric}_std"])
                n = int(row[f"{metric}_n"])
                if mean is None or std is None or n == 0:
                    continue
                lo, hi = normal_mean_ci(mean, std, n)
                iteration = int(row["iteration"])
                x.append(iteration)
                y.append(mean)
                lower.append(max(lo, 0.0))
                upper.append(hi)
                source_rows.append(
                    {
                        "model": label,
                        "metric": metric,
                        "iteration": iteration,
                        "mean": mean,
                        "std": std,
                        "n_geometries": n,
                        "ci95_lower_normal": lo,
                        "ci95_upper_normal": hi,
                    }
                )
            axis.fill_between(x, lower, upper, color=style["color"], alpha=0.12, linewidth=0, zorder=1)
            axis.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0 if label == "CT-PoreFlow" else 1.55,
                markersize=3.0,
                markevery=3,
                label=label,
                zorder=3,
            )
    for panel, axis, (_, ylabel) in zip("ab", axes, metrics):
        finish_time_axis(axis, ylabel)
        add_panel_label(axis, panel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=4)
    base = output / "temporal_field_performance_v2.png"
    files = save_publication_figure(fig, base)
    close_figure(fig)
    csv_path = base.with_suffix(".csv")
    write_rows(csv_path, source_rows)
    return {"label": base.stem, "files": [str(path) for path in files], "source": str(csv_path)}


def physical_dynamics_figure(output: Path) -> dict[str, object]:
    definitions = [
        ("normalized_mean_flux_error", "Mean-flux error / terminal flux", 1.0, True),
        ("pred_divergence_l1_lattice", r"Divergence $L_1$ ($\times10^{-8}$)", 1.0e8, False),
        ("relative_error_hydraulic_tortuosity", "Tortuosity relative error", 1.0, True),
    ]
    source_rows: list[dict[str, object]] = []
    apply_nature_style(font_size=8.4)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.62), constrained_layout=True)
    for axis in axes:
        add_stage_background(axis)
    for model_index, (label, run) in enumerate(MODELS):
        rows = read_rows(run / "analysis" / "per_time_metrics.csv")
        style = method_style(label, model_index)
        for axis, (metric, _, scale, log_y) in zip(axes, definitions):
            x: list[int] = []
            y: list[float] = []
            lo_values: list[float] = []
            hi_values: list[float] = []
            for row in rows:
                mean = finite_float(row[f"{metric}_mean"])
                std = finite_float(row[f"{metric}_std"])
                n = int(row[f"{metric}_n"])
                if mean is None or std is None or n == 0:
                    continue
                lo, hi = normal_mean_ci(mean, std, n)
                iteration = int(row["iteration"])
                x.append(iteration)
                y.append(mean * scale)
                lo_values.append(max(lo * scale, 1e-12 if mean > 0 else 0.0))
                hi_values.append(hi * scale)
                source_rows.append(
                    {
                        "model": label,
                        "metric": metric,
                        "iteration": iteration,
                        "mean": mean,
                        "std": std,
                        "n_geometries": n,
                        "ci95_lower_normal": lo,
                        "ci95_upper_normal": hi,
                    }
                )
            # Symmetric normal intervals may cross zero; clipping them on a
            # logarithmic axis creates a false, enormous uncertainty wedge.
            # The interval remains available in the source-data CSV.
            if not log_y:
                axis.fill_between(x, lo_values, hi_values, color=style["color"], alpha=0.10, linewidth=0)
            axis.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.9 if label == "CT-PoreFlow" else 1.45,
                markersize=2.8,
                markevery=3,
                label=label,
            )
    for panel, axis, (_, ylabel, _, log_y) in zip("abc", axes, definitions):
        finish_time_axis(axis, ylabel)
        if log_y:
            axis.set_yscale("log")
            axis.axhline(1.0, color="#7A818C", linestyle=":", linewidth=0.75, zorder=0)
        add_panel_label(axis, panel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=4)
    base = output / "physical_dynamics_v2.png"
    files = save_publication_figure(fig, base)
    close_figure(fig)
    csv_path = base.with_suffix(".csv")
    write_rows(csv_path, source_rows)
    return {"label": base.stem, "files": [str(path) for path in files], "source": str(csv_path)}


def load_final_records(run: Path) -> dict[str, dict[str, float]]:
    latest: dict[str, dict[str, float]] = {}
    for row in read_rows(run / "evaluation" / "test" / "per_sample_time_metrics.csv"):
        true_perm = finite_float(row["true_permeability_lattice"])
        if true_perm is None:
            continue
        iteration = int(row["iteration"])
        previous = latest.get(row["sample_id"])
        if previous is not None and int(previous["iteration"]) >= iteration:
            continue
        latest[row["sample_id"]] = {
            "iteration": float(iteration),
            "true_permeability": true_perm,
            "pred_permeability": float(row["pred_permeability_lattice"]),
            "true_flux": float(row["true_mean_flux_lattice"]),
            "pred_flux": float(row["pred_mean_flux_lattice"]),
            "true_tortuosity": float(row["true_hydraulic_tortuosity"]),
            "pred_tortuosity": float(row["pred_hydraulic_tortuosity"]),
        }
    return latest


def regression_stats(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - true
    denominator = float(np.sum((true - true.mean()) ** 2))
    return {
        "r2": 1.0 - float(np.sum(residual**2)) / max(denominator, 1e-30),
        "pearson_r": float(np.corrcoef(true, pred)[0, 1]),
        "mdape_percent": float(np.median(np.abs(residual) / np.maximum(np.abs(true), 1e-30)) * 100.0),
    }


def normalized_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fast dependency-free 2-D histogram density for scatter colouring."""

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=36)
    # A compact 3x3 smoothing kernel avoids blocky density transitions without
    # adding SciPy as a plotting-only dependency.
    padded = np.pad(counts, 1, mode="edge")
    smooth = sum(
        padded[dx : dx + counts.shape[0], dy : dy + counts.shape[1]]
        for dx in range(3)
        for dy in range(3)
    ) / 9.0
    x_bin = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, counts.shape[0] - 1)
    y_bin = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, counts.shape[1] - 1)
    density = smooth[x_bin, y_bin]
    span = float(density.max() - density.min())
    return (density - density.min()) / span if span > 0 else np.zeros_like(density)


def rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling (one-based ranks)."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def spearman_with_approx_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and a large-sample Fisher-z two-sided approximation."""

    rho = float(np.corrcoef(rank_average(x), rank_average(y))[0, 1])
    clipped = float(np.clip(rho, -0.999999, 0.999999))
    z_value = abs(np.arctanh(clipped)) * np.sqrt(max(x.size - 3, 1))
    p_value = math.erfc(z_value / np.sqrt(2.0))
    return rho, p_value


def _reference_histogram(axis, values: np.ndarray, bins: np.ndarray, *, orientation: str) -> None:
    """Hybrid-M-inspired marginal histogram with a compact smoothed trace."""

    if orientation == "vertical":
        counts, edges, patches = axis.hist(
            values, bins=bins, density=True, edgecolor="#31363D", linewidth=0.25
        )
        base, delta = np.asarray([212, 230, 241]), np.asarray([160, 78, 22])
    else:
        counts, edges, patches = axis.hist(
            values,
            bins=bins,
            density=True,
            orientation="horizontal",
            edgecolor="#31363D",
            linewidth=0.25,
        )
        base, delta = np.asarray([250, 219, 216]), np.asarray([70, 169, 166])
    peak = max(float(np.max(counts)), 1e-12)
    for count, patch in zip(counts, patches):
        rgb = np.clip(base - delta * (float(count) / peak), 0, 255).astype(int)
        patch.set_facecolor(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(counts, np.asarray([0.2, 0.6, 0.2]), mode="same")
    if orientation == "vertical":
        axis.plot(centers, smooth, color="#D64B40", linewidth=0.9)
    else:
        axis.plot(smooth, centers, color="#D64B40", linewidth=0.9)


def _calibration_reference_panel(
    fig,
    slot,
    true: np.ndarray,
    pred: np.ndarray,
    *,
    metric: str,
    title: str,
    panel_label: str,
    color: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> dict[str, float]:
    """Reference-style single regression scatter with a top residual histogram."""

    nested = slot.subgridspec(2, 1, height_ratios=(1.0, 4.4), hspace=0.08)
    histogram = fig.add_subplot(nested[0, 0])
    main = fig.add_subplot(nested[1, 0])
    stats = regression_stats(true, pred)
    residual = true - pred
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    identity = np.linspace(x_limits[0], x_limits[1], 300)
    main.scatter(
        true,
        pred,
        color=color,
        s=10,
        alpha=0.34,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    main.plot(identity, identity, color="#111111", linewidth=1.4, label="1:1")
    main.plot(identity, 0.9 * identity, color="#111111", linestyle="--", linewidth=1.0)
    main.plot(identity, 1.1 * identity, color="#111111", linestyle="--", linewidth=1.0, label="±10%")
    main.plot(identity, 0.8 * identity, color="#111111", linestyle=":", linewidth=1.0)
    main.plot(identity, 1.2 * identity, color="#111111", linestyle=":", linewidth=1.0, label="±20%")
    main.set_xlim(*x_limits)
    main.set_ylim(*y_limits)
    if metric == "tortuosity":
        main.set_yscale("log")
    main.set_xlabel("LBM reference", fontweight="bold")
    main.set_ylabel("Prediction", fontweight="bold")
    main.grid(True, color="#D7DDE4", linewidth=0.45, alpha=0.7)
    main.tick_params(direction="in", top=True, right=True, width=0.8, length=3.0)
    for spine in main.spines.values():
        spine.set_linewidth(0.85)
    metric_text = (
        f"$R^2$={stats['r2']:.3f}\n"
        f"RMSE={rmse:.3g}\n"
        f"MAE={mae:.3g}\n"
        f"MdAPE={stats['mdape_percent']:.1f}%"
    )
    main.text(
        0.04,
        0.965,
        metric_text,
        transform=main.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        fontstyle="italic",
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#30343B", "linewidth": 0.55, "pad": 1.7},
        zorder=8,
    )

    residual_limit = max(float(np.nanpercentile(np.abs(residual), 98)), 1e-9)
    bins = np.linspace(-residual_limit, residual_limit, 15)
    counts, edges, _ = histogram.hist(
        residual,
        bins=bins,
        color=color,
        alpha=0.42,
        edgecolor="#20242B",
        linewidth=0.35,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(counts, np.asarray([0.2, 0.6, 0.2]), mode="same")
    histogram.plot(centers, smooth, color=color, linewidth=1.1)
    histogram.axvline(0.0, color="#20242B", linestyle="--", linewidth=0.7)
    histogram.set_xlim(-residual_limit, residual_limit)
    histogram.set_ylabel("Count", fontweight="bold")
    histogram.set_title(f"{panel_label}  {title}", loc="left", fontsize=8.7, fontweight="bold", color=color, pad=2)
    histogram.tick_params(axis="both", labelsize=6.3, direction="in", width=0.7, length=2.5)
    for spine in histogram.spines.values():
        spine.set_linewidth(0.75)
    return stats


def calibration_figure_reference_hist(output: Path) -> dict[str, object]:
    loaded = {label: load_final_records(run) for label, run in MODELS}
    common = sorted(set.intersection(*(set(records) for records in loaded.values())))
    definitions = (
        ("permeability", "Terminal permeability"),
        ("tortuosity", "Hydraulic tortuosity"),
    )
    source_rows: list[dict[str, object]] = []
    statistics: dict[tuple[str, str], dict[str, float]] = {}
    for label, _ in MODELS:
        for metric, _ in definitions:
            true = np.asarray([loaded[label][sample][f"true_{metric}"] for sample in common], dtype=float)
            pred = np.asarray([loaded[label][sample][f"pred_{metric}"] for sample in common], dtype=float)
            statistics[(label, metric)] = regression_stats(true, pred)
            density = normalized_density(true, pred)
            for sample, true_value, pred_value, density_value in zip(common, true, pred, density):
                source_rows.append(
                    {
                        "model": label,
                        "metric": metric,
                        "sample_id": sample,
                        "lbm_reference_display_units": true_value,
                        "prediction_display_units": pred_value,
                        "normalized_point_density": density_value,
                    }
                )

    limits: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for metric, _ in definitions:
        true_all = np.concatenate(
            [np.asarray([loaded[label][sample][f"true_{metric}"] for sample in common], dtype=float) for label, _ in MODELS]
        )
        pred_all = np.concatenate(
            [np.asarray([loaded[label][sample][f"pred_{metric}"] for sample in common], dtype=float) for label, _ in MODELS]
        )
        x_span = max(float(np.ptp(true_all)), 1e-9)
        x_limits = (max(0.0, float(np.min(true_all)) - 0.05 * x_span), float(np.max(true_all)) + 0.07 * x_span)
        if metric == "tortuosity":
            y_limits = (max(1e-3, 0.95 * float(np.min(pred_all))), 1.08 * float(np.max(pred_all)))
        else:
            pooled = np.concatenate((true_all, pred_all))
            span = max(float(np.ptp(pooled)), 1e-9)
            y_limits = (max(0.0, float(np.min(pooled)) - 0.05 * span), float(np.max(pooled)) + 0.07 * span)
        limits[metric] = (x_limits, y_limits)

    apply_nature_style(font_size=7.8)
    fig = plt.figure(figsize=(7.2, 8.25), constrained_layout=True)
    outer = fig.add_gridspec(4, 2, wspace=0.11, hspace=0.11)
    for row, (label, _) in enumerate(MODELS):
        style = method_style(label, row)
        for column, (metric, metric_title) in enumerate(definitions):
            true = np.asarray([loaded[label][sample][f"true_{metric}"] for sample in common], dtype=float)
            pred = np.asarray([loaded[label][sample][f"pred_{metric}"] for sample in common], dtype=float)
            x_limits, y_limits = limits[metric]
            _calibration_reference_panel(
                fig,
                outer[row, column],
                true,
                pred,
                metric=metric,
                title=f"{label}: {metric_title}",
                panel_label=chr(ord("a") + row * 2 + column),
                color=str(style["color"]),
                x_limits=x_limits,
                y_limits=y_limits,
            )
    fig.legend(
        handles=(
            Line2D([0], [0], color="#111111", linewidth=1.4, label="1:1"),
            Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.0, label="±10%"),
            Line2D([0], [0], color="#111111", linestyle=":", linewidth=1.0, label="±20%"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.012),
        ncol=3,
        frameon=False,
    )

    base = output / "physical_calibration_density_v2.png"
    files = save_publication_figure(fig, base)
    close_figure(fig)
    csv_path = base.with_suffix(".csv")
    write_rows(csv_path, source_rows)
    return {"label": base.stem, "files": [str(path) for path in files], "source": str(csv_path)}


def calibration_figure(output: Path) -> dict[str, object]:
    """Original compact density-calibration figure used by the current manuscript."""

    loaded = {label: load_final_records(run) for label, run in MODELS}
    common = sorted(set.intersection(*(set(records) for records in loaded.values())))
    definitions = [
        ("permeability", "Permeability (lattice units)"),
        ("tortuosity", "Hydraulic tortuosity"),
    ]
    source_rows: list[dict[str, object]] = []
    limits: dict[str, tuple[float, float]] = {}
    for metric, _ in definitions:
        pooled: list[float] = []
        for label, _ in MODELS:
            true = np.asarray([loaded[label][sample][f"true_{metric}"] for sample in common], dtype=float)
            pred = np.asarray([loaded[label][sample][f"pred_{metric}"] for sample in common], dtype=float)
            density = normalized_density(true, pred)
            pooled.extend(true)
            pooled.extend(pred)
            for sample, true_value, pred_value, density_value in zip(common, true, pred, density):
                source_rows.append(
                    {
                        "model": label,
                        "metric": metric,
                        "sample_id": sample,
                        "lbm_reference_display_units": true_value,
                        "prediction_display_units": pred_value,
                        "normalized_point_density": density_value,
                    }
                )
        pooled_array = np.asarray(pooled, dtype=float)
        span = max(float(np.ptp(pooled_array)), 1e-9)
        limits[metric] = (
            max(0.0, float(np.min(pooled_array)) - 0.04 * span),
            float(np.max(pooled_array)) + 0.06 * span,
        )

    apply_nature_style(font_size=8.0)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.10), constrained_layout=True)
    scatter = None
    for column, (label, _) in enumerate(MODELS):
        for row, (metric, metric_title) in enumerate(definitions):
            axis = axes[row, column]
            true = np.asarray([loaded[label][sample][f"true_{metric}"] for sample in common], dtype=float)
            pred = np.asarray([loaded[label][sample][f"pred_{metric}"] for sample in common], dtype=float)
            density = normalized_density(true, pred)
            stats = regression_stats(true, pred)
            lower, upper = limits[metric]
            identity = np.linspace(lower, upper, 400)
            axis.fill_between(identity, 0.8 * identity, 1.2 * identity, color="#F39B7F", alpha=0.09, linewidth=0)
            axis.fill_between(identity, 0.9 * identity, 1.1 * identity, color="#4DBBD5", alpha=0.13, linewidth=0)
            scatter = axis.scatter(
                true,
                pred,
                c=density,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=7.0,
                alpha=0.72,
                edgecolors="none",
                rasterized=True,
                zorder=3,
            )
            axis.plot(identity, identity, color="#20242B", linestyle="--", linewidth=0.9, zorder=4)
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            if metric == "tortuosity":
                axis.set_yscale("log")
            if row == 0:
                title = "CT-PoreFlow" if label == "CT-PoreFlow" else label
                axis.set_title(title, fontsize=8.2, fontweight="bold", pad=3)
            if column == 0:
                axis.set_ylabel(f"Predicted\n{metric_title}")
            if row == 1:
                axis.set_xlabel("LBM reference")
            axis.text(
                0.04,
                0.96,
                f"$R^2$={stats['r2']:.3f}\n$r$={stats['pearson_r']:.3f}\nMdAPE={stats['mdape_percent']:.1f}%",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.2,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2},
                zorder=5,
            )
            soften_axis(axis)
            add_panel_label(axis, chr(ord("a") + row * 4 + column), x=-0.18, y=1.04)

    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes, fraction=0.018, pad=0.012, aspect=35)
        colorbar.set_label("Normalized point density")
    fig.legend(
        handles=[
            Patch(facecolor="#4DBBD5", alpha=0.25, edgecolor="none", label="±10%"),
            Patch(facecolor="#F39B7F", alpha=0.20, edgecolor="none", label="±20%"),
            Line2D([0], [0], color="#20242B", linestyle="--", linewidth=0.9, label="1:1"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=3,
        frameon=False,
    )
    base = output / "physical_calibration_density_v2.png"
    files = save_publication_figure(fig, base)
    close_figure(fig)
    csv_path = base.with_suffix(".csv")
    write_rows(csv_path, source_rows)
    return {"label": base.stem, "files": [str(path) for path in files], "source": str(csv_path)}


def pareto_front(rows: list[dict[str, object]], x_key: str) -> list[dict[str, object]]:
    best = float("inf")
    front: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: float(item[x_key])):
        error = float(row["velocity_rel_l2"])
        if error < best:
            front.append(row)
            best = error
    return front


def efficiency_figure(output: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for label, run in MODELS:
        summary = json.loads((run / "evaluation" / "test" / "summary.json").read_text(encoding="utf-8"))
        model = json.loads((run / "model.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "model": label,
                "run": run.name,
                "parameters": int(model["parameters"]),
                "model_ms": float(summary["model_milliseconds_per_frame"]),
                "end_to_end_ms": float(summary["end_to_end_milliseconds_per_frame"]),
                "velocity_rel_l2": float(summary["relative_l2_velocity"]),
                "pressure_mae": float(summary["mae_p_normalized"]),
            }
        )
    apply_nature_style(font_size=8.6)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    definitions = [
        ("model_ms", "Model-only latency (ms frame$^{-1}$)"),
        ("end_to_end_ms", "End-to-end latency (ms frame$^{-1}$)"),
    ]
    offsets = {
        "Time-U-Net": (6, 8),
        "Time-FNO": (6, -15),
        "DeepONet": (6, -15),
        "CT-PoreFlow": (-6, 8),
    }
    parameters = np.asarray([float(row["parameters"]) for row in rows])
    bubble_sizes = 46 + 105 * np.sqrt(parameters / parameters.max())
    for panel, axis, (x_key, xlabel) in zip("ab", axes, definitions):
        for index, (row, size) in enumerate(zip(rows, bubble_sizes)):
            label = str(row["model"])
            style = method_style(label, index)
            x = float(row[x_key])
            y = float(row["velocity_rel_l2"])
            axis.scatter(x, y, s=size, color=style["color"], marker=style["marker"], edgecolor="white", linewidth=0.7, zorder=3)
            dx, dy = offsets[label]
            axis.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", ha="right" if dx < 0 else "left", fontsize=7.5, color="#20242B")
        front = pareto_front(rows, x_key)
        axis.plot(
            [float(row[x_key]) for row in front],
            [float(row["velocity_rel_l2"]) for row in front],
            color="#6B7280",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
            label="Pareto front",
        )
        axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(r"Velocity relative $L_2$ (lower is better)")
        axis.set_ylim(0.18, 0.92)
        axis.margins(x=0.18)
        soften_axis(axis)
        add_panel_label(axis, panel)
    size_handles = [
        plt.scatter([], [], s=46 + 105 * np.sqrt(value / parameters.max()), color="#AAB2BD", edgecolor="white", label=f"{value/1e6:.1f} M")
        for value in (parameters.min(), np.median(parameters), parameters.max())
    ]
    axes[0].legend(loc="upper right")
    axes[1].legend(handles=size_handles, title="Parameters", loc="upper right", labelspacing=0.7)
    base = output / "accuracy_efficiency_v2.png"
    files = save_publication_figure(fig, base)
    close_figure(fig)
    csv_path = base.with_suffix(".csv")
    write_rows(csv_path, rows)
    return {"label": base.stem, "files": [str(path) for path in files], "source": str(csv_path)}


def stratified_figure(output: Path) -> dict[str, object]:
    sample_path = FORMAL / "ct_poreflow_v2_transport_seed2026" / "analysis" / "stratified" / "sample_level_metrics.csv"
    quartile_path = FORMAL / "ct_poreflow_v2_transport_seed2026" / "analysis" / "stratified" / "quartile_metrics.csv"
    samples = read_rows(sample_path)
    quartile_rows = read_rows(quartile_path)
    ci_lookup = {(row["factor"], int(row["quartile"])): row for row in quartile_rows}
    factors = [
        ("porosity", "Total porosity"),
        ("connected_porosity", "Connected porosity"),
        ("dead_pore_ratio", "Dead-pore ratio"),
        ("hydraulic_tortuosity", "Hydraulic tortuosity"),
        ("iterations", "LBM convergence iterations"),
    ]
    colors = ["#D9EDF2", "#AFD9E3", "#79BACA", "#3C7F91"]
    apply_nature_style(font_size=8.1)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.05), constrained_layout=True, sharey=True)
    source_rows: list[dict[str, object]] = []
    error = np.asarray([float(row["mean_relative_l2_velocity"]) for row in samples])
    for panel, axis, (factor, title) in zip("abcde", axes.flat, factors):
        values = np.asarray([float(row[factor]) for row in samples])
        edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
        assignment = edges.copy()
        for index in range(1, len(assignment)):
            if assignment[index] <= assignment[index - 1]:
                assignment[index] = np.nextafter(assignment[index - 1], np.inf)
        bins = np.clip(np.digitize(values, assignment[1:-1], right=True), 0, 3)
        groups = [error[bins == quartile] for quartile in range(4)]
        violin = axis.violinplot(groups, positions=np.arange(1, 5), widths=0.76, showextrema=False, showmeans=False, showmedians=False)
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#4B5563")
            body.set_linewidth(0.55)
            body.set_alpha(0.88)
        box = axis.boxplot(
            groups,
            positions=np.arange(1, 5),
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#111827", "linewidth": 0.95},
            whiskerprops={"color": "#4B5563", "linewidth": 0.65},
            capprops={"color": "#4B5563", "linewidth": 0.65},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("#4B5563")
            patch.set_linewidth(0.6)
        means = []
        low_errors = []
        high_errors = []
        for quartile in range(1, 5):
            row = ci_lookup[(factor, quartile)]
            mean = float(row["mean_relative_l2_velocity"])
            lo = float(row["ci95_lower_relative_l2_velocity"])
            hi = float(row["ci95_upper_relative_l2_velocity"])
            means.append(mean)
            low_errors.append(mean - lo)
            high_errors.append(hi - mean)
        axis.errorbar(np.arange(1, 5), means, yerr=np.asarray([low_errors, high_errors]), fmt="o", color="#E64B35", markerfacecolor="white", markeredgewidth=0.8, markersize=3.5, capsize=2, linewidth=0.8, zorder=4)
        rho, _ = spearman_with_approx_p(values, error)
        axis.text(0.97, 0.96, f"$\\rho_s$={rho:.2f}", transform=axis.transAxes, ha="right", va="top", fontsize=7.1, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.0})
        axis.set_title(title)
        axis.set_xticks(np.arange(1, 5), ["Q1", "Q2", "Q3", "Q4"])
        axis.set_ylabel(r"Geometry-mean velocity relative $L_2$")
        soften_axis(axis)
        add_panel_label(axis, panel)
        for row_index, (sample, value, err, bin_index) in enumerate(zip(samples, values, error, bins)):
            source_rows.append(
                {
                    "factor": factor,
                    "sample_id": sample["sample_id"],
                    "factor_value": value,
                    "quartile": int(bin_index) + 1,
                    "geometry_mean_velocity_relative_l2": err,
                    "quartile_lower": edges[int(bin_index)],
                    "quartile_upper": edges[int(bin_index) + 1],
                }
            )
    key_axis = axes.flat[-1]
    key_axis.set_axis_off()
    key_axis.legend(
        handles=[
            Patch(facecolor="#79BACA", edgecolor="#4B5563", label="Distribution"),
            Line2D([0], [0], marker="s", color="#4B5563", markerfacecolor="white", linestyle="", label="Median and IQR"),
            Line2D([0], [0], marker="o", color="#E64B35", markerfacecolor="white", label="Mean ± 95% bootstrap CI"),
        ],
        loc="upper left",
        title="Visual encoding",
    )
    key_axis.text(0.03, 0.55, "Quartiles and bootstrap intervals use\nthe same 764 held-out geometries.\nExact bounds and sample counts are\nprovided in the source-data CSV.", transform=key_axis.transAxes, va="top", color="#4B5563", linespacing=1.35)
    base = output / "topology_stratified_distributions_v2.png"
    files = save_publication_figure(fig, base)
    close_figure(fig)
    csv_path = base.with_suffix(".csv")
    write_rows(csv_path, source_rows)
    return {"label": base.stem, "files": [str(path) for path in files], "source": str(csv_path)}


def warm_start_figure(output: Path) -> dict[str, object]:
    runs = [
        ("Time-U-Net", FORMAL / "unet_time_seed2026" / "warm_start" / "warm_start_samples.csv"),
        ("CT-PoreFlow", FORMAL / "ct_poreflow_seed2026" / "warm_start" / "warm_start_samples.csv"),
    ]
    definitions = [
        ("iteration_reduction_fraction", "Iteration reduction (%)", 100.0, False, 0.0),
        ("end_to_end_wall_clock_speedup", "End-to-end speedup", 1.0, False, 1.0),
        ("refined_velocity_relative_l2", r"Refined velocity relative $L_2$", 1.0, True, None),
    ]
    loaded = {label: read_rows(path) for label, path in runs}
    colors = [method_style(label, index)["color"] for index, (label, _) in enumerate(runs)]
    rng = np.random.default_rng(20260805)
    apply_nature_style(font_size=8.5)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.68), constrained_layout=True)
    source_rows: list[dict[str, object]] = []
    for panel, axis, (metric, ylabel, scale, log_y, reference) in zip("abc", axes, definitions):
        groups = []
        for label, _ in runs:
            values = np.asarray([float(row[metric]) * scale for row in loaded[label]])
            groups.append(values)
            jitter = rng.normal(0.0, 0.045, size=values.size)
            axis.scatter(
                np.full(values.size, len(groups)) + jitter,
                values,
                s=5.2,
                color=colors[len(groups) - 1],
                alpha=0.22,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )
            for row, value in zip(loaded[label], values):
                source_rows.append({"model": label, "metric": metric, "sample_id": row["sample_id"], "display_value": value})
        violin = axis.violinplot(groups, positions=[1, 2], widths=0.68, showextrema=False, showmeans=False, showmedians=False)
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#4B5563")
            body.set_linewidth(0.65)
            body.set_alpha(0.45)
        box = axis.boxplot(
            groups,
            positions=[1, 2],
            widths=0.19,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#111827", "linewidth": 1.0},
            whiskerprops={"color": "#4B5563", "linewidth": 0.7},
            capprops={"color": "#4B5563", "linewidth": 0.7},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("#4B5563")
            patch.set_linewidth(0.65)
        for position, values, color in zip([1, 2], groups, colors):
            axis.scatter(position, float(np.mean(values)), s=25, marker="D", color=color, edgecolor="white", linewidth=0.7, zorder=4)
        if reference is not None:
            axis.axhline(reference, color="#6B7280", linestyle="--", linewidth=0.85, zorder=0)
        if metric == "iteration_reduction_fraction":
            axis.set_yscale("symlog", linthresh=20.0, linscale=1.0)
        if log_y:
            axis.set_yscale("log")
        axis.set_xticks([1, 2], ["Time-U-Net", "CT-PoreFlow"])
        axis.set_ylabel(ylabel)
        soften_axis(axis)
        add_panel_label(axis, panel)
    fig.legend(
        handles=[
            Patch(facecolor="#AAB2BD", edgecolor="#4B5563", alpha=0.45, label="Distribution"),
            Line2D([0], [0], marker="s", color="#4B5563", markerfacecolor="white", linestyle="", label="Median and IQR"),
            Line2D([0], [0], marker="D", color="#E64B35", markeredgecolor="white", linestyle="", label="Mean"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
    )
    base = output / "warm_start_outcomes_v2.png"
    files = save_publication_figure(fig, base)
    close_figure(fig)
    csv_path = base.with_suffix(".csv")
    write_rows(csv_path, source_rows)
    return {"label": base.stem, "files": [str(path) for path in files], "source": str(csv_path)}


def paired_effect_figure(output: Path) -> dict[str, object]:
    rows = read_rows(FORMAL / "analysis" / "ct_v2_matched_statistics" / "paired_tests.csv")
    baseline_order = ["Time-U-Net-matched", "Time-FNO", "DeepONet"]
    metric_order = [
        ("relative_l2_velocity", r"Velocity relative $L_2$"),
        ("mae_p_normalized", "Normalized pressure MAE"),
        ("mae_speed_lattice", "Speed MAE"),
        ("relative_error_permeability_final", "Permeability relative error"),
        ("relative_error_hydraulic_tortuosity_final", "Tortuosity relative error"),
    ]
    lookup = {(row["baseline"], row["metric"]): row for row in rows}
    matrix = np.asarray(
        [[float(lookup[(baseline, metric)]["matched_rank_biserial"]) for baseline in baseline_order] for metric, _ in metric_order]
    )
    apply_nature_style(font_size=8.3)
    fig, axis = plt.subplots(figsize=(5.45, 3.35), constrained_layout=True)
    image = axis.imshow(matrix, cmap="RdBu", vmin=-1.0, vmax=1.0, aspect="auto")
    for row_index, (metric, _) in enumerate(metric_order):
        for column, baseline in enumerate(baseline_order):
            result = lookup[(baseline, metric)]
            significant = result["holm_reject_at_0_05"].lower() == "true"
            value = matrix[row_index, column]
            color = "white" if abs(value) >= 0.58 else "#111827"
            axis.text(column, row_index, f"{value:+.2f}{'*' if significant else ''}", ha="center", va="center", color=color, fontsize=8.0, weight="bold" if significant else "normal")
    axis.set_xticks(
        np.arange(len(baseline_order)),
        ["Time-U-Net", "Time-FNO", "DeepONet"],
    )
    axis.set_yticks(np.arange(len(metric_order)), [label for _, label in metric_order])
    axis.set_xlabel("Baseline compared with CT-PoreFlow")
    axis.tick_params(axis="x", rotation=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.035)
    colorbar.set_label("Matched rank-biserial effect")
    axis.text(0.0, -0.19, "Positive: lower error for CT-PoreFlow    * Holm-adjusted $p<0.05$", transform=axis.transAxes, ha="left", va="top", fontsize=7.5, color="#4B5563")
    add_panel_label(axis, "a", x=-0.30, y=1.03)
    base = output / "paired_effect_heatmap_v2.png"
    files = save_publication_figure(fig, base)
    close_figure(fig)
    csv_path = base.with_suffix(".csv")
    write_rows(csv_path, [lookup[(baseline, metric)] for metric, _ in metric_order for baseline in baseline_order])
    return {"label": base.stem, "files": [str(path) for path in files], "source": str(csv_path)}


def temporal_holdout_figure(output: Path) -> dict[str, object]:
    metrics = [
        ("relative_l2_velocity", r"Velocity relative $L_2$"),
        ("mae_p_normalized", "Normalized pressure MAE"),
    ]
    source_rows: list[dict[str, object]] = []
    apply_nature_style(font_size=8.6)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    for axis in axes:
        add_stage_background(axis)
    for model_index, (label, run) in enumerate(HOLDOUT_MODELS):
        rows = read_rows(run / "evaluation" / "temporal_holdout" / "per_sample_time_metrics.csv")
        grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            iteration = int(row["iteration"])
            for metric, _ in metrics:
                value = finite_float(row[metric])
                if value is not None:
                    grouped[iteration][metric].append(value)
        style = method_style(label, model_index)
        for axis, (metric, _) in zip(axes, metrics):
            x: list[int] = []
            mean_values: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for iteration in sorted(grouped):
                values = np.asarray(grouped[iteration][metric])
                if values.size == 0:
                    continue
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if values.size > 1 else 0.0
                lo, hi = normal_mean_ci(mean, std, int(values.size))
                x.append(iteration)
                mean_values.append(mean)
                lower.append(max(lo, 0.0))
                upper.append(hi)
                source_rows.append(
                    {
                        "model": label,
                        "metric": metric,
                        "heldout_iteration": iteration,
                        "mean": mean,
                        "std": std,
                        "n_geometries": int(values.size),
                        "ci95_lower_normal": lo,
                        "ci95_upper_normal": hi,
                    }
                )
            axis.fill_between(x, lower, upper, color=style["color"], alpha=0.14, linewidth=0)
            axis.plot(x, mean_values, color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=2.0 if label == "CT-PoreFlow" else 1.7, markersize=3.5, label=label)
    for panel, axis, (_, ylabel) in zip("ab", axes, metrics):
        finish_time_axis(axis, ylabel)
        axis.set_xlabel("Held-out LBM iteration")
        add_panel_label(axis, panel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=2)
    base = output / "temporal_holdout_error_v2.png"
    files = save_publication_figure(fig, base)
    close_figure(fig)
    csv_path = base.with_suffix(".csv")
    write_rows(csv_path, source_rows)
    return {"label": base.stem, "files": [str(path) for path in files], "source": str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        temporal_field_figure(args.output_dir),
        physical_dynamics_figure(args.output_dir),
        calibration_figure(args.output_dir),
        efficiency_figure(args.output_dir),
        stratified_figure(args.output_dir),
        paired_effect_figure(args.output_dir),
        temporal_holdout_figure(args.output_dir),
    ]
    manifest = {
        "results_mode": "data_aware",
        "source_root": str(FORMAL),
        "style_basis": "Ref_code journal grammar: Arial, inward ticks, NPG palette, density scatter, uncertainty bands, editable vector export",
        "figures": [
            {"label": artifact["label"], "type": "results", "engine": "matplotlib", "source_data": artifact["source"], "files": artifact["files"]}
            for artifact in artifacts
        ],
    }
    manifest_path = args.output_dir / "figures.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
