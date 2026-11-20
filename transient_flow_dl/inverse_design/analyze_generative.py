from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.lines import Line2D


COLORS = {"cgan": "#0072B2", "diffusion": "#D55E00"}
MARKERS = {"guided": "o", "random": "s"}
DISPLAY = {"cgan": "Conditional GAN", "diffusion": "Conditional diffusion"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.7,
            "axes.labelsize": 8.9,
            "axes.titlesize": 9.4,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def group_number(target_id: str) -> int:
    return int(target_id[1])


def summarize(diversity: list[dict[str, str]], lbm: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    generation_rows: list[dict[str, object]] = []
    for method in ("cgan", "diffusion"):
        rows = [row for row in diversity if row["method"] == method]
        generation_rows.append(
            {
                "method": method,
                "generated": sum(int(row["generated_count"]) for row in rows),
                "targets": len(rows),
                "mean_spanning_rate": float(np.mean([float(row["spanning_rate"]) for row in rows])),
                "mean_unique_rate": float(np.mean([float(row["unique_rate"]) for row in rows])),
                "mean_pairwise_hamming": float(np.mean([float(row["pairwise_hamming_mean"]) for row in rows])),
                "mean_proxy_distance": float(np.mean([float(row["mean_proxy_target_distance"]) for row in rows])),
                "mean_best_proxy_distance": float(np.mean([float(row["best_proxy_target_distance"]) for row in rows])),
                "total_sampling_screening_seconds": float(sum(float(row["sampling_and_screening_seconds"]) for row in rows)),
                "milliseconds_per_candidate": 1000.0 * float(sum(float(row["sampling_and_screening_seconds"]) for row in rows)) / sum(int(row["generated_count"]) for row in rows),
            }
        )

    verification_rows: list[dict[str, object]] = []
    for method in ("cgan", "diffusion"):
        for strategy in ("guided", "random"):
            rows = [row for row in lbm if row["method"] == method and row["strategy"] == strategy]
            if not rows:
                continue
            passing = [row for row in rows if row["quality_pass"].lower() == "true"]
            distances = np.asarray([float(row["verified_target_distance"]) for row in passing])
            k_error = np.asarray(
                [abs(float(row["verified_permeability_lattice"]) - float(row["target_permeability_lattice"])) / float(row["target_permeability_lattice"]) for row in passing]
            )
            tau_error = np.asarray(
                [abs(float(row["verified_hydraulic_tortuosity"]) - float(row["target_hydraulic_tortuosity"])) / float(row["target_hydraulic_tortuosity"]) for row in passing]
            )
            phi_error = np.asarray([abs(float(row["selected_porosity"]) - float(row["target_porosity"])) for row in passing])
            success_20 = (phi_error <= 0.03) & (k_error <= 0.20) & (tau_error <= 0.10)
            verification_rows.append(
                {
                    "method": method,
                    "strategy": strategy,
                    "assigned": len(rows),
                    "qc_passing": len(passing),
                    "quality_pass_rate": float(np.mean([row["quality_pass"].lower() == "true" for row in rows])),
                    "mean_verified_distance": float(distances.mean()),
                    "median_verified_distance": float(np.median(distances)),
                    "mean_permeability_relative_error": float(k_error.mean()),
                    "mean_tortuosity_relative_error": float(tau_error.mean()),
                    "success_20_rate": float(success_20.mean()),
                    "mean_lbm_seconds_all_assigned": float(np.mean([float(row["seconds"]) for row in rows])),
                }
            )
    return generation_rows, verification_rows


def exact_sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        count += statistic >= observed - 1e-15
        total += 1
    return count / total


def paired_statistics(rows: list[dict[str, str]], seed: int) -> dict[str, object]:
    result: dict[str, object] = {}
    rng = np.random.default_rng(seed)
    for method in ("cgan", "diffusion"):
        target_values: defaultdict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["method"] == method and row["quality_pass"].lower() == "true":
                target_values[row["target_id"]][row["strategy"]].append(float(row["verified_target_distance"]))
        differences = []
        per_target = []
        for target, strategies in sorted(target_values.items()):
            guided = float(np.mean(strategies["guided"]))
            random = float(np.mean(strategies["random"]))
            differences.append(random - guided)
            per_target.append({"target_id": target, "guided": guided, "random": random, "reduction": random - guided})
        difference = np.asarray(differences)
        indexes = rng.integers(0, len(difference), size=(20_000, len(difference)))
        bootstrap = difference[indexes].mean(axis=1)
        result[method] = {
            "targets": len(difference),
            "mean_guided_distance": float(np.mean([row["guided"] for row in per_target])),
            "mean_random_distance": float(np.mean([row["random"] for row in per_target])),
            "mean_distance_reduction": float(difference.mean()),
            "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
            "exact_sign_flip_p": exact_sign_flip(difference),
            "per_target": per_target,
        }
    return result


def augment_mach_gate(rows: list[dict[str, str]], root: Path) -> list[dict[str, str]]:
    sound_speed = np.sqrt(1.0 / 3.0)
    enriched: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        with np.load(root / "lbm_fields" / f"{row['candidate_id']}.npz") as fields:
            ux = np.asarray(fields["velocity_x_lattice"], dtype=np.float64)
            uy = np.asarray(fields["velocity_y_lattice"], dtype=np.float64)
        max_mach = float(np.sqrt(ux * ux + uy * uy).max() / sound_speed)
        original = row["quality_pass"].lower() == "true"
        row["quality_pass_without_mach"] = str(original)
        row["max_mach"] = f"{max_mach:.12g}"
        row["quality_pass"] = str(original and max_mach < 0.10)
        enriched.append(row)
    return enriched


def plot_generation(diversity: list[dict[str, str]], output: Path) -> None:
    paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), constrained_layout=True)
    target_order = sorted({row["target_id"] for row in diversity}, key=lambda x: (group_number(x), x))
    x = np.arange(len(target_order))
    for method in ("cgan", "diffusion"):
        lookup = {row["target_id"]: row for row in diversity if row["method"] == method}
        axes[0, 0].plot(x, [100 * float(lookup[target]["spanning_rate"]) for target in target_order], marker="o", ms=3, lw=1.2, color=COLORS[method], label=DISPLAY[method])
        axes[0, 1].plot(x, [float(lookup[target]["best_proxy_target_distance"]) for target in target_order], marker="o", ms=3, lw=1.2, color=COLORS[method], label=DISPLAY[method])
        axes[1, 0].scatter(
            [float(lookup[target]["pairwise_hamming_mean"]) for target in target_order],
            [float(lookup[target]["mean_proxy_target_distance"]) for target in target_order],
            s=24,
            color=COLORS[method],
            alpha=0.8,
            label=DISPLAY[method],
        )
    for boundary in (5.5, 11.5):
        axes[0, 0].axvline(boundary, color="#BBBBBB", lw=0.7, ls="--")
        axes[0, 1].axvline(boundary, color="#BBBBBB", lw=0.7, ls="--")
    axes[0, 0].set(ylabel="Through-connected samples (%)", ylim=(-2, 104), xticks=[2.5, 8.5, 14.5], xticklabels=["Low $\\phi$", "Mid $\\phi$", "High $\\phi$"])
    axes[0, 1].set(ylabel="Best surrogate target distance", xticks=[2.5, 8.5, 14.5], xticklabels=["Low $\\phi$", "Mid $\\phi$", "High $\\phi$"])
    axes[1, 0].set(xlabel="Pairwise binary Hamming distance", ylabel="Mean surrogate target distance")
    generation, _ = summarize(diversity, [])
    positions = np.arange(2)
    axes[1, 1].bar(positions - 0.18, [100 * row["mean_spanning_rate"] for row in generation], 0.36, color=[COLORS[row["method"]] for row in generation], alpha=0.9)
    diversity_axis = axes[1, 1].twinx()
    diversity_axis.plot(positions + 0.18, [row["mean_pairwise_hamming"] for row in generation], marker="D", color="#222222", lw=1.2)
    axes[1, 1].set(xticks=positions, xticklabels=["cGAN", "Diffusion"], ylabel="Mean spanning rate (%)", ylim=(0, 105))
    diversity_axis.set_ylabel("Mean Hamming diversity")
    axes[0, 0].set_title("a  Hydraulic validity across targets", loc="left", fontweight="bold")
    axes[0, 1].set_title("b  Best attainable surrogate match", loc="left", fontweight="bold")
    axes[1, 0].set_title("c  Diversity--fidelity trade-off", loc="left", fontweight="bold")
    axes[1, 1].set_title("d  Aggregate validity and diversity", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    for axis in axes.ravel():
        axis.grid(True, color="#E5E5E5", lw=0.45)
    save_figure(fig, output, "generative_inverse_quality")


def _histogram_with_gradient(
    axis: plt.Axes,
    values: np.ndarray,
    bins: np.ndarray,
    orientation: str,
    cmap_name: str,
) -> None:
    counts, _, patches = axis.hist(
        values,
        bins=bins,
        orientation=orientation,
        density=False,
        edgecolor="#222222",
        linewidth=0.25,
    )
    scale = max(float(np.max(counts)) if counts.size else 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    for count, patch in zip(counts, patches):
        patch.set_facecolor(cmap(0.22 + 0.70 * float(count) / scale))


def _round_trip_metrics(x: np.ndarray, y: np.ndarray, log_scale: bool) -> tuple[float, float, float, np.ndarray]:
    if log_scale:
        metric_x = np.log10(x)
        metric_y = np.log10(y)
    else:
        metric_x = x
        metric_y = y
    residual = metric_y - metric_x
    denominator = float(np.sum((metric_x - np.mean(metric_x)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0.0 else float("nan")
    return r_squared, float(np.sqrt(np.mean(residual**2))), float(np.mean(np.abs(residual))), residual


def _plot_round_trip_reference(
    fig: plt.Figure,
    spec,
    rows: list[dict[str, str]],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    log_scale: bool,
) -> None:
    selected = [row for row in rows if row["quality_pass"].lower() == "true"]
    x = np.asarray([float(row[x_key]) for row in selected], dtype=np.float64)
    y = np.asarray([float(row[y_key]) for row in selected], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if log_scale:
        finite &= (x > 0.0) & (y > 0.0)
    x, y = x[finite], y[finite]
    selected = [row for row, keep in zip(selected, finite) if keep]
    r_squared, rmse, mae, residual = _round_trip_metrics(x, y, log_scale)

    grid = spec.subgridspec(
        3,
        2,
        width_ratios=(6.0, 1.15),
        height_ratios=(1.15, 6.0, 2.0),
        hspace=0.06,
        wspace=0.05,
    )
    ax_hist_x = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[1, 0], sharex=ax_hist_x)
    ax_hist_y = fig.add_subplot(grid[1, 1], sharey=ax_main)
    ax_residual = fig.add_subplot(grid[2, 0], sharex=ax_main)
    for empty_spec in (grid[0, 1], grid[2, 1]):
        fig.add_subplot(empty_spec).axis("off")

    pooled_min = float(min(np.min(x), np.min(y)))
    pooled_max = float(max(np.max(x), np.max(y)))
    if log_scale:
        plot_min = pooled_min / 1.18
        plot_max = pooled_max * 1.18
        bins = np.geomspace(plot_min, plot_max, 24)
    else:
        span = max(pooled_max - pooled_min, 1.0e-6)
        plot_min = max(0.0, pooled_min - 0.06 * span)
        plot_max = pooled_max + 0.06 * span
        bins = np.linspace(plot_min, plot_max, 24)

    for method in ("cgan", "diffusion"):
        for strategy in ("guided", "random"):
            indexes = [
                index
                for index, row in enumerate(selected)
                if row["method"] == method and row["strategy"] == strategy
            ]
            ax_main.scatter(
                x[indexes],
                y[indexes],
                s=15,
                color=COLORS[method],
                marker=MARKERS[strategy],
                alpha=0.68,
                edgecolors="#222222",
                linewidths=0.22,
                zorder=3,
            )
            ax_residual.scatter(
                x[indexes],
                residual[indexes],
                s=10,
                color=COLORS[method],
                marker=MARKERS[strategy],
                alpha=0.62,
                edgecolors="#222222",
                linewidths=0.18,
                zorder=3,
            )

    identity = np.geomspace(plot_min, plot_max, 200) if log_scale else np.linspace(plot_min, plot_max, 200)
    ax_main.plot(identity, identity, color="#222222", linestyle="--", linewidth=0.9, zorder=4)
    ax_main.set_xlim(plot_min, plot_max)
    ax_main.set_ylim(plot_min, plot_max)
    ax_main.set_ylabel(y_label, fontweight="bold")
    ax_main.tick_params(axis="x", labelbottom=False)
    ax_main.grid(True, color="#E5E5E5", linewidth=0.4)

    _histogram_with_gradient(ax_hist_x, x, bins, "vertical", "Blues")
    _histogram_with_gradient(ax_hist_y, y, bins, "horizontal", "Oranges")
    ax_hist_x.set_title(title, loc="left", fontweight="bold", pad=3)
    ax_hist_x.tick_params(labelbottom=False, labelleft=False, length=0)
    ax_hist_y.tick_params(labelbottom=False, labelleft=False, length=0)
    for spine in ("top", "right", "left"):
        ax_hist_x.spines[spine].set_visible(False)
    for spine in ("top", "right", "bottom"):
        ax_hist_y.spines[spine].set_visible(False)

    if log_scale:
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_hist_x.set_xscale("log")
        ax_hist_y.set_yscale("log")
        residual_label = r"$\log_{10}$(LBM/surrogate)"
        metric_unit = " log units"
    else:
        residual_label = "LBM - surrogate"
        metric_unit = ""
    residual_limit = max(float(np.quantile(np.abs(residual), 0.98)) * 1.15, 1.0e-3)
    residual_std = float(np.std(residual, ddof=1))
    ax_residual.axhline(0.0, color="#222222", linestyle="--", linewidth=0.75)
    for multiplier in (-2.0, -1.0, 1.0, 2.0):
        ax_residual.axhline(multiplier * residual_std, color="#999999", linestyle="--", linewidth=0.45)
    ax_residual.set_ylim(-residual_limit, residual_limit)
    ax_residual.set_xlabel(x_label, fontweight="bold")
    ax_residual.set_ylabel(residual_label, fontweight="bold")
    ax_residual.grid(True, color="#EAEAEA", linewidth=0.35)

    metrics = (
        f"QC-passing candidates\n"
        f"$R^2$={r_squared:.3f}\n"
        f"RMSE={rmse:.3f}{metric_unit}\n"
        f"MAE={mae:.3f}{metric_unit}\n"
        f"$n$={len(x)}"
    )
    ax_main.text(
        0.035,
        0.965,
        metrics,
        transform=ax_main.transAxes,
        va="top",
        fontsize=7.4,
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#222222", "linewidth": 0.65},
        zorder=8,
    )
    inset = ax_main.inset_axes([0.59, 0.055, 0.37, 0.16])
    box = inset.boxplot(
        [residual],
        vert=False,
        patch_artist=True,
        widths=0.58,
        showfliers=True,
        medianprops={"color": "#222222", "linewidth": 0.8},
        boxprops={"edgecolor": "#222222", "linewidth": 0.65},
        whiskerprops={"color": "#222222", "linewidth": 0.6},
        capprops={"color": "#222222", "linewidth": 0.6},
        flierprops={"marker": "o", "markersize": 1.5, "markerfacecolor": "#999999", "markeredgewidth": 0.0},
    )
    box["boxes"][0].set_facecolor("#9ECAE1")
    inset.axvline(0.0, color="#222222", linestyle="--", linewidth=0.6)
    inset.set_xlim(-residual_limit, residual_limit)
    inset.set_yticks([])
    inset.tick_params(axis="x", labelsize=5.1, pad=1)
    inset.set_title("Residuals", fontsize=7.0, pad=1)

def plot_lbm(rows: list[dict[str, str]], statistics: dict[str, object], output: Path) -> None:
    paper_style()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix"})
    fig = plt.figure(figsize=(7.15, 4.65), constrained_layout=True)
    outer = fig.add_gridspec(2, 2, height_ratios=(0.12, 1.0), hspace=0.02)
    legend_axis = fig.add_subplot(outer[0, :])
    legend_axis.axis("off")
    legend_handles = [
        Line2D([], [], color=COLORS["cgan"], marker="o", linestyle="None", markersize=4.5, label="GAN guided"),
        Line2D([], [], color=COLORS["cgan"], marker="s", linestyle="None", markersize=4.5, label="GAN random"),
        Line2D([], [], color=COLORS["diffusion"], marker="o", linestyle="None", markersize=4.5, label="Diff. guided"),
        Line2D([], [], color=COLORS["diffusion"], marker="s", linestyle="None", markersize=4.5, label="Diff. random"),
    ]
    legend_axis.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        frameon=False,
        fontsize=8.0,
        handletextpad=0.35,
        columnspacing=1.4,
    )
    _plot_round_trip_reference(
        fig,
        outer[1, 0],
        rows,
        "predicted_permeability_lattice",
        "verified_permeability_lattice",
        "a  Permeability round trip",
        "Surrogate permeability",
        "LBM permeability",
        True,
    )
    _plot_round_trip_reference(
        fig,
        outer[1, 1],
        rows,
        "predicted_hydraulic_tortuosity",
        "verified_hydraulic_tortuosity",
        "b  Tortuosity round trip",
        "Surrogate tortuosity",
        "LBM tortuosity",
        False,
    )
    save_figure(fig, output, "generative_inverse_lbm")


def _plot_predicted_streamlines(
    axis: plt.Axes,
    ux: np.ndarray,
    uy: np.ndarray,
    pore: np.ndarray,
    speed_scale: float,
) -> plt.AxesImage:
    # Stored arrays place the pressure-driven x direction on image axis 0.
    # A 90-degree counter-clockwise rotation moves the inlet to the left.
    ux_display = np.rot90(ux, k=1)
    uy_display = np.rot90(uy, k=1)
    pore_display = np.rot90(pore.astype(bool), k=1)
    speed = np.sqrt(ux_display * ux_display + uy_display * uy_display)
    image = axis.imshow(speed, cmap="turbo", vmin=0.0, vmax=speed_scale, origin="upper")
    height, width = speed.shape
    # After the same spatial rotation, physical +x points right and physical
    # +y points upward; streamplot's displayed vertical coordinate points down.
    display_u = np.ma.masked_where(~pore_display, ux_display)
    display_v = np.ma.masked_where(~pore_display, -uy_display)
    stream = axis.streamplot(
        np.arange(width),
        np.arange(height),
        display_u,
        display_v,
        density=0.62,
        color="#FFFFFF",
        linewidth=0.55,
        arrowsize=0.42,
        minlength=0.12,
        maxlength=4.0,
        integration_direction="both",
        zorder=2,
    )
    halo = [path_effects.Stroke(linewidth=1.15, foreground="#111111", alpha=0.60), path_effects.Normal()]
    stream.lines.set_path_effects(halo)
    stream.arrows.set_path_effects(halo)
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    return image


def plot_mosaic(rows: list[dict[str, str]], root: Path, output: Path) -> None:
    paper_style()
    target_ids = ("P1-KH-TL", "P2-KM-TL", "P3-KH-TL")
    cases: list[dict[str, str]] = []
    for target in target_ids:
        for method in ("cgan", "diffusion"):
            options = [
                row
                for row in rows
                if row["target_id"] == target
                and row["method"] == method
                and row["strategy"] == "guided"
                and row["quality_pass"].lower() == "true"
                and (root / "lbm_fields" / f"{row['candidate_id']}.npz").exists()
            ]
            if not options:
                options = [
                    row
                    for row in rows
                    if row["target_id"] == target
                    and row["method"] == method
                    and row["strategy"] == "guided"
                    and (root / "lbm_fields" / f"{row['candidate_id']}.npz").exists()
                ]
            cases.append(min(options, key=lambda row: float(row["verified_target_distance"])))
    prediction_path = output / "fig17_ct_predictions.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing CT-PoreFlow field predictions: {prediction_path}. "
            "Run export_fig17_ct_predictions.py before plotting Fig. 17."
        )
    with np.load(prediction_path) as predictions:
        prediction_ids = [str(value) for value in np.asarray(predictions["candidate_ids"])]
        predicted_ux = np.asarray(predictions["velocity_x_lattice"], dtype=np.float64)
        predicted_uy = np.asarray(predictions["velocity_y_lattice"], dtype=np.float64)
        speed_scale = float(np.asarray(predictions["velocity_scale"]))
        checkpoint_name = str(np.asarray(predictions["checkpoint"]))
    if checkpoint_name != "ct_poreflow_v2_transport_seed2026":
        raise ValueError(f"Fig. 17 requires the audited CT-PoreFlow checkpoint, found {checkpoint_name}")
    prediction_index = {candidate_id: index for index, candidate_id in enumerate(prediction_ids)}
    missing_predictions = [row["candidate_id"] for row in cases if row["candidate_id"] not in prediction_index]
    if missing_predictions:
        raise ValueError(f"CT-PoreFlow predictions do not match the plotted candidates: {missing_predictions}")

    # Five compact columns: target/property text, geometry, CT-PoreFlow
    # streamlines, LBM speed, and LBM pressure.  The speed
    # colorbar is shared by the prediction and LBM columns.
    fig = plt.figure(figsize=(7.15, 7.72))
    grid = fig.add_gridspec(
        len(cases) + 1,
        5,
        width_ratios=(1.08, 1.0, 1.0, 1.0, 1.0),
        height_ratios=(*([1.0] * len(cases)), 0.055),
        left=0.012,
        right=0.992,
        bottom=0.072,
        top=0.956,
        wspace=0.060,
        hspace=0.040,
    )
    axes = np.empty((len(cases), 5), dtype=object)
    for row_index in range(len(cases)):
        for column_index in range(5):
            axes[row_index, column_index] = fig.add_subplot(grid[row_index, column_index])
    speed_cax = fig.add_subplot(grid[-1, 2:4])
    pressure_cax = fig.add_subplot(grid[-1, 4])
    speed_image = None
    pressure_image = None
    for index, row in enumerate(cases):
        method = row["method"]
        target = row["target_id"]
        with np.load(root / "generated" / method / f"{target}.npz") as generated:
            pore = np.asarray(generated["pore_masks"][int(row["candidate_index"])])
        with np.load(root / "lbm_fields" / f"{row['candidate_id']}.npz") as fields:
            ux = np.asarray(fields["velocity_x_lattice"])
            uy = np.asarray(fields["velocity_y_lattice"])
            density = np.asarray(fields["density_lattice"])
        speed = np.sqrt(ux * ux + uy * uy)
        pressure = np.clip((density - 1.0) / 1.0e-5, 0.0, 1.0)
        prediction_row = prediction_index[row["candidate_id"]]
        axes[index, 0].axis("off")
        axes[index, 0].text(0.0, 0.98, f"{DISPLAY[method]}\n{target}", va="top", fontsize=8.1, fontweight="bold", color=COLORS[method], transform=axes[index, 0].transAxes)
        axes[index, 0].text(0.0, 0.61, f"Target $k$={float(row['target_permeability_lattice']):.3f}\nTarget $\\tau_h$={float(row['target_hydraulic_tortuosity']):.3f}\nLBM $k$={float(row['verified_permeability_lattice']):.3f}\nLBM $\\tau_h$={float(row['verified_hydraulic_tortuosity']):.3f}\n$D$={float(row['verified_target_distance']):.3f}", va="top", fontsize=7.4, transform=axes[index, 0].transAxes, linespacing=1.12)
        # Dataset convention: pore=1 and solid=0.  Standard grayscale therefore
        # renders pore space white and the solid skeleton black.
        axes[index, 1].imshow(np.rot90(pore, k=1), cmap="gray", vmin=0, vmax=1)
        _plot_predicted_streamlines(
            axes[index, 2],
            predicted_ux[prediction_row],
            predicted_uy[prediction_row],
            pore,
            speed_scale,
        )
        speed_image = axes[index, 3].imshow(np.rot90(speed, k=1), cmap="turbo", vmin=0, vmax=speed_scale)
        pressure_image = axes[index, 4].imshow(np.rot90(pressure, k=1), cmap="viridis", vmin=0, vmax=1)
        for column in (1, 2, 3, 4):
            axes[index, column].set_xticks([]); axes[index, column].set_yticks([])
    axes[0, 1].set_title("Generated\npore mask", linespacing=0.95)
    axes[0, 2].set_title("CT-PoreFlow\npredicted streamlines", linespacing=0.95)
    axes[0, 3].set_title("LBM\nspeed", linespacing=0.95)
    axes[0, 4].set_title("LBM\npressure $p^*$", linespacing=0.95)
    speed_bar = fig.colorbar(
        speed_image,
        cax=speed_cax,
        orientation="horizontal",
        ticks=(0.0, 2.0e-6, 4.0e-6),
    )
    speed_bar.set_label("Speed (lattice units; shared scale)", labelpad=1.5)
    speed_bar.ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
    speed_bar.ax.xaxis.get_offset_text().set_fontsize(7.5)

    pressure_bar = fig.colorbar(
        pressure_image,
        cax=pressure_cax,
        orientation="horizontal",
        ticks=(0.0, 0.5, 1.0),
    )
    pressure_bar.set_label("Normalized pressure", labelpad=1.5)
    pressure_bar.ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
    save_figure(fig, output, "generative_inverse_mosaic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Optional root containing lbm_fields/ when the audited CSV snapshot is stored separately.",
    )
    parser.add_argument(
        "--enriched-csv",
        type=Path,
        help="Use an existing audited enriched LBM table instead of recomputing the Mach gate.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    diversity = read_csv(args.input_root / "diversity_summary.csv")
    asset_root = args.asset_root or args.input_root
    lbm = read_csv(args.enriched_csv) if args.enriched_csv else augment_mach_gate(
        read_csv(args.input_root / "lbm_verification.csv"), asset_root
    )
    generation, verification = summarize(diversity, lbm)
    statistics = paired_statistics(lbm, args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "generative_inverse_generation_summary.csv", generation)
    write_csv(args.output_root / "generative_inverse_lbm_summary.csv", verification)
    write_csv(args.output_root / "generative_inverse_lbm_enriched.csv", lbm)
    (args.output_root / "generative_inverse_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    plot_generation(diversity, args.output_root)
    plot_lbm(lbm, statistics, args.output_root)
    plot_mosaic(lbm, asset_root, args.output_root)
    print(json.dumps({"status": "complete", "generation_rows": len(generation), "verification_rows": len(verification), "lbm_rows": len(lbm)}))


if __name__ == "__main__":
    main()
