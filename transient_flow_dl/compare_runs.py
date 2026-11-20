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


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != len(args.label):
        raise SystemExit("--run and --label counts must match")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    physics_fig, physics_axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    for index, (run, label) in enumerate(zip(args.run, args.label)):
        rows = read_table(run / "analysis" / "per_time_metrics.csv")
        x = np.asarray([int(row["iteration"]) for row in rows], dtype=float)
        velocity = [float(row["relative_l2_velocity_mean"]) for row in rows]
        pressure = [float(row["mae_p_normalized_mean"]) for row in rows]
        style = method_style(label, index)
        plot_style = {
            **style,
            "linewidth": 1.6 if label != "CT-PoreFlow" else 2.0,
            "markersize": 3.1,
            "markevery": 3,
            "label": label,
        }
        axes[0].plot(x, velocity, **plot_style)
        axes[1].plot(x, pressure, **plot_style)
        physics_axes[0].plot(
            x,
            [float(row["normalized_mean_flux_error_mean"]) for row in rows],
            **plot_style,
        )
        physics_axes[1].plot(
            x, [float(row["pred_divergence_l1_lattice_mean"]) for row in rows], **plot_style
        )
        physics_axes[2].plot(
            x,
            [float(row["relative_error_hydraulic_tortuosity_mean"]) for row in rows],
            **plot_style,
        )
    axes[0].set_ylabel(r"Velocity relative $L_2$")
    axes[1].set_ylabel("Normalized pressure MAE")
    # Keep the terminal 50,000-iteration anchor explicit without crowding the
    # adjacent decade label after the three-panel figure is set at journal width.
    time_ticks = [0, 10, 100, 1_000, 50_000]
    time_labels = ["0", "10", r"$10^2$", r"$10^3$", r"$5\times10^4$"]
    for panel, axis in zip("ab", axes):
        axis.set_xlabel("LBM iteration")
        axis.set_xscale("symlog", linthresh=1.0)
        axis.set_xticks(time_ticks, time_labels)
        axis.margins(y=0.08)
        soften_axis(axis)
        add_panel_label(axis, panel)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=min(4, len(legend_labels)))
    png = args.output_dir / "baseline_error_vs_time.png"
    save_publication_figure(fig, png)
    close_figure(fig)
    physics_axes[0].set_ylabel("Flux error / final true flux")
    physics_axes[1].set_ylabel(r"Divergence $L_1$ (lattice)")
    physics_axes[2].set_ylabel("Tortuosity relative error")
    for panel, axis in zip("abc", physics_axes):
        axis.set_xlabel("LBM iteration")
        axis.set_xscale("symlog", linthresh=1.0)
        axis.set_xticks(time_ticks, time_labels)
        axis.margins(y=0.08)
        soften_axis(axis)
        add_panel_label(axis, panel)
    handles, legend_labels = physics_axes[0].get_legend_handles_labels()
    physics_fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=min(4, len(legend_labels)),
    )
    physics_png = args.output_dir / "baseline_physics_vs_time.png"
    save_publication_figure(physics_fig, physics_png)
    close_figure(physics_fig)
    print(png)
    print(physics_png)


if __name__ == "__main__":
    main()
