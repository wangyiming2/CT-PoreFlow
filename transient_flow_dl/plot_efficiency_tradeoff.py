from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import (
    add_panel_label,
    apply_nature_style,
    close_figure,
    method_style,
    save_publication_figure,
    soften_axis,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pareto_latency_error(rows: list[dict]) -> list[dict]:
    front: list[dict] = []
    best_error = float("inf")
    for row in sorted(rows, key=lambda item: item["model_ms"]):
        if row["velocity_rel_l2"] < best_error:
            front.append(row)
            best_error = row["velocity_rel_l2"]
    return front


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="PNG output path; PDF and CSV are adjacent.")
    args = parser.parse_args()
    if len(args.run) != len(args.label):
        raise ValueError("--run and --label counts must match")
    rows = []
    for run, label in zip(args.run, args.label):
        summary = load_json(run / "evaluation" / "test" / "summary.json")
        model = load_json(run / "model.json")
        rows.append(
            {
                "label": label,
                "run": run.name,
                "parameters": int(model["parameters"]),
                "model_ms": float(summary["model_milliseconds_per_frame"]),
                "end_to_end_ms": float(summary["end_to_end_milliseconds_per_frame"]),
                "fps": float(summary["model_frames_per_second"]),
                "velocity_rel_l2": float(summary["relative_l2_velocity"]),
                "pressure_mae": float(summary["mae_p_normalized"]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    apply_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    max_latency = max(row["model_ms"] for row in rows)
    max_parameters = max(row["parameters"] / 1e6 for row in rows)
    max_error = max(row["velocity_rel_l2"] for row in rows)
    for index, row in enumerate(rows):
        style = method_style(row["label"], index)
        color = style["color"]
        marker = style["marker"]
        axes[0].scatter(
            row["model_ms"], row["velocity_rel_l2"], s=48, color=color, marker=marker,
            edgecolor="white", linewidth=0.55, zorder=3
        )
        axes[1].scatter(
            row["parameters"] / 1e6, row["velocity_rel_l2"], s=48, color=color, marker=marker,
            edgecolor="white", linewidth=0.55, zorder=3
        )
        for axis, x, x_max in (
            (axes[0], row["model_ms"], max_latency),
            (axes[1], row["parameters"] / 1e6, max_parameters),
        ):
            at_right_edge = x == x_max
            y_offset = -16 if row["velocity_rel_l2"] == max_error else (6 if index % 2 == 0 else -14)
            axis.annotate(
                row["label"],
                (x, row["velocity_rel_l2"]),
                xytext=(-6 if at_right_edge else 6, y_offset),
                textcoords="offset points",
                fontsize=6.8,
                ha="right" if at_right_edge else "left",
                color="#222831",
            )

    front = pareto_latency_error(rows)
    if len(front) > 1:
        axes[0].plot(
            [row["model_ms"] for row in front],
            [row["velocity_rel_l2"] for row in front],
            linestyle="--",
            linewidth=1.0,
            color="#6B7280",
            label="Latency-error Pareto front",
            zorder=2,
        )
        axes[0].legend(loc="upper right")

    axes[0].set_xlabel("Model-only latency (ms per frame; lower is faster)")
    axes[1].set_xlabel("Trainable parameters (million)")
    axes[0].set_ylabel("Velocity relative $L_2$ (lower is better)")
    axes[1].set_ylabel("Velocity relative $L_2$ (lower is better)")
    axes[0].set_xscale("log")
    errors = [row["velocity_rel_l2"] for row in rows]
    error_span = max(errors) - min(errors)
    lower = max(0.0, min(errors) - max(0.06, 0.12 * error_span))
    upper = max(errors) + max(0.06, 0.12 * error_span)
    for panel, axis in zip("ab", axes):
        axis.set_ylim(lower, upper)
        axis.margins(x=0.08, y=0.08)
        soften_axis(axis)
        add_panel_label(axis, panel)
    save_publication_figure(fig, args.output)
    close_figure(fig)
    print(args.output)
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".svg"))
    print(csv_path)


if __name__ == "__main__":
    main()
