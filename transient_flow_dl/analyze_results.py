from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    "relative_l2_velocity",
    "mae_speed_lattice",
    "mae_p_normalized",
]
PHYSICS_METRICS = [
    "normalized_mean_flux_error",
    "pressure_drop_mae_normalized",
    "pred_divergence_l1_lattice",
    "relative_error_hydraulic_tortuosity",
]
ALL_METRICS = METRICS + PHYSICS_METRICS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    source = args.run_dir / "evaluation" / args.split / "per_sample_time_metrics.csv"
    groups: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    records: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    final_flux: dict[str, float] = {}
    final_iteration: dict[str, int] = {}
    for row in records:
        sample_id = row["sample_id"]
        iteration = int(row["iteration"])
        if iteration >= final_iteration.get(sample_id, -1):
            final_iteration[sample_id] = iteration
            final_flux[sample_id] = float(row["true_mean_flux_lattice"])
    for row in records:
        iteration = int(row["iteration"])
        for metric in ALL_METRICS:
            if metric == "normalized_mean_flux_error":
                denominator = max(abs(final_flux[row["sample_id"]]), np.finfo(float).eps)
                value = abs(
                    float(row["pred_mean_flux_lattice"])
                    - float(row["true_mean_flux_lattice"])
                ) / denominator
            else:
                value = float(row[metric])
            if np.isfinite(value):
                groups[iteration][metric].append(value)

    output = args.run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for iteration in sorted(groups):
        row: dict[str, float | int] = {"iteration": iteration}
        for metric in ALL_METRICS:
            values = np.asarray(groups[iteration][metric], dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
            row[f"{metric}_n"] = int(values.size)
        rows.append(row)
    table_path = output / "per_time_metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.log1p([row["iteration"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    labels = ["Velocity relative L2", "Speed MAE (lattice)", "Pressure-proxy MAE"]
    for axis, metric, label in zip(axes, METRICS, labels):
        mean = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float)
        std = np.asarray([row[f"{metric}_std"] for row in rows], dtype=float)
        axis.plot(x, mean, color="#2468b4", linewidth=2)
        axis.fill_between(x, np.maximum(0.0, mean - std), mean + std, color="#2468b4", alpha=0.18)
        axis.set_xlabel("log(1 + LBM iteration)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure_path = output / "error_vs_time.png"
    fig.savefig(figure_path, dpi=300)
    fig.savefig(output / "error_vs_time.pdf")
    plt.close(fig)

    physics_labels = [
        "Flux error / final true flux",
        "Pressure-drop MAE (normalized)",
        "Predicted divergence L1 (lattice)",
        "Tortuosity relative error",
    ]
    physics_fig, physics_axes = plt.subplots(2, 2, figsize=(9.0, 6.5), constrained_layout=True)
    for axis, metric, label in zip(physics_axes.flat, PHYSICS_METRICS, physics_labels):
        mean = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float)
        std = np.asarray([row[f"{metric}_std"] for row in rows], dtype=float)
        axis.plot(x, mean, color="#2A9D8F", linewidth=2)
        axis.fill_between(x, np.maximum(0.0, mean - std), mean + std, color="#2A9D8F", alpha=0.18)
        axis.set_xlabel("log(1 + LBM iteration)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    physics_path = output / "physics_vs_time.png"
    physics_fig.savefig(physics_path, dpi=300)
    physics_fig.savefig(output / "physics_vs_time.pdf")
    plt.close(physics_fig)
    print(table_path)
    print(figure_path)
    print(physics_path)


if __name__ == "__main__":
    main()
