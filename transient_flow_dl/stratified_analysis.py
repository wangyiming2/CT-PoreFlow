from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import (
    add_panel_label,
    apply_nature_style,
    close_figure,
    save_publication_figure,
    soften_axis,
)
from src.utils import load_json


FACTORS = [
    ("porosity", "Total porosity"),
    ("connected_porosity", "Connected porosity"),
    ("dead_pore_ratio", "Dead-pore ratio"),
    ("hydraulic_tortuosity", "Hydraulic tortuosity"),
    ("iterations", "LBM convergence iterations"),
]


def _read_metadata(steady_root: Path, split: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in sorted((steady_root / "samples" / split).rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample_file = str(payload.get("sample_file", "")).replace("\\", "/")
        if sample_file:
            sample_id = sample_file.split("samples/", 1)[-1]
        else:
            relative = path.relative_to(steady_root / "samples").with_suffix(".npz")
            sample_id = relative.as_posix()
        pore_fraction = float(payload["porosity"])
        connected = float(payload["connected_porosity"])
        payload["dead_pore_ratio"] = (pore_fraction - connected) / max(pore_fraction, 1e-30)
        result[sample_id] = payload
    if not result:
        raise RuntimeError(f"No metadata found under {steady_root / 'samples' / split}")
    return result


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _mean(values: list[float]) -> float:
    array = _finite(values)
    return float(array.mean()) if array.size else float("nan")


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int = 10_000,
) -> tuple[float, float]:
    """Geometry-bootstrap percentile interval for a quartile mean."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    boot_means = np.empty(draws, dtype=float)
    chunk = 1_000
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        boot_means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(boot_means, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    config = load_json(args.run_dir / "config.json")
    rng = np.random.default_rng(int(config.get("seed", 2026)))
    metadata = _read_metadata(Path(config["steady_root"]), args.split)
    metric_path = args.run_dir / "evaluation" / args.split / "per_sample_time_metrics.csv"
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with metric_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["sample_id"]].append(row)

    sample_rows: list[dict[str, float | int | str]] = []
    for sample_id, time_rows in grouped.items():
        if sample_id not in metadata:
            raise KeyError(f"Missing geometry metadata for {sample_id}")
        meta = metadata[sample_id]
        nonzero = [row for row in time_rows if int(row["iteration"]) > 0]
        final = max(time_rows, key=lambda row: int(row["iteration"]))
        rel_velocity = [float(row["relative_l2_velocity"]) for row in nonzero]
        pressure_mae = [float(row["mae_p_normalized"]) for row in time_rows]
        speed_mae = [float(row["mae_speed_lattice"]) for row in time_rows]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "porosity": float(meta["porosity"]),
                "connected_porosity": float(meta["connected_porosity"]),
                "dead_pore_ratio": float(meta["dead_pore_ratio"]),
                "permeability_lattice": float(meta["permeability_lattice"]),
                "hydraulic_tortuosity": float(meta["hydraulic_tortuosity"]),
                "iterations": int(meta["iterations"]),
                "mean_relative_l2_velocity": _mean(rel_velocity),
                "mean_mae_p_normalized": _mean(pressure_mae),
                "mean_mae_speed_lattice": _mean(speed_mae),
                "final_flux_relative_error": float(final["relative_error_mean_flux_final"]),
                "final_permeability_relative_error": float(
                    final["relative_error_permeability_final"]
                ),
                "final_tortuosity_relative_error": float(
                    final["relative_error_hydraulic_tortuosity_final"]
                ),
            }
        )
    output = args.run_dir / "analysis" / "stratified"
    output.mkdir(parents=True, exist_ok=True)
    sample_table = output / "sample_level_metrics.csv"
    with sample_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    apply_nature_style(font_size=7.0)
    stratified_rows: list[dict[str, float | int | str]] = []
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15), constrained_layout=True, sharey=True)
    palette = ["#D8E0F0", "#AEBBD6", "#788DB8", "#405A8A"]
    for panel, axis, (factor, label) in zip("abcde", axes.flat, FACTORS):
        factor_values = np.asarray([float(row[factor]) for row in sample_rows])
        edges = np.quantile(factor_values, [0.0, 0.25, 0.5, 0.75, 1.0])
        # Repeated integer quantiles are nudged only for bin assignment; reported
        # boundaries retain the real values.
        assignment_edges = edges.copy()
        for index in range(1, len(assignment_edges)):
            if assignment_edges[index] <= assignment_edges[index - 1]:
                assignment_edges[index] = np.nextafter(assignment_edges[index - 1], np.inf)
        bins = np.clip(np.digitize(factor_values, assignment_edges[1:-1], right=True), 0, 3)
        means: list[float] = []
        lower_errors: list[float] = []
        upper_errors: list[float] = []
        labels: list[str] = []
        for quartile in range(4):
            selected = np.asarray(
                [
                    float(row["mean_relative_l2_velocity"])
                    for row, bin_index in zip(sample_rows, bins)
                    if bin_index == quartile
                ],
                dtype=float,
            )
            selected = selected[np.isfinite(selected)]
            mean = float(selected.mean()) if selected.size else float("nan")
            ci_lower, ci_upper = _bootstrap_mean_ci(
                selected, rng, draws=args.bootstrap_draws
            )
            lower = float(edges[quartile])
            upper = float(edges[quartile + 1])
            means.append(mean)
            lower_errors.append(max(mean - ci_lower, 0.0))
            upper_errors.append(max(ci_upper - mean, 0.0))
            labels.append(f"Q{quartile + 1}")
            stratified_rows.append(
                {
                    "factor": factor,
                    "quartile": quartile + 1,
                    "lower": lower,
                    "upper": upper,
                    "n": int(selected.size),
                    "mean_relative_l2_velocity": mean,
                    "ci95_lower_relative_l2_velocity": ci_lower,
                    "ci95_upper_relative_l2_velocity": ci_upper,
                    "bootstrap_draws": args.bootstrap_draws,
                }
            )
        axis.bar(
            labels,
            means,
            yerr=np.asarray([lower_errors, upper_errors]),
            color=palette,
            edgecolor="#303846",
            linewidth=0.45,
            capsize=2.0,
            error_kw={"elinewidth": 0.75, "capthick": 0.75},
        )
        axis.set_title(label)
        axis.set_ylabel(r"Mean velocity relative $L_2$")
        soften_axis(axis)
        add_panel_label(axis, panel)
    key_axis = axes.flat[-1]
    key_axis.set_axis_off()
    key_axis.text(
        0.04,
        0.92,
        "Quartile key",
        transform=key_axis.transAxes,
        fontweight="bold",
        va="top",
    )
    key_axis.text(
        0.04,
        0.74,
        "Bars: geometry-level mean\nWhiskers: 95% geometry-bootstrap CI\n10,000 resamples; quartiles: test set\nExact bounds and n: source-data CSV",
        transform=key_axis.transAxes,
        va="top",
        linespacing=1.45,
        color="#374151",
    )
    figure_path = output / "stratified_error.png"
    save_publication_figure(fig, figure_path)
    close_figure(fig)

    stratified_table = output / "quartile_metrics.csv"
    with stratified_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stratified_rows[0]))
        writer.writeheader()
        writer.writerows(stratified_rows)
    worst = sorted(
        sample_rows,
        key=lambda row: float(row["mean_relative_l2_velocity"]),
        reverse=True,
    )[:20]
    worst_table = output / "worst_20_samples.csv"
    with worst_table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(worst[0]))
        writer.writeheader()
        writer.writerows(worst)
    print(sample_table)
    print(stratified_table)
    print(worst_table)
    print(figure_path)


if __name__ == "__main__":
    main()
