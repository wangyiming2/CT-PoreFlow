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
from scipy.stats import wilcoxon

from figure_style import (
    add_panel_label,
    apply_nature_style,
    close_figure,
    method_style,
    save_publication_figure,
    soften_axis,
)
from statistics_utils import add_holm_fields, matched_rank_biserial


METRICS = ("relative_l2_velocity", "mae_p_normalized", "mae_speed_lattice")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_temporal_grid(
    rows: list[dict[str, str]], expected_geometries: int, expected_time_points: int
) -> None:
    """Require one complete, duplicate-free held-out time grid per geometry."""
    grids: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grids[row["sample_id"]].append(int(row["iteration"]))
    if len(grids) != expected_geometries:
        raise ValueError(
            f"Temporal-holdout CSV has {len(grids)} geometries, "
            f"expected {expected_geometries}"
        )
    reference_grid: tuple[int, ...] | None = None
    for sample_id, iterations in grids.items():
        unique_grid = tuple(sorted(set(iterations)))
        if len(iterations) != expected_time_points or len(unique_grid) != expected_time_points:
            raise ValueError(
                f"{sample_id}: expected {expected_time_points} unique held-out times, "
                f"found {len(iterations)} rows and {len(unique_grid)} unique iterations"
            )
        if reference_grid is None:
            reference_grid = unique_grid
        elif unique_grid != reference_grid:
            raise ValueError(f"{sample_id}: held-out iteration grid differs across geometries")


def load_rows(run: Path) -> tuple[dict, list[dict[str, str]]]:
    config = load_json(run / "config.json")
    heldout = [int(value) for value in config["heldout_time_indices"]]
    training = [int(value) for value in config["training_time_indices"]]
    validation = [int(value) for value in config["validation_time_indices"]]
    if set(heldout) & set(training):
        raise ValueError(f"{run}: held-out and training time indices overlap")
    if set(heldout) & set(validation):
        raise ValueError(f"{run}: held-out indices were used for checkpoint selection")
    evaluation = run / "evaluation" / "temporal_holdout"
    summary = load_json(evaluation / "summary.json")
    if [int(value) for value in summary.get("time_indices", [])] != heldout:
        raise ValueError(f"{run}: evaluation time indices do not match the holdout declaration")
    expected = 764 * len(heldout)
    if int(summary.get("records", -1)) != expected or int(summary.get("geometries", -1)) != 764:
        raise ValueError(f"{run}: temporal holdout scope is not 764 x {len(heldout)}")
    with (evaluation / "per_sample_time_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected:
        raise ValueError(f"{run}: temporal holdout CSV has {len(rows)} rows, expected {expected}")
    _validate_temporal_grid(rows, expected_geometries=764, expected_time_points=len(heldout))
    return summary, rows


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def grouped_samples(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value):
            values[row["sample_id"]].append(value)
    return {sample: float(np.mean(sample_values)) for sample, sample_values in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze genuine unseen-time interpolation.")
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--proposed-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()

    labels = ["Time-U-Net", "CT-PoreFlow"]
    runs = [args.baseline_run, args.proposed_run]
    loaded = [load_rows(run) for run in runs]
    summaries = {label: summary for label, (summary, _) in zip(labels, loaded)}
    rows_by_label = {label: rows for label, (_, rows) in zip(labels, loaded)}
    heldout = [int(value) for value in summaries[labels[0]]["time_indices"]]
    if heldout != [int(value) for value in summaries[labels[1]]["time_indices"]]:
        raise ValueError("The two models were not evaluated at identical held-out times")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for label in labels:
        summary = summaries[label]
        summary_rows.append(
            {
                "model": label,
                "records": int(summary["records"]),
                "geometries": int(summary["geometries"]),
                **{metric: float(summary[metric]) for metric in METRICS},
            }
        )
    with (args.output_dir / "temporal_holdout_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    per_time_rows = []
    for label in labels:
        grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows_by_label[label]:
            iteration = int(row["iteration"])
            for metric in METRICS:
                value = float(row[metric])
                if np.isfinite(value):
                    grouped[iteration][metric].append(value)
        for iteration in sorted(grouped):
            per_time_rows.append(
                {
                    "model": label,
                    "iteration": iteration,
                    **{
                        metric: float(np.mean(grouped[iteration][metric]))
                        for metric in METRICS
                    },
                }
            )
    with (args.output_dir / "temporal_holdout_per_time.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_time_rows[0]))
        writer.writeheader()
        writer.writerows(per_time_rows)

    rng = np.random.default_rng(2026)
    paired_tests: list[dict[str, object]] = []
    for metric in METRICS:
        baseline = grouped_samples(rows_by_label[labels[0]], metric)
        proposed = grouped_samples(rows_by_label[labels[1]], metric)
        common = sorted(set(baseline) & set(proposed))
        if len(common) != 764:
            raise ValueError(f"{metric}: expected 764 paired geometries, found {len(common)}")
        baseline_values = np.asarray([baseline[sample] for sample in common])
        proposed_values = np.asarray([proposed[sample] for sample in common])
        difference = baseline_values - proposed_values
        lower, upper = bootstrap_mean_ci(difference, rng, args.bootstrap_draws)
        try:
            test = wilcoxon(baseline_values, proposed_values, alternative="two-sided")
            statistic, raw_p = float(test.statistic), float(test.pvalue)
        except ValueError:
            statistic, raw_p = 0.0, 1.0
        paired_tests.append(
            {
                "metric": metric,
                "n": len(common),
                "baseline_mean": float(baseline_values.mean()),
                "proposed_mean": float(proposed_values.mean()),
                "mean_baseline_minus_proposed": float(difference.mean()),
                "mean_difference_ci95_lower": lower,
                "mean_difference_ci95_upper": upper,
                "proposed_better_fraction": float(np.mean(proposed_values < baseline_values)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": raw_p,
                "matched_rank_biserial": matched_rank_biserial(difference),
            }
        )
    add_holm_fields(paired_tests)
    result = {
        "experiment": "unseen stored-time interpolation",
        "training_time_indices": load_json(args.baseline_run / "config.json")[
            "training_time_indices"
        ],
        "heldout_time_indices": heldout,
        "validation_time_indices": load_json(args.baseline_run / "config.json")[
            "validation_time_indices"
        ],
        "records_per_model": 764 * len(heldout),
        "paired_geometries": 764,
        "bootstrap_draws": args.bootstrap_draws,
        "multiplicity_correction": "Holm-Bonferroni across the three temporal-holdout endpoints",
        "family_wise_alpha": 0.05,
        "summaries": summary_rows,
        "paired_tests": paired_tests,
    }
    (args.output_dir / "temporal_holdout_statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    apply_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    for index, label in enumerate(labels):
        subset = [row for row in per_time_rows if row["model"] == label]
        iterations = [int(row["iteration"]) for row in subset]
        style = method_style(label, index)
        axes[0].plot(
            iterations,
            [float(row["relative_l2_velocity"]) for row in subset],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=3.5,
            linewidth=1.8 if label != "CT-PoreFlow" else 2.0,
            label=label,
            color=style["color"],
        )
        axes[1].plot(
            iterations,
            [float(row["mae_p_normalized"]) for row in subset],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=3.5,
            linewidth=1.8 if label != "CT-PoreFlow" else 2.0,
            label=label,
            color=style["color"],
        )
    for panel, axis, ylabel in zip(
        "ab", axes, [r"Velocity relative $L_2$", "Normalized pressure MAE"]
    ):
        axis.set_xscale("log")
        axis.set_xlabel("Held-out LBM iteration")
        axis.set_ylabel(ylabel)
        axis.margins(y=0.08)
        soften_axis(axis)
        add_panel_label(axis, panel)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=2)
    save_publication_figure(fig, args.output_dir / "temporal_holdout_error.png")
    close_figure(fig)
    print(args.output_dir)


if __name__ == "__main__":
    main()
