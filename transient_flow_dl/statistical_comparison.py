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
    apply_nature_style,
    close_figure,
    method_style,
    save_publication_figure,
    soften_axis,
)
from statistics_utils import add_holm_fields, matched_rank_biserial


METRICS = [
    "relative_l2_velocity",
    "mae_p_normalized",
    "mae_speed_lattice",
    "relative_error_mean_flux_final",
    "relative_error_permeability_final",
    "relative_error_hydraulic_tortuosity_final",
]
FINAL_METRICS = {
    "relative_error_mean_flux_final",
    "relative_error_permeability_final",
    "relative_error_hydraulic_tortuosity_final",
}


def _load_sample_metrics(run: Path) -> dict[str, dict[str, float]]:
    path = run / "evaluation" / "test" / "per_sample_time_metrics.csv"
    rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows_by_sample[row["sample_id"]].append(row)
    result: dict[str, dict[str, float]] = {}
    for sample_id, rows in rows_by_sample.items():
        final_iteration = max(int(row["iteration"]) for row in rows)
        sample_metrics: dict[str, float] = {}
        for metric in METRICS:
            values = []
            for row in rows:
                iteration = int(row["iteration"])
                if metric == "relative_l2_velocity" and iteration == 0:
                    continue
                if metric in FINAL_METRICS and iteration != final_iteration:
                    continue
                value = float(row[metric])
                if np.isfinite(value):
                    values.append(value)
            if metric in FINAL_METRICS and len(values) != 1:
                raise ValueError(
                    f"{path}: {sample_id} has {len(values)} finite final values for {metric}; "
                    "expected exactly one at the maximum iteration"
                )
            sample_metrics[metric] = float(np.mean(values)) if values else float("nan")
        result[sample_id] = sample_metrics
    return result


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(draws, values.size))
    estimates = values[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    if len(args.run) != len(args.label):
        raise ValueError("Each --run requires one --label")
    if args.reference not in args.label:
        raise ValueError("--reference must exactly match one --label")
    all_metrics = {
        label: _load_sample_metrics(run) for run, label in zip(args.run, args.label)
    }
    common_samples = sorted(set.intersection(*(set(values) for values in all_metrics.values())))
    if not common_samples:
        raise RuntimeError("No common test geometries across runs")
    rng = np.random.default_rng(2026)
    summaries: list[dict[str, float | int | str]] = []
    paired: list[dict[str, float | int | str]] = []
    reference_data = all_metrics[args.reference]
    for label in args.label:
        for metric in METRICS:
            values = np.asarray(
                [all_metrics[label][sample].get(metric, np.nan) for sample in common_samples],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            lower, upper = _bootstrap_ci(values, rng, args.bootstrap_draws)
            summaries.append(
                {
                    "model": label,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "median": float(np.median(values)),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
            if label == args.reference:
                continue
            baseline_values = []
            reference_values = []
            for sample in common_samples:
                baseline_value = all_metrics[label][sample].get(metric, np.nan)
                reference_value = reference_data[sample].get(metric, np.nan)
                if np.isfinite(baseline_value) and np.isfinite(reference_value):
                    baseline_values.append(baseline_value)
                    reference_values.append(reference_value)
            baseline_array = np.asarray(baseline_values, dtype=float)
            reference_array = np.asarray(reference_values, dtype=float)
            difference = baseline_array - reference_array
            difference_lower, difference_upper = _bootstrap_ci(
                difference, rng, args.bootstrap_draws
            )
            try:
                test = wilcoxon(baseline_array, reference_array, alternative="two-sided")
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            except ValueError:
                statistic = 0.0
                p_value = 1.0
            paired.append(
                {
                    "reference": args.reference,
                    "baseline": label,
                    "metric": metric,
                    "n": int(difference.size),
                    "baseline_mean": float(baseline_array.mean()),
                    "reference_mean": float(reference_array.mean()),
                    "mean_absolute_reduction": float(difference.mean()),
                    "mean_absolute_reduction_ci95_lower": difference_lower,
                    "mean_absolute_reduction_ci95_upper": difference_upper,
                    "relative_reduction_fraction": float(
                        difference.mean() / max(float(baseline_array.mean()), 1e-30)
                    ),
                    "reference_better_fraction": float(np.mean(reference_array < baseline_array)),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                    "matched_rank_biserial": matched_rank_biserial(difference),
                }
            )

    # Principal models and ablations are launched as separate invocations, so
    # every model-by-metric test emitted here forms one pre-specified family.
    add_holm_fields(paired)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "statistical_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    paired_path = args.output_dir / "paired_tests.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    json_path = args.output_dir / "statistical_results.json"
    json_path.write_text(
        json.dumps(
            {
                "reference": args.reference,
                "common_test_geometries": len(common_samples),
                "bootstrap_draws": args.bootstrap_draws,
                "multiplicity_correction": "Holm-Bonferroni across all paired tests in this analysis",
                "family_wise_alpha": 0.05,
                "summaries": summaries,
                "paired_tests": paired,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    velocity_data = [
        np.asarray(
            [all_metrics[label][sample]["relative_l2_velocity"] for sample in common_samples],
            dtype=float,
        )
        for label in args.label
    ]
    apply_nature_style()
    fig, axis = plt.subplots(figsize=(7.2, 3.35), constrained_layout=True)
    parts = axis.violinplot(velocity_data, showmeans=False, showmedians=False, showextrema=False)
    for index, (body, label) in enumerate(zip(parts["bodies"], args.label)):
        body.set_facecolor(method_style(label, index)["color"])
        body.set_edgecolor("#374151")
        body.set_linewidth(0.55)
        body.set_alpha(0.76 if label == args.reference else 0.58)
    axis.boxplot(
        velocity_data,
        widths=0.16,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "#1F2937"},
        medianprops={"color": "#111827", "linewidth": 1.15},
        whiskerprops={"color": "#1F2937"},
        capprops={"color": "#1F2937"},
    )
    axis.set_xticks(np.arange(1, len(args.label) + 1), args.label, rotation=18, ha="right")
    axis.set_ylabel(r"Per-geometry mean velocity relative $L_2$")
    soften_axis(axis)
    figure_path = args.output_dir / "paired_error_distribution.png"
    save_publication_figure(fig, figure_path)
    close_figure(fig)
    print(summary_path)
    print(paired_path)
    print(json_path)
    print(figure_path)


if __name__ == "__main__":
    main()
