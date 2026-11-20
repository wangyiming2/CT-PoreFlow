from __future__ import annotations

import argparse
import csv
import json
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
from statistics_utils import add_holm_fields, matched_rank_biserial


METRICS = [
    "warm_iterations",
    "iteration_reduction_fraction",
    "iteration_speedup",
    "solver_wall_clock_speedup",
    "end_to_end_wall_clock_speedup",
    "refined_velocity_relative_l2",
    "refined_pressure_relative_l2",
    "warm_mass_balance_error",
    "warm_permeability_relative_difference_vs_cold",
]


def load(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
    for row in rows.values():
        cold = float(row["cold_permeability_lattice"])
        warm = float(row["warm_permeability_lattice"])
        row["warm_permeability_relative_difference_vs_cold"] = str(
            abs(warm - cold) / max(abs(cold), 1e-30)
        )
    return rows


def bootstrap_ci(values: np.ndarray, draws: int, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, values.size, size=(draws, values.size))
    estimates = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def within_model_description(
    rows: dict[str, dict[str, str]],
    samples: list[str],
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float | int | list[float]]:
    """Describe warm-versus-cold utility without adding post-hoc hypothesis tests."""

    cold_iterations = np.asarray(
        [float(rows[sample]["cold_iterations_measured"]) for sample in samples]
    )
    warm_iterations = np.asarray([float(rows[sample]["warm_iterations"]) for sample in samples])
    iteration_reduction = cold_iterations - warm_iterations
    end_to_end_speedup = np.asarray(
        [float(rows[sample]["end_to_end_wall_clock_speedup"]) for sample in samples]
    )
    return {
        "n": int(len(samples)),
        "cold_iterations_mean": float(cold_iterations.mean()),
        "warm_iterations_mean": float(warm_iterations.mean()),
        "cold_minus_warm_iterations_mean": float(iteration_reduction.mean()),
        "cold_minus_warm_iterations_mean_ci95": bootstrap_ci(
            iteration_reduction, draws, rng
        ),
        "end_to_end_speedup_mean": float(end_to_end_speedup.mean()),
        "end_to_end_speedup_mean_ci95": bootstrap_ci(end_to_end_speedup, draws, rng),
        "fraction_end_to_end_faster_than_cold": float(np.mean(end_to_end_speedup > 1.0)),
        "warm_convergence_rate": float(
            np.mean([rows[sample]["warm_converged"].lower() == "true" for sample in samples])
        ),
    }


def main() -> None:
    from scipy.stats import wilcoxon

    parser = argparse.ArgumentParser(description="Paired comparison of two neural LBM warm starts.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Time-U-Net")
    parser.add_argument("--proposed-label", default="CT-PoreFlow")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()

    baseline = load(args.baseline)
    proposed = load(args.proposed)
    common = sorted(set(baseline) & set(proposed))
    if len(common) != 100:
        raise RuntimeError(f"Expected 100 paired warm-start samples, found {len(common)}")
    rng = np.random.default_rng(2026)
    rows: list[dict[str, float | int | str | list[float]]] = []
    for metric in METRICS:
        base = np.asarray([float(baseline[sample][metric]) for sample in common])
        prop = np.asarray([float(proposed[sample][metric]) for sample in common])
        finite = np.isfinite(base) & np.isfinite(prop)
        base = base[finite]
        prop = prop[finite]
        difference = prop - base
        try:
            test = wilcoxon(prop, base, alternative="two-sided")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        except ValueError:
            statistic = 0.0
            p_value = 1.0
        rows.append(
            {
                "metric": metric,
                "n": int(difference.size),
                "baseline_mean": float(base.mean()),
                "proposed_mean": float(prop.mean()),
                "mean_proposed_minus_baseline": float(difference.mean()),
                "difference_ci95": bootstrap_ci(difference, args.bootstrap_draws, rng),
                "proposed_better_fraction": float(
                    np.mean(prop < base)
                    if metric
                    in {
                        "warm_iterations",
                        "refined_velocity_relative_l2",
                        "refined_pressure_relative_l2",
                        "warm_mass_balance_error",
                        "warm_permeability_relative_difference_vs_cold",
                    }
                    else np.mean(prop > base)
                ),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
                "matched_rank_biserial_proposed_minus_baseline": matched_rank_biserial(difference),
            }
        )

    # All nine warm-start endpoints form one multiplicity family.
    add_holm_fields(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "warm_start_paired_tests.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [key for key in rows[0] if key != "difference_ci95"] + [
            "difference_ci95_lower",
            "difference_ci95_upper",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            interval = row["difference_ci95"]
            writer.writerow(
                {
                    **{key: value for key, value in row.items() if key != "difference_ci95"},
                    "difference_ci95_lower": interval[0],
                    "difference_ci95_upper": interval[1],
                }
            )
    json_path = args.output_dir / "warm_start_comparison.json"
    within_model = {
        args.baseline_label: within_model_description(
            baseline, common, args.bootstrap_draws, rng
        ),
        args.proposed_label: within_model_description(
            proposed, common, args.bootstrap_draws, rng
        ),
    }
    json_path.write_text(
        json.dumps(
            {
                "baseline": args.baseline_label,
                "proposed": args.proposed_label,
                "paired_samples": len(common),
                "bootstrap_draws": args.bootstrap_draws,
                "multiplicity_correction": "Holm-Bonferroni across all warm-start endpoints",
                "family_wise_alpha": 0.05,
                "tests": rows,
                "within_model_descriptive": within_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    panels = [
        ("warm_iterations", "Warm iterations", False),
        ("end_to_end_wall_clock_speedup", "End-to-end speedup", True),
        ("refined_velocity_relative_l2", "Refined velocity relative L2", False),
    ]
    apply_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    box_colors = [
        method_style(args.baseline_label, 0)["color"],
        method_style(args.proposed_label, 1)["color"],
    ]
    for panel, axis, (metric, title, higher_better) in zip("bcd", axes, panels):
        base = np.asarray([float(baseline[sample][metric]) for sample in common])
        prop = np.asarray([float(proposed[sample][metric]) for sample in common])
        if metric == "warm_iterations":
            cold = np.asarray(
                [float(baseline[sample]["cold_iterations_measured"]) for sample in common]
            )
            values = [cold, base, prop]
            positions = [0, 1, 2]
            labels = ["Cold", args.baseline_label, args.proposed_label]
            colors = ["#D1D5DB", *box_colors]
            for cold_value, base_value, prop_value in zip(cold, base, prop):
                axis.plot(
                    [0, 1], [cold_value, base_value], color="#9CA3AF",
                    alpha=0.10, linewidth=0.4, zorder=1,
                )
                axis.plot(
                    [0, 2], [cold_value, prop_value], color="#9CA3AF",
                    alpha=0.10, linewidth=0.4, zorder=1,
                )
        else:
            values = [base, prop]
            positions = [0, 1]
            labels = [args.baseline_label, args.proposed_label]
            colors = box_colors
            for left, right in zip(base, prop):
                axis.plot(
                    [0, 1], [left, right], color="#9CA3AF",
                    alpha=0.16, linewidth=0.45, zorder=1,
                )
        box = axis.boxplot(
            values,
            positions=positions,
            widths=0.34,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "#111827", "linewidth": 1.1},
            whiskerprops={"color": "#4B5563", "linewidth": 0.8},
            capprops={"color": "#4B5563", "linewidth": 0.8},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
            patch.set_edgecolor("#374151")
            patch.set_linewidth(0.7)
        axis.set_xticks(positions, labels, rotation=12, ha="right")
        axis.set_title(title)
        axis.text(
            0.02,
            0.98,
            "higher is better" if higher_better else "lower is better",
            transform=axis.transAxes,
            va="top",
            fontsize=6.2,
            color="#4B5563",
        )
        soften_axis(axis)
        add_panel_label(axis, panel)
    figure_path = args.output_dir / "warm_start_paired_comparison.png"
    save_publication_figure(fig, figure_path)
    close_figure(fig)
    print(csv_path)
    print(json_path)
    print(figure_path)


if __name__ == "__main__":
    main()
