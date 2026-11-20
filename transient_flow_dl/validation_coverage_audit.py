from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

EXPECTED_GEOMETRIES = 764
EXPECTED_TIME_INDICES = [0, 5, 10, 15, 20, 25, 29]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must use non-empty LABEL=PATH")
    return label.strip(), Path(path.strip())


def bin_counts(files: list[Path], validation_root: Path) -> Counter[str]:
    return Counter(path.relative_to(validation_root).parts[0] for path in files)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figure_style import apply_nature_style, close_figure, save_publication_figure, soften_axis
    parser = argparse.ArgumentParser(
        description="Audit checkpoint-selection coverage against full validation performance."
    )
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_rows: list[dict[str, str | int | float]] = []
    reference_config: dict | None = None
    for label, run_dir in args.run:
        config = load_json(run_dir / "config.json")
        summary = load_json(run_dir / "evaluation" / "validation_full" / "summary.json")
        if int(summary.get("geometries", -1)) != EXPECTED_GEOMETRIES:
            raise ValueError(f"{run_dir}: full-validation geometry count is not 764")
        if int(summary.get("records", -1)) != EXPECTED_GEOMETRIES * len(EXPECTED_TIME_INDICES):
            raise ValueError(f"{run_dir}: full-validation record count is not 5,348")
        if [int(value) for value in summary.get("time_indices", [])] != EXPECTED_TIME_INDICES:
            raise ValueError(f"{run_dir}: unexpected validation time indices")
        reference_config = reference_config or config
        run_rows.append(
            {
                "model": label,
                "run": run_dir.name,
                "checkpoint_selection_geometries": int(config["max_validation_samples"]),
                "full_validation_geometries": int(summary["geometries"]),
                "full_validation_records": int(summary["records"]),
                "relative_l2_velocity": float(summary["relative_l2_velocity"]),
                "mae_p_normalized": float(summary["mae_p_normalized"]),
                "mae_speed_lattice": float(summary["mae_speed_lattice"]),
            }
        )

    if reference_config is None:
        raise RuntimeError("No runs supplied")
    validation_root = Path(reference_config["steady_root"]) / "samples" / "validation"
    files = sorted(validation_root.rglob("*.npz"))
    if len(files) != EXPECTED_GEOMETRIES:
        raise ValueError(f"Expected 764 validation geometries, found {len(files)}")
    subset_size = int(reference_config["max_validation_samples"])
    selected = files[:subset_size]
    all_counts = bin_counts(files, validation_root)
    selected_counts = bin_counts(selected, validation_root)
    bins = sorted(all_counts)

    coverage_rows: list[dict[str, str | int | float]] = []
    for bin_name in bins:
        for scope, counts, total in (
            ("checkpoint_selection", selected_counts, len(selected)),
            ("full_validation", all_counts, len(files)),
        ):
            count = int(counts.get(bin_name, 0))
            coverage_rows.append(
                {
                    "scope": scope,
                    "porosity_bin": bin_name,
                    "count": count,
                    "fraction": count / float(total),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_csv = args.output_dir / "validation_bin_coverage.csv"
    with coverage_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)
    performance_csv = args.output_dir / "full_validation_performance.csv"
    with performance_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    x = list(range(len(bins)))
    width = 0.36
    apply_nature_style()
    fig, axis = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    subset_bars = axis.bar(
        [value - width / 2 for value in x],
        [selected_counts.get(name, 0) / len(selected) for name in bins],
        width,
        label=f"Checkpoint subset (n={len(selected)})",
        color="#484878",
        edgecolor="#303846",
        linewidth=0.55,
    )
    full_bars = axis.bar(
        [value + width / 2 for value in x],
        [all_counts.get(name, 0) / len(files) for name in bins],
        width,
        label=f"Full validation (n={len(files)})",
        color="#B4C0E4",
        edgecolor="#303846",
        linewidth=0.55,
    )
    for bar in subset_bars:
        bar.set_hatch("///")
    for bar in full_bars:
        bar.set_hatch("...")
    axis.set_xticks(x, [f"{0.30 + 0.05 * index:.2f}-{0.35 + 0.05 * index:.2f}" for index in x])
    axis.set_xlabel("Target porosity bin")
    axis.set_ylabel("Geometry fraction")
    axis.legend(loc="upper right")
    soften_axis(axis)
    figure = args.output_dir / "validation_coverage.png"
    save_publication_figure(fig, figure)
    close_figure(fig)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint_selection_rule": "first max_validation_samples paths after deterministic sort",
                "checkpoint_selection_geometries": len(selected),
                "full_validation_geometries": len(files),
                "validation_time_indices": EXPECTED_TIME_INDICES,
                "checkpoint_selection_bin_counts": dict(sorted(selected_counts.items())),
                "full_validation_bin_counts": dict(sorted(all_counts.items())),
                "runs": run_rows,
                "interpretation": (
                    "The common checkpoint subset emphasizes low-porosity geometries; "
                    "full-validation best-checkpoint performance is reported as a sensitivity audit."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(coverage_csv)
    print(performance_csv)
    print(figure)
    print(summary_path)


if __name__ == "__main__":
    main()
