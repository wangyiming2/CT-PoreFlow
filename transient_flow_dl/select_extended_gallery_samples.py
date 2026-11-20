"""Select a deterministic 16-case gallery spanning morphology and error behaviour."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_records(topology_path: Path, unet_path: Path, ct_path: Path) -> list[dict]:
    topology = {row["sample_id"]: row for row in read_csv(topology_path)}
    unet = {row["sample_id"]: row for row in read_csv(unet_path)}
    ct = {row["sample_id"]: row for row in read_csv(ct_path)}
    records: list[dict] = []
    for sample_id in sorted(set(topology) & set(unet) & set(ct)):
        base = topology[sample_id]
        u_error = float(unet[sample_id]["mean_relative_l2_velocity"])
        c_error = float(ct[sample_id]["mean_relative_l2_velocity"])
        records.append(
            {
                "sample_index": int(base["sample_index"]),
                "sample_id": sample_id,
                "porosity": float(base["porosity"]),
                "connected_porosity": float(base["connected_porosity"]),
                "dead_pore_ratio": float(base["dead_pore_ratio"]),
                "permeability": float(base["permeability"]),
                "tortuosity": float(base["tortuosity"]),
                "unet_mean_velocity_l2": u_error,
                "ct_mean_velocity_l2": c_error,
                "combined_mean_velocity_l2": 0.5 * (u_error + c_error),
                "model_error_difference": c_error - u_error,
            }
        )
    if len(records) != 764:
        raise RuntimeError(f"Expected 764 common test geometries, found {len(records)}")
    return records


def choose(records: list[dict]) -> list[dict]:
    selected: list[dict] = []
    used: set[int] = set()

    def take(label: str, candidates: list[dict], score) -> None:
        available = [row for row in candidates if row["sample_index"] not in used]
        if not available:
            raise RuntimeError(f"No unused candidate remains for {label}")
        row = min(available, key=score)
        used.add(row["sample_index"])
        item = dict(row)
        item["selection_label"] = label
        selected.append(item)

    # One typical-error example per predefined porosity bin.
    for bin_index in range(8):
        token = f"bin_{bin_index:02d}_"
        group = [row for row in records if token in row["sample_id"]]
        median_error = float(np.median([row["combined_mean_velocity_l2"] for row in group]))
        median_phi = float(np.median([row["porosity"] for row in group]))
        take(
            f"porosity_bin_{bin_index:02d}",
            group,
            lambda row, me=median_error, mp=median_phi: abs(row["combined_mean_velocity_l2"] - me)
            + 0.15 * abs(row["porosity"] - mp),
        )

    def quantile_subset(key: str, q: float, upper: bool) -> list[dict]:
        values = np.asarray([row[key] for row in records], dtype=float)
        threshold = float(np.quantile(values, q))
        return [row for row in records if (row[key] >= threshold if upper else row[key] <= threshold)]

    global_median = float(np.median([row["combined_mean_velocity_l2"] for row in records]))
    median_score = lambda row: abs(row["combined_mean_velocity_l2"] - global_median)
    take("high_tortuosity", quantile_subset("tortuosity", 0.90, True), median_score)
    take("high_dead_pore", quantile_subset("dead_pore_ratio", 0.90, True), median_score)
    take("low_permeability", quantile_subset("permeability", 0.10, False), median_score)
    take("high_permeability", quantile_subset("permeability", 0.90, True), median_score)

    take("easy_case", records, lambda row: row["combined_mean_velocity_l2"])
    take("hard_case", records, lambda row: -row["combined_mean_velocity_l2"])
    # Positive (CT - U-Net) means Time-U-Net is better; negative means CT-PoreFlow is better.
    take("ct_advantage", records, lambda row: row["model_error_difference"])
    take("unet_advantage", records, lambda row: -row["model_error_difference"])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology",
        type=Path,
        default=Path("output/representative_samples/all_test_topology_metrics.csv"),
    )
    parser.add_argument(
        "--unet",
        type=Path,
        default=Path("output/formal_results/unet_time_seed2026/analysis/stratified/sample_level_metrics.csv"),
    )
    parser.add_argument(
        "--ct",
        type=Path,
        default=Path("output/formal_results/ct_poreflow_seed2026/analysis/stratified/sample_level_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/extended_prediction_gallery/selected_samples_16.json"),
    )
    args = parser.parse_args()
    records = load_records(args.topology, args.unet, args.ct)
    selected = choose(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(",".join(str(row["sample_index"]) for row in selected))
    for row in selected:
        print(
            f"{row['selection_label']:>18s} index={row['sample_index']:03d} "
            f"phi={row['porosity']:.3f} k={row['permeability']:.3f} "
            f"tau={row['tortuosity']:.3f} U={row['unet_mean_velocity_l2']:.3f} "
            f"CT={row['ct_mean_velocity_l2']:.3f}"
        )


if __name__ == "__main__":
    main()
