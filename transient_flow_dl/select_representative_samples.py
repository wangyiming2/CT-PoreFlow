"""Select topology-balanced test samples without subjective hand-picking.

The selector deliberately rejects geometries whose through-connected pore space
is concentrated on one lateral side.  It then chooses four complementary cases
near predefined distributional targets for porosity, permeability and hydraulic
tortuosity.  The resulting sample indices match ``TransientFlowDataset`` order.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import add_panel_label, apply_nature_style, save_publication_figure


def _metadata(npz: np.lib.npyio.NpzFile) -> dict:
    return json.loads(str(npz["metadata_json"].item()))


def describe(path: Path, root: Path, index: int) -> dict:
    with np.load(path) as sample:
        pore = np.asarray(sample["pore"], dtype=bool)
        connected = np.asarray(sample["through_connected_pore"], dtype=bool)
        meta = _metadata(sample)
    rows, cols = np.nonzero(connected)
    lateral_profile = connected.mean(axis=0)
    half = connected.shape[1] // 2
    left = float(connected[:, :half].sum())
    right = float(connected[:, half:].sum())
    total = max(left + right, 1.0)
    lateral_span = float((cols.max() - cols.min() + 1) / connected.shape[1])
    lateral_centroid = float(cols.mean() / max(connected.shape[1] - 1, 1))
    lateral_imbalance = float(abs(left - right) / total)
    occupied_lateral_fraction = float(np.mean(lateral_profile > 0))
    flow_strip_occupancy = [
        float(block.mean()) for block in np.array_split(connected, 10, axis=0)
    ]
    porosity = float(meta.get("porosity", pore.mean()))
    connected_porosity = float(meta.get("connected_porosity", connected.mean()))
    return {
        "sample_index": index,
        "sample_id": path.relative_to(root).as_posix(),
        "file": str(path.resolve()),
        "porosity": porosity,
        "connected_porosity": connected_porosity,
        "dead_pore_ratio": 1.0 - connected_porosity / max(porosity, 1e-12),
        "permeability": float(meta.get("permeability_lattice", np.nan)),
        "tortuosity": float(meta.get("hydraulic_tortuosity", np.nan)),
        "lateral_span": lateral_span,
        "lateral_centroid": lateral_centroid,
        "lateral_imbalance": lateral_imbalance,
        "occupied_lateral_fraction": occupied_lateral_fraction,
        "min_flow_strip_occupancy": min(flow_strip_occupancy),
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values, axis=0)
    scale = np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (values - median) / scale


def choose(records: list[dict]) -> list[dict]:
    eligible = [
        record
        for record in records
        if record["lateral_span"] >= 0.80
        and record["occupied_lateral_fraction"] >= 0.80
        and abs(record["lateral_centroid"] - 0.5) <= 0.12
        and record["lateral_imbalance"] <= 0.25
        and record["min_flow_strip_occupancy"] > 0.01
        and np.isfinite(record["permeability"])
        and np.isfinite(record["tortuosity"])
    ]
    if len(eligible) < 20:
        raise RuntimeError(f"Only {len(eligible)} topology-balanced samples passed the filter")
    features = np.asarray(
        [[r["porosity"], np.log1p(r["permeability"]), r["tortuosity"]] for r in eligible]
    )
    z = robust_z(features)
    # Complementary, interpretable distributional targets.
    targets = [
        ("low_porosity", [0.20, 0.35, 0.50]),
        ("median_case", [0.50, 0.50, 0.50]),
        ("high_tortuosity", [0.50, 0.50, 0.85]),
        ("high_permeability", [0.80, 0.85, 0.35]),
    ]
    raw = np.asarray(
        [
            [
                np.nanquantile(features[:, 0], q[0]),
                np.nanquantile(features[:, 1], q[1]),
                np.nanquantile(features[:, 2], q[2]),
            ]
            for _, q in targets
        ]
    )
    target_z = robust_z(np.vstack([features, raw]))[-len(targets) :]
    selected: list[dict] = []
    used: set[int] = set()
    for target_index, (label, _) in enumerate(targets):
        distance = np.sum((z - target_z[target_index]) ** 2, axis=1)
        for candidate in np.argsort(distance):
            if int(candidate) not in used:
                used.add(int(candidate))
                record = dict(eligible[int(candidate)])
                record["selection_label"] = label
                record["selection_distance"] = float(distance[int(candidate)])
                selected.append(record)
                break
    return selected


def plot_selection(selected: list[dict], output: Path) -> None:
    apply_nature_style()
    fig, axes = plt.subplots(len(selected), 3, figsize=(7.2, 8.0), constrained_layout=True)
    for row, record in enumerate(selected):
        with np.load(record["file"]) as sample:
            pore = np.asarray(sample["pore"], dtype=float)
            connected = np.asarray(sample["through_connected_pore"], dtype=float)
            edt = np.asarray(sample["edt_lattice"], dtype=float)
        for col, (field, title, cmap) in enumerate(
            ((pore, "Pore mask", "gray_r"), (connected, "Through-connected pore", "Blues"), (edt, "EDT", "magma"))
        ):
            image = axes[row, col].imshow(field, cmap=cmap, interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 2:
                fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.03)
            if row == 0:
                axes[row, col].set_title(title)
        axes[row, 0].set_ylabel(
            f"{record['selection_label'].replace('_', ' ')}\n"
            f"$\\phi$={record['porosity']:.3f}, $k$={record['permeability']:.2f}",
            fontweight="bold",
        )
        add_panel_label(axes[row, 0], chr(ord("a") + row), x=-0.47, y=1.03)
    fig.suptitle("Objective topology-balanced test cases", y=1.015)
    save_publication_figure(fig, output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steady-root", type=Path, default=Path("Dataset/LBM_8000_20260714_dataset")
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=Path("output/representative_samples"))
    args = parser.parse_args()
    sample_root = args.steady_root / "samples"
    files = sorted((sample_root / args.split).rglob("*.npz"))
    records = [describe(path, sample_root, index) for index, path in enumerate(files)]
    selected = choose(records)
    args.output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with (args.output / "all_test_topology_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    (args.output / "selected_samples.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_selection(selected, args.output / "selected_geometry_panel.png")
    for record in selected:
        print(
            f"{record['selection_label']}: index={record['sample_index']} "
            f"id={record['sample_id']} phi={record['porosity']:.4f} "
            f"k={record['permeability']:.4g} tau={record['tortuosity']:.4f}"
        )


if __name__ == "__main__":
    main()
