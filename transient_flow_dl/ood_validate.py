from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from figure_style import apply_nature_style


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "remote_source"))

from lbm_dataset_generator import (  # noqa: E402
    SolverConfig,
    euclidean_distance_to_solid,
    normalized_sample_fields,
    run_lbm,
    through_connected_components,
)
from src.ood_geometry import OOD_FAMILIES, geometry_statistics, make_rule_mask  # noqa: E402
from precompute_topology import geodesic  # noqa: E402


FAMILY_LABELS = {
    "circle_array": "Circular array",
    "snowflakes": "Snowflakes",
    "trees": "Tree obstacles",
    "staggered_baffles": "Staggered baffles",
}
MODEL_LABELS = {
    "unet_v2_transport_matched_seed2026": "Time-U-Net",
    "fno_time_seed2026": "Time-FNO",
    "deeponet_seed2026": "DeepONet",
    "ct_poreflow_v2_transport_seed2026": "CT-PoreFlow",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_geometries(
    output_root: Path,
    porosities: list[float],
    samples_per_condition: int,
    seed: int,
) -> list[dict[str, Any]]:
    geometry_root = output_root / "geometries"
    geometry_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(OOD_FAMILIES):
        for porosity_index, target_porosity in enumerate(porosities):
            for replicate in range(samples_per_condition):
                sample_seed = seed + family_index * 10000 + porosity_index * 1000 + replicate
                sample_id = f"{family}_phi{int(round(target_porosity * 100)):02d}_r{replicate:02d}"
                solid = make_rule_mask(
                    family,
                    seed=sample_seed,
                    severity=1,
                    target_porosity=target_porosity,
                )
                connected, connected_porosity, connected_fraction = through_connected_components(solid)
                edt = euclidean_distance_to_solid(solid)
                distance_in = geodesic(connected, True)
                distance_out = geodesic(connected, False)
                statistics = geometry_statistics(solid)
                destination = geometry_root / f"{sample_id}.npz"
                np.savez_compressed(
                    destination,
                    solid=solid,
                    pore=(solid == 0).astype(np.uint8),
                    through_connected_pore=connected.astype(np.uint8),
                    edt_lattice=edt.astype(np.float32),
                    distance_from_inlet=distance_in,
                    distance_from_outlet=distance_out,
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "family": family,
                        "target_porosity": target_porosity,
                        "replicate": replicate,
                        "seed": sample_seed,
                        "porosity": statistics["porosity"],
                        "connected_porosity": connected_porosity,
                        "connected_fraction_of_pore": connected_fraction,
                        "dead_pore_ratio": statistics["dead_pore_ratio"],
                        "geometry_file": destination.relative_to(output_root).as_posix(),
                    }
                )
    _write_csv(output_root / "geometry_manifest.csv", rows)
    _write_json(
        output_root / "manifest.json",
        {
            "benchmark": "rule-based morphology OOD for transient pore flow",
            "seed": seed,
            "families": list(OOD_FAMILIES),
            "target_porosities": porosities,
            "samples_per_condition": samples_per_condition,
            "sample_count": len(rows),
            "solid_convention": "1=solid, 0=fluid",
            "flow_axis": "array axis 0",
            "training_distribution": "QSGS random irregular porous media",
            "ood_principle": "regular motifs absent from training, porosity matched within the training range",
        },
    )
    plot_geometry_atlas(output_root, rows)
    return rows


def _load_manifest(output_root: Path) -> list[dict[str, str]]:
    with (output_root / "geometry_manifest.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run_lbm_truth(
    output_root: Path,
    max_samples: int,
    snapshots: int,
    max_iterations: int,
) -> list[dict[str, Any]]:
    rows = _load_manifest(output_root)
    if max_samples > 0:
        rows = rows[:max_samples]
    truth_root = output_root / "lbm_truth"
    truth_root.mkdir(parents=True, exist_ok=True)
    solver = SolverConfig(max_iterations=max_iterations, transient_snapshots=snapshots)
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for number, row in enumerate(rows, 1):
        destination = truth_root / f"{row['sample_id']}.npz"
        metadata_path = destination.with_suffix(".json")
        if destination.exists() and metadata_path.exists():
            records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            continue
        with np.load(output_root / row["geometry_file"]) as geometry:
            solid = np.asarray(geometry["solid"], dtype=np.uint8)
            connected = np.asarray(geometry["through_connected_pore"], dtype=bool)
            edt = np.asarray(geometry["edt_lattice"], dtype=np.float64)
            distance_in = np.asarray(geometry["distance_from_inlet"], dtype=np.float32)
            distance_out = np.asarray(geometry["distance_from_outlet"], dtype=np.float32)
        sample_start = time.perf_counter()
        lbm = run_lbm(solid, solver)
        arrays, factors = normalized_sample_fields(solid, connected, edt, lbm)
        arrays["distance_from_inlet"] = distance_in
        arrays["distance_from_outlet"] = distance_out
        np.savez_compressed(destination, **arrays)
        metadata = {
            "sample_id": row["sample_id"],
            "family": row["family"],
            "target_porosity": float(row["target_porosity"]),
            "porosity": float(row["porosity"]),
            "connected_porosity": float(row["connected_porosity"]),
            "converged": bool(lbm["converged"]),
            "iterations": int(lbm["iterations"]),
            "velocity_residual": float(lbm["velocity_residual"]),
            "mass_balance_error": float(lbm["mass_balance_error"]),
            "max_mach": float(lbm["max_mach"]),
            "permeability_lattice": float(lbm["permeability_lattice"]),
            "hydraulic_tortuosity": float(factors["hydraulic_tortuosity"]),
            "seconds": time.perf_counter() - sample_start,
            "truth_file": destination.relative_to(output_root).as_posix(),
        }
        _write_json(metadata_path, metadata)
        records.append(metadata)
        print(
            f"LBM {number}/{len(rows)} {row['sample_id']} converged={metadata['converged']} "
            f"iterations={metadata['iterations']} seconds={metadata['seconds']:.1f}",
            flush=True,
        )
    _write_csv(output_root / "lbm_summary.csv", records)
    _write_json(
        output_root / "lbm_run.json",
        {
            "samples": len(records),
            "snapshots": snapshots,
            "max_iterations": max_iterations,
            "elapsed_seconds": float(sum(float(record["seconds"]) for record in records)),
            "wall_seconds_this_invocation": time.perf_counter() - start,
            "converged_samples": sum(bool(row["converged"]) for row in records),
        },
    )
    plot_lbm_atlas(output_root, records)
    return records


def _finite_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1.0e-12 else float("nan")


def evaluate_models(output_root: Path, run_dirs: list[Path]) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required only for model evaluation. Geometry generation and LBM truth are complete."
        ) from exc
    from src.models import build_model

    truth_files = sorted((output_root / "lbm_truth").glob("*.npz"))
    if not truth_files:
        raise FileNotFoundError("No LBM truth files found; run --stage lbm first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_rows: list[dict[str, Any]] = []
    prediction_root = output_root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model = build_model(config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        velocity_scale = float(checkpoint["velocity_scale"])
        input_channels = int(config.get("input_channels", 3))
        model_name = run_dir.name
        with torch.inference_mode():
            for truth_file in truth_files:
                metadata = json.loads(truth_file.with_suffix(".json").read_text(encoding="utf-8"))
                with np.load(truth_file) as sample:
                    pore = np.asarray(sample["pore"], dtype=np.float32)
                    connected = np.asarray(sample["through_connected_pore"], dtype=np.float32)
                    edt = np.asarray(sample["edt_lattice"], dtype=np.float32) / float(pore.shape[0])
                    channels = [pore, connected, edt]
                    if input_channels == 5:
                        channels.extend(
                            [
                                np.asarray(sample["distance_from_inlet"], dtype=np.float32),
                                np.asarray(sample["distance_from_outlet"], dtype=np.float32),
                            ]
                        )
                    geometry = torch.from_numpy(np.stack(channels)[None]).to(device)
                    iterations = np.asarray(sample["transient_iterations"], dtype=np.int64)
                    ux = np.asarray(sample["transient_velocity_x_raw_lattice"], dtype=np.float32)
                    uy = np.asarray(sample["transient_velocity_y_raw_lattice"], dtype=np.float32)
                    density = np.asarray(sample["transient_density_raw_lattice"], dtype=np.float32)
                query = np.log1p(iterations) / math.log1p(int(config.get("max_iteration", 50000)))
                times = torch.from_numpy(query.astype(np.float32)).to(device)
                geometry_batch = geometry.expand(len(times), -1, -1, -1)
                prediction = model(geometry_batch, times).float().cpu().numpy()
                target = np.stack(
                    [
                        ux / velocity_scale,
                        uy / velocity_scale,
                        ((density - float(config["rho_out"])) / (float(config["rho_in"]) - float(config["rho_out"])))
                        * pore[None],
                    ],
                    axis=1,
                )
                prediction *= pore[None, None]
                error = prediction - target
                valid = np.broadcast_to(pore[None, None] > 0.5, target.shape)
                for time_index, iteration in enumerate(iterations):
                    velocity_valid = valid[time_index, :2]
                    velocity_error = error[time_index, :2][velocity_valid]
                    velocity_target = target[time_index, :2][velocity_valid]
                    relative_velocity = _finite_ratio(
                        float(np.linalg.norm(velocity_error)), float(np.linalg.norm(velocity_target))
                    )
                    result_rows.append(
                        {
                            "model": model_name,
                            "sample_id": metadata["sample_id"],
                            "family": metadata["family"],
                            "target_porosity": metadata["target_porosity"],
                            "porosity": metadata["porosity"],
                            "time_index": time_index,
                            "iteration": int(iteration),
                            "relative_l2_velocity": relative_velocity,
                            "mae_ux_normalized": float(np.abs(error[time_index, 0][pore > 0.5]).mean()),
                            "mae_uy_normalized": float(np.abs(error[time_index, 1][pore > 0.5]).mean()),
                            "mae_p_normalized": float(np.abs(error[time_index, 2][pore > 0.5]).mean()),
                        }
                    )
                np.savez_compressed(
                    prediction_root / f"{model_name}__{metadata['sample_id']}.npz",
                    iterations=iterations,
                    prediction_normalized=prediction.astype(np.float32),
                    target_normalized=target.astype(np.float32),
                    velocity_scale=np.float64(velocity_scale),
                )
        print(f"evaluated {model_name} on {len(truth_files)} OOD geometries", flush=True)
    _write_csv(output_root / "per_sample_time_metrics.csv", result_rows)
    summary = summarize_results(result_rows)
    _write_csv(output_root / "model_family_summary.csv", summary)
    plot_model_summary(output_root, summary)
    return result_rows


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if math.isfinite(float(row["relative_l2_velocity"])):
            grouped[(str(row["model"]), str(row["family"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, family), group in sorted(grouped.items()):
        terminal = [item for item in group if int(item["time_index"]) == 29]
        values = np.asarray([float(item["relative_l2_velocity"]) for item in group])
        terminal_values = np.asarray([float(item["relative_l2_velocity"]) for item in terminal])
        output.append(
            {
                "model": model,
                "family": family,
                "frames": len(group),
                "geometries": len({str(item["sample_id"]) for item in group}),
                "mean_relative_l2_velocity": float(values.mean()),
                "median_relative_l2_velocity": float(np.median(values)),
                "terminal_mean_relative_l2_velocity": float(terminal_values.mean()),
                "terminal_median_relative_l2_velocity": float(np.median(terminal_values)),
            }
        )
    return output


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_geometry_atlas(output_root: Path, rows: list[dict[str, Any]]) -> None:
    apply_nature_style(font_size=8.5)
    selected: dict[tuple[str, float], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["family"]), float(row["target_porosity"]))
        selected.setdefault(key, row)
    porosities = sorted({float(row["target_porosity"]) for row in rows})
    fig, axes = plt.subplots(len(OOD_FAMILIES), len(porosities), figsize=(2.2 * len(porosities), 7.8))
    axes = np.atleast_2d(axes)
    for row_index, family in enumerate(OOD_FAMILIES):
        for column_index, porosity in enumerate(porosities):
            row = selected[(family, porosity)]
            with np.load(output_root / row["geometry_file"]) as sample:
                solid = np.asarray(sample["solid"])
            ax = axes[row_index, column_index]
            ax.imshow(solid.T, cmap="gray_r", origin="lower", interpolation="nearest", vmin=0, vmax=1)
            if row_index == 0:
                ax.set_title(rf"Target $\phi={porosity:.2f}$", fontsize=9)
            if column_index == 0:
                ax.set_ylabel(FAMILY_LABELS[family], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
    fig.text(0.5, 0.02, "Flow direction: left to right", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.97, bottom=0.06, wspace=0.05, hspace=0.08)
    _save_figure(fig, output_root / "figures" / "rule_ood_geometry_atlas")


def plot_lbm_atlas(output_root: Path, rows: list[dict[str, Any]]) -> None:
    apply_nature_style(font_size=8.5)
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if abs(float(row["target_porosity"]) - 0.55) < 0.02:
            selected.setdefault(str(row["family"]), row)
    if len(selected) != len(OOD_FAMILIES):
        for row in rows:
            selected.setdefault(str(row["family"]), row)
    families = [family for family in OOD_FAMILIES if family in selected]
    if not families:
        return
    speed_limit = 5.024437996326014e-06
    fig, axes = plt.subplots(len(families), 4, figsize=(9.0, 1.85 * len(families)), squeeze=False)
    for row_index, family in enumerate(families):
        row = selected[family]
        with np.load(output_root / row["truth_file"]) as sample:
            solid = np.asarray(sample["solid"])
            connected = np.asarray(sample["through_connected_pore"])
            edt = np.asarray(sample["edt_lattice"]) / 100.0
            ux = np.asarray(sample["transient_velocity_x_raw_lattice"][-1])
            uy = np.asarray(sample["transient_velocity_y_raw_lattice"][-1])
            speed = np.sqrt(ux * ux + uy * uy) / speed_limit
        fields = (solid, connected, edt, speed)
        cmaps = ("gray_r", "gray", "viridis", "magma")
        limits = ((0, 1), (0, 1), (0, 0.2), (0, 1))
        for column_index, (field, cmap, limit) in enumerate(zip(fields, cmaps, limits)):
            ax = axes[row_index, column_index]
            image = ax.imshow(field.T, origin="lower", cmap=cmap, vmin=limit[0], vmax=limit[1])
            if row_index == 0:
                ax.set_title(("Solid mask", "Connected pore", "EDT / 100", r"Terminal $|u|/s_u$")[column_index])
            if column_index == 0:
                ax.set_ylabel(FAMILY_LABELS[family])
            ax.set_xticks([])
            ax.set_yticks([])
    fig.subplots_adjust(left=0.13, right=0.975, top=0.96, bottom=0.035, wspace=0.08, hspace=0.045)
    _save_figure(fig, output_root / "figures" / "rule_ood_lbm_atlas")


def plot_model_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    apply_nature_style(font_size=8.7)
    models = [name for name in MODEL_LABELS if any(row["model"] == name for row in rows)]
    x = np.arange(len(OOD_FAMILIES), dtype=float)
    width = 0.78 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8.0, 3.45))
    colors = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    for index, model in enumerate(models):
        values = []
        for family in OOD_FAMILIES:
            match = next(row for row in rows if row["model"] == model and row["family"] == family)
            values.append(float(match["terminal_mean_relative_l2_velocity"]))
        ax.bar(x + (index - (len(models) - 1) / 2) * width, values, width, label=MODEL_LABELS[model], color=colors[index])
    ax.set_xticks(x, [FAMILY_LABELS[family] for family in OOD_FAMILIES])
    ax.set_ylabel(r"Terminal relative $L_2$ velocity error")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, output_root / "figures" / "rule_ood_model_comparison")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-geometry OOD benchmark for transient pore-flow surrogates")
    parser.add_argument("--stage", choices=("generate", "lbm", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output" / "ood_rule_validation")
    parser.add_argument("--porosities", type=float, nargs="+", default=[0.60, 0.68])
    parser.add_argument("--samples-per-condition", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snapshots", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=50000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Formal trained run directory; repeat for each model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    if args.stage in {"generate", "all"}:
        generate_geometries(output_root, args.porosities, args.samples_per_condition, args.seed)
    if args.stage in {"lbm", "all"}:
        run_lbm_truth(output_root, args.max_samples, args.snapshots, args.max_iterations)
    if args.stage in {"evaluate", "all"}:
        run_dirs = args.run_dir or [
            ROOT / "output" / "formal_results" / name for name in MODEL_LABELS
        ]
        evaluate_models(output_root, [path.resolve() for path in run_dirs])


if __name__ == "__main__":
    main()
