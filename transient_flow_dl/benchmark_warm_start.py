from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.data import TransientFlowDataset
from src.models import build_model
from src.utils import load_json
from src.warm_start_lbm import WarmStartConfig, run_lbm_from_state


def _metadata(npz: np.lib.npyio.NpzFile) -> dict:
    raw = npz["metadata_json"].item()
    return json.loads(str(raw))


def _relative_l2(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    difference = (prediction - truth)[:, mask]
    reference = truth[:, mask]
    return float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_cold_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark neural-prediction warm starts for the original D2Q9-BGK solver."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cold-cache",
        type=Path,
        help="Shared measured cold-start cache. Defaults to project analysis/cold_start_<N>.csv.",
    )
    args = parser.parse_args()

    config = load_json(args.run_dir / "config.json")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    velocity_scale = float(checkpoint["velocity_scale"])
    dataset = TransientFlowDataset(
        config["steady_root"],
        config["transient_root"],
        args.split,
        velocity_scale,
        config["rho_in"],
        config["rho_out"],
        config["max_iteration"],
        mode="fixed_times",
        time_indices=[29],
        max_samples=args.samples,
        seed=config["seed"],
        topology_root=config.get("topology_root"),
        time_transform=config.get("time_encoding", "log"),
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    solver_config = WarmStartConfig(
        rhoin=float(config["rho_in"]),
        rhoout=float(config["rho_out"]),
        tau=float(config.get("tau", 1.0)),
        max_iterations=int(config["max_iteration"]),
    )
    output = args.output or args.run_dir / "warm_start"
    output.mkdir(parents=True, exist_ok=True)
    cold_cache_path = args.cold_cache or (
        args.run_dir.parent.parent / "analysis" / f"cold_start_{args.split}_{args.samples}.csv"
    )
    cold_cache = _load_cold_cache(cold_cache_path)
    cold_rows: list[dict] = list(cold_cache.values())
    rows: list[dict[str, float | int | str | bool]] = []

    for index in range(len(dataset)):
        item = dataset[index]
        geometry = item["geometry"][None].to(device)
        query_time = item["time"][None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_inference = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            prediction = model(geometry, query_time)[0].float()
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start_inference
        prediction = prediction.cpu().numpy()

        steady_path, _, sample_id = dataset.pairs[index]
        with np.load(steady_path) as steady:
            solid = np.asarray(steady["solid"], dtype=np.uint8)
            pore = solid == 0
            truth = np.stack(
                [
                    np.asarray(steady["velocity_x_raw_lattice"], dtype=np.float64),
                    np.asarray(steady["velocity_y_raw_lattice"], dtype=np.float64),
                    np.asarray(steady["density_raw_lattice"], dtype=np.float64),
                ]
            )
            metadata = _metadata(steady)

        if sample_id in cold_cache:
            cached = cold_cache[sample_id]
            cold_iterations_measured = int(float(cached["cold_iterations_measured"]))
            cold_solver_seconds = float(cached["cold_solver_seconds"])
            cold_converged = cached["cold_converged"].lower() == "true"
            cold_finite_state = cached["cold_finite_state"].lower() == "true"
            cold_mach_valid = cached["cold_mach_valid"].lower() == "true"
            cold_max_mach = float(cached["cold_max_mach"])
            cold_mass_balance_error = float(cached["cold_mass_balance_error"])
            cold_permeability_lattice = float(cached["cold_permeability_lattice"])
        else:
            cold_start = time.perf_counter()
            cold_result = run_lbm_from_state(solid, solver_config, None)
            cold_solver_seconds = time.perf_counter() - cold_start
            cold_iterations_measured = int(cold_result["iterations"])
            cold_converged = bool(cold_result["converged"])
            cold_finite_state = bool(cold_result["finite_state"])
            cold_mach_valid = bool(cold_result["mach_valid"])
            cold_max_mach = float(cold_result["max_mach"])
            cold_mass_balance_error = float(cold_result["mass_balance_error"])
            cold_permeability_lattice = float(cold_result["permeability_lattice"])
            cold_row = {
                "sample_id": sample_id,
                "cold_iterations_measured": cold_iterations_measured,
                "cold_solver_seconds": cold_solver_seconds,
                "cold_converged": cold_converged,
                "cold_finite_state": cold_finite_state,
                "cold_mach_valid": cold_mach_valid,
                "cold_max_mach": cold_max_mach,
                "cold_mass_balance_error": cold_mass_balance_error,
                "cold_permeability_lattice": cold_permeability_lattice,
            }
            cold_rows.append(cold_row)
            cold_cache[sample_id] = {key: str(value) for key, value in cold_row.items()}
            _write_rows(cold_cache_path, cold_rows)
        initial = {
            "ux": prediction[0] * velocity_scale,
            "uy": prediction[1] * velocity_scale,
            "rho": float(config["rho_out"])
            + prediction[2] * (float(config["rho_in"]) - float(config["rho_out"])),
        }
        start_solver = time.perf_counter()
        result = run_lbm_from_state(solid, solver_config, initial)
        solver_seconds = time.perf_counter() - start_solver
        final = np.stack([result["ux_core"], result["uy_core"], result["rho_core"]])
        initial_stack = np.stack([initial["ux"], initial["uy"], initial["rho"]])
        cold_iterations = int(metadata.get("iterations", metadata.get("lbm_iterations", 0)))
        warm_iterations = int(result["iterations"])
        total_neural_warm_seconds = inference_seconds + solver_seconds
        true_velocity = truth[:2]
        true_pressure = truth[2] - float(config["rho_out"])
        initial_pressure = initial_stack[2] - float(config["rho_out"])
        refined_pressure = final[2] - float(config["rho_out"])
        rows.append(
            {
                "sample_id": sample_id,
                "cold_iterations_recorded": cold_iterations,
                "cold_iterations_measured": cold_iterations_measured,
                "cold_solver_seconds": cold_solver_seconds,
                "cold_converged": cold_converged,
                "cold_finite_state": cold_finite_state,
                "cold_mach_valid": cold_mach_valid,
                "cold_max_mach": cold_max_mach,
                "cold_mass_balance_error": cold_mass_balance_error,
                "cold_permeability_lattice": cold_permeability_lattice,
                "warm_iterations": warm_iterations,
                "iteration_reduction_fraction": (
                    1.0 - warm_iterations / cold_iterations_measured
                    if cold_iterations_measured > 0
                    else math.nan
                ),
                "iteration_speedup": cold_iterations_measured / warm_iterations,
                "solver_wall_clock_speedup": cold_solver_seconds / solver_seconds,
                "end_to_end_wall_clock_speedup": cold_solver_seconds
                / total_neural_warm_seconds,
                "warm_converged": bool(result["converged"]),
                "warm_finite_state": bool(result["finite_state"]),
                "warm_mach_valid": bool(result["mach_valid"]),
                "warm_max_mach": float(result["max_mach"]),
                "inference_seconds": inference_seconds,
                "warm_solver_seconds": solver_seconds,
                "total_neural_warm_seconds": total_neural_warm_seconds,
                "initial_velocity_relative_l2": _relative_l2(
                    initial_stack[:2], true_velocity, pore
                ),
                "refined_velocity_relative_l2": _relative_l2(
                    final[:2], true_velocity, pore
                ),
                "initial_pressure_relative_l2": float(
                    np.linalg.norm((initial_pressure - true_pressure)[pore])
                    / max(np.linalg.norm(true_pressure[pore]), 1e-30)
                ),
                "refined_pressure_relative_l2": float(
                    np.linalg.norm((refined_pressure - true_pressure)[pore])
                    / max(np.linalg.norm(true_pressure[pore]), 1e-30)
                ),
                "warm_mass_balance_error": float(result["mass_balance_error"]),
                "warm_permeability_lattice": float(result["permeability_lattice"]),
                "true_permeability_lattice": float(
                    metadata.get("permeability_lattice", metadata.get("permeability", math.nan))
                ),
            }
        )
        print(
            f"[{index + 1}/{len(dataset)}] {sample_id} "
            f"cold={cold_iterations_measured} warm={warm_iterations} "
            f"converged={result['converged']}"
        )

    csv_path = output / "warm_start_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric_keys = [
        "cold_iterations_recorded",
        "cold_iterations_measured",
        "cold_solver_seconds",
        "cold_max_mach",
        "warm_iterations",
        "iteration_reduction_fraction",
        "iteration_speedup",
        "solver_wall_clock_speedup",
        "end_to_end_wall_clock_speedup",
        "inference_seconds",
        "warm_solver_seconds",
        "total_neural_warm_seconds",
        "initial_velocity_relative_l2",
        "refined_velocity_relative_l2",
        "initial_pressure_relative_l2",
        "refined_pressure_relative_l2",
        "warm_mass_balance_error",
        "warm_max_mach",
    ]
    summary = {
        "run_dir": str(args.run_dir),
        "split": args.split,
        "sample_count": len(rows),
        "cold_cache": str(cold_cache_path),
        "cold_convergence_rate": float(np.mean([bool(row["cold_converged"]) for row in rows])),
        "warm_convergence_rate": float(np.mean([bool(row["warm_converged"]) for row in rows])),
        "cold_mach_valid_rate": float(np.mean([bool(row["cold_mach_valid"]) for row in rows])),
        "warm_mach_valid_rate": float(np.mean([bool(row["warm_mach_valid"]) for row in rows])),
        "metrics": {
            key: {
                "mean": float(np.nanmean([float(row[key]) for row in rows])),
                "median": float(np.nanmedian([float(row[key]) for row in rows])),
                "std": float(np.nanstd([float(row[key]) for row in rows])),
            }
            for key in numeric_keys
        },
    }
    summary_path = output / "warm_start_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
