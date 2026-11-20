from __future__ import annotations

import argparse
import importlib.util
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

import numpy as np


def load_generator(path: Path):
    module_name = "lbm_dataset_generator_dynamic"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def process_sample(
    source: str,
    destination: str,
    generator_path: str,
    solver_values: dict[str, Any],
) -> dict[str, Any]:
    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.exists():
        return {"source": source, "status": "skipped", "error": ""}

    generator = load_generator(Path(generator_path))
    solver_names = {item.name for item in fields(generator.SolverConfig)}
    config = generator.SolverConfig(
        **{key: value for key, value in solver_values.items() if key in solver_names}
    )

    try:
        with np.load(source_path) as sample:
            solid = np.asarray(sample["solid"], dtype=np.uint8)

        result = generator.run_lbm(solid, config)
        snapshots = result["transient_snapshots"]
        start = int(result["core_start"])
        end = int(result["core_end"])
        pore = solid == 0
        transient_ux = np.stack(
            [np.asarray(item["ux"][start:end], dtype=np.float32) for item in snapshots]
        )
        transient_uy = np.stack(
            [np.asarray(item["uy"][start:end], dtype=np.float32) for item in snapshots]
        )
        transient_rho = np.stack(
            [np.asarray(item["rho"][start:end], dtype=np.float32) for item in snapshots]
        )
        transient_ux[:, ~pore] = 0.0
        transient_uy[:, ~pore] = 0.0
        transient_rho[:, ~pore] = 0.0
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                transient_iterations=np.asarray(
                    [item["iteration"] for item in snapshots], dtype=np.int32
                ),
                transient_source_iterations=np.asarray(
                    [item["source_iteration"] for item in snapshots], dtype=np.int32
                ),
                transient_steady_fill_mask=np.asarray(
                    [item["source_iteration"] != item["iteration"] for item in snapshots],
                    dtype=np.uint8,
                ),
                velocity_x_raw_lattice=transient_ux,
                velocity_y_raw_lattice=transient_uy,
                density_raw_lattice=transient_rho,
                converged=np.uint8(result["converged"]),
                iterations=np.int32(result["iterations"]),
                velocity_residual=np.float64(result["velocity_residual"]),
                mass_balance_error=np.float64(result["mass_balance_error"]),
                permeability_lattice=np.float64(result["permeability_lattice"]),
            )
        os.replace(temporary, destination_path)
        return {
            "source": source,
            "status": "complete",
            "snapshots": len(snapshots),
            "iterations": int(result["iterations"]),
            "converged": bool(result["converged"]),
            "error": "",
        }
    except Exception as exc:
        return {
            "source": source,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add log-spaced transient LBM snapshots beside an existing dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output-name", default="transient_snapshots")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Independent output directory; defaults to DATASET_ROOT/OUTPUT_NAME.",
    )
    parser.add_argument("--snapshots", type=int, default=30)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    if not 2 <= args.snapshots <= 100:
        raise SystemExit("--snapshots must be between 2 and 100")
    config_path = args.dataset_root / "config.json"
    samples_root = args.dataset_root / "samples"
    if not config_path.exists() or not samples_root.is_dir():
        raise SystemExit("dataset root must contain config.json and samples/")

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    solver_values = dict(config_payload["solver"])
    solver_values["transient_snapshots"] = args.snapshots
    sources = sorted(samples_root.rglob("*.npz"))
    if args.max_samples > 0:
        sources = sources[: args.max_samples]

    output_root = args.output_root or (args.dataset_root / args.output_name)
    tasks = [
        (
            str(source),
            str(output_root / source.relative_to(samples_root)),
            str(args.generator),
            solver_values,
        )
        for source in sources
    ]
    counts = {"complete": 0, "skipped": 0, "failed": 0}
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_sample, *task) for task in tasks]
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            counts[result["status"]] += 1
            if result["status"] == "failed":
                failures.append(result)
            if number == 1 or number % 25 == 0 or number == len(futures):
                print(
                    f"{number}/{len(futures)} complete={counts['complete']} "
                    f"skipped={counts['skipped']} failed={counts['failed']}",
                    flush=True,
                )

    summary = {
        "requested_snapshots": args.snapshots,
        "fixed_transient_iterations": list(
            (0, *load_generator(args.generator).logarithmic_snapshot_iterations(
                int(solver_values["max_iterations"]), args.snapshots
            ))
        ),
        "tensor_shape_per_sample": [args.snapshots, 3, int(solver_values["nx"]), int(solver_values["ny"])],
        "tensor_fields": [
            "velocity_x_raw_lattice",
            "velocity_y_raw_lattice",
            "density_raw_lattice",
        ],
        "tensor_dtype": "float32",
        "normalization": "none; fit normalization using the training split only",
        "source_samples": len(sources),
        **counts,
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
