from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()
    config = load_json(args.config)
    steady_samples = Path(config["steady_root"]) / "samples"
    transient_root = Path(config["transient_root"])
    expected_iterations = None
    report = {"splits": {}, "missing_pairs": [], "bad_files": []}
    for split in ["train", "validation", "test"]:
        files = sorted((steady_samples / split).rglob("*.npz"))
        if args.max_files:
            files = files[: args.max_files]
        checked = 0
        for steady in files:
            relative = steady.relative_to(steady_samples)
            transient = transient_root / relative
            if not transient.exists():
                report["missing_pairs"].append(relative.as_posix())
                continue
            try:
                with np.load(steady) as s, np.load(transient) as t:
                    iterations = np.asarray(t["transient_iterations"])
                    if expected_iterations is None:
                        expected_iterations = iterations
                    valid = (
                        s["solid"].shape == (100, 100)
                        and t["velocity_x_raw_lattice"].shape == (30, 100, 100)
                        and t["velocity_y_raw_lattice"].shape == (30, 100, 100)
                        and t["density_raw_lattice"].shape == (30, 100, 100)
                        and np.array_equal(iterations, expected_iterations)
                        and np.isfinite(t["velocity_x_raw_lattice"]).all()
                        and np.isfinite(t["velocity_y_raw_lattice"]).all()
                        and np.isfinite(t["density_raw_lattice"]).all()
                    )
                    if not valid:
                        report["bad_files"].append(relative.as_posix())
                checked += 1
            except Exception as exc:
                report["bad_files"].append(f"{relative.as_posix()}: {type(exc).__name__}: {exc}")
        report["splits"][split] = {"source_files": len(files), "checked_pairs": checked}
    report["expected_iterations"] = [] if expected_iterations is None else expected_iterations.tolist()
    report["ok"] = not report["missing_pairs"] and not report["bad_files"]
    output = Path(config["output_root"]).parent / "artifacts" / "dataset_audit.json"
    save_json(output, report)
    print(output)
    print(report)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
