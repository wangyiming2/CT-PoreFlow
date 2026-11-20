from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples-per-file", type=int, default=512)
    args = parser.parse_args()
    config = load_json(args.config)
    steady_root = Path(config["steady_root"]) / "samples" / "train"
    transient_root = Path(config["transient_root"])
    rng = np.random.default_rng(int(config["seed"]))
    reservoir: list[np.ndarray] = []
    sum_sq = 0.0
    count = 0
    files = sorted(steady_root.rglob("*.npz"))
    for number, steady in enumerate(files, 1):
        relative = steady.relative_to(Path(config["steady_root"]) / "samples")
        with np.load(steady) as s, np.load(transient_root / relative) as t:
            pore = np.asarray(s["pore"], dtype=bool)
            ux = np.asarray(t["velocity_x_raw_lattice"], dtype=np.float64)
            uy = np.asarray(t["velocity_y_raw_lattice"], dtype=np.float64)
            speed = np.sqrt(ux[:, pore] ** 2 + uy[:, pore] ** 2).reshape(-1)
            sum_sq += float(np.dot(speed, speed))
            count += speed.size
            take = min(args.samples_per_file, speed.size)
            reservoir.append(speed[rng.choice(speed.size, size=take, replace=False)].astype(np.float32))
        if number % 250 == 0:
            print(f"{number}/{len(files)}", flush=True)
    sampled = np.concatenate(reservoir)
    velocity_scale = float(np.quantile(sampled, 0.995))
    payload = {
        "method": "training-pore speed q99.5 from deterministic per-file subsample",
        "seed": int(config["seed"]),
        "training_files": len(files),
        "sampled_points": int(sampled.size),
        "all_pore_time_points": int(count),
        "speed_rms": float(np.sqrt(sum_sq / count)),
        "speed_q995": velocity_scale,
        "velocity_scale": velocity_scale,
    }
    save_json(config["normalization_file"], payload)
    print(payload)


if __name__ == "__main__":
    main()
