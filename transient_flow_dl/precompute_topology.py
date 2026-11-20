from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


def geodesic(mask: np.ndarray, from_inlet: bool) -> np.ndarray:
    height, width = mask.shape
    distance = np.full((height, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    row = 0 if from_inlet else height - 1
    for column in np.flatnonzero(mask[row]):
        distance[row, column] = 0
        queue.append((row, int(column)))
    while queue:
        x, y = queue.popleft()
        next_distance = distance[x, y] + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < height and 0 <= ny < width and mask[nx, ny] and distance[nx, ny] < 0:
                distance[nx, ny] = next_distance
                queue.append((nx, ny))
    normalized = np.zeros((height, width), dtype=np.float32)
    valid = distance >= 0
    normalized[valid] = distance[valid].astype(np.float32) / float(max(height, width))
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_root = Path(config["steady_root"]) / "samples"
    output_root = Path(config["topology_root"])
    files = sorted(source_root.rglob("*.npz"))
    for number, source in enumerate(files, 1):
        relative = source.relative_to(source_root)
        destination = output_root / relative
        if destination.exists():
            continue
        with np.load(source) as sample:
            connected = np.asarray(sample["through_connected_pore"], dtype=bool)
        inlet = geodesic(connected, True)
        outlet = geodesic(connected, False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, distance_from_inlet=inlet, distance_from_outlet=outlet)
        temporary.replace(destination)
        if number % 250 == 0:
            print(f"{number}/{len(files)}", flush=True)
    print(f"complete={len(files)} output={output_root}")


if __name__ == "__main__":
    main()
