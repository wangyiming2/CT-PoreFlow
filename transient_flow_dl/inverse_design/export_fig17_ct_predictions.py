from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DL_ROOT))

from src.models import build_model  # noqa: E402


TARGET_IDS = ("P1-KH-TL", "P2-KM-TL", "P3-KH-TL")
METHODS = ("cgan", "diffusion")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_cases(rows: list[dict[str, str]], asset_root: Path) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for target_id in TARGET_IDS:
        for method in METHODS:
            options = [
                row
                for row in rows
                if row["target_id"] == target_id
                and row["method"] == method
                and row["strategy"] == "guided"
                and row["quality_pass"].lower() == "true"
                and (asset_root / "lbm_fields" / f"{row['candidate_id']}.npz").exists()
            ]
            if not options:
                options = [
                    row
                    for row in rows
                    if row["target_id"] == target_id
                    and row["method"] == method
                    and row["strategy"] == "guided"
                    and (asset_root / "lbm_fields" / f"{row['candidate_id']}.npz").exists()
                ]
            if not options:
                raise RuntimeError(f"No saved candidate fields for {method}/{target_id}")
            cases.append(min(options, key=lambda row: float(row["verified_target_distance"])))
    return cases


def through_connected_pore(pore: np.ndarray) -> np.ndarray:
    height, width = pore.shape
    visited = np.zeros_like(pore, dtype=bool)
    through = np.zeros_like(pore, dtype=bool)
    for start_x in range(height):
        for start_y in range(width):
            if not pore[start_x, start_y] or visited[start_x, start_y]:
                continue
            queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
            visited[start_x, start_y] = True
            component: list[tuple[int, int]] = []
            touches_inlet = False
            touches_outlet = False
            while queue:
                x, y = queue.popleft()
                component.append((x, y))
                touches_inlet = touches_inlet or x == 0
                touches_outlet = touches_outlet or x == height - 1
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < height and 0 <= ny < width and pore[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        queue.append((nx, ny))
            if touches_inlet and touches_outlet:
                for x, y in component:
                    through[x, y] = True
    return through


def squared_distance_transform_1d(values: np.ndarray) -> np.ndarray:
    size = int(values.size)
    locations = np.zeros(size, dtype=np.int32)
    boundaries = np.empty(size + 1, dtype=np.float64)
    output = np.empty(size, dtype=np.float64)
    k = 0
    locations[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf
    for q in range(1, size):
        location = locations[k]
        intersection = ((values[q] + q * q) - (values[location] + location * location)) / (
            2.0 * q - 2.0 * location
        )
        while intersection <= boundaries[k]:
            k -= 1
            location = locations[k]
            intersection = ((values[q] + q * q) - (values[location] + location * location)) / (
                2.0 * q - 2.0 * location
            )
        k += 1
        locations[k] = q
        boundaries[k] = intersection
        boundaries[k + 1] = np.inf
    k = 0
    for q in range(size):
        while boundaries[k + 1] < q:
            k += 1
        location = locations[k]
        output[q] = (q - location) ** 2 + values[location]
    return output


def euclidean_distance_to_solid(pore: np.ndarray) -> np.ndarray:
    solid = ~pore
    if not np.any(solid):
        return np.full(pore.shape, math.hypot(*pore.shape), dtype=np.float64)
    distance_squared = np.where(solid, 0.0, 1.0e12)
    first_pass = np.empty_like(distance_squared)
    second_pass = np.empty_like(distance_squared)
    for row in range(pore.shape[0]):
        first_pass[row, :] = squared_distance_transform_1d(distance_squared[row, :])
    for column in range(pore.shape[1]):
        second_pass[:, column] = squared_distance_transform_1d(first_pass[:, column])
    distance = np.sqrt(second_pass)
    distance[solid] = 0.0
    return distance


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


def geometry_features(pore: np.ndarray) -> np.ndarray:
    pore_bool = np.asarray(pore, dtype=bool)
    through = through_connected_pore(pore_bool)
    edt = euclidean_distance_to_solid(pore_bool).astype(np.float32) / float(pore.shape[0])
    inlet = geodesic(through, True)
    outlet = geodesic(through, False)
    return np.stack([pore_bool, through, edt, inlet, outlet], axis=0).astype(np.float32)


def main() -> None:
    formal_root = PROJECT_ROOT / "output" / "formal_results" / "inverse_formal_evaluation_v2_seed2026"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=formal_root / "analysis" / "generative_inverse_lbm_enriched.csv",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "inverse_generative",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "formal_results" / "ct_poreflow_v2_transport_seed2026",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=formal_root / "analysis" / "fig17_ct_predictions.npz",
    )
    args = parser.parse_args()

    cases = select_cases(read_csv(args.rows), args.asset_root)
    pore_masks: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
    candidate_ids: list[str] = []
    for row in cases:
        with np.load(args.asset_root / "generated" / row["method"] / f"{row['target_id']}.npz") as generated:
            pore = np.asarray(generated["pore_masks"][int(row["candidate_index"])], dtype=np.uint8)
        pore_masks.append(pore)
        geometries.append(geometry_features(pore))
        candidate_ids.append(row["candidate_id"])

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    geometry_tensor = torch.from_numpy(np.stack(geometries)).to(device)
    query_time = torch.ones(len(cases), dtype=torch.float32, device=device)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            normalized = model(geometry_tensor, query_time).float().cpu().numpy()

    velocity_scale = float(checkpoint["velocity_scale"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        candidate_ids=np.asarray(candidate_ids),
        pore_masks=np.stack(pore_masks).astype(np.uint8),
        query_time=np.ones(len(cases), dtype=np.float32),
        normalized_fields=normalized.astype(np.float32),
        velocity_x_lattice=(normalized[:, 0] * velocity_scale).astype(np.float32),
        velocity_y_lattice=(normalized[:, 1] * velocity_scale).astype(np.float32),
        pressure_normalized=normalized[:, 2].astype(np.float32),
        velocity_scale=np.asarray(velocity_scale, dtype=np.float64),
        checkpoint=np.asarray(args.run_dir.name),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "device": str(device),
                "samples": len(cases),
                "output": str(args.output),
                "checkpoint": args.run_dir.name,
                "velocity_scale": velocity_scale,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
