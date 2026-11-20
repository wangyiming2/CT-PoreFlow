from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "remote_source"))
sys.path.insert(0, str(PROJECT_ROOT))

from lbm_dataset_generator import (  # noqa: E402
    SolverConfig,
    euclidean_distance_to_solid,
    through_connected_components,
)
from inverse_design.data import ConditionStats  # noqa: E402
from inverse_design.models import ConditionalDiffusionUNet, ConditionalPoreGenerator  # noqa: E402
from inverse_design.qsgs_closed_loop import (  # noqa: E402
    ConditionStats as DistanceStats,
    TARGET_WEIGHTS,
    _distance,
    _novel_vector,
    _target_vector,
    screen_novel_pool,
    verify_candidate,
)
from inverse_design.sample import diffusion_sample  # noqa: E402
from precompute_topology import geodesic  # noqa: E402


METHODS = ("cgan", "diffusion")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def target_raw(target: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [
            float(target["target_porosity"]),
            float(target["target_connected_porosity"]),
            float(target["target_dead_pore_ratio"]),
            math.log1p(float(target["target_permeability_lattice"])),
            float(target["target_hydraulic_tortuosity"]),
        ],
        dtype=np.float32,
    )


def load_inverse_model(method: str, run_dir: Path, device: torch.device):
    checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if method == "cgan":
        model = ConditionalPoreGenerator(
            latent_dim=int(config["latent_dim"]), base=int(config["base_channels"])
        )
        model.load_state_dict(checkpoint["generator"])
    else:
        model = ConditionalDiffusionUNet(base=int(config["base_channels"]))
        model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config


@torch.inference_mode()
def generate_masks(
    method: str,
    model,
    config: dict,
    normalized_condition: np.ndarray,
    count: int,
    batch_size: int,
    guidance: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        condition = torch.from_numpy(normalized_condition).to(device)[None].repeat(size, 1)
        if method == "cgan":
            generated = model(
                torch.randn(size, int(config["latent_dim"]), device=device), condition
            )
        else:
            generated = diffusion_sample(
                model,
                condition,
                int(config["diffusion_steps"]),
                guidance,
                device,
            )
        batches.append((generated[:, 0].cpu().numpy() > 0).astype(np.uint8))
    return np.concatenate(batches, axis=0)


def candidate_from_mask(mask: np.ndarray, identifier: str) -> dict[str, object]:
    solid = (1 - mask).astype(np.uint8)
    through, connected_porosity, connected_fraction = through_connected_components(solid)
    porosity = float(mask.mean())
    dead_ratio = 1.0 - connected_fraction if porosity > 0 else 1.0
    edt_raw = euclidean_distance_to_solid(solid)
    geometry = np.stack(
        [
            mask.astype(np.float32),
            through.astype(np.float32),
            (edt_raw / 100.0).astype(np.float32),
            geodesic(through.astype(bool), True),
            geodesic(through.astype(bool), False),
        ]
    )
    return {
        "candidate_id": identifier,
        "solid": solid,
        "through": through.astype(np.uint8),
        "edt_raw": edt_raw,
        "geometry": geometry,
        "porosity": porosity,
        "connected_porosity": connected_porosity,
        "dead_pore_ratio": float(dead_ratio),
        "spans_flow": bool(np.any(through)),
    }


def diversity_metrics(masks: np.ndarray, seed: int, pair_count: int = 2048) -> dict[str, float]:
    fingerprints = [hashlib.sha256(mask.tobytes()).hexdigest() for mask in masks]
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(masks), size=pair_count)
    right = rng.integers(0, len(masks), size=pair_count)
    same = left == right
    right[same] = (right[same] + 1) % len(masks)
    hamming: list[float] = []
    iou: list[float] = []
    for a, b in zip(left, right):
        first = masks[int(a)].astype(bool)
        second = masks[int(b)].astype(bool)
        hamming.append(float(np.mean(first != second)))
        union = np.count_nonzero(first | second)
        iou.append(float(np.count_nonzero(first & second) / union) if union else 1.0)
    return {
        "unique_rate": len(set(fingerprints)) / len(fingerprints),
        "pairwise_hamming_mean": float(np.mean(hamming)),
        "pairwise_hamming_std": float(np.std(hamming)),
        "pairwise_iou_mean": float(np.mean(iou)),
        "pairwise_iou_std": float(np.std(iou)),
    }


def sample_and_screen(args: argparse.Namespace, targets: list[dict[str, object]]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qc_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        run_dir = args.cgan_run if method == "cgan" else args.diffusion_run
        model, config = load_inverse_model(method, run_dir, device)
        condition_stats = ConditionStats.load(run_dir / "condition_stats.json")
        distance_stats = DistanceStats(condition_stats.mean.astype(np.float64), condition_stats.std.astype(np.float64))
        for target_index, target in enumerate(targets):
            target_id = str(target["target_id"])
            destination = args.output_root / "generated" / method / f"{target_id}.npz"
            if destination.exists():
                print(json.dumps({"stage": "sample", "method": method, "target": target_id, "status": "cached"}), flush=True)
                continue
            seed = args.seed + method_index * 1_000_000 + target_index * 10_000
            raw = target_raw(target)
            normalized = condition_stats.normalize(raw).astype(np.float32)
            started = time.perf_counter()
            reused = (
                args.reuse_root / "generated" / method / f"{target_id}.npz"
                if args.reuse_root is not None
                else None
            )
            if reused is not None and reused.exists():
                with np.load(reused) as data:
                    masks = np.asarray(data["pore_masks"], dtype=np.uint8)
                if len(masks) != args.count:
                    raise ValueError(
                        f"{reused}: expected {args.count} cached masks, found {len(masks)}"
                    )
                sampling_source = "reused_masks_rescored"
            else:
                masks = generate_masks(
                    method,
                    model,
                    config,
                    normalized,
                    args.count,
                    args.cgan_batch if method == "cgan" else args.diffusion_batch,
                    args.guidance,
                    seed,
                    device,
                )
                sampling_source = "generated"
            candidates = [
                candidate_from_mask(mask, f"{method}_{target_id}_{index:04d}")
                for index, mask in enumerate(masks)
            ]
            velocity_scale, proxy_device = screen_novel_pool(candidates, args.forward_run, args.proxy_batch)
            target_vector = _target_vector(target)
            arrays: dict[str, np.ndarray] = {
                "pore_masks": masks,
                "porosity": np.asarray([item["porosity"] for item in candidates], dtype=np.float32),
                "connected_porosity": np.asarray([item["connected_porosity"] for item in candidates], dtype=np.float32),
                "dead_pore_ratio": np.asarray([item["dead_pore_ratio"] for item in candidates], dtype=np.float32),
                "spans_flow": np.asarray([item["spans_flow"] for item in candidates], dtype=np.uint8),
                "predicted_permeability_lattice": np.asarray([item["predicted_permeability_lattice"] for item in candidates], dtype=np.float32),
                "predicted_hydraulic_tortuosity": np.asarray([item["predicted_hydraulic_tortuosity"] for item in candidates], dtype=np.float32),
                "predicted_mass_balance_error": np.asarray([item["predicted_mass_balance_error"] for item in candidates], dtype=np.float32),
                "target_distance": np.asarray([_distance(_novel_vector(item), target_vector, distance_stats) for item in candidates], dtype=np.float32),
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **arrays)
            diversity = diversity_metrics(masks, seed)
            valid = arrays["spans_flow"].astype(bool) & np.isfinite(arrays["target_distance"])
            summary_rows.append(
                {
                    "method": method,
                    "target_id": target_id,
                    "generated_count": len(masks),
                    "spanning_rate": float(arrays["spans_flow"].mean()),
                    "mean_porosity": float(arrays["porosity"].mean()),
                    "mean_connected_porosity": float(arrays["connected_porosity"].mean()),
                    "mean_dead_pore_ratio": float(arrays["dead_pore_ratio"].mean()),
                    "mean_proxy_target_distance": float(arrays["target_distance"][valid].mean()) if np.any(valid) else math.nan,
                    "best_proxy_target_distance": float(arrays["target_distance"][valid].min()) if np.any(valid) else math.nan,
                    "sampling_and_screening_seconds": time.perf_counter() - started,
                    "proxy_device": proxy_device,
                    "sampling_source": sampling_source,
                    "velocity_scale": velocity_scale,
                    **diversity,
                }
            )
            for index in range(len(masks)):
                qc_rows.append(
                    {
                        "method": method,
                        "target_id": target_id,
                        "candidate_index": index,
                        "seed": seed + index,
                        **{name: float(values[index]) for name, values in arrays.items() if name != "pore_masks"},
                    }
                )
            write_csv(args.output_root / "candidate_quality.csv", qc_rows)
            write_csv(args.output_root / "diversity_summary.csv", summary_rows)
            print(json.dumps({"stage": "sample", "method": method, "target": target_id, "seconds": time.perf_counter() - started}), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_selections(args: argparse.Namespace, targets: list[dict[str, object]]) -> list[dict[str, object]]:
    rng = np.random.default_rng(args.seed + 77)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        for target in targets:
            target_id = str(target["target_id"])
            path = args.output_root / "generated" / method / f"{target_id}.npz"
            with np.load(path) as data:
                arrays = {name: np.asarray(data[name]) for name in data.files}
            valid = np.flatnonzero(
                arrays["spans_flow"].astype(bool)
                & np.isfinite(arrays["target_distance"])
                & (arrays["predicted_permeability_lattice"] > 0)
                & np.isfinite(arrays["predicted_hydraulic_tortuosity"])
                & (arrays["predicted_hydraulic_tortuosity"] >= 0.8)
                & (arrays["predicted_hydraulic_tortuosity"] <= 8.0)
            )
            ranked = valid[np.argsort(arrays["target_distance"][valid])]
            guided_indexes = ranked[: args.verify_per_strategy]
            guided_set = {int(index) for index in guided_indexes}
            random_order = np.asarray(
                [index for index in valid if int(index) not in guided_set], dtype=np.int64
            )
            rng.shuffle(random_order)
            for strategy, indexes in (("guided", guided_indexes), ("random", random_order[: args.verify_per_strategy])):
                for rank, index in enumerate(indexes, 1):
                    rows.append(
                        {
                            "method": method,
                            "target_id": target_id,
                            "strategy": strategy,
                            "selection_rank": rank,
                            "candidate_index": int(index),
                            "candidate_id": f"{method}_{target_id}_{int(index):04d}",
                            "proxy_target_distance": float(arrays["target_distance"][index]),
                            "predicted_permeability_lattice": float(arrays["predicted_permeability_lattice"][index]),
                            "predicted_hydraulic_tortuosity": float(arrays["predicted_hydraulic_tortuosity"][index]),
                            "selected_porosity": float(arrays["porosity"][index]),
                            "selected_connected_porosity": float(arrays["connected_porosity"][index]),
                            "selected_dead_pore_ratio": float(arrays["dead_pore_ratio"][index]),
                            **target,
                        }
                    )
            if len(ranked) < args.verify_per_strategy:
                print(json.dumps({"stage": "select", "method": method, "target": target_id, "valid": len(ranked), "warning": "insufficient_valid_candidates"}), flush=True)
    write_csv(args.output_root / "selected_candidates.csv", rows)
    return rows


def lbm_worker(payload: tuple[dict[str, object], np.ndarray]) -> tuple[dict[str, object], dict[str, object]]:
    row, pore = payload
    candidate = candidate_from_mask(pore, str(row["candidate_id"]))
    result = verify_candidate(candidate, SolverConfig())
    return row, result


def run_lbm_roundtrip(args: argparse.Namespace, selections: list[dict[str, object]]) -> None:
    tasks: list[tuple[dict[str, object], np.ndarray]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in selections:
        grouped.setdefault((str(row["method"]), str(row["target_id"])), []).append(row)
    existing_path = args.output_root / "lbm_verification.csv"
    completed: list[dict[str, object]] = []
    if existing_path.exists():
        with existing_path.open(newline="", encoding="utf-8-sig") as handle:
            completed.extend(csv.DictReader(handle))
    completed_ids = {str(row["candidate_id"]) for row in completed}
    physical_keys = (
        "verified_target_distance",
        "verified_permeability_lattice",
        "verified_hydraulic_tortuosity",
        "converged",
        "quality_pass",
        "iterations",
        "seconds",
        "mass_balance_error",
        "velocity_residual",
        "permeability_relative_change",
        "max_mach",
    )
    for (method, target_id), rows in grouped.items():
        with np.load(args.output_root / "generated" / method / f"{target_id}.npz") as data:
            masks = np.asarray(data["pore_masks"], dtype=np.uint8)
        for row in rows:
            destination = args.output_root / "lbm_fields" / f"{row['candidate_id']}.npz"
            candidate_id = str(row["candidate_id"])
            if candidate_id in completed_ids:
                continue
            reused = (
                args.reuse_root / "lbm_fields" / f"{candidate_id}.npz"
                if args.reuse_root is not None
                else None
            )
            if reused is not None and reused.exists():
                with np.load(reused) as fields:
                    metadata = json.loads(str(fields["metadata_json"].item()))
                completed.append(
                    {
                        **row,
                        **{key: metadata[key] for key in physical_keys if key in metadata},
                    }
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(reused, destination)
                completed_ids.add(candidate_id)
            else:
                tasks.append((row, masks[int(row["candidate_index"])]))
    if completed:
        write_csv(existing_path, completed)
    with ProcessPoolExecutor(max_workers=args.lbm_workers) as executor:
        futures = {executor.submit(lbm_worker, task): task[0] for task in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            row, result = future.result()
            target_vector = target_raw(row)
            verified_vector = np.asarray(
                [
                    float(row["selected_porosity"]),
                    float(row["selected_connected_porosity"]),
                    float(row["selected_dead_pore_ratio"]),
                    math.log1p(float(result["verified_permeability_lattice"])),
                    float(result["verified_hydraulic_tortuosity"]),
                ],
                dtype=np.float64,
            )
            stats = ConditionStats.load(args.cgan_run / "condition_stats.json")
            distance = float(np.linalg.norm(((verified_vector - target_vector) / stats.std) * TARGET_WEIGHTS))
            scalar = {
                **row,
                "verified_target_distance": distance,
                "verified_permeability_lattice": result["verified_permeability_lattice"],
                "verified_hydraulic_tortuosity": result["verified_hydraulic_tortuosity"],
                "converged": result["converged"],
                "quality_pass": result["quality_pass"],
                "iterations": result["iterations"],
                "seconds": result["seconds"],
                "mass_balance_error": result["mass_balance_error"],
                "velocity_residual": result["velocity_residual"],
                "permeability_relative_change": result["permeability_relative_change"],
                "max_mach": result["max_mach"],
            }
            completed.append(scalar)
            field_dir = args.output_root / "lbm_fields"
            field_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                field_dir / f"{row['candidate_id']}.npz",
                velocity_x_lattice=result["verified_velocity_x_lattice"],
                velocity_y_lattice=result["verified_velocity_y_lattice"],
                density_lattice=result["verified_density_lattice"],
                metadata_json=np.asarray(json.dumps(scalar, ensure_ascii=False)),
            )
            write_csv(existing_path, completed)
            print(json.dumps({"stage": "lbm", "done": number, "total_new": len(tasks), "candidate": row["candidate_id"], "quality_pass": bool(result["quality_pass"])}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--cgan-run", type=Path, required=True)
    parser.add_argument("--diffusion-run", type=Path, required=True)
    parser.add_argument("--forward-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reuse-root",
        type=Path,
        help="Reuse generated masks and any matching LBM fields, while rescoring with the new forward model.",
    )
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--verify-per-strategy", type=int, default=6)
    parser.add_argument("--cgan-batch", type=int, default=64)
    parser.add_argument("--diffusion-batch", type=int, default=16)
    parser.add_argument("--proxy-batch", type=int, default=16)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--lbm-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-targets", type=int, default=0)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    targets = json.loads(args.targets.read_text(encoding="utf-8-sig"))
    if args.max_targets > 0:
        targets = targets[: args.max_targets]
    write_json(args.output_root / "targets.json", targets)
    write_json(args.output_root / "manifest.json", {"status": "sampling", "target_count": len(targets), "count_per_target": args.count, "methods": METHODS, "seed": args.seed})
    sample_and_screen(args, targets)
    write_json(args.output_root / "manifest.json", {"status": "selecting", "target_count": len(targets), "count_per_target": args.count, "methods": METHODS, "seed": args.seed})
    selections = build_selections(args, targets)
    write_json(args.output_root / "manifest.json", {"status": "lbm", "selection_count": len(selections), "seed": args.seed})
    run_lbm_roundtrip(args, selections)
    write_json(args.output_root / "manifest.json", {"status": "complete", "target_count": len(targets), "count_per_target": args.count, "generated_total": len(targets) * args.count * len(METHODS), "selection_count": len(selections), "seed": args.seed})


if __name__ == "__main__":
    main()
