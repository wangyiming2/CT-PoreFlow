from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DL_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "remote_source"))
sys.path.insert(0, str(PROJECT_ROOT))

from lbm_dataset_generator import (  # noqa: E402
    SolverConfig,
    euclidean_distance_to_solid,
    generate_qsgs,
    normalized_sample_fields,
    run_lbm,
    sample_qsgs_parameters,
    through_connected_components,
)
from precompute_topology import geodesic  # noqa: E402


CONDITION_NAMES = (
    "porosity",
    "connected_porosity",
    "dead_pore_ratio",
    "log1p_permeability",
    "hydraulic_tortuosity",
)
DISPLAY_NAMES = {
    "Random-QSGS": "Random QSGS",
    "Morphology-only": "Morphology only",
    "Time-U-Net": "Time-U-Net",
    "CT-PoreFlow": "CT-PoreFlow",
    "No-anchor": "Direct transient decoder",
    "LBM-oracle": "LBM oracle",
}
METHOD_COLORS = {
    "Random-QSGS": "#9E9E9E",
    "Morphology-only": "#CC79A7",
    "Time-U-Net": "#0072B2",
    "CT-PoreFlow": "#D55E00",
    "No-anchor": "#009E73",
    "LBM-oracle": "#222222",
}
TARGET_WEIGHTS = np.asarray([1.0, 0.7, 0.7, 2.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class ConditionStats:
    mean: np.ndarray
    std: np.ndarray

    def standardized(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_from_metadata(metadata: dict) -> np.ndarray:
    porosity = float(metadata["porosity"])
    connected = float(metadata["connected_porosity"])
    dead = 1.0 - connected / max(porosity, 1e-12)
    return np.asarray(
        [
            porosity,
            connected,
            dead,
            math.log1p(float(metadata["permeability_lattice"])),
            float(metadata["hydraulic_tortuosity"]),
        ],
        dtype=np.float64,
    )


def _raw_condition(
    porosity: float,
    connected: float,
    dead: float,
    permeability: float,
    tortuosity: float,
) -> np.ndarray:
    return np.asarray(
        [porosity, connected, dead, math.log1p(max(permeability, 0.0)), tortuosity],
        dtype=np.float64,
    )


def _raw_target_dict(identifier: str, values: np.ndarray) -> dict[str, float | str]:
    return {
        "target_id": identifier,
        "target_porosity": float(values[0]),
        "target_connected_porosity": float(values[1]),
        "target_dead_pore_ratio": float(values[2]),
        "target_permeability_lattice": float(np.expm1(values[3])),
        "target_hydraulic_tortuosity": float(values[4]),
    }


def load_metadata_split(steady_root: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((steady_root / "samples" / split).rglob("*.json")):
        metadata = _json(path)
        if metadata.get("status") != "valid":
            continue
        sample_path = Path(metadata["sample_file"])
        sample_id = sample_path.as_posix()
        if not sample_id.startswith(f"samples/{split}/"):
            sample_id = f"samples/{split}/{path.with_suffix('.npz').relative_to(steady_root / 'samples' / split).as_posix()}"
        metadata["sample_id_relative"] = sample_id.removeprefix("samples/")
        metadata["npz_path"] = str(path.with_suffix(".npz"))
        rows.append(metadata)
    if not rows:
        raise RuntimeError(f"No valid metadata in {steady_root / 'samples' / split}")
    return rows


def training_statistics(train_rows: list[dict]) -> tuple[ConditionStats, np.ndarray]:
    matrix = np.stack([_condition_from_metadata(row) for row in train_rows])
    return ConditionStats(matrix.mean(0), np.maximum(matrix.std(0), 1e-8)), matrix


def build_targets(train_matrix: np.ndarray) -> list[dict[str, float | str]]:
    targets: list[dict[str, float | str]] = []
    phi_quantiles = np.quantile(train_matrix[:, 0], [0.25, 0.50, 0.75])
    for phi_index, phi in enumerate(phi_quantiles, 1):
        order = np.argsort(np.abs(train_matrix[:, 0] - phi))
        local = train_matrix[order[: min(900, len(order))]]
        morphology = np.median(local[:, :3], axis=0)
        morphology[0] = phi
        for k_label, k_quantile in (("L", 0.25), ("M", 0.50), ("H", 0.75)):
            for tau_label, tau_quantile in (("L", 0.25), ("H", 0.75)):
                values = np.asarray(
                    [
                        morphology[0],
                        morphology[1],
                        morphology[2],
                        np.quantile(local[:, 3], k_quantile),
                        np.quantile(local[:, 4], tau_quantile),
                    ],
                    dtype=np.float64,
                )
                targets.append(_raw_target_dict(f"P{phi_index}-K{k_label}-T{tau_label}", values))
    return targets


def _target_vector(target: dict[str, float | str]) -> np.ndarray:
    return _raw_condition(
        float(target["target_porosity"]),
        float(target["target_connected_porosity"]),
        float(target["target_dead_pore_ratio"]),
        float(target["target_permeability_lattice"]),
        float(target["target_hydraulic_tortuosity"]),
    )


def load_final_predictions(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["iteration"])) != 50000:
                continue
            result[row["sample_id"]] = {
                "permeability": float(row["pred_permeability_lattice"]),
                "tortuosity": float(row["pred_hydraulic_tortuosity"]),
            }
    if not result:
        raise RuntimeError(f"No terminal predictions in {path}")
    return result


def assemble_test_candidates(
    test_rows: list[dict],
    prediction_files: dict[str, Path],
) -> list[dict[str, object]]:
    predictions = {name: load_final_predictions(path) for name, path in prediction_files.items()}
    candidates: list[dict[str, object]] = []
    for metadata in test_rows:
        sample_id = str(metadata["sample_id_relative"])
        if not all(sample_id in table for table in predictions.values()):
            continue
        true = _condition_from_metadata(metadata)
        candidate: dict[str, object] = {
            "sample_id": sample_id,
            "npz_path": metadata["npz_path"],
            "true_vector": true,
            "porosity": float(true[0]),
            "connected_porosity": float(true[1]),
            "dead_pore_ratio": float(true[2]),
            "true_permeability_lattice": float(np.expm1(true[3])),
            "true_hydraulic_tortuosity": float(true[4]),
        }
        for name, table in predictions.items():
            candidate[f"{name}_permeability"] = table[sample_id]["permeability"]
            candidate[f"{name}_tortuosity"] = table[sample_id]["tortuosity"]
        candidates.append(candidate)
    if len(candidates) != len(test_rows):
        raise RuntimeError(f"Prediction/metadata mismatch: {len(candidates)} of {len(test_rows)}")
    return candidates


def _candidate_vector(candidate: dict[str, object], method: str) -> np.ndarray:
    if method == "LBM-oracle":
        return np.asarray(candidate["true_vector"], dtype=np.float64)
    if method == "Morphology-only":
        true = np.asarray(candidate["true_vector"], dtype=np.float64)
        return true.copy()
    return _raw_condition(
        float(candidate["porosity"]),
        float(candidate["connected_porosity"]),
        float(candidate["dead_pore_ratio"]),
        float(candidate[f"{method}_permeability"]),
        float(candidate[f"{method}_tortuosity"]),
    )


def _distance(values: np.ndarray, target: np.ndarray, stats: ConditionStats, morphology_only: bool = False) -> float:
    delta = stats.standardized(values) - stats.standardized(target)
    if morphology_only:
        return float(np.linalg.norm(delta[:3] * TARGET_WEIGHTS[:3]))
    return float(np.linalg.norm(delta * TARGET_WEIGHTS))


def _selection_record(
    target: dict[str, float | str],
    candidate: dict[str, object],
    method: str,
    stats: ConditionStats,
    repeat: int = 0,
) -> dict[str, object]:
    target_vector = _target_vector(target)
    truth = np.asarray(candidate["true_vector"], dtype=np.float64)
    target_k = float(target["target_permeability_lattice"])
    target_tau = float(target["target_hydraulic_tortuosity"])
    true_k = float(candidate["true_permeability_lattice"])
    true_tau = float(candidate["true_hydraulic_tortuosity"])
    k_error = abs(true_k - target_k) / max(abs(target_k), 1e-12)
    tau_error = abs(true_tau - target_tau) / max(abs(target_tau), 1e-12)
    phi_error = abs(float(candidate["porosity"]) - float(target["target_porosity"]))
    result: dict[str, object] = {
        **target,
        "method": method,
        "repeat": repeat,
        "selected_sample_id": candidate["sample_id"],
        "selected_porosity": candidate["porosity"],
        "selected_connected_porosity": candidate["connected_porosity"],
        "selected_dead_pore_ratio": candidate["dead_pore_ratio"],
        "selected_true_permeability_lattice": true_k,
        "selected_true_hydraulic_tortuosity": true_tau,
        "verified_target_distance": _distance(truth, target_vector, stats),
        "porosity_absolute_error": phi_error,
        "permeability_relative_target_error": k_error,
        "tortuosity_relative_target_error": tau_error,
        "success_10": int(phi_error <= 0.03 and k_error <= 0.10 and tau_error <= 0.05),
        "success_20": int(phi_error <= 0.03 and k_error <= 0.20 and tau_error <= 0.10),
    }
    if method not in {"Random-QSGS", "Morphology-only", "LBM-oracle"}:
        result["screened_permeability_lattice"] = candidate[f"{method}_permeability"]
        result["screened_hydraulic_tortuosity"] = candidate[f"{method}_tortuosity"]
    return result


def offline_inverse_benchmark(
    candidates: list[dict[str, object]],
    targets: list[dict[str, float | str]],
    stats: ConditionStats,
    seed: int,
    random_repeats: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    deterministic_methods = ["Morphology-only", "Time-U-Net", "CT-PoreFlow", "No-anchor", "LBM-oracle"]
    for target in targets:
        target_vector = _target_vector(target)
        phi = float(target["target_porosity"])
        pool = [item for item in candidates if abs(float(item["porosity"]) - phi) <= 0.06]
        if len(pool) < 40:
            raise RuntimeError(f"Insufficient test candidates for {target['target_id']}: {len(pool)}")
        for repeat in range(random_repeats):
            chosen = pool[int(rng.integers(0, len(pool)))]
            records.append(_selection_record(target, chosen, "Random-QSGS", stats, repeat))
        for method in deterministic_methods:
            chosen = min(
                pool,
                key=lambda item: _distance(
                    _candidate_vector(item, method),
                    target_vector,
                    stats,
                    morphology_only=method == "Morphology-only",
                ),
            )
            records.append(_selection_record(target, chosen, method, stats))

    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[str(row["method"])].append(row)
    summary: list[dict[str, object]] = []
    for method in DISPLAY_NAMES:
        values = grouped[method]
        summary.append(
            {
                "method": method,
                "selections": len(values),
                "mean_verified_target_distance": float(np.mean([float(x["verified_target_distance"]) for x in values])),
                "median_verified_target_distance": float(np.median([float(x["verified_target_distance"]) for x in values])),
                "mean_permeability_relative_target_error": float(np.mean([float(x["permeability_relative_target_error"]) for x in values])),
                "mean_tortuosity_relative_target_error": float(np.mean([float(x["tortuosity_relative_target_error"]) for x in values])),
                "success_rate_10": float(np.mean([int(x["success_10"]) for x in values])),
                "success_rate_20": float(np.mean([int(x["success_20"]) for x in values])),
            }
        )
    return records, summary


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.7,
            "axes.labelsize": 8.9,
            "axes.titlesize": 9.4,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
        }
    )


def _save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def plot_offline_benchmark(records: list[dict[str, object]], summary: list[dict[str, object]], output: Path) -> None:
    _paper_style()
    deterministic = [row for row in records if int(row["repeat"]) == 0]
    methods = list(DISPLAY_NAMES)
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.75))
    ax = axes[0, 0]
    for method in methods:
        rows = [row for row in deterministic if row["method"] == method]
        if not rows:
            continue
        ax.scatter(
            [float(row["target_permeability_lattice"]) for row in rows],
            [float(row["selected_true_permeability_lattice"]) for row in rows],
            s=18,
            alpha=0.82,
            color=METHOD_COLORS[method],
            label=DISPLAY_NAMES[method],
        )
    all_k = [float(row["target_permeability_lattice"]) for row in deterministic] + [
        float(row["selected_true_permeability_lattice"]) for row in deterministic
    ]
    lo, hi = max(min(all_k), 1e-4), max(all_k)
    line = np.geomspace(lo, hi, 200)
    ax.plot(line, line, color="black", lw=1.0)
    ax.fill_between(line, 0.8 * line, 1.2 * line, color="#D9D9D9", alpha=0.5, zorder=0)
    ax.set(xscale="log", yscale="log", xlabel="Target permeability (lattice units)", ylabel="LBM-verified permeability")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.set_title("a  Property-to-structure permeability calibration", loc="left", fontweight="bold")

    ax = axes[0, 1]
    for method in methods:
        rows = [row for row in deterministic if row["method"] == method]
        if not rows:
            continue
        ax.scatter(
            [float(row["target_hydraulic_tortuosity"]) for row in rows],
            [float(row["selected_true_hydraulic_tortuosity"]) for row in rows],
            s=18,
            alpha=0.82,
            color=METHOD_COLORS[method],
        )
    tau = [float(row["target_hydraulic_tortuosity"]) for row in deterministic] + [
        float(row["selected_true_hydraulic_tortuosity"]) for row in deterministic
    ]
    line = np.linspace(min(tau), max(tau), 200)
    ax.plot(line, line, color="black", lw=1.0)
    ax.fill_between(line, 0.9 * line, 1.1 * line, color="#D9D9D9", alpha=0.5, zorder=0)
    ax.set(xlabel="Target hydraulic tortuosity", ylabel="LBM-verified tortuosity")
    ax.set_title("b  Tortuosity calibration", loc="left", fontweight="bold")

    ax = axes[1, 0]
    data = []
    for method in methods:
        values = [float(row["verified_target_distance"]) for row in deterministic if row["method"] == method]
        data.append(values)
    boxes = ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=False, medianprops={"color": "black"})
    for patch, method in zip(boxes["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.72)
    ax.set_xticks(range(1, len(methods) + 1), [DISPLAY_NAMES[m] for m in methods], rotation=25, ha="right")
    ax.set_ylabel("Verified standardized target distance")
    ax.set_title("c  Target error across 18 specifications", loc="left", fontweight="bold")

    ax = axes[1, 1]
    summary_map = {str(row["method"]): row for row in summary}
    rates = [100.0 * float(summary_map[m]["success_rate_20"]) for m in methods]
    ax.bar(range(len(methods)), rates, color=[METHOD_COLORS[m] for m in methods], width=0.72)
    ax.set_xticks(range(len(methods)), [DISPLAY_NAMES[m] for m in methods], rotation=25, ha="right")
    ax.set_ylabel("Target success at 20% tolerance (%)")
    ax.set_ylim(0, 105)
    for index, value in enumerate(rates):
        ax.text(index, value + 2, f"{value:.0f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_title("d  Closed-loop target success", loc="left", fontweight="bold")
    for axis in axes.flat:
        axis.grid(True, color="#E7E7E7", lw=0.45, zorder=0)
    fig.tight_layout()
    _save_figure(fig, output, "inverse_design_target_calibration")


def _geometry_features(solid: np.ndarray) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    through, connected, connected_fraction = through_connected_components(solid)
    pore = solid == 0
    porosity = float(pore.mean())
    dead = 1.0 - connected / max(porosity, 1e-12)
    edt = euclidean_distance_to_solid(solid).astype(np.float32) / float(solid.shape[0])
    inlet = geodesic(through, True)
    outlet = geodesic(through, False)
    geometry = np.stack([pore, through, edt, inlet, outlet], axis=0).astype(np.float32)
    return geometry, {
        "porosity": porosity,
        "connected_porosity": connected,
        "connected_fraction_of_pore": connected_fraction,
        "dead_pore_ratio": dead,
    }, through, edt


def generate_novel_pool(count: int, seed: int, porosity_target: float) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    parameters = sample_qsgs_parameters(rng)
    candidates: list[dict[str, object]] = []
    fingerprints: set[bytes] = set()
    attempts = 0
    while len(candidates) < count and attempts < count * 60:
        attempts += 1
        requested = float(np.clip(rng.normal(porosity_target, 0.018), 0.50, 0.72))
        solid = generate_qsgs(100, 100, requested, parameters, rng)
        if solid is None:
            continue
        geometry, morphology, through, edt = _geometry_features(solid)
        if morphology["connected_porosity"] <= 0.0:
            continue
        fingerprint = np.packbits(solid).tobytes()
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        candidates.append(
            {
                "candidate_id": f"qsgs_{len(candidates):04d}",
                "requested_porosity": requested,
                "solid": solid.astype(np.uint8),
                "geometry": geometry,
                "through": through.astype(np.uint8),
                "edt_raw": edt * 100.0,
                **morphology,
            }
        )
    if len(candidates) != count:
        raise RuntimeError(f"Generated only {len(candidates)} of {count} candidates after {attempts} attempts")
    return candidates


def _transport_from_prediction(prediction: np.ndarray, pore: np.ndarray, velocity_scale: float) -> dict[str, float]:
    ux = prediction[0].astype(np.float64) * velocity_scale
    uy = prediction[1].astype(np.float64) * velocity_scale
    width = ux.shape[1]
    mean_flux = 0.5 * (ux[0].sum() + ux[-1].sum())
    viscosity = 1.0 / 6.0
    imposed_pressure_drop = (1.00001 - 1.0) / 3.0
    permeability = (mean_flux / width) * viscosity * ux.shape[0] / imposed_pressure_drop
    speed = np.sqrt(ux * ux + uy * uy)
    denominator = abs(float(ux[pore].sum()))
    tortuosity = float(speed[pore].sum()) / denominator if denominator > 1e-14 else math.nan
    inlet = float(ux[0].sum())
    outlet = float(ux[-1].sum())
    balance_scale = 0.5 * (abs(inlet) + abs(outlet))
    balance = abs(inlet - outlet) / balance_scale if balance_scale > 1e-14 else math.nan
    return {
        "predicted_permeability_lattice": float(permeability),
        "predicted_hydraulic_tortuosity": float(tortuosity),
        "predicted_mass_balance_error": float(balance),
        "predicted_mean_flux_lattice": float(mean_flux),
    }


def screen_novel_pool(candidates: list[dict[str, object]], run_dir: Path, batch_size: int = 8) -> tuple[float, str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Novel-candidate screening requires the PyTorch environment used for the forward runs; "
            "the offline inverse benchmark remains available with --skip-novel."
        ) from exc
    from src.models import build_model

    config = _json(run_dir / "config.json")
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    velocity_scale = float(checkpoint["velocity_scale"])
    input_channels = int(config.get("input_channels", 3))
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            geometry = torch.from_numpy(
                np.stack([np.asarray(item["geometry"])[:input_channels] for item in batch])
            ).to(device)
            query = torch.ones(len(batch), device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                predicted = model(geometry, query).float().cpu().numpy()
            for item, field in zip(batch, predicted):
                item["predicted_field"] = field.astype(np.float32)
                item.update(_transport_from_prediction(field, np.asarray(item["solid"]) == 0, velocity_scale))
    return velocity_scale, str(device)


def _novel_vector(candidate: dict[str, object]) -> np.ndarray:
    return _raw_condition(
        float(candidate["porosity"]),
        float(candidate["connected_porosity"]),
        float(candidate["dead_pore_ratio"]),
        float(candidate["predicted_permeability_lattice"]),
        float(candidate["predicted_hydraulic_tortuosity"]),
    )


def novel_targets(train_matrix: np.ndarray, phi: float) -> list[dict[str, float | str]]:
    order = np.argsort(np.abs(train_matrix[:, 0] - phi))
    local = train_matrix[order[: min(1200, len(order))]]
    base = np.median(local, axis=0)
    cases = []
    specifications = (
        ("high-permeability", 0.80, 0.50),
        ("low-tortuosity", 0.50, 0.20),
        ("joint-transport", 0.80, 0.20),
    )
    for identifier, kq, tq in specifications:
        values = base.copy()
        values[0] = phi
        values[3] = np.quantile(local[:, 3], kq)
        values[4] = np.quantile(local[:, 4], tq)
        if identifier == "joint-transport":
            values[2] = np.quantile(local[:, 2], 0.20)
        cases.append(_raw_target_dict(identifier, values))
    return cases


def verify_candidate(candidate: dict[str, object], solver: SolverConfig) -> dict[str, object]:
    started = time.perf_counter()
    lbm = run_lbm(np.asarray(candidate["solid"], dtype=np.uint8), solver)
    arrays, factors = normalized_sample_fields(
        np.asarray(candidate["solid"], dtype=np.uint8),
        np.asarray(candidate["through"], dtype=bool),
        np.asarray(candidate["edt_raw"], dtype=np.float64),
        lbm,
    )
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "converged": bool(lbm["converged"]),
        "iterations": int(lbm["iterations"]),
        "seconds": elapsed,
        "mass_balance_error": float(lbm["mass_balance_error"]),
        "velocity_residual": float(lbm["velocity_residual"]),
        "permeability_relative_change": float(lbm["permeability_relative_change"]),
        "max_mach": float(lbm["max_mach"]),
        "verified_permeability_lattice": float(lbm["permeability_lattice"]),
        "verified_hydraulic_tortuosity": float(factors["hydraulic_tortuosity"]),
        "verified_velocity_x_lattice": arrays["velocity_x_raw_lattice"].astype(np.float32),
        "verified_velocity_y_lattice": arrays["velocity_y_raw_lattice"].astype(np.float32),
        "verified_density_lattice": arrays["density_raw_lattice"].astype(np.float32),
    }
    result["quality_pass"] = bool(
        result["converged"]
        and result["mass_balance_error"] < solver.mass_balance_criterion
        and result["velocity_residual"] < solver.velocity_residual_criterion
        and result["permeability_relative_change"] < solver.permeability_change_criterion
        and result["max_mach"] < solver.max_mach
    )
    return result


def run_novel_closed_loop(
    candidates: list[dict[str, object]],
    targets: list[dict[str, float | str]],
    stats: ConditionStats,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed + 19)
    solver = SolverConfig()
    rows: list[dict[str, object]] = []
    verified: list[dict[str, object]] = []
    used: set[str] = set()
    for target in targets:
        vector = _target_vector(target)
        valid = [
            item
            for item in candidates
            if np.isfinite(float(item["predicted_permeability_lattice"]))
            and float(item["predicted_permeability_lattice"]) > 0
            and np.isfinite(float(item["predicted_hydraulic_tortuosity"]))
            and 0.8 <= float(item["predicted_hydraulic_tortuosity"]) <= 8.0
        ]
        ranked = sorted(valid, key=lambda item: _distance(_novel_vector(item), vector, stats))
        random_order = list(valid)
        rng.shuffle(random_order)
        for strategy, ordered in (("surrogate-guided", ranked), ("random", random_order)):
            selected = None
            result = None
            attempts = 0
            for candidate in ordered:
                identifier = str(candidate["candidate_id"])
                if identifier in used:
                    continue
                attempts += 1
                trial = verify_candidate(candidate, solver)
                if trial["quality_pass"]:
                    selected = candidate
                    result = trial
                    used.add(identifier)
                    break
                if attempts >= 5:
                    break
            if selected is None or result is None:
                raise RuntimeError(f"No QC-passing {strategy} candidate for {target['target_id']}")
            true_vector = _raw_condition(
                float(selected["porosity"]),
                float(selected["connected_porosity"]),
                float(selected["dead_pore_ratio"]),
                float(result["verified_permeability_lattice"]),
                float(result["verified_hydraulic_tortuosity"]),
            )
            target_k = float(target["target_permeability_lattice"])
            target_tau = float(target["target_hydraulic_tortuosity"])
            row = {
                **target,
                "strategy": strategy,
                "candidate_id": selected["candidate_id"],
                "porosity": selected["porosity"],
                "connected_porosity": selected["connected_porosity"],
                "dead_pore_ratio": selected["dead_pore_ratio"],
                "predicted_permeability_lattice": selected["predicted_permeability_lattice"],
                "verified_permeability_lattice": result["verified_permeability_lattice"],
                "predicted_hydraulic_tortuosity": selected["predicted_hydraulic_tortuosity"],
                "verified_hydraulic_tortuosity": result["verified_hydraulic_tortuosity"],
                "verified_target_distance": _distance(true_vector, vector, stats),
                "permeability_relative_target_error": abs(float(result["verified_permeability_lattice"]) - target_k) / max(target_k, 1e-12),
                "tortuosity_relative_target_error": abs(float(result["verified_hydraulic_tortuosity"]) - target_tau) / max(target_tau, 1e-12),
                "lbm_iterations": result["iterations"],
                "lbm_seconds": result["seconds"],
                "mass_balance_error": result["mass_balance_error"],
                "velocity_residual": result["velocity_residual"],
                "permeability_relative_change": result["permeability_relative_change"],
                "quality_pass": result["quality_pass"],
                "verification_attempts": attempts,
            }
            rows.append(row)
            selected_copy = dict(selected)
            selected_copy.update(result)
            selected_copy["target"] = target
            selected_copy["strategy"] = strategy
            verified.append(selected_copy)
    return rows, verified


def _pareto_indices(permeability: np.ndarray, tortuosity: np.ndarray) -> np.ndarray:
    order = np.argsort(permeability)[::-1]
    best_tau = math.inf
    selected: list[int] = []
    for index in order:
        if tortuosity[index] < best_tau:
            selected.append(int(index))
            best_tau = float(tortuosity[index])
    return np.asarray(selected, dtype=np.int64)


def plot_novel_pareto(candidates: list[dict[str, object]], verified: list[dict[str, object]], output: Path) -> None:
    _paper_style()
    k = np.asarray([float(item["predicted_permeability_lattice"]) for item in candidates])
    tau = np.asarray([float(item["predicted_hydraulic_tortuosity"]) for item in candidates])
    dead = np.asarray([float(item["dead_pore_ratio"]) for item in candidates])
    finite = np.isfinite(k) & np.isfinite(tau) & (k > 0) & (tau > 0) & (tau < 8)
    indexes = np.flatnonzero(finite)
    pareto_local = _pareto_indices(k[finite], tau[finite])
    pareto = indexes[pareto_local]
    pareto = pareto[np.argsort(k[pareto])]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.00), gridspec_kw={"width_ratios": [1.15, 0.85]})
    ax = axes[0]
    scatter = ax.scatter(k[finite], tau[finite], c=dead[finite], cmap="viridis", s=17, alpha=0.65, edgecolors="none")
    ax.plot(k[pareto], tau[pareto], color="#D55E00", lw=1.3, label="Predicted Pareto front")
    for item in verified:
        if item["strategy"] != "surrogate-guided":
            continue
        pred_k = float(item["predicted_permeability_lattice"])
        pred_tau = float(item["predicted_hydraulic_tortuosity"])
        true_k = float(item["verified_permeability_lattice"])
        true_tau = float(item["verified_hydraulic_tortuosity"])
        ax.annotate("", xy=(true_k, true_tau), xytext=(pred_k, pred_tau), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "black"})
        ax.scatter([true_k], [true_tau], marker="*", s=85, color="#E69F00", edgecolor="black", linewidth=0.5, zorder=5)
    ax.set(xlabel="Predicted / verified permeability", ylabel="Predicted / verified tortuosity", xscale="log")
    ax.set_title("a  Novel-QSGS design landscape", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Dead-pore ratio")

    ax = axes[1]
    labels = []
    guided = []
    random = []
    target_ids = sorted({str(item["target"]["target_id"]) for item in verified})
    for target_id in target_ids:
        labels.append(target_id.replace("-", "\n"))
        guided.append(next(float(item["verified_target_distance"]) for item in _novel_rows_cache if item["target_id"] == target_id and item["strategy"] == "surrogate-guided"))
        random.append(next(float(item["verified_target_distance"]) for item in _novel_rows_cache if item["target_id"] == target_id and item["strategy"] == "random"))
    positions = np.arange(len(labels))
    width = 0.36
    ax.bar(positions - width / 2, random, width, color="#9E9E9E", label="Random")
    ax.bar(positions + width / 2, guided, width, color="#0072B2", label="Surrogate guided")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("LBM-verified target distance")
    ax.set_title("b  Physical round-trip comparison", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#E7E7E7", lw=0.45)
    fig.tight_layout()
    _save_figure(fig, output, "inverse_design_novel_pareto")


def plot_novel_cases(verified: list[dict[str, object]], velocity_scale: float, output: Path) -> None:
    _paper_style()
    cases = [item for item in verified if item["strategy"] == "surrogate-guided"]
    fig, axes = plt.subplots(len(cases), 4, figsize=(7.15, 5.35), constrained_layout=True)
    last_image = None
    for row, item in enumerate(cases):
        target = item["target"]
        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.02,
            0.92,
            str(target["target_id"]).replace("-", " ").title(),
            transform=axes[row, 0].transAxes,
            fontsize=9.2,
            fontweight="bold",
            va="top",
        )
        axes[row, 0].text(
            0.02,
            0.72,
            f"Target\n$\\phi$={float(target['target_porosity']):.3f}\n$k$={float(target['target_permeability_lattice']):.3g}\n$\\tau_h$={float(target['target_hydraulic_tortuosity']):.3f}",
            transform=axes[row, 0].transAxes,
            va="top",
            linespacing=1.35,
        )
        display_solid = np.rot90(np.asarray(item["solid"]), k=1)
        axes[row, 1].imshow(display_solid, cmap="gray_r", vmin=0, vmax=1)
        axes[row, 1].set_title("Generated pore geometry")
        predicted = np.asarray(item["predicted_field"])
        predicted_speed = np.sqrt(predicted[0] ** 2 + predicted[1] ** 2) * velocity_scale
        true_speed = np.sqrt(np.asarray(item["verified_velocity_x_lattice"]) ** 2 + np.asarray(item["verified_velocity_y_lattice"]) ** 2)
        display_predicted_speed = np.rot90(predicted_speed, k=1)
        display_true_speed = np.rot90(true_speed, k=1)
        axes[row, 2].imshow(display_predicted_speed, cmap="turbo", vmin=0, vmax=velocity_scale)
        axes[row, 2].set_title("Surrogate terminal speed")
        last_image = axes[row, 3].imshow(display_true_speed, cmap="turbo", vmin=0, vmax=velocity_scale)
        axes[row, 3].set_title("LBM-verified speed")
        axes[row, 3].text(
            0.02,
            0.02,
            f"$k$={float(item['verified_permeability_lattice']):.3g}\n$\\tau_h$={float(item['verified_hydraulic_tortuosity']):.3f}",
            transform=axes[row, 3].transAxes,
            color="white",
            fontsize=8.0,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.18", "fc": "black", "alpha": 0.55, "ec": "none"},
        )
        for column in range(1, 4):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes[:, 2:].ravel().tolist(), fraction=0.025, pad=0.015)
        colorbar.set_label(f"Speed (lattice units); global scale 0-{velocity_scale:.2g}")
    _save_figure(fig, output, "inverse_design_verified_cases")


_novel_rows_cache: list[dict[str, object]] = []


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop QSGS pore inverse design with LBM round-trip verification")
    parser.add_argument("--steady-root", type=Path, default=PROJECT_ROOT / "Dataset" / "LBM_8000_20260714_dataset")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output" / "inverse_design")
    parser.add_argument("--time-unet-run", type=Path, default=PROJECT_ROOT / "output" / "formal_results" / "unet_time_seed2026")
    parser.add_argument("--ct-run", type=Path, default=PROJECT_ROOT / "output" / "formal_results" / "ct_poreflow_seed2026")
    parser.add_argument("--no-anchor-run", type=Path, default=PROJECT_ROOT / "output" / "formal_results" / "ablation_no_steady_anchor_seed2026")
    parser.add_argument(
        "--novel-run",
        type=Path,
        help="Forward checkpoint used to screen newly generated QSGS candidates. Defaults to --time-unet-run.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--novel-candidates", type=int, default=120)
    parser.add_argument("--skip-novel", action="store_true")
    args = parser.parse_args()
    novel_run = args.novel_run or args.time_unet_run
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train_rows = load_metadata_split(args.steady_root, "train")
    test_rows = load_metadata_split(args.steady_root, "test")
    stats, train_matrix = training_statistics(train_rows)
    targets = build_targets(train_matrix)
    prediction_files = {
        "Time-U-Net": args.time_unet_run / "evaluation" / "test" / "per_sample_time_metrics.csv",
        "CT-PoreFlow": args.ct_run / "evaluation" / "test" / "per_sample_time_metrics.csv",
        "No-anchor": args.no_anchor_run / "evaluation" / "test" / "per_sample_time_metrics.csv",
    }
    test_candidates = assemble_test_candidates(test_rows, prediction_files)
    offline_rows, offline_summary = offline_inverse_benchmark(
        test_candidates, targets, stats, args.seed, args.random_repeats
    )
    _write_csv(output / "offline_inverse_design_selections.csv", offline_rows)
    _write_csv(output / "offline_inverse_design_summary.csv", offline_summary)
    _write_json(output / "inverse_design_targets.json", targets)
    plot_offline_benchmark(offline_rows, offline_summary, output)

    novel_manifest: dict[str, object] = {"status": "skipped"}
    if not args.skip_novel:
        novel = generate_novel_pool(args.novel_candidates, args.seed + 1000, float(np.median(train_matrix[:, 0])))
        velocity_scale, device = screen_novel_pool(novel, novel_run)
        pool_rows = []
        for item in novel:
            pool_rows.append(
                {
                    "candidate_id": item["candidate_id"],
                    "requested_porosity": item["requested_porosity"],
                    "porosity": item["porosity"],
                    "connected_porosity": item["connected_porosity"],
                    "dead_pore_ratio": item["dead_pore_ratio"],
                    "predicted_permeability_lattice": item["predicted_permeability_lattice"],
                    "predicted_hydraulic_tortuosity": item["predicted_hydraulic_tortuosity"],
                    "predicted_mass_balance_error": item["predicted_mass_balance_error"],
                }
            )
        _write_csv(output / "novel_qsgs_candidate_pool.csv", pool_rows)
        np.savez_compressed(
            output / "novel_qsgs_candidate_pool.npz",
            candidate_ids=np.asarray([str(item["candidate_id"]) for item in novel]),
            solid=np.stack([np.asarray(item["solid"], dtype=np.uint8) for item in novel]),
            predicted_fields=np.stack([np.asarray(item["predicted_field"], dtype=np.float32) for item in novel]),
        )
        targets_novel = novel_targets(train_matrix, float(np.median(train_matrix[:, 0])))
        novel_rows, verified = run_novel_closed_loop(novel, targets_novel, stats, args.seed)
        global _novel_rows_cache
        _novel_rows_cache = novel_rows
        _write_csv(output / "novel_qsgs_lbm_verification.csv", novel_rows)
        for item in verified:
            candidate_dir = output / "verified_candidates"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                candidate_dir / f"{item['target']['target_id']}_{item['strategy']}_{item['candidate_id']}.npz",
                solid=np.asarray(item["solid"], dtype=np.uint8),
                through_connected_pore=np.asarray(item["through"], dtype=np.uint8),
                predicted_field=np.asarray(item["predicted_field"], dtype=np.float32),
                verified_velocity_x_lattice=np.asarray(item["verified_velocity_x_lattice"], dtype=np.float32),
                verified_velocity_y_lattice=np.asarray(item["verified_velocity_y_lattice"], dtype=np.float32),
                verified_density_lattice=np.asarray(item["verified_density_lattice"], dtype=np.float32),
                metadata_json=np.asarray(json.dumps({k: v for k, v in item.items() if not isinstance(v, np.ndarray)}, ensure_ascii=False)),
            )
        plot_novel_pareto(novel, verified, output)
        plot_novel_cases(verified, velocity_scale, output)
        novel_manifest = {
            "status": "complete",
            "candidate_count": len(novel),
            "verified_count": len(verified),
            "forward_model": novel_run.name,
            "forward_run": str(novel_run.resolve()),
            "device": device,
            "velocity_scale": velocity_scale,
            "targets": targets_novel,
        }

    manifest = {
        "status": "complete",
        "seed": args.seed,
        "steady_root": str(args.steady_root.resolve()),
        "train_geometries": len(train_rows),
        "test_geometries": len(test_rows),
        "offline_target_count": len(targets),
        "random_repeats": args.random_repeats,
        "prediction_files": {name: str(path.resolve()) for name, path in prediction_files.items()},
        "checkpoint_sha256": {name: _sha256(path.parent.parent.parent / "best.pt") for name, path in prediction_files.items()},
        "condition_names": CONDITION_NAMES,
        "condition_mean": stats.mean.tolist(),
        "condition_std": stats.std.tolist(),
        "target_weights": TARGET_WEIGHTS.tolist(),
        "novel_closed_loop": novel_manifest,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(output / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
