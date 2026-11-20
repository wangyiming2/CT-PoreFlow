#!/usr/bin/env python3
"""Generate a quality-controlled 2-D QSGS-LBM dataset for deep learning.

The matrix convention follows ``lbm2d_cx_tecplot.m``:

* axis 0 / rows: physical X and the main flow direction;
* axis 1 / columns: physical Y and the transverse direction;
* solid == 1, pore fluid == 0.

Each accepted sample contains the binary geometry, connected-pore mask,
Euclidean distance transform (EDT), raw LBM fields, recoverable normalized
fields, physical/scalar labels, QSGS parameters, normalization factors and
quality-control metrics. The QSGS growth parameters are fixed to the original
MATLAB values; target porosity is stratified and random seeds provide distinct
realizations. Invalid attempts are kept outside the valid dataset.

Runtime dependencies: numpy and Pillow. SciPy is deliberately not required.

Examples
--------
Quick implementation check::

    python lbm_dataset_generator.py --self-test

Small end-to-end trial::

    python lbm_dataset_generator.py --samples-per-bin 1 --workers 1

Requested full stratified dataset (8 bins x 1000 valid samples)::

    python lbm_dataset_generator.py --samples-per-bin 1000 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Prevent each worker from starting a second layer of BLAS threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - dependency message
    raise SystemExit("Pillow is required: python -m pip install pillow") from exc


POROSITY_BINS: tuple[tuple[float, float], ...] = (
    (0.30, 0.35),
    (0.35, 0.40),
    (0.40, 0.45),
    (0.45, 0.50),
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
)

CX = np.asarray([1, 0, -1, 0, 1, -1, -1, 1, 0], dtype=np.int8)
CY = np.asarray([0, 1, 0, -1, 1, 1, -1, -1, 0], dtype=np.int8)
WEIGHTS = np.asarray(
    [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 4 / 9],
    dtype=np.float64,
)
OPPOSITE = np.asarray([2, 3, 0, 1, 6, 7, 4, 5, 8], dtype=np.int8)
CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class SolverConfig:
    nx: int = 100
    ny: int = 100
    buffer_rows: int = 5
    rhoin: float = 1.00001
    rhoout: float = 1.0
    tau: float = 1.0
    max_iterations: int = 50_000
    check_interval: int = 500
    velocity_residual_criterion: float = 1e-4
    mass_balance_criterion: float = 1e-3
    permeability_change_criterion: float = 1e-4
    max_mach: float = 0.10
    transient_snapshots: int = 0

    @property
    def nu_lattice(self) -> float:
        return CS2 * (self.tau - 0.5)


@dataclass(frozen=True)
class DatasetConfig:
    output_root: str
    samples_per_bin: int = 1
    workers: int = 1
    base_seed: int = 20260705
    max_geometry_attempts: int = 50
    save_invalid: bool = True
    max_invalid_files_per_task: int = 3
    visualize_every: int = 1
    train_fraction: float = 0.80
    validation_fraction: float = 0.10


@dataclass(frozen=True)
class SampleTask:
    bin_index: int
    sample_index: int
    samples_per_bin: int
    task_seed: int
    split: str
    run_dir: str
    solver: SolverConfig
    dataset: DatasetConfig


def python_scalar(value: Any) -> Any:
    """Convert NumPy values into JSON/CSV-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: python_scalar(value) for key, value in metadata.items()}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)


def sample_qsgs_parameters(rng: np.random.Generator) -> dict[str, float | int]:
    """Return the fixed QSGS parameters used by the original MATLAB model."""
    del rng  # Geometry remains stochastic through the separate realization seed.
    return {
        "seed_probability": 0.01,
        "growth_p_x_plus": 0.01,
        "growth_p_y_plus": 0.01,
        "growth_p_x_minus": 0.01,
        "growth_p_y_minus": 0.01,
        "growth_p_xplus_yplus": 0.03,
        "growth_p_xminus_yplus": 0.03,
        "growth_p_xminus_yminus": 0.03,
        "growth_p_xplus_yminus": 0.05,
        "base_growth_probability": 0.01,
        "anisotropy_ratio_x_over_y": 1.0,
        "directional_bias_x": 0.0,
        "directional_bias_y": 0.0,
        "diagonal_growth_factor": 3.5,
        "smoothing_kernel": 3,
        "smoothing_iterations": 2,
    }


def median_filter_binary(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Apply MATLAB-like 3x3 zero-padded median filtering to a binary mask."""
    result = mask.astype(np.uint8, copy=True)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        result = (windows.sum(axis=(-2, -1)) >= 5).astype(np.uint8)
    return result


def generate_qsgs(
    nx: int,
    ny: int,
    target_porosity: float,
    parameters: dict[str, float | int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Generate a QSGS solid mask using the eight-neighbour MATLAB rule."""
    total = nx * ny
    target_solid = int(round((1.0 - target_porosity) * total))
    solid = (rng.random((nx, ny)) < float(parameters["seed_probability"])).astype(np.uint8)
    coordinates = [tuple(item) for item in np.argwhere(solid == 1)]

    if not coordinates:
        return None
    if len(coordinates) > target_solid:
        keep = rng.choice(len(coordinates), size=target_solid, replace=False)
        selected = [coordinates[int(index)] for index in keep]
        solid.fill(0)
        for i, j in selected:
            solid[i, j] = 1
        coordinates = selected

    directions = (
        (1, 0, float(parameters["growth_p_x_plus"])),
        (0, 1, float(parameters["growth_p_y_plus"])),
        (-1, 0, float(parameters["growth_p_x_minus"])),
        (0, -1, float(parameters["growth_p_y_minus"])),
        (1, 1, float(parameters["growth_p_xplus_yplus"])),
        (-1, 1, float(parameters["growth_p_xminus_yplus"])),
        (-1, -1, float(parameters["growth_p_xminus_yminus"])),
        (1, -1, float(parameters["growth_p_xplus_yminus"])),
    )

    # A large guard prevents malformed parameters from creating an endless loop.
    # Low-probability QSGS growth can legitimately add no cell in one pass, so
    # require repeated stalls before rejecting the geometry.
    stalled_passes = 0
    for _growth_pass in range(20_000):
        if len(coordinates) >= target_solid:
            break
        count_before = len(coordinates)
        active_count = count_before
        order = rng.permutation(active_count)

        for coordinate_index in order:
            i, j = coordinates[int(coordinate_index)]
            for di, dj, probability in directions:
                if len(coordinates) >= target_solid:
                    break
                ni = i + di
                nj = j + dj
                if 0 <= ni < nx and 0 <= nj < ny and solid[ni, nj] == 0:
                    if rng.random() < probability:
                        solid[ni, nj] = 1
                        coordinates.append((ni, nj))

        if len(coordinates) == count_before:
            stalled_passes += 1
            if stalled_passes >= 50:
                return None
        else:
            stalled_passes = 0
    else:
        return None

    if len(coordinates) < target_solid:
        return None

    return median_filter_binary(solid, int(parameters["smoothing_iterations"]))


def through_connected_components(solid: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return pore components touching both X inlet and X outlet.

    Four-neighbour connectivity is deliberately used: a diagonal point contact
    has zero throat width and is not treated as a conducting connection.
    Connected porosity is through-connected pore cells divided by all cells.
    """
    nx, ny = solid.shape
    pore = solid == 0
    visited = np.zeros_like(pore, dtype=bool)
    through_mask = np.zeros_like(pore, dtype=bool)

    from collections import deque

    for start_i in range(nx):
        for start_j in range(ny):
            if not pore[start_i, start_j] or visited[start_i, start_j]:
                continue
            queue: deque[tuple[int, int]] = deque([(start_i, start_j)])
            visited[start_i, start_j] = True
            component: list[tuple[int, int]] = []
            touches_inlet = False
            touches_outlet = False

            while queue:
                i, j = queue.popleft()
                component.append((i, j))
                touches_inlet = touches_inlet or i == 0
                touches_outlet = touches_outlet or i == nx - 1
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni = i + di
                    nj = j + dj
                    if (
                        0 <= ni < nx
                        and 0 <= nj < ny
                        and pore[ni, nj]
                        and not visited[ni, nj]
                    ):
                        visited[ni, nj] = True
                        queue.append((ni, nj))

            if touches_inlet and touches_outlet:
                for i, j in component:
                    through_mask[i, j] = True

    connected_cells = int(np.count_nonzero(through_mask))
    pore_cells = int(np.count_nonzero(pore))
    connected_porosity = connected_cells / solid.size
    connected_fraction_of_pore = connected_cells / pore_cells if pore_cells else 0.0
    return through_mask, float(connected_porosity), float(connected_fraction_of_pore)


def squared_distance_transform_1d(values: np.ndarray) -> np.ndarray:
    """Felzenszwalb-Huttenlocher exact squared Euclidean transform in 1-D."""
    n = int(values.size)
    locations = np.zeros(n, dtype=np.int32)
    boundaries = np.empty(n + 1, dtype=np.float64)
    output = np.empty(n, dtype=np.float64)
    k = 0
    locations[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf

    for q in range(1, n):
        location = locations[k]
        intersection = (
            (values[q] + q * q) - (values[location] + location * location)
        ) / (2.0 * q - 2.0 * location)
        while intersection <= boundaries[k]:
            k -= 1
            location = locations[k]
            intersection = (
                (values[q] + q * q) - (values[location] + location * location)
            ) / (2.0 * q - 2.0 * location)
        k += 1
        locations[k] = q
        boundaries[k] = intersection
        boundaries[k + 1] = np.inf

    k = 0
    for q in range(n):
        while boundaries[k + 1] < q:
            k += 1
        location = locations[k]
        output[q] = (q - location) ** 2 + values[location]
    return output


def euclidean_distance_to_solid(solid: np.ndarray) -> np.ndarray:
    """Exact EDT for pore cells, zero in solid cells, without SciPy."""
    if not np.any(solid == 1):
        return np.full(solid.shape, math.hypot(*solid.shape), dtype=np.float64)
    large = 1.0e12
    distance_squared = np.where(solid == 1, 0.0, large)
    first_pass = np.empty_like(distance_squared)
    second_pass = np.empty_like(distance_squared)
    for i in range(solid.shape[0]):
        first_pass[i, :] = squared_distance_transform_1d(distance_squared[i, :])
    for j in range(solid.shape[1]):
        second_pass[:, j] = squared_distance_transform_1d(first_pass[:, j])
    result = np.sqrt(second_pass)
    result[solid == 1] = 0.0
    return result


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    speed_squared = ux * ux + uy * uy
    feq = np.empty((*rho.shape, 9), dtype=np.float64)
    for q in range(9):
        cu = ux * float(CX[q]) + uy * float(CY[q])
        feq[:, :, q] = WEIGHTS[q] * rho * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared
        )
    return feq


def logarithmic_snapshot_iterations(max_iterations: int, count: int) -> tuple[int, ...]:
    """Return exactly ``count - 1`` unique positive log-spaced iterations."""
    if count <= 1 or max_iterations <= 0:
        return ()
    positive_count = count - 1
    if positive_count > max_iterations:
        raise ValueError("snapshot count exceeds the available integer iterations")
    values = np.geomspace(1, max_iterations, num=positive_count)
    iterations: list[int] = []
    for index, value in enumerate(values):
        lower = 1 if not iterations else iterations[-1] + 1
        remaining = positive_count - index - 1
        upper = max_iterations - remaining
        iterations.append(min(max(int(round(value)), lower), upper))
    return tuple(iterations)


def apply_pressure_boundaries(
    f: np.ndarray,
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    rhoin: float,
    rhoout: float,
) -> None:
    """Non-equilibrium extrapolation at X inlet and outlet."""
    rho[0, :] = rhoin
    rho[-1, :] = rhoout
    ux[0, :] = ux[1, :]
    ux[-1, :] = ux[-2, :]
    uy[0, :] = uy[1, :]
    uy[-1, :] = uy[-2, :]

    for boundary_row, neighbour_row in ((0, 1), (-1, -2)):
        boundary_speed2 = ux[boundary_row] ** 2 + uy[boundary_row] ** 2
        neighbour_speed2 = ux[neighbour_row] ** 2 + uy[neighbour_row] ** 2
        for q in range(9):
            cu_boundary = ux[boundary_row] * CX[q] + uy[boundary_row] * CY[q]
            cu_neighbour = ux[neighbour_row] * CX[q] + uy[neighbour_row] * CY[q]
            feq_boundary = WEIGHTS[q] * rho[boundary_row] * (
                1.0
                + 3.0 * cu_boundary
                + 4.5 * cu_boundary**2
                - 1.5 * boundary_speed2
            )
            feq_neighbour = WEIGHTS[q] * rho[neighbour_row] * (
                1.0
                + 3.0 * cu_neighbour
                + 4.5 * cu_neighbour**2
                - 1.5 * neighbour_speed2
            )
            f[boundary_row, :, q] = (
                feq_boundary + f[neighbour_row, :, q] - feq_neighbour
            )


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.sum(f, axis=2)
    safe_rho = np.where(np.abs(rho) > 1e-30, rho, 1.0)
    ux = np.sum(f * CX.reshape(1, 1, 9), axis=2) / safe_rho
    uy = np.sum(f * CY.reshape(1, 1, 9), axis=2) / safe_rho
    return rho, ux, uy


def compute_velocity_residual(
    ux: np.ndarray,
    uy: np.ndarray,
    previous_ux: np.ndarray,
    previous_uy: np.ndarray,
    fluid: np.ndarray,
) -> float:
    difference = (ux - previous_ux) ** 2 + (uy - previous_uy) ** 2
    magnitude = ux * ux + uy * uy
    numerator = float(np.sum(difference[fluid]))
    denominator = float(np.sum(magnitude[fluid]))
    return math.sqrt(numerator) / (math.sqrt(denominator) + 1e-30)


def permeability_metrics(
    ux: np.ndarray,
    rho: np.ndarray,
    solid_full: np.ndarray,
    core_start: int,
    core_end: int,
    nu: float,
) -> dict[str, float]:
    """Darcy permeability using mean inlet/outlet velocity and dynamic viscosity."""
    inlet_row = core_start - 1 if core_start > 0 else core_start
    outlet_row = core_end if core_end < solid_full.shape[0] else core_end - 1
    height = solid_full.shape[1]
    length = core_end - core_start
    inlet_fluid = solid_full[inlet_row] == 0
    outlet_fluid = solid_full[outlet_row] == 0

    density_in = float(np.mean(rho[inlet_row, inlet_fluid])) if inlet_fluid.any() else math.nan
    density_out = (
        float(np.mean(rho[outlet_row, outlet_fluid])) if outlet_fluid.any() else math.nan
    )
    pressure_in = density_in * CS2
    pressure_out = density_out * CS2
    delta_pressure = pressure_in - pressure_out
    inlet_flux = float(np.sum(ux[inlet_row, inlet_fluid])) if inlet_fluid.any() else 0.0
    outlet_flux = float(np.sum(ux[outlet_row, outlet_fluid])) if outlet_fluid.any() else 0.0
    darcy_inlet = inlet_flux / height
    darcy_outlet = outlet_flux / height
    darcy_mean = 0.5 * (darcy_inlet + darcy_outlet)
    reference_density = 0.5 * (density_in + density_out)
    dynamic_viscosity = reference_density * nu

    if (
        math.isfinite(delta_pressure)
        and abs(delta_pressure) > np.finfo(float).eps
        and math.isfinite(dynamic_viscosity)
    ):
        permeability = abs(dynamic_viscosity * darcy_mean * length / delta_pressure)
    else:
        permeability = math.nan

    mass_scale = max(abs(inlet_flux), abs(outlet_flux), np.finfo(float).eps)
    mass_balance_error = abs(inlet_flux - outlet_flux) / mass_scale
    return {
        "density_in_sample_face": density_in,
        "density_out_sample_face": density_out,
        "pressure_in_sample_face": pressure_in,
        "pressure_out_sample_face": pressure_out,
        "delta_pressure_sample_lattice": delta_pressure,
        "inlet_total_flux_lattice": inlet_flux,
        "outlet_total_flux_lattice": outlet_flux,
        "mean_total_flux_lattice": 0.5 * (inlet_flux + outlet_flux),
        "darcy_velocity_inlet_lattice": darcy_inlet,
        "darcy_velocity_outlet_lattice": darcy_outlet,
        "apparent_darcy_velocity_lattice": darcy_mean,
        "reference_density_lattice": reference_density,
        "dynamic_viscosity_lattice": dynamic_viscosity,
        "permeability_lattice": permeability,
        "mass_balance_error": mass_balance_error,
    }


def run_lbm(solid_core: np.ndarray, config: SolverConfig) -> dict[str, Any]:
    """Run D2Q9 BGK LBM; BGK collision is applied only to fluid nodes."""
    nx, ny = solid_core.shape
    buffer_rows = config.buffer_rows
    nx_full = nx + 2 * buffer_rows
    core_start = buffer_rows
    core_end = buffer_rows + nx
    solid_full = np.zeros((nx_full, ny), dtype=np.uint8)
    solid_full[core_start:core_end, :] = solid_core
    fluid = solid_full == 0

    rho = np.ones((nx_full, ny), dtype=np.float64)
    rho[0, :] = config.rhoin
    rho[-1, :] = config.rhoout
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    f = equilibrium(rho, ux, uy)
    previous_ux = np.zeros_like(ux)
    previous_uy = np.zeros_like(uy)
    previous_permeability = math.nan
    permeability_change = math.inf
    velocity_residual = math.inf
    metrics: dict[str, float] = {}
    converged = False
    iterations = 0
    snapshot_schedule = set(
        logarithmic_snapshot_iterations(config.max_iterations, config.transient_snapshots)
    )
    transient_snapshots: list[dict[str, Any]] = []
    if config.transient_snapshots > 0:
        transient_snapshots.append(
            {
                "iteration": 0,
                "source_iteration": 0,
                "rho": rho.astype(np.float32, copy=True),
                "ux": ux.astype(np.float32, copy=True),
                "uy": uy.astype(np.float32, copy=True),
            }
        )

    for iteration in range(1, config.max_iterations + 1):
        speed_squared = ux * ux + uy * uy
        f_post = f.copy()

        # Fluid-only BGK collision. Solid populations are preserved for bounce-back.
        for q in range(9):
            cu = ux * float(CX[q]) + uy * float(CY[q])
            feq_q = WEIGHTS[q] * rho * (
                1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared
            )
            collided = f[:, :, q] - (f[:, :, q] - feq_q) / config.tau
            layer = f_post[:, :, q]
            layer[fluid] = collided[fluid]

        streamed = np.empty_like(f)
        for q in range(9):
            streamed[:, :, q] = np.roll(
                f_post[:, :, q], shift=(int(CX[q]), int(CY[q])), axis=(0, 1)
            )

        # Full-way on-node bounce-back after streaming, matching the MATLAB core.
        solid_populations = streamed[solid_full == 1, :].copy()
        streamed[solid_full == 1, :] = solid_populations[:, OPPOSITE]
        f = streamed

        rho, ux, uy = macroscopic(f)
        apply_pressure_boundaries(f, rho, ux, uy, config.rhoin, config.rhoout)
        ux[solid_full == 1] = 0.0
        uy[solid_full == 1] = 0.0
        rho[solid_full == 1] = 1.0

        if iteration in snapshot_schedule:
            transient_snapshots.append(
                {
                    "iteration": iteration,
                    "source_iteration": iteration,
                    "rho": rho.astype(np.float32, copy=True),
                    "ux": ux.astype(np.float32, copy=True),
                    "uy": uy.astype(np.float32, copy=True),
                }
            )

        if iteration % config.check_interval != 0:
            continue

        iterations = iteration
        velocity_residual = compute_velocity_residual(
            ux, uy, previous_ux, previous_uy, fluid
        )
        metrics = permeability_metrics(
            ux, rho, solid_full, core_start, core_end, config.nu_lattice
        )
        permeability = metrics["permeability_lattice"]
        if math.isfinite(previous_permeability) and math.isfinite(permeability):
            permeability_scale = max(
                abs(previous_permeability),
                abs(permeability),
                np.finfo(float).eps,
            )
            permeability_change = abs(permeability - previous_permeability) / permeability_scale
        else:
            permeability_change = math.inf

        finite_state = (
            np.isfinite(rho).all()
            and np.isfinite(ux).all()
            and np.isfinite(uy).all()
            and np.isfinite(f).all()
        )
        if not finite_state:
            break

        velocity_ok = velocity_residual < config.velocity_residual_criterion
        mass_ok = metrics["mass_balance_error"] < config.mass_balance_criterion
        permeability_ok = permeability_change < config.permeability_change_criterion
        if velocity_ok and mass_ok and permeability_ok:
            converged = True
            break

        previous_ux[...] = ux
        previous_uy[...] = uy
        previous_permeability = permeability

    if iterations == 0:
        iterations = config.max_iterations
        velocity_residual = compute_velocity_residual(
            ux, uy, previous_ux, previous_uy, fluid
        )
        metrics = permeability_metrics(
            ux, rho, solid_full, core_start, core_end, config.nu_lattice
        )

    if config.transient_snapshots > 0:
        captured = {int(item["iteration"]): item for item in transient_snapshots}
        requested_iterations = (0, *sorted(snapshot_schedule))
        final_rho = rho.astype(np.float32, copy=True)
        final_ux = ux.astype(np.float32, copy=True)
        final_uy = uy.astype(np.float32, copy=True)
        fixed_snapshots: list[dict[str, Any]] = []
        for requested_iteration in requested_iterations:
            if requested_iteration in captured:
                fixed_snapshots.append(captured[requested_iteration])
            else:
                # Once convergence has been reached, the steady field is the
                # physical state at all later requested times. Keeping the
                # requested time and the actual source time makes padding explicit.
                fixed_snapshots.append(
                    {
                        "iteration": requested_iteration,
                        "source_iteration": iterations,
                        "rho": final_rho,
                        "ux": final_ux,
                        "uy": final_uy,
                    }
                )
        transient_snapshots = fixed_snapshots

    speed = np.sqrt(ux * ux + uy * uy)
    max_speed = float(np.max(speed[fluid])) if np.any(fluid) else math.nan
    max_mach = max_speed / math.sqrt(CS2) if math.isfinite(max_speed) else math.nan

    return {
        "rho_full": rho,
        "ux_full": ux,
        "uy_full": uy,
        "speed_full": speed,
        "solid_full": solid_full,
        "core_start": core_start,
        "core_end": core_end,
        "converged": converged,
        "iterations": iterations,
        "velocity_residual": velocity_residual,
        "permeability_relative_change": permeability_change,
        "max_speed_lattice": max_speed,
        "max_mach": max_mach,
        "transient_snapshots": transient_snapshots,
        **metrics,
    }


def normalized_sample_fields(
    solid: np.ndarray,
    through_mask: np.ndarray,
    edt: np.ndarray,
    lbm: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    start = int(lbm["core_start"])
    end = int(lbm["core_end"])
    pore = solid == 0

    ux = np.asarray(lbm["ux_full"][start:end], dtype=np.float64).copy()
    uy = np.asarray(lbm["uy_full"][start:end], dtype=np.float64).copy()
    speed = np.asarray(lbm["speed_full"][start:end], dtype=np.float64).copy()
    density = np.asarray(lbm["rho_full"][start:end], dtype=np.float64).copy()
    pressure = density * CS2

    # Solid-node fields are undefined; store them as zero for direct DL loading.
    for field in (ux, uy, speed, density, pressure):
        field[~pore] = 0.0

    fluid_speed = speed[pore]
    u_ref = float(np.max(fluid_speed)) if fluid_speed.size else 0.0
    if not math.isfinite(u_ref) or u_ref <= np.finfo(float).eps:
        u_ref = math.nan

    pressure_offset = float(lbm["pressure_out_sample_face"])
    pressure_scale = float(lbm["delta_pressure_sample_lattice"])
    density_offset = float(lbm["density_out_sample_face"])
    density_scale = float(
        lbm["density_in_sample_face"] - lbm["density_out_sample_face"]
    )
    edt_ref = float(np.max(edt)) if np.any(edt > 0) else 1.0

    if math.isfinite(u_ref):
        ux_normalized = ux / u_ref
        uy_normalized = uy / u_ref
        speed_normalized = speed / u_ref
    else:
        ux_normalized = np.zeros_like(ux)
        uy_normalized = np.zeros_like(uy)
        speed_normalized = np.zeros_like(speed)

    if math.isfinite(pressure_scale) and abs(pressure_scale) > np.finfo(float).eps:
        pressure_normalized = (pressure - pressure_offset) / pressure_scale
    else:
        pressure_normalized = np.zeros_like(pressure)
    if math.isfinite(density_scale) and abs(density_scale) > np.finfo(float).eps:
        density_normalized = (density - density_offset) / density_scale
    else:
        density_normalized = np.zeros_like(density)

    for field in (ux_normalized, uy_normalized, speed_normalized, pressure_normalized, density_normalized):
        field[~pore] = 0.0

    velocity_sum = float(np.sum(ux[pore]))
    hydraulic_tortuosity = (
        float(np.sum(speed[pore])) / abs(velocity_sum)
        if abs(velocity_sum) > np.finfo(float).eps
        else math.nan
    )

    arrays = {
        "solid": solid.astype(np.uint8),
        "pore": pore.astype(np.uint8),
        "through_connected_pore": through_mask.astype(np.uint8),
        "edt_lattice": edt.astype(np.float32),
        "edt_normalized": (edt / edt_ref).astype(np.float32),
        "velocity_x_raw_lattice": ux.astype(np.float32),
        "velocity_y_raw_lattice": uy.astype(np.float32),
        "velocity_magnitude_raw_lattice": speed.astype(np.float32),
        "density_raw_lattice": density.astype(np.float32),
        "pressure_raw_lattice": pressure.astype(np.float32),
        "velocity_x_normalized": ux_normalized.astype(np.float32),
        "velocity_y_normalized": uy_normalized.astype(np.float32),
        "velocity_magnitude_normalized": speed_normalized.astype(np.float32),
        "density_normalized": density_normalized.astype(np.float32),
        "pressure_normalized": pressure_normalized.astype(np.float32),
    }
    factors = {
        "U_ref_lattice": u_ref,
        "pressure_offset_lattice": pressure_offset,
        "pressure_scale_lattice": pressure_scale,
        "density_offset_lattice": density_offset,
        "density_scale_lattice": density_scale,
        "EDT_ref_lattice": edt_ref,
        "hydraulic_tortuosity": hydraulic_tortuosity,
    }

    snapshots = lbm.get("transient_snapshots", [])
    if snapshots:
        transient_ux = np.stack(
            [np.asarray(item["ux"][start:end], dtype=np.float32) for item in snapshots]
        )
        transient_uy = np.stack(
            [np.asarray(item["uy"][start:end], dtype=np.float32) for item in snapshots]
        )
        transient_density = np.stack(
            [np.asarray(item["rho"][start:end], dtype=np.float32) for item in snapshots]
        )
        transient_ux[:, ~pore] = 0.0
        transient_uy[:, ~pore] = 0.0
        transient_density[:, ~pore] = 0.0
        arrays.update(
            {
                "transient_iterations": np.asarray(
                    [item["iteration"] for item in snapshots], dtype=np.int32
                ),
                "transient_source_iterations": np.asarray(
                    [item["source_iteration"] for item in snapshots], dtype=np.int32
                ),
                "transient_steady_fill_mask": np.asarray(
                    [item["source_iteration"] != item["iteration"] for item in snapshots],
                    dtype=np.uint8,
                ),
                "transient_velocity_x_raw_lattice": transient_ux,
                "transient_velocity_y_raw_lattice": transient_uy,
                "transient_density_raw_lattice": transient_density,
            }
        )
        factors["transient_snapshot_count"] = len(snapshots)
    return arrays, factors


def jet_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return (255.0 * np.stack((red, green, blue), axis=-1)).astype(np.uint8)


def diverging_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -1.0, 1.0)
    magnitude = np.abs(values)
    white = 1.0 - magnitude
    red = np.where(values >= 0.0, 1.0, white)
    blue = np.where(values <= 0.0, 1.0, white)
    green = white
    return (255.0 * np.stack((red, green, blue), axis=-1)).astype(np.uint8)


def visualization_panel(
    data: np.ndarray,
    title: str,
    signed: bool = False,
    binary: bool = False,
    size: int = 220,
) -> Image.Image:
    finite = np.isfinite(data)
    if binary:
        normalized = np.clip(data, 0.0, 1.0)
        rgb = np.repeat((255.0 * normalized[..., None]).astype(np.uint8), 3, axis=2)
    elif signed:
        scale = float(np.max(np.abs(data[finite]))) if finite.any() else 1.0
        normalized = data / scale if scale > 0.0 else np.zeros_like(data)
        rgb = diverging_colormap(normalized)
    else:
        low = float(np.min(data[finite])) if finite.any() else 0.0
        high = float(np.max(data[finite])) if finite.any() else 1.0
        normalized = (data - low) / (high - low) if high > low else np.zeros_like(data)
        rgb = jet_colormap(normalized)

    image = Image.fromarray(rgb, mode="RGB")
    resampling = getattr(Image, "Resampling", Image)
    method = resampling.NEAREST if binary else resampling.BILINEAR
    image = image.resize((size, size), method)
    panel = Image.new("RGB", (size, size + 24), "white")
    panel.paste(image, (0, 24))
    ImageDraw.Draw(panel).text((6, 5), title, fill="black")
    return panel


def save_visualization(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]
) -> None:
    panels = (
        visualization_panel(1 - arrays["solid"], "Pore-solid binary", binary=True),
        visualization_panel(arrays["through_connected_pore"], "Through-connected pore", binary=True),
        visualization_panel(arrays["edt_normalized"], "EDT normalized"),
        visualization_panel(arrays["velocity_x_normalized"], "Ux / U_ref", signed=True),
        visualization_panel(arrays["velocity_y_normalized"], "Uy / U_ref", signed=True),
        visualization_panel(arrays["velocity_magnitude_normalized"], "|U| / U_ref"),
        visualization_panel(arrays["pressure_normalized"], "Pressure normalized"),
        visualization_panel(arrays["density_normalized"], "Density normalized"),
    )
    width = 4 * panels[0].width
    height = 2 * panels[0].height + 54
    canvas = Image.new("RGB", (width, height), "white")
    for index, panel in enumerate(panels):
        x = (index % 4) * panel.width
        y = (index // 4) * panel.height
        canvas.paste(panel, (x, y))
    text = (
        f"{metadata['sample_id']}  split={metadata['split']}  "
        f"phi={metadata['porosity']:.5f}  connected_phi={metadata['connected_porosity']:.5f}  "
        f"k={metadata['permeability_lattice']:.6g}  tau_h={metadata['hydraulic_tortuosity']:.5g}"
    )
    ImageDraw.Draw(canvas).text((8, height - 38), text, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def save_npz_and_metadata(
    npz_path: Path,
    json_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = clean_metadata(metadata)
    np.savez_compressed(
        npz_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=True)),
    )
    write_json(json_path, metadata)


def assemble_metadata(
    task: SampleTask,
    attempt: int,
    attempt_seed: int,
    target_porosity: float,
    actual_porosity: float,
    qsgs: dict[str, float | int],
    connected_porosity: float,
    connected_fraction: float,
    lbm: dict[str, Any] | None,
    factors: dict[str, float] | None,
    status: str,
    invalid_reason: str,
) -> dict[str, Any]:
    low, high = POROSITY_BINS[task.bin_index]
    metadata: dict[str, Any] = {
        "sample_id": f"b{task.bin_index:02d}_s{task.sample_index:06d}",
        "split": task.split,
        "status": status,
        "invalid_reason": invalid_reason,
        "valid_sample": int(status == "valid"),
        "bin_index": task.bin_index,
        "porosity_bin_low": low,
        "porosity_bin_high": high,
        "target_porosity": target_porosity,
        "porosity": actual_porosity,
        "connected_porosity": connected_porosity,
        "connected_fraction_of_pore": connected_fraction,
        "task_seed": task.task_seed,
        "attempt_seed": attempt_seed,
        "generation_attempt": attempt,
        "flow_direction": "x_row_axis0",
        **qsgs,
        "rhoin": task.solver.rhoin,
        "rhoout": task.solver.rhoout,
        "delta_rho": task.solver.rhoin - task.solver.rhoout,
        "tau": task.solver.tau,
        "nu_lattice": task.solver.nu_lattice,
        "buffer_rows_each_side": task.solver.buffer_rows,
        "velocity_residual_criterion": task.solver.velocity_residual_criterion,
        "mass_balance_criterion": task.solver.mass_balance_criterion,
        "permeability_change_criterion": task.solver.permeability_change_criterion,
        "requested_transient_snapshots": task.solver.transient_snapshots,
    }
    if lbm is not None:
        excluded = {
            "rho_full",
            "ux_full",
            "uy_full",
            "speed_full",
            "solid_full",
            "core_start",
            "core_end",
            "transient_snapshots",
        }
        metadata.update({key: value for key, value in lbm.items() if key not in excluded})
    if factors is not None:
        metadata.update(factors)
    return clean_metadata(metadata)


def save_invalid_attempt(
    task: SampleTask,
    metadata: dict[str, Any],
    solid: np.ndarray,
    edt: np.ndarray | None,
    lbm: dict[str, Any] | None,
) -> str:
    run_dir = Path(task.run_dir)
    stem = f"{metadata['sample_id']}_attempt_{metadata['generation_attempt']:02d}"
    folder = run_dir / "invalid_samples" / f"bin_{task.bin_index:02d}"
    arrays: dict[str, np.ndarray] = {"solid": solid.astype(np.uint8)}
    if edt is not None:
        arrays["edt_lattice"] = edt.astype(np.float32)
    if lbm is not None:
        start = int(lbm["core_start"])
        end = int(lbm["core_end"])
        arrays.update(
            {
                "velocity_x_final_lattice": lbm["ux_full"][start:end].astype(np.float32),
                "velocity_y_final_lattice": lbm["uy_full"][start:end].astype(np.float32),
                "density_final_lattice": lbm["rho_full"][start:end].astype(np.float32),
            }
        )
    npz_path = folder / f"{stem}.npz"
    json_path = folder / f"{stem}.json"
    save_npz_and_metadata(npz_path, json_path, arrays, metadata)
    return str(npz_path.relative_to(run_dir))


def generate_sample_task(task: SampleTask) -> dict[str, Any]:
    """Worker: retry randomized QSGS candidates until one valid sample is saved."""
    master_rng = np.random.default_rng(task.task_seed)
    low, high = POROSITY_BINS[task.bin_index]
    run_dir = Path(task.run_dir)
    invalid_records: list[dict[str, Any]] = []
    invalid_files_saved = 0

    for attempt in range(1, task.dataset.max_geometry_attempts + 1):
        attempt_seed = int(master_rng.integers(1, np.iinfo(np.int64).max))
        rng = np.random.default_rng(attempt_seed)
        target_porosity = float(rng.uniform(low, high))
        qsgs = sample_qsgs_parameters(rng)
        solid = generate_qsgs(
            task.solver.nx, task.solver.ny, target_porosity, qsgs, rng
        )

        if solid is None:
            metadata = assemble_metadata(
                task,
                attempt,
                attempt_seed,
                target_porosity,
                math.nan,
                qsgs,
                0.0,
                0.0,
                None,
                None,
                "invalid",
                "qsgs_growth_failed",
            )
            invalid_records.append(metadata)
            continue

        actual_porosity = float(np.count_nonzero(solid == 0) / solid.size)
        upper_ok = actual_porosity <= high if task.bin_index == len(POROSITY_BINS) - 1 else actual_porosity < high
        if not (actual_porosity >= low and upper_ok):
            metadata = assemble_metadata(
                task,
                attempt,
                attempt_seed,
                target_porosity,
                actual_porosity,
                qsgs,
                0.0,
                0.0,
                None,
                None,
                "invalid",
                "post_smoothing_porosity_outside_bin",
            )
            invalid_records.append(metadata)
            continue

        through_mask, connected_porosity, connected_fraction = through_connected_components(solid)
        edt = euclidean_distance_to_solid(solid)
        if connected_porosity <= 0.0:
            metadata = assemble_metadata(
                task,
                attempt,
                attempt_seed,
                target_porosity,
                actual_porosity,
                qsgs,
                connected_porosity,
                connected_fraction,
                None,
                None,
                "invalid",
                "no_inlet_to_outlet_connection",
            )
            if task.dataset.save_invalid and invalid_files_saved < task.dataset.max_invalid_files_per_task:
                metadata["invalid_file"] = save_invalid_attempt(task, metadata, solid, edt, None)
                invalid_files_saved += 1
            invalid_records.append(metadata)
            continue

        lbm = run_lbm(solid, task.solver)
        arrays, factors = normalized_sample_fields(solid, through_mask, edt, lbm)
        finite_scalars = all(
            math.isfinite(float(value))
            for value in (
                lbm["permeability_lattice"],
                lbm["mass_balance_error"],
                lbm["velocity_residual"],
                lbm["permeability_relative_change"],
                factors["U_ref_lattice"],
                factors["hydraulic_tortuosity"],
            )
        )
        valid = bool(
            lbm["converged"]
            and finite_scalars
            and lbm["mass_balance_error"] < task.solver.mass_balance_criterion
            and lbm["permeability_relative_change"]
            < task.solver.permeability_change_criterion
            and lbm["velocity_residual"] < task.solver.velocity_residual_criterion
            and lbm["max_mach"] < task.solver.max_mach
        )

        if not valid:
            reasons: list[str] = []
            if not lbm["converged"]:
                reasons.append("lbm_not_converged")
            if not finite_scalars:
                reasons.append("nonfinite_output")
            if lbm["mass_balance_error"] >= task.solver.mass_balance_criterion:
                reasons.append("mass_balance_failed")
            if lbm["max_mach"] >= task.solver.max_mach:
                reasons.append("mach_number_too_high")
            metadata = assemble_metadata(
                task,
                attempt,
                attempt_seed,
                target_porosity,
                actual_porosity,
                qsgs,
                connected_porosity,
                connected_fraction,
                lbm,
                factors,
                "invalid",
                ";".join(reasons) or "quality_control_failed",
            )
            if task.dataset.save_invalid and invalid_files_saved < task.dataset.max_invalid_files_per_task:
                metadata["invalid_file"] = save_invalid_attempt(task, metadata, solid, edt, lbm)
                invalid_files_saved += 1
            invalid_records.append(metadata)
            continue

        metadata = assemble_metadata(
            task,
            attempt,
            attempt_seed,
            target_porosity,
            actual_porosity,
            qsgs,
            connected_porosity,
            connected_fraction,
            lbm,
            factors,
            "valid",
            "",
        )
        bin_folder = f"bin_{task.bin_index:02d}_{low:.2f}_{high:.2f}".replace(".", "p")
        sample_folder = run_dir / "samples" / task.split / bin_folder
        visualization_folder = run_dir / "visualizations" / task.split / bin_folder
        stem = str(metadata["sample_id"])
        npz_path = sample_folder / f"{stem}.npz"
        json_path = sample_folder / f"{stem}.json"
        metadata["sample_file"] = str(npz_path.relative_to(run_dir))
        metadata["metadata_file"] = str(json_path.relative_to(run_dir))

        if task.dataset.visualize_every > 0 and task.sample_index % task.dataset.visualize_every == 0:
            visualization_path = visualization_folder / f"{stem}.png"
            metadata["visualization_file"] = str(visualization_path.relative_to(run_dir))
        else:
            visualization_path = None
            metadata["visualization_file"] = ""

        save_npz_and_metadata(npz_path, json_path, arrays, metadata)
        if visualization_path is not None:
            save_visualization(visualization_path, arrays, metadata)

        return {
            "success": True,
            "metadata": clean_metadata(metadata),
            "invalid_records": invalid_records,
            "error": "",
        }

    return {
        "success": False,
        "metadata": {
            "sample_id": f"b{task.bin_index:02d}_s{task.sample_index:06d}",
            "bin_index": task.bin_index,
            "split": task.split,
            "status": "failed",
            "valid_sample": 0,
            "invalid_reason": "maximum_geometry_attempts_exhausted",
            "task_seed": task.task_seed,
        },
        "invalid_records": invalid_records,
        "error": "maximum_geometry_attempts_exhausted",
    }


def write_csv_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "sample_id",
        "split",
        "status",
        "valid_sample",
        "invalid_reason",
        "bin_index",
        "porosity_bin_low",
        "porosity_bin_high",
        "target_porosity",
        "porosity",
        "connected_porosity",
        "permeability_lattice",
        "hydraulic_tortuosity",
        "outlet_total_flux_lattice",
        "apparent_darcy_velocity_lattice",
        "converged",
        "iterations",
        "velocity_residual",
        "mass_balance_error",
        "permeability_relative_change",
    ]
    all_fields = set().union(*(record.keys() for record in records))
    fieldnames = [name for name in preferred if name in all_fields]
    fieldnames.extend(sorted(all_fields - set(fieldnames)))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(clean_metadata(record))


def split_for_index(index: int, total: int, train: float, validation: float) -> str:
    fraction = (index + 0.5) / total
    if fraction <= train:
        return "train"
    if fraction <= train + validation:
        return "validation"
    return "test"


def dataset_schema() -> dict[str, Any]:
    return {
        "matrix_convention": {
            "axis_0": "physical X / row / flow direction",
            "axis_1": "physical Y / column / transverse direction",
            "solid_value": 1,
            "pore_value": 0,
        },
        "array_keys": {
            "solid": "uint8 binary solid mask",
            "pore": "uint8 binary pore mask",
            "through_connected_pore": "4-neighbour pore components touching inlet and outlet",
            "edt_lattice": "Euclidean distance to nearest solid node in lattice units",
            "edt_normalized": "edt_lattice / EDT_ref_lattice",
            "velocity_x_raw_lattice": "raw steady X velocity; zero in solid",
            "velocity_y_raw_lattice": "raw steady Y velocity; zero in solid",
            "velocity_magnitude_raw_lattice": "raw speed; zero in solid",
            "density_raw_lattice": "raw fluid density; zero in solid",
            "pressure_raw_lattice": "rho/3; zero in solid",
            "velocity_x_normalized": "velocity_x_raw_lattice / U_ref_lattice",
            "velocity_y_normalized": "velocity_y_raw_lattice / U_ref_lattice",
            "velocity_magnitude_normalized": "speed / U_ref_lattice",
            "density_normalized": "(rho-density_offset)/density_scale; zero in solid",
            "pressure_normalized": "(p-pressure_offset)/pressure_scale; zero in solid",
            "transient_iterations": "int32 globally fixed logarithmic LBM iteration targets",
            "transient_source_iterations": "int32 actual source iteration; convergence iteration for steady-filled frames",
            "transient_steady_fill_mask": "uint8 1 where an after-convergence target uses the final steady field",
            "transient_velocity_x_raw_lattice": "float32 transient raw ux in lattice units; zero in solid",
            "transient_velocity_y_raw_lattice": "float32 transient raw uy in lattice units; zero in solid",
            "transient_density_raw_lattice": "float32 transient raw density in lattice units; zero in solid",
        },
        "scalar_definitions": {
            "connected_porosity": "through-connected pore cells / all sample cells",
            "permeability_lattice": "abs(mu*q_mean*L/delta_p), mu=rho_ref*nu",
            "hydraulic_tortuosity": "sum_pore(|u|) / abs(sum_pore(u_x))",
            "apparent_darcy_velocity_lattice": "mean of inlet/outlet total flux divided by sample width",
            "mass_balance_error": "abs(Qin-Qout)/max(abs(Qin),abs(Qout),eps)",
            "permeability_relative_change": "abs(Kn-Kprev)/max(abs(Kn),abs(Kprev),eps)",
        },
        "normalization_recovery": {
            "velocity": "u_raw = u_normalized * U_ref_lattice",
            "pressure": "p_raw = p_normalized * pressure_scale_lattice + pressure_offset_lattice",
            "density": "rho_raw = density_normalized * density_scale_lattice + density_offset_lattice",
            "edt": "edt_lattice = edt_normalized * EDT_ref_lattice",
        },
        "validity_rule": (
            "through connection exists AND all three convergence criteria pass "
            "AND outputs are finite AND max Mach is below the configured limit"
        ),
    }


def run_self_test() -> int:
    rng = np.random.default_rng(12345)
    qsgs = sample_qsgs_parameters(rng)
    solid = generate_qsgs(32, 32, 0.60, qsgs, rng)
    if solid is None or solid.shape != (32, 32):
        raise RuntimeError("QSGS self-test failed")
    edt = euclidean_distance_to_solid(solid)
    if edt.shape != solid.shape or not np.isfinite(edt).all() or np.any(edt[solid == 1] != 0):
        raise RuntimeError("EDT self-test failed")
    _, connected_porosity, _ = through_connected_components(solid)

    # A small parallel-plate channel tests the LBM execution path quickly.
    channel = np.zeros((24, 24), dtype=np.uint8)
    channel[:, 0] = 1
    channel[:, -1] = 1
    config = SolverConfig(
        nx=24,
        ny=24,
        buffer_rows=3,
        max_iterations=100,
        check_interval=20,
    )
    result = run_lbm(channel, config)
    for key in ("permeability_lattice", "mass_balance_error", "velocity_residual"):
        if not math.isfinite(float(result[key])):
            raise RuntimeError(f"LBM self-test produced nonfinite {key}")
    print(
        "SELF-TEST PASSED | "
        f"QSGS porosity={np.mean(solid == 0):.4f}, "
        f"connected_porosity={connected_porosity:.4f}, "
        f"short-run k={result['permeability_lattice']:.6g}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a stratified, quality-controlled 2-D QSGS-LBM dataset."
    )
    parser.add_argument(
        "--output-root",
        default="LBM_DL_Datasets",
        help="Parent directory for timestamped dataset runs.",
    )
    parser.add_argument("--run-name", default="", help="Optional stable run directory name.")
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=1,
        help="Valid samples per porosity bin. Safe default: 1 (8 samples total).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--max-geometry-attempts", type=int, default=50)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--buffer-rows", type=int, default=5)
    parser.add_argument("--rhoin", type=float, default=1.00001)
    parser.add_argument("--rhoout", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=50_000)
    parser.add_argument("--check-interval", type=int, default=500)
    parser.add_argument("--velocity-residual-criterion", type=float, default=1e-4)
    parser.add_argument("--mass-balance-criterion", type=float, default=1e-3)
    parser.add_argument("--permeability-change-criterion", type=float, default=1e-4)
    parser.add_argument("--max-mach", type=float, default=0.10)
    parser.add_argument(
        "--transient-snapshots",
        type=int,
        default=0,
        help="Save 10-20 recommended log-spaced LBM states per accepted sample; 0 disables.",
    )
    parser.add_argument("--visualize-every", type=int, default=1)
    parser.add_argument("--no-save-invalid", action="store_true")
    parser.add_argument("--max-invalid-files-per-task", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.samples_per_bin <= 0:
        raise SystemExit("--samples-per-bin must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.transient_snapshots < 0 or args.transient_snapshots > 100:
        raise SystemExit("--transient-snapshots must be between 0 and 100")

    if args.tau <= 0.5:
        raise SystemExit("--tau must be greater than 0.5 so nu_lattice is positive")
    if args.rhoin == args.rhoout:
        raise SystemExit("--rhoin and --rhoout must differ to drive the flow")
    solver = SolverConfig(
        nx=args.nx,
        ny=args.ny,
        buffer_rows=args.buffer_rows,
        rhoin=args.rhoin,
        rhoout=args.rhoout,
        tau=args.tau,
        max_iterations=args.max_iterations,
        check_interval=args.check_interval,
        velocity_residual_criterion=args.velocity_residual_criterion,
        mass_balance_criterion=args.mass_balance_criterion,
        permeability_change_criterion=args.permeability_change_criterion,
        max_mach=args.max_mach,
        transient_snapshots=args.transient_snapshots,
    )
    run_name = args.run_name or f"LBM_DL_Dataset_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    dataset = DatasetConfig(
        output_root=args.output_root,
        samples_per_bin=args.samples_per_bin,
        workers=args.workers,
        base_seed=args.seed,
        max_geometry_attempts=args.max_geometry_attempts,
        save_invalid=not args.no_save_invalid,
        max_invalid_files_per_task=args.max_invalid_files_per_task,
        visualize_every=args.visualize_every,
    )

    write_json(
        run_dir / "config.json",
        {
            "solver": asdict(solver),
            "nu_lattice": solver.nu_lattice,
            "dataset": asdict(dataset),
            "porosity_bins": [list(item) for item in POROSITY_BINS],
            "qsgs_parameter_mode": "fixed_original_matlab",
            "qsgs_parameters": sample_qsgs_parameters(np.random.default_rng(0)),
        },
    )
    write_json(run_dir / "dataset_schema.json", dataset_schema())

    tasks: list[SampleTask] = []
    for bin_index in range(len(POROSITY_BINS)):
        for sample_index in range(args.samples_per_bin):
            seed_sequence = np.random.SeedSequence([args.seed, bin_index, sample_index])
            task_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
            split = split_for_index(
                sample_index,
                args.samples_per_bin,
                dataset.train_fraction,
                dataset.validation_fraction,
            )
            tasks.append(
                SampleTask(
                    bin_index=bin_index,
                    sample_index=sample_index,
                    samples_per_bin=args.samples_per_bin,
                    task_seed=task_seed,
                    split=split,
                    run_dir=str(run_dir),
                    solver=solver,
                    dataset=dataset,
                )
            )

    total = len(tasks)
    print(f"Dataset run: {run_dir}")
    print(
        f"Requested valid samples: {total} = {len(POROSITY_BINS)} bins x "
        f"{args.samples_per_bin}; workers={args.workers}"
    )
    valid_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_sample_task, task): task for task in tasks}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # keep the full batch running
                result = {
                    "success": False,
                    "metadata": {
                        "sample_id": f"b{task.bin_index:02d}_s{task.sample_index:06d}",
                        "bin_index": task.bin_index,
                        "split": task.split,
                        "status": "failed",
                        "valid_sample": 0,
                        "invalid_reason": "worker_exception",
                        "task_seed": task.task_seed,
                    },
                    "invalid_records": [],
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            invalid_records.extend(result.get("invalid_records", []))
            if result["success"]:
                valid_records.append(result["metadata"])
            else:
                failure = dict(result["metadata"])
                failure["worker_error"] = result.get("error", "")
                failed_records.append(failure)

            if completed_count == 1 or completed_count % max(1, total // 100) == 0 or completed_count == total:
                print(
                    f"Progress {completed_count}/{total} | valid={len(valid_records)} "
                    f"failed={len(failed_records)} invalid_attempts={len(invalid_records)}",
                    flush=True,
                )

    valid_records.sort(key=lambda item: str(item.get("sample_id", "")))
    failed_records.sort(key=lambda item: str(item.get("sample_id", "")))
    write_csv_records(run_dir / "manifest_valid.csv", valid_records)
    write_csv_records(run_dir / "manifest_invalid_attempts.csv", invalid_records)
    write_csv_records(run_dir / "manifest_failed_tasks.csv", failed_records)

    counts_by_bin = {
        str(index): sum(int(row.get("bin_index", -1)) == index for row in valid_records)
        for index in range(len(POROSITY_BINS))
    }
    summary = {
        "requested_valid_samples": total,
        "valid_samples": len(valid_records),
        "failed_tasks": len(failed_records),
        "invalid_attempts": len(invalid_records),
        "valid_counts_by_bin": counts_by_bin,
        "complete": len(valid_records) == total,
        "run_directory": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
