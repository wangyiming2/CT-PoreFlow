from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


CX = np.asarray([1, 0, -1, 0, 1, -1, -1, 1, 0], dtype=np.int8)
CY = np.asarray([0, 1, 0, -1, 1, 1, -1, -1, 0], dtype=np.int8)
WEIGHTS = np.asarray(
    [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 4 / 9],
    dtype=np.float64,
)
OPPOSITE = np.asarray([2, 3, 0, 1, 6, 7, 4, 5, 8], dtype=np.int8)
CS2 = 1.0 / 3.0


@dataclass(frozen=True)
class WarmStartConfig:
    buffer_rows: int = 5
    rhoin: float = 1.00001
    rhoout: float = 1.0
    tau: float = 1.0
    max_iterations: int = 50_000
    check_interval: int = 500
    velocity_residual_criterion: float = 1e-4
    mass_balance_criterion: float = 1e-3
    permeability_change_criterion: float = 1e-4
    max_mach_criterion: float = 0.1

    @property
    def nu_lattice(self) -> float:
        return CS2 * (self.tau - 0.5)


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    speed_squared = ux * ux + uy * uy
    feq = np.empty((*rho.shape, 9), dtype=np.float64)
    for q in range(9):
        cu = ux * float(CX[q]) + uy * float(CY[q])
        feq[:, :, q] = WEIGHTS[q] * rho * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared
        )
    return feq


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.sum(f, axis=2)
    safe_rho = np.where(np.abs(rho) > 1e-30, rho, 1.0)
    ux = np.sum(f * CX.reshape(1, 1, 9), axis=2) / safe_rho
    uy = np.sum(f * CY.reshape(1, 1, 9), axis=2) / safe_rho
    return rho, ux, uy


def apply_pressure_boundaries(
    f: np.ndarray,
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    rhoin: float,
    rhoout: float,
) -> None:
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
                1.0 + 3.0 * cu_boundary + 4.5 * cu_boundary**2 - 1.5 * boundary_speed2
            )
            feq_neighbour = WEIGHTS[q] * rho[neighbour_row] * (
                1.0 + 3.0 * cu_neighbour + 4.5 * cu_neighbour**2 - 1.5 * neighbour_speed2
            )
            f[boundary_row, :, q] = feq_boundary + f[neighbour_row, :, q] - feq_neighbour


def _velocity_residual(
    ux: np.ndarray,
    uy: np.ndarray,
    previous_ux: np.ndarray,
    previous_uy: np.ndarray,
    fluid: np.ndarray,
) -> float:
    difference = (ux - previous_ux) ** 2 + (uy - previous_uy) ** 2
    magnitude = ux * ux + uy * uy
    return math.sqrt(float(np.sum(difference[fluid]))) / (
        math.sqrt(float(np.sum(magnitude[fluid]))) + 1e-30
    )


def _transport_metrics(
    ux: np.ndarray,
    rho: np.ndarray,
    solid_full: np.ndarray,
    core_start: int,
    core_end: int,
    nu: float,
) -> dict[str, float]:
    inlet_row = core_start - 1 if core_start > 0 else core_start
    outlet_row = core_end if core_end < solid_full.shape[0] else core_end - 1
    inlet_fluid = solid_full[inlet_row] == 0
    outlet_fluid = solid_full[outlet_row] == 0
    density_in = float(np.mean(rho[inlet_row, inlet_fluid]))
    density_out = float(np.mean(rho[outlet_row, outlet_fluid]))
    delta_pressure = (density_in - density_out) * CS2
    inlet_flux = float(np.sum(ux[inlet_row, inlet_fluid]))
    outlet_flux = float(np.sum(ux[outlet_row, outlet_fluid]))
    mean_flux = 0.5 * (inlet_flux + outlet_flux)
    darcy_velocity = mean_flux / float(solid_full.shape[1])
    dynamic_viscosity = 0.5 * (density_in + density_out) * nu
    length = core_end - core_start
    permeability = abs(dynamic_viscosity * darcy_velocity * length / delta_pressure)
    flux_scale = max(abs(inlet_flux), abs(outlet_flux), np.finfo(float).eps)
    return {
        "permeability_lattice": permeability,
        "mass_balance_error": abs(inlet_flux - outlet_flux) / flux_scale,
        "inlet_total_flux_lattice": inlet_flux,
        "outlet_total_flux_lattice": outlet_flux,
        "mean_total_flux_lattice": mean_flux,
    }


def _initial_state(
    solid_core: np.ndarray,
    config: WarmStartConfig,
    initial_core: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    nx, ny = solid_core.shape
    core_start = config.buffer_rows
    core_end = core_start + nx
    solid_full = np.zeros((nx + 2 * config.buffer_rows, ny), dtype=np.uint8)
    solid_full[core_start:core_end] = solid_core
    rho = np.ones_like(solid_full, dtype=np.float64)
    ux = np.zeros_like(rho)
    uy = np.zeros_like(rho)
    rho[0] = config.rhoin
    rho[-1] = config.rhoout

    if initial_core is not None:
        for name in ("rho", "ux", "uy"):
            value = np.asarray(initial_core[name], dtype=np.float64)
            if value.shape != solid_core.shape:
                raise ValueError(f"initial {name} shape {value.shape} != {solid_core.shape}")
        rho_core = np.clip(
            np.asarray(initial_core["rho"], dtype=np.float64),
            config.rhoout - abs(config.rhoin - config.rhoout),
            config.rhoin + abs(config.rhoin - config.rhoout),
        )
        ux_core = np.asarray(initial_core["ux"], dtype=np.float64)
        uy_core = np.asarray(initial_core["uy"], dtype=np.float64)
        speed = np.sqrt(ux_core * ux_core + uy_core * uy_core)
        velocity_cap = 0.08 * math.sqrt(CS2)
        factor = np.minimum(1.0, velocity_cap / np.maximum(speed, 1e-30))
        rho[core_start:core_end] = rho_core
        ux[core_start:core_end] = ux_core * factor
        uy[core_start:core_end] = uy_core * factor

        # Extend the nearest predicted fluid row through each empty buffer so the
        # warm start does not introduce a velocity discontinuity at the core edge.
        ux[:core_start] = ux[core_start]
        uy[:core_start] = uy[core_start]
        ux[core_end:] = ux[core_end - 1]
        uy[core_end:] = uy[core_end - 1]
        for row in range(core_start):
            alpha = row / float(core_start)
            rho[row] = (1.0 - alpha) * config.rhoin + alpha * rho[core_start]
        for offset, row in enumerate(range(core_end, rho.shape[0])):
            alpha = (offset + 1) / float(config.buffer_rows)
            rho[row] = (1.0 - alpha) * rho[core_end - 1] + alpha * config.rhoout

    solid = solid_full == 1
    rho[solid] = 1.0
    ux[solid] = 0.0
    uy[solid] = 0.0
    f = equilibrium(rho, ux, uy)
    return solid_full, rho, ux, uy, f, core_start, core_end


def run_lbm_from_state(
    solid_core: np.ndarray,
    config: WarmStartConfig,
    initial_core: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the dataset's D2Q9-BGK solver from rest or a neural field prediction."""
    solid_core = np.asarray(solid_core, dtype=np.uint8)
    solid_full, rho, ux, uy, f, core_start, core_end = _initial_state(
        solid_core, config, initial_core
    )
    fluid = solid_full == 0
    previous_ux = ux.copy()
    previous_uy = uy.copy()
    previous_permeability = math.nan
    permeability_change = math.inf
    velocity_residual = math.inf
    metrics: dict[str, float] = {}
    converged = False
    finite_state = True
    mach_valid = True
    max_mach = 0.0
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        speed_squared = ux * ux + uy * uy
        f_post = f.copy()
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
        solid_populations = streamed[solid_full == 1, :].copy()
        streamed[solid_full == 1, :] = solid_populations[:, OPPOSITE]
        f = streamed
        rho, ux, uy = macroscopic(f)
        apply_pressure_boundaries(f, rho, ux, uy, config.rhoin, config.rhoout)
        ux[solid_full == 1] = 0.0
        uy[solid_full == 1] = 0.0
        rho[solid_full == 1] = 1.0
        if iteration % config.check_interval != 0:
            continue

        iterations = iteration
        velocity_residual = _velocity_residual(ux, uy, previous_ux, previous_uy, fluid)
        metrics = _transport_metrics(
            ux, rho, solid_full, core_start, core_end, config.nu_lattice
        )
        permeability = metrics["permeability_lattice"]
        if math.isfinite(previous_permeability) and math.isfinite(permeability):
            scale = max(abs(previous_permeability), abs(permeability), np.finfo(float).eps)
            permeability_change = abs(permeability - previous_permeability) / scale
        else:
            permeability_change = math.inf
        finite_state = bool(
            np.isfinite(rho).all()
            and np.isfinite(ux).all()
            and np.isfinite(uy).all()
            and np.isfinite(f).all()
        )
        # Apply the same low-Mach quality gate as dataset generation at the
        # same cadence as residual, permeability, and mass-balance checks.
        speed = np.sqrt(ux * ux + uy * uy)
        max_mach = float(np.max(speed[fluid]) / math.sqrt(CS2))
        mach_valid = bool(math.isfinite(max_mach) and max_mach < config.max_mach_criterion)
        if not finite_state or not mach_valid:
            break
        if (
            velocity_residual < config.velocity_residual_criterion
            and metrics["mass_balance_error"] < config.mass_balance_criterion
            and permeability_change < config.permeability_change_criterion
        ):
            converged = True
            break
        previous_ux[...] = ux
        previous_uy[...] = uy
        previous_permeability = permeability

    if iterations == 0:
        iterations = config.max_iterations
        velocity_residual = _velocity_residual(ux, uy, previous_ux, previous_uy, fluid)
        metrics = _transport_metrics(
            ux, rho, solid_full, core_start, core_end, config.nu_lattice
        )
    return {
        "rho_core": rho[core_start:core_end].copy(),
        "ux_core": ux[core_start:core_end].copy(),
        "uy_core": uy[core_start:core_end].copy(),
        "converged": converged,
        "finite_state": finite_state,
        "mach_valid": mach_valid,
        "max_mach": max_mach,
        "iterations": iterations,
        "velocity_residual": velocity_residual,
        "permeability_relative_change": permeability_change,
        **metrics,
    }
