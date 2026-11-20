from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_warm_start import _load_cold_cache, _write_rows
from src.warm_start_lbm import WarmStartConfig, run_lbm_from_state


def test_cold_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cold.csv"
    rows = [
        {
            "sample_id": "s0",
            "cold_iterations_measured": 100,
            "cold_solver_seconds": 1.2,
            "cold_converged": True,
            "cold_finite_state": True,
            "cold_mach_valid": True,
            "cold_max_mach": 0.001,
            "cold_mass_balance_error": 0.01,
            "cold_permeability_lattice": 0.4,
        }
    ]
    _write_rows(path, rows)
    loaded = _load_cold_cache(path)
    assert loaded["s0"]["cold_converged"] == "True"
    assert float(loaded["s0"]["cold_solver_seconds"]) == 1.2


def test_short_warm_start_solver_remains_finite() -> None:
    solid = np.zeros((12, 12), dtype=np.uint8)
    solid[4:8, 5:7] = 1
    result = run_lbm_from_state(
        solid,
        WarmStartConfig(buffer_rows=2, max_iterations=20, check_interval=10),
    )
    assert result["iterations"] == 20
    assert result["finite_state"]
    assert result["mach_valid"]
    assert result["max_mach"] < 0.1


def test_mach_gate_prevents_false_convergence() -> None:
    solid = np.zeros((12, 12), dtype=np.uint8)
    result = run_lbm_from_state(
        solid,
        WarmStartConfig(
            buffer_rows=2,
            max_iterations=20,
            check_interval=10,
            max_mach_criterion=1e-8,
        ),
    )
    assert result["finite_state"]
    assert not result["mach_valid"]
    assert not result["converged"]
