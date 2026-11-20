from __future__ import annotations

import numpy as np

from stratified_analysis import _bootstrap_mean_ci


def test_geometry_bootstrap_is_deterministic_and_contains_mean() -> None:
    values = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float)
    first = _bootstrap_mean_ci(values, np.random.default_rng(2026), draws=2_000)
    second = _bootstrap_mean_ci(values, np.random.default_rng(2026), draws=2_000)
    assert first == second
    assert first[0] < float(values.mean()) < first[1]


def test_single_geometry_interval_collapses_to_value() -> None:
    assert _bootstrap_mean_ci(
        np.asarray([0.25]), np.random.default_rng(2026), draws=10
    ) == (0.25, 0.25)
