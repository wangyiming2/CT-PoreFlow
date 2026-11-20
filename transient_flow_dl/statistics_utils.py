from __future__ import annotations

from collections.abc import Sequence


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm--Bonferroni family-wise-error adjusted p-values."""

    count = len(p_values)
    if count == 0:
        return []
    checked = [float(value) for value in p_values]
    if any(value < 0.0 or value > 1.0 for value in checked):
        raise ValueError("p-values must lie in [0, 1]")
    order = sorted(range(count), key=checked.__getitem__)
    adjusted = [1.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * checked[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def add_holm_fields(
    rows: list[dict[str, object]],
    *,
    raw_key: str = "wilcoxon_p_value",
    alpha: float = 0.05,
) -> None:
    """Add adjusted p-values and decisions to one pre-specified test family."""

    adjusted = holm_adjust([float(row[raw_key]) for row in rows])
    family_size = len(rows)
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = value
        row["holm_reject_at_0_05"] = bool(value < alpha)
        row["multiplicity_family_size"] = family_size


def matched_rank_biserial(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial effect size with average ranks for ties."""

    nonzero = [float(value) for value in differences if float(value) != 0.0]
    if not nonzero:
        return 0.0
    order = sorted(range(len(nonzero)), key=lambda index: abs(nonzero[index]))
    ranks = [0.0] * len(nonzero)
    start = 0
    while start < len(order):
        end = start + 1
        magnitude = abs(nonzero[order[start]])
        while end < len(order) and abs(nonzero[order[end]]) == magnitude:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    positive = sum(rank for rank, value in zip(ranks, nonzero) if value > 0.0)
    negative = sum(rank for rank, value in zip(ranks, nonzero) if value < 0.0)
    return (positive - negative) / max(positive + negative, 1e-30)
