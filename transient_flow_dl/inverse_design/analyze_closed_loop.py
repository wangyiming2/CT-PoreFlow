from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def aggregate_by_target(rows: list[dict[str, str]], method: str, metric: str) -> dict[str, float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            grouped[row["target_id"]].append(float(row[metric]))
    return {target: float(np.mean(values)) for target, values in grouped.items()}


def exact_sign_flip_p(difference: np.ndarray) -> float:
    observed = abs(float(difference.mean()))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(difference)):
        statistic = abs(float(np.mean(difference * np.asarray(signs))))
        count += statistic >= observed - 1e-15
        total += 1
    return count / total


def bootstrap_interval(difference: np.ndarray, seed: int, repeats: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(repeats, len(difference)))
    values = difference[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[int(index)])
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--novel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows = read_rows(args.input)
    methods = ["Random-QSGS", "Morphology-only", "CT-PoreFlow", "No-anchor"]
    proposed = aggregate_by_target(rows, "Time-U-Net", "verified_target_distance")
    comparisons = []
    for offset, baseline in enumerate(methods):
        reference = aggregate_by_target(rows, baseline, "verified_target_distance")
        targets = sorted(set(proposed) & set(reference))
        difference = np.asarray([reference[target] - proposed[target] for target in targets])
        lower, upper = bootstrap_interval(difference, args.seed + offset)
        comparisons.append(
            {
                "comparison": f"Time-U-Net vs {baseline}",
                "targets": len(targets),
                "mean_distance_reduction": float(difference.mean()),
                "median_distance_reduction": float(np.median(difference)),
                "bootstrap_95_ci_low": lower,
                "bootstrap_95_ci_high": upper,
                "exact_sign_flip_p": exact_sign_flip_p(difference),
            }
        )
    adjusted = holm_adjust([float(row["exact_sign_flip_p"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = value

    novel_rows = read_rows(args.novel)
    novel_grouped: defaultdict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in novel_rows:
        novel_grouped[row["target_id"]][row["strategy"]] = row
    novel_pairs = []
    for target, strategies in sorted(novel_grouped.items()):
        guided = float(strategies["surrogate-guided"]["verified_target_distance"])
        random = float(strategies["random"]["verified_target_distance"])
        novel_pairs.append(
            {
                "target_id": target,
                "random_verified_distance": random,
                "guided_verified_distance": guided,
                "relative_distance_reduction": (random - guided) / random,
            }
        )
    payload = {
        "offline_paired_comparisons": comparisons,
        "novel_qsgs_pairs": novel_pairs,
        "novel_mean_random_distance": float(np.mean([row["random_verified_distance"] for row in novel_pairs])),
        "novel_mean_guided_distance": float(np.mean([row["guided_verified_distance"] for row in novel_pairs])),
        "novel_mean_relative_reduction": float(
            1.0
            - np.mean([row["guided_verified_distance"] for row in novel_pairs])
            / np.mean([row["random_verified_distance"] for row in novel_pairs])
        ),
        "note": "The novel-QSGS comparison contains only three target cases and is descriptive, not an inferential test.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inverse_design_statistics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "inverse_design_paired_tests.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
