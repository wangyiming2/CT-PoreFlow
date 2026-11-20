from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_warm_starts import METRICS, load


class WarmStartComparisonTest(unittest.TestCase):
    def test_matched_cold_permeability_difference_is_derived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "warm.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "cold_permeability_lattice",
                        "warm_permeability_lattice",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "test/sample.npz",
                        "cold_permeability_lattice": 2.0,
                        "warm_permeability_lattice": 2.5,
                    }
                )
            rows = load(path)
            self.assertAlmostEqual(
                float(
                    rows["test/sample.npz"][
                        "warm_permeability_relative_difference_vs_cold"
                    ]
                ),
                0.25,
            )
            self.assertEqual(len(METRICS), 9)


if __name__ == "__main__":
    unittest.main()
