from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ood_geometry import OOD_FAMILIES, geometry_statistics, make_rule_mask


class OODGeometryTests(unittest.TestCase):
    def test_all_families_are_deterministic_binary_and_connected(self) -> None:
        for family in OOD_FAMILIES:
            first = make_rule_mask(family, seed=2026, severity=1, target_porosity=0.65)
            second = make_rule_mask(family, seed=2026, severity=1, target_porosity=0.65)
            self.assertTrue(np.array_equal(first, second), family)
            self.assertEqual(first.shape, (100, 100))
            self.assertEqual(set(np.unique(first)), {0, 1})
            self.assertFalse(first[0].any())
            self.assertFalse(first[-1].any())
            statistics = geometry_statistics(first)
            self.assertGreater(statistics["connected_porosity"], 0.0)
            self.assertGreater(statistics["porosity"], 0.50)
            self.assertLess(statistics["porosity"], 0.72)


if __name__ == "__main__":
    unittest.main()
