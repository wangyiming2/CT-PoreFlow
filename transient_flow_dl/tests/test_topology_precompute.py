from __future__ import annotations

import unittest

import numpy as np

from precompute_topology import geodesic


class GeodesicNormalizationTests(unittest.TestCase):
    def test_distance_uses_fixed_lattice_scale(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        path = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (2, 3),
            (2, 2),
            (2, 1),
            (2, 0),
            (3, 0),
            (4, 0),
        ]
        for row, column in path:
            mask[row, column] = True

        distance = geodesic(mask, from_inlet=True)

        # Every open inlet pixel is a source; the shortest route begins at
        # (0,4) and takes eight graph edges to reach the outlet pixel.
        self.assertAlmostEqual(float(distance[4, 0]), 8.0 / 5.0, places=6)
        self.assertGreater(float(distance.max()), 1.0)
        self.assertEqual(float(distance[1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
