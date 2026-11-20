from __future__ import annotations

import unittest

import numpy as np

from src.data import TransientFlowDataset


class TemporalSubsetTest(unittest.TestCase):
    @staticmethod
    def dataset(mode: str, indices: list[int]) -> TransientFlowDataset:
        dataset = object.__new__(TransientFlowDataset)
        dataset.mode = mode
        dataset.time_indices = indices
        dataset.iterations = np.arange(30)
        dataset.seed = 2026
        dataset.pairs = [(None, None, "sample")] * 3
        return dataset

    def test_random_training_never_samples_heldout_index(self) -> None:
        allowed = [0, 3, 11, 29]
        dataset = self.dataset("random_time", allowed)
        observed = {dataset._index(0)[1] for _ in range(200)}
        self.assertTrue(observed)
        self.assertTrue(observed.issubset(set(allowed)))

    def test_fixed_evaluation_preserves_requested_index_order(self) -> None:
        requested = [2, 7, 27]
        dataset = self.dataset("fixed_times", requested)
        self.assertEqual(dataset._index(0), (0, 2))
        self.assertEqual(dataset._index(1), (0, 7))
        self.assertEqual(dataset._index(2), (0, 27))
        self.assertEqual(dataset._index(3), (1, 2))


if __name__ == "__main__":
    unittest.main()
