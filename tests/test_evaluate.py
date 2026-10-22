from __future__ import annotations

import unittest

import numpy as np

from predict_badminton.evaluate import reliability_stats


class ReliabilityTests(unittest.TestCase):
    def test_calibration_error_is_weighted_by_bin_population(self) -> None:
        labels = np.array([0] * 9 + [1])
        probabilities = np.array([0.1] * 9 + [0.6])
        stats = reliability_stats(labels, probabilities, n_bins=2)
        self.assertAlmostEqual(stats["ece"], 0.13)

    def test_bin_boundaries_match_the_reliability_curve(self) -> None:
        stats = reliability_stats(np.array([0, 0, 0, 1]), np.array([0.1, 0.5, 0.5, 1.0]), n_bins=2)
        self.assertAlmostEqual(stats["ece"], 1.1 / 4)


if __name__ == "__main__":
    unittest.main()
