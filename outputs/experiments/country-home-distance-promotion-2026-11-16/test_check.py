import unittest

import numpy as np
import pandas as pd

import check as c


def frame():
    return pd.DataFrame({"fixture_id": list("abcdefghijkl"), "label": [0, 1]*6,
                         "tournament_code": ["a"]*4+["b"]*4+["c"]*4,
                         "tournament_name": ["event"]*12,
                         "prediction_time": pd.to_datetime(["2026-01-01"]*12)})


class PromotionTests(unittest.TestCase):
    def test_metrics_include_finite_calibration_report(self):
        data = frame()
        probabilities = np.where(data["label"].to_numpy(), .7, .3)
        result = c.metrics(data, probabilities)
        self.assertEqual(result["n"], len(data))
        self.assertEqual(result["accuracy"], 1)
        self.assertAlmostEqual(result["ece"], .3)
        self.assertTrue(all(np.isfinite(result[name]) for name in c.METRICS))

    def test_only_home_and_distance_are_added(self):
        self.assertEqual(set(c.VARIANTS["home_distance"])-set(c.VARIANTS["country_only"]), set(c.s.HOME+c.s.DISTANCE))
        self.assertTrue(set(c.s.STRENGTH+c.s.IDENTITY+c.s.LOCATION) <= set(c.VARIANTS["country_only"]))

    def test_training_and_test_boundaries_are_strict(self):
        data = pd.DataFrame({"fixture_id": ["early", "late_label", "boundary", "end"],
                             "prediction_time": pd.to_datetime(["2025-12-20", "2025-12-21", "2026-01-01", "2026-04-01"]),
                             "available_at": pd.to_datetime(["2025-12-21", "2026-01-01", "2026-01-02", "2026-04-02"])})
        train, test = c.split_window(data, "2026-01-01", "2026-04-01")
        self.assertEqual(train["fixture_id"].tolist(), ["early"])
        self.assertEqual(test["fixture_id"].tolist(), ["boundary"])

    def test_quarters_partition_the_year_without_duplicate_boundary_rows(self):
        data = pd.DataFrame({"fixture_id": list("abcde"),
                             "prediction_time": pd.to_datetime(["2026-01-01", "2026-04-01", "2026-07-01", "2026-10-01", "2027-01-01"]),
                             "available_at": pd.to_datetime(["2026-01-02", "2026-04-02", "2026-07-02", "2026-10-02", "2027-01-02"])})
        seen = [value for start, end in c.QUARTERS for value in c.split_window(data, start, end)[1]["fixture_id"]]
        self.assertEqual(seen, list("abcd"))

    def test_identical_predictions_have_zero_differences_and_intervals(self):
        data = frame()
        probabilities = np.tile(np.where(data["label"].to_numpy(), .7, .3), (3, 1))
        result, evidence = c.paired_check(data, probabilities, probabilities, seeds=(1, 2, 3), repeats=100)
        self.assertEqual(result["n"], 12)
        self.assertEqual(result["clusters"], 3)
        self.assertEqual(sum(evidence["match_counts"]), 12)
        for name in c.DELTA_METRICS:
            self.assertEqual(result[name]["mean"], 0)
            self.assertEqual(result[name]["crossed_seed_cluster_ci95"], [0, 0])

    def test_better_predictions_improve_accuracy_loss_and_brier(self):
        data = frame()
        y = data["label"].to_numpy()
        wrong = np.tile(np.where(y, .4, .6), (2, 1))
        correct = np.tile(np.where(y, .8, .2), (2, 1))
        result, _ = c.paired_check(data, wrong, correct, seeds=(1, 2), repeats=100)
        self.assertEqual(result["accuracy"]["mean"], 1)
        self.assertLess(result["log_loss"]["mean"], 0)
        self.assertLess(result["brier_score"]["mean"], 0)
        self.assertGreater(result["accuracy"]["crossed_seed_cluster_ci95"][0], 0)

    def test_seed_averaging_is_not_probability_ensembling(self):
        data = frame()
        y = data["label"].to_numpy()
        baseline = np.tile(np.where(y, .6, .4), (2, 1))
        mixed = np.stack([np.where(y, .9, .1), np.where(y, .4, .6)])
        result, _ = c.paired_check(data, baseline, mixed, seeds=(1, 2), repeats=100)
        self.assertEqual(result["accuracy"]["mean"], -.5)
        self.assertLess(result["accuracy"]["crossed_seed_cluster_ci95"][0], result["accuracy"]["cluster_ci95"][0])

    def test_missing_tournament_code_uses_name_and_year_clusters(self):
        data = frame()
        data["tournament_code"] = None
        data["tournament_name"] = ["a"]*6+["b"]*6
        p = np.full((1, len(data)), .6)
        result, _ = c.paired_check(data, p, p, seeds=(1,), repeats=100)
        self.assertEqual(result["clusters"], 2)

    def test_no_forward_evidence_means_not_production_ready(self):
        stats = {"accuracy": {"mean": .01, "crossed_seed_cluster_ci95": [.001, .02]},
                 "log_loss": {"mean": -.01, "crossed_seed_cluster_ci95": [-.02, -.001]},
                 "brier_score": {"mean": -.01}}
        gate = c.promotion_gate(stats, stats, stats)
        self.assertTrue(gate["historical_gate_passed"])
        self.assertFalse(gate["evidence_ready_for_promotion"])
        self.assertFalse(gate["production_changed"])

    def test_accuracy_interval_touching_zero_does_not_pass(self):
        stats = {"accuracy": {"mean": .01, "crossed_seed_cluster_ci95": [0, .02]},
                 "log_loss": {"mean": -.01, "crossed_seed_cluster_ci95": [-.02, -.001]},
                 "brier_score": {"mean": -.01}}
        self.assertFalse(c.promotion_gate(stats, stats, stats, forward_available=True)["historical_gate_passed"])


if __name__ == "__main__":
    unittest.main()
