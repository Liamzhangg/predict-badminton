from datetime import datetime, timedelta
import unittest

import numpy as np
import pandas as pd

import run as r
from tests.test_feature_state import _match


class ProductionComparisonTests(unittest.TestCase):
    def test_baseline_is_exact_production_schema(self):
        self.assertEqual(r.VARIANTS["production_recipe"], [])
        self.assertEqual(len(r.p.s.e.FEATURE_COLS), 44)

    def test_candidate_is_unchanged_country_bundle_without_home_or_distance(self):
        expected = r.p.s.LOCATION+r.p.s.STRENGTH+r.p.s.IDENTITY
        self.assertEqual(r.VARIANTS["country_candidate"], expected)
        self.assertEqual(len(r.p.s.e.FEATURE_COLS+expected), 55)
        self.assertFalse(set(expected) & set(r.p.s.HOME+r.p.s.DISTANCE))

    def test_replay_preserves_model_hyperparameters_except_seed_and_threads(self):
        original = {"random_state": 42, "n_jobs": -1, "n_estimators": 500, "max_depth": 6, "reg_alpha": .1}
        result = r.training_params(original, 99)
        self.assertEqual(result, {**original, "random_state": 99, "n_jobs": 4})
        self.assertEqual(original["random_state"], 42)
        self.assertEqual(original["n_jobs"], -1)

    def test_actual_artifact_evaluation_excludes_all_pre_snapshot_targets(self):
        cutoff = datetime(2026, 11, 12, 5, 42)
        data = pd.DataFrame({"prediction_time": [cutoff-timedelta(seconds=1), cutoff, cutoff+timedelta(seconds=1)]})
        self.assertEqual(r.post_training_rows(data, cutoff).index.tolist(), [2])

    def test_training_digest_is_order_independent_but_tracks_labels_and_timing(self):
        when = datetime(2026, 1, 1)
        first = {"fixture_id": "a", "winner": 1, "prediction_time": when, "available_at": when+timedelta(hours=1)}
        second = {**first, "fixture_id": "b"}
        digest = r.cohort_digest([first, second])
        self.assertEqual(digest, r.cohort_digest([second, first]))
        self.assertNotEqual(digest, r.cohort_digest([{**first, "winner": 2}, second]))
        self.assertNotEqual(digest, r.cohort_digest([{**first, "available_at": when}, second]))
        self.assertNotEqual(digest, r.cohort_digest([{**first, "team1_country": "JPN"}, second]))

    def test_both_feature_schemas_reverse_and_score_symmetrically(self):
        base = r.p.s.e.FeatureBuilder().preview(_match("future", datetime(2026, 1, 1), 1))
        frame = pd.DataFrame([base])
        frame["discipline"] = pd.Categorical(frame["discipline"], categories=["MS", "WS", "MD", "WD", "XD", "OTHER"])
        for column in r.VARIANTS["country_candidate"]:
            frame[column] = "AAA" if column == "country_id_t1" else "BBB" if column == "country_id_t2" else 0.0

        class Model:
            def predict_proba(self, data):
                score = data["elo_diff_disc"].to_numpy()/400
                if "country_id_t1" in data:
                    score = score+data["country_id_t1"].astype(object).map({"AAA": 1, "BBB": -1}).to_numpy()/4
                p = 1/(1+np.exp(-score))
                return np.column_stack((1-p, p))

        for variant in r.VARIANTS:
            first, _ = r.p.s.matrices(frame, frame, variant)
            swapped = r.p.s.reverse(first, variant)
            pd.testing.assert_frame_equal(first, r.p.s.reverse(swapped, variant))
            np.testing.assert_allclose(r.p.s.predict(Model(), first, variant), 1-r.p.s.predict(Model(), swapped, variant), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
