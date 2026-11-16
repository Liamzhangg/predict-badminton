from datetime import datetime, timedelta
import pickle
import unittest

import numpy as np
import pandas as pd

import experiment as e
from predict_badminton.features import FeatureBuilder
from predict_badminton.config import FEATURE_COLS
from tests.test_feature_state import _match


def match(when, edition="next", ident="m", available=None):
    return {"match_id": ident, "fixture_id": ident, "match_time": when, "prediction_time": when,
            "available_at": available or when+timedelta(hours=1), "edition": edition,
            "tournament_code": edition, "tournament_name": edition, "discipline": "MS",
            "team1_ids": ["1"], "team2_ids": ["2"], "team1_country": "AAA", "team2_country": "BBB"}


class TravelTests(unittest.TestCase):
    def setUp(self):
        self.points = {"AA": (0, 0), "BB": (0, 90), "CC": (0, 180)}
        self.codes = {"AAA": "AA", "BBB": "BB", "CCC": "CC"}
        self.history = e.Travel(self.points, self.codes)
        self.now = datetime(2025, 1, 20, 12)

    def test_great_circle_distance(self):
        self.assertEqual(e.distance("AA", "AA", self.points), 0)
        self.assertAlmostEqual(e.distance("AA", "BB", self.points), 10007.557221, places=5)
        self.assertAlmostEqual(e.distance("AA", "CC", self.points), 20015.114442, places=5)
        self.assertTrue(np.isnan(e.distance(None, "AA", self.points)))

    def test_no_prior_tournament_uses_current_home_proxy(self):
        row, _ = self.history.preview(match(self.now), "BB", 14)
        self.assertAlmostEqual(row["km_t1"], e.distance("AA", "BB", self.points))
        self.assertEqual(row["km_t2"], 0)
        self.assertEqual(row["travel_home_origin_t1"], 1)

    def test_recent_previous_tournament_is_used_across_disciplines(self):
        old = match(self.now-timedelta(days=7), "old")
        old["discipline"] = "XD"
        self.history.observe(old, "CC")
        row, _ = self.history.preview(match(self.now), "AA", 14)
        self.assertAlmostEqual(row["km_t1"], e.distance("CC", "AA", self.points))
        self.assertEqual(row["travel_home_origin_t1"], 0)

    def test_exact_14_days_keeps_last_location_but_later_falls_back(self):
        self.history.observe(match(self.now-timedelta(days=14), "old"), "CC")
        same, _ = self.history.preview(match(self.now), "BB", 14)
        later, _ = self.history.preview(match(self.now+timedelta(seconds=1)), "BB", 14)
        self.assertEqual(same["travel_home_origin_t1"], 0)
        self.assertEqual(later["travel_home_origin_t1"], 1)

    def test_unknown_recent_location_does_not_turn_into_zero_or_assumed_home(self):
        self.history.observe(match(self.now-timedelta(days=1), "old"), None)
        row, _ = self.history.preview(match(self.now), "AA", 14)
        self.assertTrue(np.isnan(row["km_t1"]))
        self.assertEqual(row["travel_origin_missing_t1"], 1)
        self.assertEqual(row["travel_home_origin_t1"], 0)

    def test_future_location_and_out_of_order_results_cannot_override_recent_past(self):
        self.history.observe(match(self.now-timedelta(days=1), "recent"), "CC")
        self.history.observe(match(self.now-timedelta(days=10), "older"), "BB")
        row, _ = self.history.preview(match(self.now), "AA", 14)
        self.assertAlmostEqual(row["km_t1"], e.distance("CC", "AA", self.points))
        self.history.observe(match(self.now+timedelta(days=1), "future"), "CC")
        row, _ = self.history.preview(match(self.now), "AA", 14)
        self.assertEqual(row["travel_home_origin_t1"], 1)

    def test_inbound_distance_persists_for_later_rounds_in_same_tournament(self):
        first = match(self.now)
        row, changes = self.history.preview(first, "CC", 14)
        self.history.commit_targets(changes, 14)
        self.history.observe(first, "CC")
        later, _ = self.history.preview(match(self.now+timedelta(days=1)), "CC", 14)
        self.assertEqual(row["km_t1"], later["km_t1"])
        self.assertFalse(later["first_entry"])

    def test_long_break_within_same_edition_restarts_from_home(self):
        _, changes = self.history.preview(match(self.now), "CC", 14)
        self.history.commit_targets(changes, 14)
        self.history.observe(match(self.now), "CC")
        later, _ = self.history.preview(match(self.now+timedelta(days=15)), "BB", 14)
        self.assertEqual(later["travel_home_origin_t1"], 1)
        self.assertTrue(later["first_entry"])

    def test_preview_is_read_only(self):
        before = pickle.dumps((self.history.previous, self.history.trips))
        self.history.preview(match(self.now), "BB", 14)
        self.assertEqual(before, pickle.dumps((self.history.previous, self.history.trips)))

    def test_doubles_average_partners_and_missing_partner_stays_missing(self):
        target = match(self.now)
        target.update(discipline="MD", team1_ids=["1", "3"], team2_ids=["2", "4"])
        old = match(self.now-timedelta(days=1), "old")
        old["team1_ids"], old["team2_ids"] = ["1"], ["2"]
        self.history.observe(old, "CC")
        row, _ = self.history.preview(target, "AA", 14)
        self.assertAlmostEqual(row["km_t1"], e.distance("CC", "AA", self.points)/2)
        self.history.previous["3"] = {"time": self.now-timedelta(hours=1), "host": None, "edition": "unknown"}
        row, _ = self.history.preview(target, "AA", 14)
        self.assertTrue(np.isnan(row["km_t1"]))

    def test_availability_cutoff_is_strict_and_tied_targets_do_not_share_outcomes(self):
        old = match(self.now-timedelta(days=2), "old", "old", available=self.now)
        target = match(self.now, "target", "target")
        locations = (self.points, self.codes, {"old": "CC", "target": "BB"}, {})
        frame = e.travel_frame([old, target], locations).set_index("fixture_id")
        self.assertEqual(frame.loc["target", "d14_travel_home_origin_t1"], 1)
        old["available_at"] = self.now-timedelta(seconds=1)
        frame = e.travel_frame([old, target], locations).set_index("fixture_id")
        self.assertEqual(frame.loc["target", "d14_travel_home_origin_t1"], 0)

    def test_prediction_side_swap_is_complementary_for_augmented_features(self):
        base = FeatureBuilder().preview(_match("future", self.now, 1))
        frame = pd.DataFrame([base])
        frame["discipline"] = pd.Categorical(frame["discipline"], categories=["MS", "WS", "MD", "WD", "XD", "OTHER"])
        for column in e.GEO + e.DISTANCE:
            frame["d14_"+column] = 1 if column.endswith("t1") else 0

        class Model:
            def predict_proba(self, value):
                p = 1/(1+np.exp(value["travel_log_km_diff"].to_numpy()))
                return np.column_stack((1-p, p))

        original = e.predict(Model(), frame, "travel_14d")
        swapped = frame.copy()
        for name in FEATURE_COLS: swapped[name] = e.swap_features(frame[FEATURE_COLS])[name]
        for column in e.GEO + e.DISTANCE:
            swapped["d14_"+column] = -frame["d14_"+column] if column.endswith("diff") else frame["d14_"+column[:-1]+("2" if column.endswith("1") else "1")]
        np.testing.assert_allclose(original, 1-e.predict(Model(), swapped, "travel_14d"), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
