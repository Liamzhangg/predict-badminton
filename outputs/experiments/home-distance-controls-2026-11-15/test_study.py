from datetime import datetime, timedelta
import pickle
import unittest

import numpy as np
import pandas as pd

import study as s
from tests.test_feature_state import _match


def match(when, ident="m", first="AAA", second="BBB", discipline="MS", available=None):
    return {"match_id": ident, "fixture_id": ident, "match_time": when, "prediction_time": when,
            "available_at": available or when+timedelta(hours=1), "tournament_code": "event",
            "tournament_name": "event", "discipline": discipline, "winner": 1,
            "team1_country": first, "team2_country": second}


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2025, 1, 20, 12)
        self.points = {"AA": (0, 0), "BB": (0, 90)}
        self.codes = {"AAA": "AA", "BBB": "BB", "CCC": "AA"}
        self.locations = self.points, self.codes, {"event": "AA"}, {}

    def test_country_strength_update_is_uniform_and_discipline_specific(self):
        history = s.CountryStrength(self.codes)
        history.observe(match(self.now))
        self.assertEqual(history.preview("AAA", "MS"), (1510, np.log(2)))
        self.assertEqual(history.preview("BBB", "MS"), (1490, np.log(2)))
        self.assertEqual(history.preview("AAA", "WS"), (1500, 0))

    def test_domestic_and_unknown_matches_do_not_update_country_strength(self):
        history = s.CountryStrength(self.codes)
        before = pickle.dumps(history.engine.get_state())
        history.observe(match(self.now, second="AAA"))
        history.observe(match(self.now, second="unknown"))
        self.assertEqual(before, pickle.dumps(history.engine.get_state()))

    def test_country_preview_is_read_only(self):
        history = s.CountryStrength(self.codes)
        before = pickle.dumps(history.engine.get_state())
        history.preview("AAA", "MS")
        history.preview(None, "MS")
        self.assertEqual(before, pickle.dumps(history.engine.get_state()))

    def test_results_at_target_timestamp_are_withheld(self):
        earlier = match(self.now-timedelta(days=2), "old", available=self.now)
        target = match(self.now, "target")
        frame = s.context_frame([earlier, target], self.locations).set_index("fixture_id")
        self.assertEqual(frame.loc["target", "country_rating_t1"], 1500)
        earlier["available_at"] = self.now-timedelta(seconds=1)
        frame = s.context_frame([earlier, target], self.locations).set_index("fixture_id")
        self.assertEqual(frame.loc["target", "country_rating_t1"], 1510)

    def test_tied_predictions_cannot_share_outcomes(self):
        first, second = match(self.now, "a"), match(self.now, "b")
        frame = s.context_frame([first, second], self.locations)
        self.assertTrue((frame["country_rating_t1"] == 1500).all())
        later = match(self.now+timedelta(days=1), "c")
        combined = s.context_frame([first, second, later], self.locations).set_index("fixture_id")
        self.assertGreater(combined.loc["c", "country_rating_t1"], 1510)

    def test_home_distance_and_missingness(self):
        row = s.context_frame([match(self.now)], self.locations).iloc[0]
        self.assertEqual(row["home_t1"], 1)
        self.assertEqual(row["home_t2"], 0)
        self.assertEqual(row["home_log_km_t1"], 0)
        self.assertAlmostEqual(row["home_log_km_t2"], np.log1p(10007.557221), places=8)
        unknown = s.context_frame([match(self.now, first="unknown")], self.locations).iloc[0]
        self.assertTrue(np.isnan(unknown["home_t1"]))
        self.assertTrue(np.isnan(unknown["home_log_km_t1"]))
        self.assertTrue(np.isnan(unknown["country_rating_t1"]))
        self.assertEqual(unknown["country_known_t1"], 0)

    def test_geographic_alias_does_not_merge_country_strength_identities(self):
        history = s.CountryStrength(self.codes)
        history.observe(match(self.now, second="CCC"))
        self.assertEqual(history.preview("AAA", "MS")[0], 1510)
        self.assertEqual(history.preview("CCC", "MS")[0], 1490)

    def feature_rows(self):
        base = s.e.FeatureBuilder().preview(_match("future", self.now, 1))
        frame = pd.DataFrame([base])
        frame["discipline"] = pd.Categorical(frame["discipline"], categories=["MS", "WS", "MD", "WD", "XD", "OTHER"])
        context = s.context_frame([match(self.now)], self.locations).iloc[0]
        for column in set(s.VARIANTS["all"]):
            frame[column] = context[column]
        return frame

    def test_country_categories_are_training_only_and_shared_across_sides(self):
        train = self.feature_rows()
        test = train.copy()
        test["country_id_t1"] = "CCC"
        first, second = s.matrices(train, test, "all")
        self.assertEqual(list(first["country_id_t1"].cat.categories), ["AAA", "BBB"])
        self.assertEqual(first["country_id_t1"].dtype, first["country_id_t2"].dtype)
        self.assertTrue(second["country_id_t1"].isna().all())

    def test_primary_comparison_adds_only_distance(self):
        self.assertEqual(s.VARIANTS["all"], s.VARIANTS["home_country_strength"]+s.DISTANCE)
        for name in s.VARIANTS:
            if name != "raw_control":
                self.assertEqual(s.VARIANTS[name][:len(s.LOCATION)], s.LOCATION)

    def test_reverse_is_involution_and_predictions_are_complementary(self):
        frame = self.feature_rows()
        first, _ = s.matrices(frame, frame, "all")
        reverse = s.reverse(first, "all")
        pd.testing.assert_frame_equal(first, s.reverse(reverse, "all"))

        class Model:
            def predict_proba(self, value):
                p = 1/(1+np.exp(value["home_log_km_diff"].to_numpy()/10))
                return np.column_stack((1-p, p))

        np.testing.assert_allclose(s.predict(Model(), first, "all"), 1-s.predict(Model(), reverse, "all"), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
