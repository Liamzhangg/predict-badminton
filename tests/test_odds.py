from __future__ import annotations

import copy
import json
import math
import pickle
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from predict_badminton.features import FeatureBuilder
from predict_badminton.odds import american_to_decimal, expected_return, fair_american, first_round_predictions, positive_ev_from, round_predictions


class FixedModel:
    def __init__(self, history):
        self.feature_name_ = history.feature_columns
        self.feature_policy_ = history.model_policy
        self.trained_through_ = history.history_cutoff

    def predict_proba(self, frame):
        return np.tile([0.4, 0.6], (len(frame), 1))


def snapshot():
    return {"name": "Example Open", "tournament_code": "example", "tournament_tier": 5,
            "matches": [{"match_id": "1", "discipline": "MS", "round_name": "R32", "slot": 1,
                         "match_time": "2030-01-02 00:00:00", "time_estimated": True, "winner": 0,
                         "team1_name": "Player A", "team2_name": "Player B",
                         "team1_ids": ["1"], "team2_ids": ["2"],
                         "team1_country": "CAN", "team2_country": "USA",
                         "team1_seed": 0, "team2_seed": 0}]}


class OddsTests(unittest.TestCase):
    def test_standard_conversions_and_even_money(self):
        for odds, decimal in ((150, 2.5), (-150, 5 / 3), (100, 2), (-100, 2)):
            with self.subTest(odds=odds):
                self.assertAlmostEqual(american_to_decimal(odds), decimal)
        for probability, odds in ((0.5, 100), (0.4, 150), (0.6, -150), (0.8, -400)):
            self.assertAlmostEqual(fair_american(probability), odds)

    def test_user_example_is_positive_ev_not_a_guaranteed_win(self):
        self.assertAlmostEqual(expected_return(0.5, 150), 0.25)
        self.assertAlmostEqual(expected_return(0.5, -110), -1 / 22)
        self.assertAlmostEqual(expected_return(0.4, 150), 0)
        self.assertAlmostEqual(expected_return(0.6, -150), 0)

    def test_minimum_prices_are_strictly_positive_not_rounded_break_even(self):
        for probability, price in ((0.5, 101), (0.4, 151), (0.6, -149), (0.8, -399)):
            self.assertEqual(positive_ev_from(probability), price)
        for probability in np.linspace(0.001, 0.999, 1001):
            price = positive_ev_from(float(probability))
            self.assertGreater(expected_return(probability, price), 0)
            previous = -100 if price == 100 else price - 1
            self.assertLessEqual(expected_return(probability, previous), 1e-12)

    def test_invalid_prices_and_probabilities_are_rejected(self):
        for odds in (0, 99, -99, math.inf, math.nan):
            with self.subTest(odds=odds), self.assertRaises(ValueError):
                american_to_decimal(odds)
        for probability in (-1, 0, 1, 2, math.inf, math.nan):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                fair_american(probability)
        for probability in (-1, 2, math.inf, math.nan):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                expected_return(probability, 100)

    def test_predictions_preserve_history_and_complementary_probabilities(self):
        data = snapshot()
        original = copy.deepcopy(data)
        history = FeatureBuilder()
        history.elo._last_match["known"] = datetime(2030, 1, 1)
        before = pickle.dumps(history.get_state())
        table, cutoff = first_round_predictions(data, (FixedModel(history), None, None, history))
        self.assertEqual(table["probability"].tolist(), [0.5, 0.5])
        self.assertEqual(table["positive_ev_from"].tolist(), [101, 101])
        self.assertEqual(cutoff, "2030-01-01T00:00:00")
        self.assertEqual(pickle.dumps(history.get_state()), before)
        self.assertEqual(data, original)

    def test_history_cutoff_is_checked_in_utc(self):
        data = snapshot()
        data["matches"][0]["match_time"] = "2030-01-02 09:00:00+09:00"
        history = FeatureBuilder()
        history.elo._last_match["known"] = datetime(2030, 1, 2)
        with self.assertRaisesRegex(ValueError, "cutoff"):
            first_round_predictions(data, (FixedModel(history), None, None, history))

    def test_completed_duplicate_missing_or_nonstandard_fixtures_are_rejected(self):
        history = FeatureBuilder()
        for field, value in (("winner", 1), ("round_name", "R16"), ("team1_ids", []),
                             ("team1_ids", ["Q1"]), ("team1_ids", [""]),
                             ("team1_ids", ["2"]), ("discipline", "OTHER")):
            data = snapshot()
            data["matches"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                first_round_predictions(data, (FixedModel(history), None, None, history))
        data = snapshot()
        data["matches"] *= 2
        with self.assertRaisesRegex(ValueError, "unique"):
            first_round_predictions(data, (FixedModel(history), None, None, history))

    def test_official_snapshot_contains_80_unplayed_main_draw_matches(self):
        path = Path(__file__).resolve().parents[1] / "data/draws/kumamoto-masters-japan-2026-first-round.json"
        data = json.loads(path.read_text())
        self.assertEqual(Counter(m["discipline"] for m in data["matches"]),
                         {event: 16 for event in ("MS", "WS", "MD", "WD", "XD")})
        self.assertEqual(len({m["match_id"] for m in data["matches"]}), 80)
        self.assertEqual(len(data["sources"]), 5)
        self.assertTrue(all(len(source["sha256"]) == 64 for source in data["sources"]))
        self.assertEqual(sum(m["time_estimated"] for m in data["matches"]), 59)
        self.assertTrue(all(m["winner"] == 0 and m["round_name"] == "R32" for m in data["matches"]))
        history = FeatureBuilder()
        table, _ = first_round_predictions(data, (FixedModel(history), None, None, history))
        self.assertEqual(len(table), 160)
        np.testing.assert_allclose(table.groupby("match_id")["probability"].sum(), 1)

    def test_r16_snapshot_covers_all_events_and_excludes_the_official_walkover(self):
        path = Path(__file__).resolve().parents[1] / "data/draws/kumamoto-masters-japan-2026-r16.json"
        data = json.loads(path.read_text())
        self.assertEqual(Counter(m["discipline"] for m in data["matches"]),
                         {"MS": 8, "WS": 8, "MD": 8, "WD": 8, "XD": 7})
        self.assertEqual(len({m["match_id"] for m in data["matches"]}), 39)
        self.assertEqual(len(data["sources"]), 6)
        self.assertTrue(all(not m["time_estimated"] and m["winner"] == 0 for m in data["matches"]))
        self.assertEqual(data["excluded_matches"][0]["match_id"], "1565902")
        self.assertEqual(data["excluded_matches"][0]["status"], "Walkover")
        history = FeatureBuilder()
        before = pickle.dumps(history.get_state())
        table, _ = round_predictions(data, (FixedModel(history), None, None, history), round_name="R16")
        self.assertEqual(len(table), 78)
        self.assertNotIn("1565902", set(table["match_id"]))
        np.testing.assert_allclose(table.groupby("match_id")["probability"].sum(), 1)
        self.assertEqual(before, pickle.dumps(history.get_state()))

    def test_round_predictions_reject_wrong_rounds_and_walkovers_even_without_a_winner(self):
        history = FeatureBuilder()
        model = FixedModel(history)
        for changes in ({"round_name": "R32"}, {"score_status": 1}, {"status": "Walkover"}, {"winner": 2}):
            data = snapshot()
            data["matches"][0]["round_name"] = "R16"
            data["matches"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Only unplayed R16"):
                round_predictions(data, (model, None, None, history), round_name="R16")
        with self.assertRaisesRegex(ValueError, "Unsupported knockout round"):
            round_predictions(snapshot(), (model, None, None, history), round_name="qualifying")
