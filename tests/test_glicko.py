from __future__ import annotations

import math
import pickle
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from glicko2.math import phi_to_rd, rating_to_mu, rd_to_phi, update_rating

from predict_badminton.config import GLICKO_MODEL_FEATURE_COLS, GLICKO_FEATURE_COLS, LEGACY_FEATURE_COLS
from predict_badminton.features import FeatureBuilder
from predict_badminton.glicko import GlickoEngine
from tests.test_feature_state import _match


def _glicko_builder():
    builder = FeatureBuilder()
    builder._state_version = 3
    builder.glicko = GlickoEngine()
    builder.trueskill = None
    return builder


class GlickoTests(unittest.TestCase):
    def test_library_matches_published_worked_example(self):
        opponents = [(rating_to_mu(rating), rd_to_phi(rd), score)
                     for rating, rd, score in ((1400, 30, 1), (1550, 100, 0), (1700, 300, 0))]
        result = update_rating(0, rd_to_phi(200), 0.06, opponents, tau=0.5)
        self.assertAlmostEqual(result.mu * 173.7178 + 1500, 1464.05067, places=4)
        self.assertAlmostEqual(phi_to_rd(result.phi), 151.51652, places=4)
        self.assertAlmostEqual(result.sigma, 0.059995984, places=8)

    def test_current_week_outcomes_do_not_change_current_week_features(self):
        engine = GlickoEngine()
        match = _match("m1", datetime(2024, 1, 1), 1)
        before = engine.features(match)
        engine.observe(match)
        self.assertEqual(before, engine.features({**match, "match_time": datetime(2024, 1, 7, 23, 59)}))
        next_week = engine.features({**match, "match_time": datetime(2024, 1, 8)})
        self.assertGreater(next_week["glicko_rating_t1"], 1500)
        self.assertLess(next_week["glicko_rating_t2"], 1500)

    def test_both_sides_use_frozen_opponent_ratings(self):
        engine = GlickoEngine()
        match = _match("m1", datetime(2024, 1, 1), 1)
        engine.observe(match)
        result = engine.features({**match, "match_time": datetime(2024, 1, 8)})
        expected = update_rating(0, rd_to_phi(350), 0.06, [(0, rd_to_phi(350), 1)])
        self.assertAlmostEqual(result["glicko_rating_t1"] - 1500, 1500 - result["glicko_rating_t2"])
        self.assertAlmostEqual(result["glicko_rating_t1"], 1500 + expected.mu * 173.7178)

    def test_week_aggregates_all_observed_results(self):
        engine = GlickoEngine()
        match = _match("m1", datetime(2024, 1, 1), 1)
        for day, winner in ((1, 1), (2, 1), (7, 2)):
            engine.observe({**match, "match_time": datetime(2024, 1, day), "winner": winner})
        expected = update_rating(0, rd_to_phi(350), 0.06,
                                 [(0, rd_to_phi(350), score) for score in (1, 1, 0)])
        result = engine.features({**match, "match_time": datetime(2024, 1, 8)})
        self.assertAlmostEqual(result["glicko_rating_t1"], 1500 + expected.mu * 173.7178)

    def test_inactivity_grows_uncertainty_not_rating_mean(self):
        engine = GlickoEngine()
        match = _match("m1", datetime(2024, 1, 1), 1)
        engine.observe(match)
        immediate = engine.features({**match, "match_time": datetime(2024, 1, 8)})
        later = engine.features({**match, "match_time": datetime(2024, 1, 8) + timedelta(weeks=10)})
        self.assertEqual(immediate["glicko_rating_t1"], later["glicko_rating_t1"])
        prior = update_rating(0, rd_to_phi(350), 0.06, [(0, rd_to_phi(350), 1)])
        for _ in range(10):
            prior = update_rating(*prior, [])
        self.assertAlmostEqual(later["glicko_rd_t1"], phi_to_rd(prior.phi))

    def test_forecasts_are_read_only_and_independent_of_horizon_order(self):
        engine = GlickoEngine()
        match = _match("m1", datetime(2024, 1, 1), 1)
        engine.observe(match)
        before = pickle.dumps(engine.get_state())
        future = {**match, "match_time": datetime(2024, 1, 15)}
        expected = engine.features(future)
        engine.features({**match, "match_time": datetime(2030, 1, 1)})
        engine.features({**match, "team1_ids": ["new"], "match_time": datetime(2024, 1, 8)})
        self.assertEqual(engine.features(future), expected)
        self.assertEqual(pickle.dumps(engine.get_state()), before)

    def test_exact_partnerships_are_canonical_and_independent_of_discipline(self):
        engine = GlickoEngine()
        match = {**_match("m1", datetime(2024, 1, 1), 1), "discipline": "MD",
                 "team1_ids": ["a", "b"], "team2_ids": ["c", "d"]}
        engine.observe(match)
        future = {**match, "match_time": datetime(2024, 1, 8)}
        expected = engine.features(future)
        self.assertEqual(expected, engine.features({**future, "team1_ids": ["b", "a"]}))
        for change in ({"team1_ids": ["a", "c"]}, {"discipline": "XD"}):
            self.assertEqual(engine.features({**future, **change})["glicko_rating_t1"], 1500)

    def test_side_swap_negates_all_difference_features(self):
        engine = GlickoEngine()
        match = _match("m1", datetime(2024, 1, 1), 1)
        engine.observe(match)
        future = {**match, "match_time": datetime(2024, 1, 8)}
        left = engine.features(future)
        right = engine.features({**future, "team1_ids": match["team2_ids"], "team2_ids": match["team1_ids"]})
        for quantity in ("rating", "rd", "volatility"):
            self.assertEqual(left[f"glicko_{quantity}_diff"], -right[f"glicko_{quantity}_diff"])

    def test_out_of_order_observations_are_rejected_without_changes(self):
        engine = GlickoEngine()
        engine.observe(_match("m1", datetime(2024, 1, 8), 1))
        before = pickle.dumps(engine.get_state())
        with self.assertRaisesRegex(ValueError, "chronological"):
            engine.observe(_match("past", datetime(2024, 1, 1), 1))
        self.assertEqual(pickle.dumps(engine.get_state()), before)

    def test_partial_week_survives_save_load_and_continued_processing(self):
        builder = _glicko_builder()
        builder.process(_match("m1", datetime(2024, 1, 1), 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature_state.pkl"
            builder.save(path)
            restored = FeatureBuilder.load(path)
        self.assertEqual(restored.get_state(), builder.get_state())
        for day in (2, 8, 22):
            match = _match(f"m{day}", datetime(2024, 1, day), 2)
            self.assertEqual(restored.process(match), builder.process(match))
            self.assertEqual(restored.get_state(), builder.get_state())
        self.assertEqual(builder.get_state()["version"], 3)
        self.assertEqual(builder.feature_columns, GLICKO_MODEL_FEATURE_COLS)
        self.assertEqual(len(GLICKO_MODEL_FEATURE_COLS), 47)
        self.assertTrue(all(math.isfinite(value) for value in builder.glicko.features(match).values()))

    def test_complete_legacy_state_retains_original_features(self):
        builder = _glicko_builder()
        builder.process(_match("m1", datetime(2024, 1, 1), 1))
        state = builder.get_state()
        state.pop("glicko")
        state["version"] = 1
        state["elo"].pop("score_margin_weight")
        state["elo"].pop("score_margin_reference")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pkl"
            path.write_bytes(pickle.dumps(state))
            legacy = FeatureBuilder.load(path)
        self.assertIsNone(legacy.glicko)
        self.assertEqual(legacy.feature_columns, LEGACY_FEATURE_COLS)
        self.assertEqual(legacy.get_state(), state)
        row = legacy.preview(_match("future", datetime(2024, 1, 8), 1))
        self.assertFalse(set(GLICKO_FEATURE_COLS).intersection(row))


if __name__ == "__main__":
    unittest.main()
