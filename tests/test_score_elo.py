from __future__ import annotations

import math
import pickle
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from predict_badminton.config import FEATURE_COLS, TRUESKILL_FEATURE_COLS
from predict_badminton.features import FeatureBuilder
from predict_badminton.glicko import GlickoEngine
from predict_badminton.match_rules import point_margin, score_multiplier, standard_match_exclusion
from tests.test_feature_state import _match


def _raw(games, winner=1, status="Normal"):
    return {"winner": winner, "scoreStatusValue": status,
            "score": [{"home": a, "away": b} for a, b in games]}


class ScoreEloTests(unittest.TestCase):
    def test_training_rejects_an_entirely_unsupported_dataset(self):
        import train

        match = {**_match("one", datetime(2024, 1, 1), 1), "discipline": "OTHER"}
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(smoke_test=False, artifacts_dir=Path(tmp), data_dir=Path(tmp),
                                   max_files=None, include_para=False)
            with patch("train.parse_args", return_value=args), \
                    patch("predict_badminton.data_parser.load_all_matches", return_value=([match], {}, {})):
                self.assertEqual(train.main(), 1)

    def test_complete_scores_including_shutouts_and_deuce(self):
        for games in ([(21, 19), (21, 19)], [(21, 0), (21, 0)],
                      [(30, 29), (29, 30), (30, 28)], [(21, 10), (21, 10), (0, 0)]):
            self.assertIsNotNone(point_margin(_raw(games)))

    def test_missing_invalid_retired_and_nonstandard_scores_fall_back(self):
        for games, winner, status in (([(21, 10), (21, 10)], 1, "Retired"),
                                     ([(15, 10), (15, 10)], 1, "Normal"),
                                     ([(21, 20), (21, 10)], 1, "Normal"),
                                     ([(21, 10), (19, 12)], 1, "Normal"),
                                     ([(21, 10), (21, 10)], 2, "Normal"),
                                     ([(21, 10), (21, 10), (10, 21)], 1, "Normal"),
                                     ([(21, 10), (0, 0), (21, 10)], 1, "Normal"),
                                     ([(True, 0), (21, 0)], 1, "Normal"),
                                     ([(float("nan"), 0), (21, 0)], 1, "Normal")):
            with self.subTest(games=games, winner=winner, status=status):
                self.assertIsNone(point_margin(_raw(games, winner, status)))
        for raw in ({}, {"score": None}, {"score": [None]}):
            self.assertIsNone(point_margin(raw))

    def test_multiplier_examples_bounds_and_side_swap(self):
        close = point_margin(_raw([(21, 19), (21, 19)]))
        clear = point_margin(_raw([(21, 10), (21, 10)]))
        self.assertAlmostEqual(score_multiplier(close, 0.5, 0.2), 0.625)
        self.assertAlmostEqual(score_multiplier(clear, 0.5, 0.2), 1.3870967741935485)
        swapped = point_margin(_raw([(19, 21), (19, 21)], 2))
        self.assertEqual(close, swapped)
        negative = point_margin(_raw([(21, 19), (1, 21), (21, 19)]))
        self.assertLess(negative, 0)
        self.assertEqual(score_multiplier(negative, 0.5, 0.2), 0.5)
        self.assertEqual(score_multiplier(1, 0.5, 0.2), 1.5)
        for margin in (None, math.nan, math.inf, -2, 2, True):
            self.assertEqual(score_multiplier(margin, 0.5, 0.2), 1)

    def test_score_is_only_used_after_current_features(self):
        close, clear = FeatureBuilder(), FeatureBuilder()
        match = _match("one", datetime(2024, 1, 1), 1)
        a = close.process({**match, "score_margin": 0.05})
        b = clear.process({**match, "score_margin": 0.35})
        pd.testing.assert_series_equal(pd.Series(a), pd.Series(b))
        future = _match("two", datetime(2024, 1, 2), 1)
        self.assertLess(close.preview(future)["elo_diff_disc"], clear.preview(future)["elo_diff_disc"])

    def test_score_updates_are_zero_sum_for_players_and_raw_pairs(self):
        for doubles in (False, True):
            builder = FeatureBuilder()
            match = {**_match("one", datetime(2024, 1, 1), 1), "score_margin": 0.4}
            if doubles:
                match.update(discipline="MD", team1_ids=["a", "b"], team2_ids=["c", "d"])
            builder.process(match)
            ids = match["team1_ids"] + match["team2_ids"]
            self.assertAlmostEqual(sum(builder.elo._disc[pid][match["discipline"]] for pid in ids), 1500*len(ids))
            self.assertAlmostEqual(sum(builder.elo._global[pid] for pid in ids), 1500*len(ids))
            self.assertAlmostEqual(sum(v[match["discipline"]] for v in builder.elo._pair.values()), 0)

    def test_only_elo_features_change_with_score_policy(self):
        current, control = FeatureBuilder(), FeatureBuilder(score_margin_weight=0)
        for day, winner in ((1, 1), (2, 2), (9, 1)):
            match = {**_match(str(day), datetime(2024, 1, day), winner), "score_margin": 0.35}
            a, b = current.process(match), control.process(match)
            for key in a:
                if key.startswith(("elo_", "pair_")):
                    continue
                if isinstance(a[key], float) and math.isnan(a[key]):
                    self.assertTrue(math.isnan(b[key]))
                else:
                    self.assertEqual(a[key], b[key], key)
        self.assertEqual(current.trueskill.features(match), control.trueskill.features(match))
        self.assertEqual(len(TRUESKILL_FEATURE_COLS), 6)

    def test_missing_score_matches_original_update(self):
        current, control = FeatureBuilder(), FeatureBuilder(score_margin_weight=0)
        match = _match("one", datetime(2024, 1, 1), 1)
        current.process(match)
        control.process(match)
        self.assertEqual(current.elo.team_elo(["p1"], "MS"), control.elo.team_elo(["p1"], "MS"))

    def test_current_policy_survives_persistence_and_continued_updates(self):
        builder = FeatureBuilder()
        builder.process({**_match("one", datetime(2024, 1, 1), 1), "score_margin": 0.35})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.pkl"
            builder.save(path)
            restored = FeatureBuilder.load(path)
        self.assertEqual(restored.get_state(), builder.get_state())
        match = {**_match("two", datetime(2024, 1, 9), 2), "score_margin": 0.05}
        self.assertEqual(restored.process(match), builder.process(match))
        self.assertEqual(restored.get_state(), builder.get_state())
        self.assertEqual(restored.get_state()["version"], 4)

    def test_legacy_states_keep_original_elo_updates(self):
        for version in (1, 2):
            state = FeatureBuilder(score_margin_weight=0).get_state()
            state["version"] = version
            state.pop("trueskill")
            state["elo"].pop("score_margin_weight")
            state["elo"].pop("score_margin_reference")
            if version == 2:
                state["glicko"] = GlickoEngine().get_state()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "history.pkl"
                path.write_bytes(pickle.dumps(state))
                restored = FeatureBuilder.load(path)
            self.assertEqual(restored.get_state(), state)
            self.assertEqual(restored.elo.score_margin_weight, 0)
            match = {**_match("one", datetime(2024, 1, 1), 1), "score_margin": 0.4}
            restored.process(match)
            baseline = FeatureBuilder(score_margin_weight=0)
            baseline.process(match)
            self.assertEqual(restored.elo.team_elo(["p1"], "MS"), baseline.elo.team_elo(["p1"], "MS"))

    def test_model_and_history_policy_must_match_even_with_same_columns(self):
        history = FeatureBuilder()
        model = SimpleNamespace(feature_name_=FEATURE_COLS)
        with self.assertRaisesRegex(ValueError, "rating policy"):
            history.validate_model(model)
        model.feature_policy_ = FeatureBuilder(score_margin_weight=0).model_policy
        with self.assertRaisesRegex(ValueError, "rating policy"):
            history.validate_model(model)
        model.feature_policy_ = history.model_policy
        history.validate_model(model)
        model.feature_policy_ = FeatureBuilder(k_base=16).model_policy
        with self.assertRaisesRegex(ValueError, "rating policy"):
            history.validate_model(model)

    def test_current_missing_policy_cannot_silently_load_as_legacy(self):
        for key in ("score_margin_weight", "score_margin_reference"):
            state = FeatureBuilder().get_state()
            state["elo"].pop(key)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "history.pkl"
                path.write_bytes(pickle.dumps(state))
                with self.assertRaisesRegex(ValueError, "retrain"):
                    FeatureBuilder.load(path)

    def test_standard_scope_excludes_air_other_and_invalid_teams(self):
        match = _match("one", datetime(2024, 1, 1), 1)
        self.assertIsNone(standard_match_exclusion(match))
        cases = [{**match, "discipline": "OTHER"},
                 {**match, "tournament_name": "BWF AirBadminton World Cup"},
                 {**match, "tournament_name": "ANOC WBG Europe Qualification"},
                 {**match, "team2_ids": ["p1"]},
                 {**match, "discipline": "MD"}]
        for invalid in cases:
            self.assertIsNotNone(standard_match_exclusion(invalid))
            with self.assertRaisesRegex(ValueError, "standard"):
                FeatureBuilder().preview(invalid)


if __name__ == "__main__":
    unittest.main()
