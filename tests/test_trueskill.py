from __future__ import annotations

from datetime import datetime, timedelta
import math
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from trueskill import TrueSkill

from predict_badminton.config import FEATURE_COLS, TRUESKILL_FEATURE_COLS
from predict_badminton.features import FeatureBuilder
from predict_badminton.trueskill import TrueSkillEngine
from tests.test_feature_state import _match


class TrueSkillTests(unittest.TestCase):
    def test_library_matches_published_example(self):
        env = TrueSkill()
        (winner,), (loser,) = env.rate([(env.create_rating(),), (env.create_rating(),)], ranks=[0, 1])
        self.assertAlmostEqual(winner.mu, 29.396, places=3)
        self.assertAlmostEqual(loser.mu, 20.604, places=3)
        self.assertAlmostEqual(winner.sigma, 7.171, places=3)

    def test_no_draw_updates_match_analytic_singles_and_doubles(self):
        for count in (1, 2):
            engine = TrueSkillEngine()
            match = _match("one", datetime(2024, 1, 1), 1)
            if count == 2:
                match.update(discipline="MD", team1_ids=["a", "b"], team2_ids=["c", "d"])
            p = engine.parameters
            variance = p["sigma"]**2 + p["tau"]**2
            c = math.sqrt(2*count*(variance+p["beta"]**2))
            expected_mu = p["mu"] + variance/c*math.sqrt(2/math.pi)
            expected_sigma = math.sqrt(variance*(1-variance/c**2*2/math.pi))
            engine.observe(match)
            for key in engine._keys(match)[0]:
                self.assertAlmostEqual(engine.ratings[key].mu, expected_mu, delta=2e-6)
                self.assertAlmostEqual(engine.ratings[key].sigma, expected_sigma, delta=2e-6)

    def test_new_partners_retain_individual_skill_but_other_events_do_not(self):
        engine = TrueSkillEngine()
        match = {**_match("one", datetime(2024, 1, 1), 1), "discipline": "MD",
                 "team1_ids": ["a", "b"], "team2_ids": ["c", "d"]}
        engine.observe(match)
        future = {**match, "match_time": datetime(2024, 1, 2),
                  "team1_ids": ["a", "new"], "team2_ids": ["other", "another"]}
        self.assertGreater(engine.features(future)["trueskill_mu_diff"], 0)
        self.assertEqual(engine.features({**future, "discipline": "XD"})["trueskill_mu_diff"], 0)
        self.assertEqual(engine.features(future), engine.features({**future, "team1_ids": ["new", "a"]}))

    def test_current_outcome_only_changes_later_features(self):
        first, second = FeatureBuilder(), FeatureBuilder()
        match = _match("one", datetime(2024, 1, 1), 1)
        one = first.process(match)
        two = second.process({**match, "winner": 2, "label": 0})
        for column in TRUESKILL_FEATURE_COLS:
            self.assertEqual(one[column], two[column])
        future = _match("two", datetime(2024, 1, 2), 1)
        self.assertGreater(first.preview(future)["trueskill_mu_diff"], second.preview(future)["trueskill_mu_diff"])

    def test_preview_has_no_calendar_inflation_and_does_not_mutate_history(self):
        engine = TrueSkillEngine()
        match = _match("one", datetime(2024, 1, 1), 1)
        engine.observe(match)
        before = pickle.dumps(engine.get_state())
        tomorrow = {**match, "match_time": match["match_time"] + timedelta(days=1)}
        later = {**match, "match_time": match["match_time"] + timedelta(days=180)}
        self.assertEqual(engine.features(tomorrow), engine.features(later))
        engine.features({**later, "team1_ids": ["new"]})
        self.assertEqual(pickle.dumps(engine.get_state()), before)

    def test_out_of_order_result_rejected_without_mutation(self):
        engine = TrueSkillEngine()
        engine.observe(_match("one", datetime(2024, 1, 2), 1))
        before = pickle.dumps(engine.get_state())
        with self.assertRaisesRegex(ValueError, "chronological"):
            engine.observe(_match("past", datetime(2024, 1, 1), 1))
        self.assertEqual(pickle.dumps(engine.get_state()), before)

    def test_state_restores_saved_parameters_and_continued_updates(self):
        engine = TrueSkillEngine(beta=3, tau=0.1)
        engine.observe(_match("one", datetime(2024, 1, 1), 1))
        restored = TrueSkillEngine.from_state(engine.get_state())
        self.assertEqual(engine.get_state(), restored.get_state())
        match = _match("two", datetime(2024, 1, 2), 2)
        engine.observe(match)
        restored.observe(match)
        self.assertEqual(engine.get_state(), restored.get_state())

    def test_rest_days_remain_even_without_calendar_rating_inflation(self):
        builder = FeatureBuilder()
        builder.process(_match("one", datetime(2024, 1, 1), 1))
        near = builder.preview(_match("near", datetime(2024, 1, 2), 1))
        far = builder.preview(_match("far", datetime(2024, 7, 1), 1))
        self.assertEqual(near["trueskill_sigma_t1"], far["trueskill_sigma_t1"])
        self.assertGreater(far["rest_days_t1"], near["rest_days_t1"])

    def test_schema_and_policy_reject_mismatched_trueskill_history(self):
        builder = FeatureBuilder()
        self.assertEqual(len(FEATURE_COLS), 44)
        model = SimpleNamespace(feature_name_=FEATURE_COLS, feature_policy_=builder.model_policy)
        builder.validate_model(model)
        other = FeatureBuilder()
        other.trueskill = TrueSkillEngine(tau=0.2)
        with self.assertRaisesRegex(ValueError, "rating policy"):
            other.validate_model(model)

    def test_incomplete_and_downgraded_states_are_rejected(self):
        states = []
        state = FeatureBuilder().get_state()
        state.pop("trueskill")
        states.append(state)
        for version in (1, 2, 3):
            state = FeatureBuilder(score_margin_weight=0).get_state()
            state["version"] = version
            states.append(state)
        state = FeatureBuilder().get_state()
        state["trueskill"]["parameters"].pop("tau")
        states.append(state)
        for state in states:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.pkl"
                path.write_bytes(pickle.dumps(state))
                with self.assertRaises(ValueError):
                    FeatureBuilder.load(path)


if __name__ == "__main__":
    unittest.main()
