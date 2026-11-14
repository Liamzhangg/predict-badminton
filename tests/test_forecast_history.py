from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import observations_before, record_results
from predict_badminton.features import FeatureBuilder
from predict_badminton.forecast_history import refresh_history
from predict_badminton.inference import build_inference_features
from predict_badminton.odds import round_predictions
from predict_badminton.temporal import OBSERVED_AVAILABILITY_POLICY
from predict_badminton.tournament import forecast
from tests.test_observations import completed_match
from tests.test_odds import FixedModel, snapshot
from tests.test_tournament import draw
from tests.test_temporal import historical


class ForecastHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.db = self.root / "observations.sqlite3"
        self.seen = datetime(2026, 11, 11, 16, tzinfo=UTC)
        self.cutoff = self.seen + timedelta(hours=1)
        self.match = {**completed_match(), "matchStatus": "F"}
        self.history = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
        self.save([self.match])
        self.record(self.match, self.seen)

    def save(self, matches):
        (self.raw / "matches.json").write_text(json.dumps([matches]))

    def record(self, match, when):
        record_results([match], source_url="https://example.com/results", response_sha256="source",
                       observed_at=when, path=self.db)

    def refresh(self, **kwargs):
        return refresh_history(self.history, as_of=self.cutoff, raw_dir=self.raw,
                               observations_db=self.db, **kwargs)

    def test_latest_completed_match_updates_ratings_form_and_h2h_without_mutating_inputs(self):
        before = pickle.dumps(self.history.get_state())
        files = {p: p.read_bytes() for p in (self.raw / "matches.json", self.db)}
        history = self.refresh()
        self.assertEqual(history.elo.team_disc_count(["p1"], "MS"), 1)
        self.assertGreater(history.elo.team_elo(["p1"], "MS")[0], 1500)
        self.assertGreater(history.trueskill.ratings[("MS", "p1")].mu, 25)
        self.assertTrue(history._player_hist["p1"][-1]["won"])
        self.assertEqual(next(iter(history._h2h.values())), [1, 0, 1])
        self.assertEqual(history.history_cutoff, self.cutoff.replace(tzinfo=None))
        self.assertEqual(pickle.dumps(self.history.get_state()), before)
        self.assertTrue(all(p.read_bytes() == content for p, content in files.items()))

    def test_exact_cutoff_and_later_receipts_are_excluded(self):
        for cutoff in (self.seen - timedelta(seconds=1), self.seen):
            with self.subTest(cutoff=cutoff):
                history = refresh_history(self.history, as_of=cutoff, raw_dir=self.raw, observations_db=self.db)
                self.assertEqual(history.elo.team_disc_count(["p1"], "MS"), 0)
        self.assertEqual(self.refresh().elo.team_disc_count(["p1"], "MS"), 1)

    def test_receipt_after_forecast_cannot_enter_via_a_future_fixture_start(self):
        self.cutoff = self.seen - timedelta(minutes=1)
        history = self.refresh()
        self.assertEqual(history.elo.team_disc_count(["p1"], "MS"), 0)

    def test_repeated_refresh_does_not_double_count_saved_history(self):
        first = self.refresh()
        self.history = first
        second = self.refresh()
        self.assertEqual(first.get_state(), second.get_state())
        self.assertEqual(second.elo.team_disc_count(["p1"], "MS"), 1)

    def test_retraction_removes_a_previously_included_match(self):
        self.history = self.refresh()
        self.record({"id": "m1", "winner": 0}, self.seen + timedelta(minutes=30))
        history = self.refresh()
        self.assertEqual(history.elo.team_disc_count(["p1"], "MS"), 0)

    def test_unobserved_correction_is_not_treated_as_the_old_result(self):
        self.save([{**self.match, "winner": 2}])
        self.assertEqual(self.refresh().elo.team_disc_count(["p1"], "MS"), 0)

    def test_observed_correction_rebuilds_instead_of_appending(self):
        self.history = self.refresh()
        corrected = {**self.match, "winner": 2}
        self.record(corrected, self.seen + timedelta(minutes=30))
        self.save([corrected])
        history = self.refresh()
        self.assertEqual(history.elo.team_disc_count(["p1"], "MS"), 1)
        self.assertFalse(history._player_hist["p1"][-1]["won"])

    def test_untracked_old_history_uses_conservative_fallback_and_aliases_count_once(self):
        old = {**self.match, "id": "old", "matchTimeUtc": "2020-01-01 10:00:00"}
        self.save([old, self.match, {**self.match, "id": "alias"}])
        history = self.refresh()
        self.assertEqual(history.elo.team_disc_count(["p1"], "MS"), 2)

    def test_live_results_with_a_provisional_winner_are_not_used(self):
        self.save([{**self.match, "matchStatus": "L"}])
        self.assertEqual(self.refresh().elo.team_disc_count(["p1"], "MS"), 0)

    def test_completed_target_cannot_be_scored_as_an_upcoming_match(self):
        with self.assertRaisesRegex(ValueError, "completed target"):
            self.refresh(target_match_ids={"m1"})

    def test_completed_target_alias_is_also_rejected(self):
        self.save([self.match, {**self.match, "id": "alias"}])
        with self.assertRaisesRegex(ValueError, "completed target"):
            self.refresh(target_match_ids={"alias"})

    def test_policy_parameters_are_preserved(self):
        self.history.elo.k_base = 16
        self.history.elo.pair_lambda = 10
        self.history.elo.alpha_player = 0.6
        history = self.refresh()
        self.assertEqual(history.model_policy, self.history.model_policy)

    def test_refresh_rejects_missing_sources_legacy_policy_and_naive_cutoffs(self):
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            refresh_history(self.history, as_of=self.cutoff.replace(tzinfo=None), raw_dir=self.raw, observations_db=self.db)
        with self.assertRaisesRegex(ValueError, "observed"):
            refresh_history(FeatureBuilder(), as_of=self.cutoff, raw_dir=self.raw, observations_db=self.db)
        with self.assertRaisesRegex(ValueError, "observations"):
            refresh_history(self.history, as_of=self.cutoff, raw_dir=self.raw, observations_db=self.root / "missing.sqlite3")
        with self.assertRaisesRegex(ValueError, "raw"):
            refresh_history(self.history, as_of=self.cutoff, raw_dir=self.root / "missing", observations_db=self.db)

    def test_corrupt_receipts_do_not_silently_fall_back_to_stale_history(self):
        self.db.write_bytes(b"not a database")
        with self.assertRaisesRegex(ValueError, "result observations"):
            self.refresh()

    def test_fast_result_replay_matches_the_training_replay_state(self):
        old = {**self.match, "id": "old", "matchTimeUtc": "2020-01-01 10:00:00"}
        self.save([self.match, old])
        matches, _, _ = load_all_matches(self.raw, observed_results=observations_before(self.cutoff, self.db))
        expected = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
        expected.build_dataframe(matches, as_of=self.cutoff.replace(tzinfo=None))
        actual = self.refresh()
        expected_state, actual_state = expected.get_state(), actual.get_state()
        expected_state.pop("player_names")
        actual_state.pop("player_names")
        self.assertEqual(expected_state, actual_state)

    def test_result_only_replay_matches_training_for_tied_and_late_results(self):
        release = datetime(2026, 1, 5)
        matches = [historical("a", datetime(2026, 1, 1, 10), release=release),
                   historical("b", datetime(2026, 1, 2, 10), winner=2),
                   historical("c", datetime(2026, 1, 4, 10), release=release),
                   historical("d", datetime(2026, 1, 6, 10))]
        matches[0]["availability_source"] = "observed_completed_result"
        trained = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
        fresh = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
        cutoff = datetime(2026, 1, 8)
        trained.build_dataframe(matches, as_of=cutoff)
        fresh.replay_results(list(reversed(matches)), as_of=cutoff)
        self.assertEqual(trained.get_state(), fresh.get_state())
        with self.assertRaisesRegex(ValueError, "empty"):
            fresh.replay_results(matches, as_of=cutoff)

    def test_default_inference_refreshes_and_frozen_opt_out_preserves_old_features(self):
        state, fixtures = self.root / "state.pkl", self.root / "fixtures.csv"
        self.history.save(state)
        fixtures.write_text("match_id,match_time,discipline,team1_player_ids,team2_player_ids\n"
                            "future,2026-11-12 10:00:00,MS,p1,p2\n")
        model = FixedModel(self.history)
        frozen, _ = build_inference_features(fixtures, state, model=model, refresh=False)
        live, _ = build_inference_features(fixtures, state, model=model, as_of=self.cutoff,
                                          raw_dir=self.raw, observations_db=self.db)
        self.assertEqual(frozen["form5_cnt_t1"].iloc[0], 0)
        self.assertEqual(live["form5_cnt_t1"].iloc[0], 1)
        self.assertEqual(live["h2h_total"].iloc[0], 1)
        self.assertTrue(live.attrs["history_refreshed"])
        self.assertFalse(frozen.attrs["history_refreshed"])
        self.assertEqual(FeatureBuilder.load(state).get_state(), self.history.get_state())

    def test_default_round_and_draw_forecasts_share_the_refresh(self):
        model = FixedModel(self.history)
        bundle = (model, None, None, self.history)
        refreshed = self.refresh()
        for module, call in (("predict_badminton.odds.refresh_history", lambda: round_predictions(
                snapshot(), bundle, round_name="R32", as_of=self.cutoff)),
                ("predict_badminton.tournament.refresh_history", lambda: forecast(
                    draw(), "MS", bundle, as_of=self.cutoff))):
            with self.subTest(module=module), patch(module, return_value=refreshed) as refresh:
                result = call()
                refresh.assert_called_once()
                if isinstance(result, tuple):
                    self.assertTrue(result[0]["history_refreshed"].all())
                    self.assertEqual(result[1], refreshed.history_cutoff.isoformat())
                else:
                    self.assertTrue(result["history_refreshed"])
                    self.assertEqual(result["cutoff"], refreshed.history_cutoff.isoformat())

    def test_past_fixture_is_rejected_before_history_is_refreshed(self):
        state, fixtures = self.root / "state.pkl", self.root / "fixtures.csv"
        self.history.save(state)
        fixtures.write_text("match_id,match_time,discipline,team1_player_ids,team2_player_ids\n"
                            "future,2026-11-11 10:00:00,MS,p1,p2\n")
        with patch("predict_badminton.inference.refresh_history") as refresh:
            with self.assertRaisesRegex(ValueError, "forecast cutoff"):
                build_inference_features(fixtures, state, as_of=self.cutoff)
            refresh.assert_not_called()

    def test_newer_classifier_cannot_be_used_before_its_training_cutoff(self):
        state, fixtures = self.root / "state.pkl", self.root / "fixtures.csv"
        self.history._history_cutoff = self.cutoff.replace(tzinfo=None) + timedelta(days=1)
        self.history.save(state)
        fixtures.write_text("match_id,match_time,discipline,team1_player_ids,team2_player_ids\n"
                            "future,2026-11-13 10:00:00,MS,p1,p2\n")
        with self.assertRaisesRegex(ValueError, "training cutoff"):
            build_inference_features(fixtures, state, model=FixedModel(self.history), as_of=self.cutoff,
                                     raw_dir=self.raw, observations_db=self.db)


if __name__ == "__main__":
    unittest.main()
