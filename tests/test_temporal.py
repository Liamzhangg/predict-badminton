from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from predict_badminton.config import FEATURE_COLS
from predict_badminton.data_parser import load_all_matches
from predict_badminton.features import FeatureBuilder
from predict_badminton.inference import InferenceBuilder
from predict_badminton.model import (
    _chronological_validation_split, _split_folds, chronological_holdout_split, train_final_model, walk_forward_cv,
)
from predict_badminton.temporal import match_timing
from tests.test_data_parser import _raw_match
from tests.test_feature_state import _match
from tests.test_model_splitting import _feature_frame
from tests.test_probabilities import BiasedModel


def historical(mid, when, winner=1, release=None):
    return {**_match(mid, when, winner), "prediction_time": when,
            "available_at": release or when+timedelta(days=1), "availability_source": "test"}


class TemporalTests(unittest.TestCase):
    def test_utc_results_wait_until_after_the_recorded_end_day(self):
        start = datetime(2026, 1, 1, 23, 30)
        timing = match_timing(start, utc=True, duration=60, coarse=False)
        self.assertEqual(timing["prediction_time"], start)
        self.assertEqual(timing["available_at"], datetime(2026, 1, 3))

    def test_unverified_local_dates_use_earliest_prediction_and_late_release(self):
        start = datetime(2026, 1, 1, 20)
        timing = match_timing(start, utc=False, duration=60, coarse=False)
        self.assertEqual(timing["prediction_time"], datetime(2025, 12, 31, 10))
        self.assertEqual(timing["available_at"], datetime(2026, 1, 3))

    def test_unknown_or_invalid_duration_never_gets_invented(self):
        start = datetime(2026, 1, 1, 10)
        for duration in (None, True, 0, -1, 301, float("nan"), "30"):
            self.assertEqual(match_timing(start, utc=True, duration=duration, coarse=False)["available_at"], datetime(2026, 1, 3))
        self.assertEqual(match_timing(start, utc=True, duration=30, coarse=True)["prediction_time"], datetime(2026, 1, 1))

    def test_equal_time_future_round_result_cannot_change_earlier_round_features(self):
        stamp = datetime(2026, 1, 1)
        future = {**historical("1", stamp), "round_name": "QF"}
        earlier = {**historical("2", stamp), "round_name": "R16"}
        first = FeatureBuilder().build_dataframe([future, earlier])
        second = FeatureBuilder().build_dataframe([{**future, "winner": 2, "label": 0}, earlier])
        pd.testing.assert_frame_equal(first[FEATURE_COLS], second[FEATURE_COLS])
        self.assertEqual(first["elo_matches_t1_disc"].tolist(), [0, 0])

    def test_result_release_is_strict_and_pending_results_do_not_leak(self):
        first = historical("1", datetime(2026, 1, 1, 10), release=datetime(2026, 1, 2))
        at_release = historical("2", datetime(2026, 1, 2))
        after_release = historical("3", datetime(2026, 1, 2, 0, 1))
        frame = FeatureBuilder().build_dataframe([first, at_release, after_release])
        self.assertEqual(frame["h2h_total"].tolist(), [0, 0, 1])
        self.assertEqual(frame["form5_wr_t1"].tolist(), [0.5, 0.5, 1])

    def test_tied_input_order_does_not_change_features_or_final_history(self):
        stamp = datetime(2026, 1, 1, 10)
        matches = [historical("2", stamp, 2), historical("1", stamp),
                   historical("3", stamp+timedelta(days=2))]
        a, b = FeatureBuilder(), FeatureBuilder()
        pd.testing.assert_frame_equal(a.build_dataframe(matches), b.build_dataframe(list(reversed(matches))))
        self.assertEqual(a.get_state(), b.get_state())

    def test_released_batch_updates_follow_recorded_play_order_not_match_id(self):
        release = datetime(2026, 1, 2)
        first = historical("z-first", datetime(2026, 1, 1, 10), 1, release)
        second = historical("a-second", datetime(2026, 1, 1, 12), 2, release)
        later = historical("later", datetime(2026, 1, 3))
        history = FeatureBuilder()
        history.build_dataframe([second, later, first])
        self.assertEqual(history._player_hist["p1"][0]["won"], True)
        self.assertEqual(history._player_hist["p1"][1]["won"], False)

    def test_own_and_future_outcomes_do_not_change_prior_feature_rows(self):
        matches = [historical(str(i), datetime(2026, 1, i+1, 10)) for i in range(5)]
        baseline = FeatureBuilder().build_dataframe(matches)
        changed = [dict(m) for m in matches]
        changed[2].update(winner=2, label=0, score_margin=0.4)
        altered = FeatureBuilder().build_dataframe(changed)
        pd.testing.assert_frame_equal(baseline.loc[:2, FEATURE_COLS], altered.loc[:2, FEATURE_COLS])
        self.assertNotEqual(baseline.loc[4, "h2h_wins_t1"], altered.loc[4, "h2h_wins_t1"])

    def test_replay_rejects_missing_timing_and_unavailable_training_labels(self):
        with self.assertRaisesRegex(ValueError, "reload raw data"):
            FeatureBuilder().build_dataframe([_match("old", datetime(2026, 1, 1), 1)])
        match = historical("one", datetime(2026, 1, 1))
        history = FeatureBuilder()
        before = history.get_state()
        with self.assertRaisesRegex(ValueError, "not yet eligible"):
            history.build_dataframe([match], as_of=datetime(2026, 1, 2))
        self.assertEqual(history.get_state(), before)

    def test_history_cutoff_is_saved_and_blocks_pre_availability_fixtures(self):
        builder = FeatureBuilder()
        builder.build_dataframe([historical("one", datetime(2026, 1, 1, 10))])
        cutoff = builder.history_cutoff
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pkl"
            builder.save(path)
            restored = FeatureBuilder.load(path)
            self.assertEqual(restored.history_cutoff, cutoff)
            for stamp in (datetime(2026, 1, 1, 12), cutoff):
                with self.assertRaisesRegex(ValueError, "cutoff"):
                    InferenceBuilder(restored).build_features(pd.DataFrame([{"match_time": stamp}]))
            state = pickle.loads(path.read_bytes())
            state.pop("history_cutoff")
            path.write_bytes(pickle.dumps(state))
            with self.assertRaises(ValueError):
                FeatureBuilder.load(path)

    def test_model_cannot_be_newer_than_its_history(self):
        history = FeatureBuilder()
        history.build_dataframe([historical("one", datetime(2026, 1, 1))])
        model = SimpleNamespace(feature_name_=FEATURE_COLS, feature_policy_=history.model_policy)
        with self.assertRaisesRegex(ValueError, "training cutoff"):
            history.validate_model(model)
        model.trained_through_ = history.history_cutoff+timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "training cutoff"):
            history.validate_model(model)
        model.trained_through_ = history.history_cutoff
        history.validate_model(model)

    def test_raw_duplicates_with_different_ids_update_history_once(self):
        raw = {**_raw_match("a"), "duration": 30}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            original = json.dumps([[raw, {**raw, "id": "b"}]])
            path.write_text(original)
            matches, drops, _ = load_all_matches(Path(tmp))
            self.assertEqual(path.read_text(), original)
        builder = FeatureBuilder()
        builder.build_dataframe(matches)
        self.assertEqual(builder.elo.team_disc_count(["p1"], "MS"), 1)
        self.assertEqual(drops["duplicate_fixture"], 1)

    def test_conflicting_outcomes_are_quarantined_in_either_source_order(self):
        raw = _raw_match("a")
        for second_id in ("a", "b"):
            for records in ([raw, {**raw, "id": second_id, "winner": 2}],
                            [{**raw, "id": second_id, "winner": 2}, raw]):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "raw.json"
                    path.write_text(json.dumps([records]))
                    matches, drops, _ = load_all_matches(Path(tmp))
                self.assertEqual(matches, [])
                self.assertEqual(drops["conflicting_result"], 2)

    def test_repeated_team_game_slots_are_not_duplicates(self):
        for first_id, second_id in (("a", "b"), (None, None)):
            raw = {**_raw_match(first_id), "isTeamMatch": True, "matchOrder": 1, "matchTypeNo": 1}
            other = {**raw, "id": second_id, "winner": 2, "matchOrder": 2, "matchTypeNo": 2}
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "raw.json").write_text(json.dumps([[raw, other]]))
                matches, drops, _ = load_all_matches(Path(tmp))
            self.assertEqual(len(matches), 2)
            self.assertEqual(drops, {})

    def test_holdout_purges_labels_unavailable_at_year_boundary(self):
        frame = pd.DataFrame({"match_time": pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-10"]),
                              "available_at": pd.to_datetime(["2025-12-31", "2026-01-01", "2026-01-11"])})
        train, test, _ = chronological_holdout_split(frame)
        self.assertEqual(train.index.tolist(), [0])
        self.assertEqual(test.index.tolist(), [2])

    def test_early_stopping_groups_ties_and_purges_unavailable_labels(self):
        frame = _feature_frame()
        frame["prediction_time"] = frame["match_time"]
        frame["available_at"] = frame["match_time"]+pd.Timedelta(days=3)
        frame.loc[50:52, "prediction_time"] = frame.loc[50, "prediction_time"]
        fit, stop = _chronological_validation_split(frame)
        self.assertLess(fit["prediction_time"].max(), stop["prediction_time"].min())
        self.assertLess(fit["available_at"].max(), stop["prediction_time"].min())
        frame["prediction_time"] = frame.loc[0, "prediction_time"]
        with self.assertRaises(ValueError):
            _chronological_validation_split(frame)

    def test_duplicate_fixture_cannot_cross_holdout(self):
        frame = pd.DataFrame({"match_time": pd.to_datetime(["2025-01-01", "2026-01-01"]),
                              "fixture_id": ["same", "same"]})
        with self.assertRaisesRegex(ValueError, "Duplicate fixture_id"):
            chronological_holdout_split(frame)

    def test_final_fit_rejects_overlapping_training_and_test_data(self):
        frame = _feature_frame()
        with patch("predict_badminton.model._train_lgbm_for_evaluation") as fit:
            with self.assertRaisesRegex(ValueError, "availability-isolated"):
                train_final_model(frame, frame, oof_preds=frame["label"].to_numpy(), oof_labels=frame["label"].to_numpy())
        fit.assert_not_called()

    def test_walk_forward_fold_only_fits_results_available_before_validation(self):
        frame = _feature_frame(500)
        frame["available_at"] = frame["match_time"]+pd.Timedelta(days=5)
        folds = _split_folds(frame, [(2021, 2021)])
        self.assertEqual(len(folds), 1)
        fit, validation = folds[0]
        self.assertLess(fit["available_at"].max(), pd.Timestamp("2021-01-01"))
        self.assertGreaterEqual(validation["match_time"].min(), pd.Timestamp("2021-01-01"))

    def test_nested_tuning_only_receives_each_folds_training_rows(self):
        frame = _feature_frame(120)
        train, validation = frame.iloc[:60], frame.iloc[60:]
        with patch("predict_badminton.model._split_folds", return_value=[(train, validation)]), \
                patch("predict_badminton.model.tune_lgbm", return_value={"max_depth": 2}) as tune, \
                patch("predict_badminton.model._train_lgbm_for_evaluation", return_value=(BiasedModel(), 7)) as fit:
            walk_forward_cv(frame, n_trials=2)
        pd.testing.assert_frame_equal(tune.call_args.args[0], train)
        self.assertEqual(tune.call_args.kwargs, {"n_trials": 2})
        self.assertEqual(fit.call_args.args[1], {"max_depth": 2})

    def test_nested_tuning_uses_defaults_when_no_inner_fold_exists(self):
        frame = _feature_frame(120)
        train, validation = frame.iloc[:60], frame.iloc[60:]
        with patch("predict_badminton.model._split_folds", side_effect=[[(train, validation)], []]), \
                patch("predict_badminton.model.tune_lgbm") as tune, \
                patch("predict_badminton.model._train_lgbm_for_evaluation", return_value=(BiasedModel(), 7)):
            walk_forward_cv(frame, n_trials=2)
        tune.assert_not_called()


if __name__ == "__main__":
    unittest.main()
