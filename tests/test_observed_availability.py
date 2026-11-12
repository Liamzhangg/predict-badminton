from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from predict_badminton.config import FEATURE_COLS, REPO_ROOT
from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import observations_before, record_results
from predict_badminton.features import FeatureBuilder
from predict_badminton.inference import InferenceBuilder
from predict_badminton.temporal import AVAILABILITY_POLICY, LEGACY_OBSERVED_AVAILABILITY_POLICY, OBSERVED_AVAILABILITY_POLICY
from tests.test_observations import completed_match
import train


class ObservedAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.db = self.root / "observations.sqlite3"
        self.when = datetime(2026, 11, 11, 16, tzinfo=UTC)

    def record(self, match, when=None):
        record_results([match], source_url="https://example.com/results", response_sha256="response",
                       observed_at=when or self.when, path=self.db)

    def load(self, records, cutoff=None):
        path = self.raw / "tournament.json"
        original = json.dumps([records])
        path.write_text(original)
        receipts = observations_before(cutoff or self.when+timedelta(seconds=1), self.db)
        matches, drops, _ = load_all_matches(self.raw, observed_results=receipts)
        self.assertEqual(path.read_text(), original)
        return matches, drops

    def test_receipt_is_not_eligible_at_its_exact_timestamp(self):
        self.record(completed_match())
        self.assertIsNone(observations_before(self.when, self.db)["m1"])
        receipt = observations_before(self.when+timedelta(microseconds=1), self.db)["m1"]
        self.assertEqual(receipt["available_at"], self.when.replace(tzinfo=None))
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            observations_before(self.when.replace(tzinfo=None), self.db)

    def test_corrections_and_reversions_start_new_availability_episodes(self):
        original = completed_match()
        for hour, match in enumerate((original, original, {**original, "winner": 2}, original, original)):
            self.record(match, self.when+timedelta(hours=hour))
        first = observations_before(self.when+timedelta(hours=2), self.db)["m1"]
        corrected = observations_before(self.when+timedelta(hours=3), self.db)["m1"]
        reverted = observations_before(self.when+timedelta(hours=5), self.db)["m1"]
        self.assertEqual(first["available_at"], self.when.replace(tzinfo=None))
        self.assertEqual(corrected["available_at"], self.when.replace(tzinfo=None)+timedelta(hours=2))
        self.assertEqual(reverted["available_at"], self.when.replace(tzinfo=None)+timedelta(hours=3))
        self.assertEqual(first["result_sha256"], reverted["result_sha256"])
        self.assertNotEqual(first["result_sha256"], corrected["result_sha256"])

    def test_conflicting_receipts_at_same_time_require_a_later_clean_receipt(self):
        original = completed_match()
        self.record(original)
        self.record({**original, "winner": 2})
        self.assertIsNone(observations_before(self.when+timedelta(seconds=1), self.db)["m1"])
        later = self.when+timedelta(hours=1)
        self.record(original, later)
        self.assertEqual(observations_before(later+timedelta(seconds=1), self.db)["m1"]["available_at"], later.replace(tzinfo=None))

    def test_invalidation_obeys_cutoff_and_restoration_starts_a_new_episode(self):
        original = completed_match()
        self.record(original)
        reset_at = self.when+timedelta(hours=1)
        self.record({"id": "m1", "winner": 0}, reset_at)
        before = self.db.read_bytes()
        self.assertIsNotNone(observations_before(reset_at, self.db)["m1"])
        self.assertIsNone(observations_before(reset_at+timedelta(microseconds=1), self.db)["m1"])
        matches, _ = self.load([original], cutoff=reset_at+timedelta(seconds=1))
        self.assertEqual(matches, [])
        self.assertEqual(self.db.read_bytes(), before)
        restored_at = self.when+timedelta(hours=2)
        self.record(original, restored_at)
        receipt = observations_before(restored_at+timedelta(seconds=1), self.db)["m1"]
        self.assertEqual(receipt["available_at"], restored_at.replace(tzinfo=None))

    def test_completed_and_invalid_receipts_at_same_time_are_unavailable(self):
        original = completed_match()
        record_results([original, {"id": "m1", "winner": 0}], source_url="https://example.com/results",
                       response_sha256="conflicting", observed_at=self.when, path=self.db)
        self.assertIsNone(observations_before(self.when+timedelta(seconds=1), self.db)["m1"])

    def test_an_untracked_fixture_alias_cannot_restore_a_retracted_or_future_result(self):
        original = completed_match()
        alias = {**original, "id": "alias"}
        self.record(original)
        for records, cutoff in (([original, alias], self.when),
                                ([alias, original], self.when)):
            matches, drops = self.load(records, cutoff=cutoff)
            self.assertEqual(matches, [])
            self.assertEqual(drops["unobserved_or_conflicting_result_version"], 2)
        self.record({"id": "m1", "winner": 0}, self.when+timedelta(hours=1))
        for raw in (original, {**original, "winner": 0}, {**original, "scoreStatus": 1}):
            for records in ([raw, alias], [alias, raw]):
                matches, drops = self.load(records, cutoff=self.when+timedelta(hours=2))
                self.assertEqual(matches, [])
                self.assertEqual(drops["unobserved_or_conflicting_result_version"], 2)

    def test_matching_observed_copy_is_preferred_to_untracked_fixture_alias(self):
        original = completed_match()
        alias = {**original, "id": "a-first", "team1seed": 1}
        self.record(original)
        for records in ([original, alias], [alias, original]):
            matches, _ = self.load(records)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["match_id"], "m1")
            self.assertIsNone(matches[0]["team1_seed"])
            self.assertEqual(matches[0]["available_at"], self.when.replace(tzinfo=None))

    def test_rejected_observations_are_excluded_even_without_deduplication(self):
        self.record(completed_match())
        self.load([completed_match()])
        receipts = observations_before(self.when, self.db)
        matches, drops, _ = load_all_matches(self.raw, observed_results=receipts, deduplicate=False)
        self.assertEqual(matches, [])
        self.assertEqual(drops["unobserved_or_conflicting_result_version"], 1)

    def test_matching_result_uses_receipt_without_backdating_old_history(self):
        match = completed_match()
        self.record(match)
        old = {**match, "id": "historical", "matchTimeUtc": "2020-01-01 10:00:00"}
        matches, drops = self.load([match, old])
        self.assertEqual(drops, {})
        self.assertEqual(matches[0]["available_at"], datetime(2020, 1, 2))
        self.assertEqual(matches[0]["availability_source"], "utc_day_after_recorded_end")
        self.assertEqual(matches[1]["available_at"], self.when.replace(tzinfo=None))
        self.assertEqual(matches[1]["availability_source"], "observed_completed_result")
        default, _, _ = load_all_matches(self.raw)
        self.assertEqual(default[1]["available_at"], datetime(2026, 11, 12))

    def test_future_or_mismatched_versions_never_fall_back_to_earlier_embargo(self):
        original = completed_match()
        self.record(original)
        corrected = {**original, "winner": 2}
        self.record(corrected, self.when+timedelta(hours=1))
        matches, drops = self.load([corrected])
        self.assertEqual(matches, [])
        self.assertEqual(drops["unobserved_or_conflicting_result_version"], 1)
        matches, _ = self.load([original], cutoff=self.when)
        self.assertEqual(matches, [])
        matches, _ = self.load([corrected], cutoff=self.when+timedelta(hours=2))
        self.assertEqual(matches[0]["available_at"], datetime(2026, 11, 11, 17))

    def test_score_and_metadata_must_match_the_observed_version(self):
        original = completed_match()
        self.record(original)
        for changes in ({"duration": 46}, {"team1seed": 1}, {"score": [{"home": 21, "away": 9}]}):
            with self.subTest(changes=changes):
                matches, drops = self.load([{**original, **changes}])
                self.assertEqual(matches, [])
                self.assertEqual(drops["unobserved_or_conflicting_result_version"], 1)

    def test_impossible_observation_before_start_is_rejected(self):
        original = completed_match()
        self.record(original, self.when.replace(hour=8))
        matches, drops = self.load([original])
        self.assertEqual(matches, [])
        self.assertEqual(drops["observation_before_match"], 1)

    def test_observed_policy_is_saved_and_rejects_mixed_model_bundles(self):
        original = completed_match()
        self.record(original)
        matches, _ = self.load([original])
        history = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
        frame = history.build_dataframe(matches, as_of=self.when.replace(tzinfo=None)+timedelta(seconds=1))
        self.assertEqual(frame.loc[0, "elo_matches_t1_disc"], 0)
        state = self.root / "state.pkl"
        history.save(state)
        restored = FeatureBuilder.load(state)
        self.assertEqual(restored.model_policy, history.model_policy)
        self.assertEqual(restored.model_policy["result_availability"], OBSERVED_AVAILABILITY_POLICY)
        model = SimpleNamespace(feature_name_=FEATURE_COLS, feature_policy_=history.model_policy,
                                trained_through_=history.history_cutoff)
        restored.validate_model(model)
        model.feature_policy_ = FeatureBuilder().model_policy
        with self.assertRaisesRegex(ValueError, "policy differs"):
            restored.validate_model(model)
        payload = pickle.loads(state.read_bytes())
        payload["result_availability_policy"] = "unknown"
        state.write_bytes(pickle.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Unsupported result-availability"):
            FeatureBuilder.load(state)
        self.assertEqual(FeatureBuilder().model_policy["result_availability"], AVAILABILITY_POLICY)

    def test_v1_observed_history_remains_loadable_without_changing_its_policy(self):
        self.record(completed_match())
        matches, _ = self.load([completed_match()])
        legacy = FeatureBuilder(result_availability_policy=LEGACY_OBSERVED_AVAILABILITY_POLICY)
        legacy.build_dataframe(matches)
        state = self.root / "v1.pkl"
        legacy.save(state)
        restored = FeatureBuilder.load(state)
        self.assertEqual(restored.model_policy, legacy.model_policy)
        self.assertEqual(restored.model_policy["result_availability"], LEGACY_OBSERVED_AVAILABILITY_POLICY)

    def test_features_do_not_see_results_until_observed(self):
        raw = [{**completed_match(), "id": f"m{i}", "matchTimeUtc": f"2026-11-11 {hour}:00:00"}
               for i, hour in enumerate(("09", "12", "14"))]
        for match, hour in zip(raw, (13, 13, 15)):
            self.record(match, self.when.replace(hour=hour))
        matches, _ = self.load(raw)
        history = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
        baseline = history.build_dataframe(matches)
        self.assertEqual(baseline["elo_matches_t1_disc"].tolist(), [0, 0, 2])
        changed = [dict(m) for m in matches]
        changed[-1].update(winner=2, label=0, score_margin=0.4)
        altered = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY).build_dataframe(changed)
        pd.testing.assert_frame_equal(baseline[FEATURE_COLS], altered[FEATURE_COLS])
        fixture = {**matches[-1], "match_id": "future", "match_time": datetime(2026, 11, 12),
                   "team1_player_ids": "p1", "team2_player_ids": "p2"}
        preview = InferenceBuilder(history).build_features(pd.DataFrame([fixture]))
        self.assertEqual(preview.loc[0, "elo_matches_t1_disc"], 3)
        with self.assertRaisesRegex(ValueError, "require the observed-result"):
            FeatureBuilder().build_dataframe(matches)

    def test_observed_training_cannot_overwrite_the_active_artifacts(self):
        args = SimpleNamespace(smoke_test=False, artifacts_dir=REPO_ROOT / "outputs/artifacts", observations_db=self.db)
        with patch("train.parse_args", return_value=args):
            self.assertEqual(train.main(), 1)

    def test_training_defaults_to_observations_and_allows_explicit_conservative_history(self):
        from predict_badminton.data_sources.observations import OBSERVATIONS_PATH

        with patch("sys.argv", ["train.py"]):
            args = train.parse_args()
        self.assertEqual(args.observations_db, OBSERVATIONS_PATH)
        self.assertIsNone(args.artifacts_dir)
        with patch("sys.argv", ["train.py", "--conservative-history"]):
            self.assertIsNone(train.parse_args().observations_db)

    def test_missing_default_database_uses_conservative_fallback_without_creating_it(self):
        artifacts = self.root / "run"
        with patch("predict_badminton.data_sources.observations.OBSERVATIONS_PATH", self.db), \
                patch("sys.argv", ["train.py", "--artifacts-dir", str(artifacts)]), \
                patch("predict_badminton.data_parser.load_all_matches", return_value=([], {}, {})) as load:
            self.assertEqual(train.main(), 1)
        self.assertEqual(load.call_args.kwargs["observed_results"], {})
        self.assertEqual(json.loads((artifacts / "observed_result_receipts.json").read_text()), {})
        self.assertFalse(self.db.exists())

    def test_missing_custom_database_fails_without_creating_a_run(self):
        artifacts = self.root / "run"
        with patch("sys.argv", ["train.py", "--observations-db", str(self.db), "--artifacts-dir", str(artifacts)]):
            self.assertEqual(train.main(), 1)
        self.assertFalse(artifacts.exists())

    def test_unreadable_database_does_not_silently_use_conservative_history(self):
        self.db.write_bytes(b"not a SQLite database")
        artifacts = self.root / "run"
        with patch("sys.argv", ["train.py", "--observations-db", str(self.db), "--artifacts-dir", str(artifacts)]):
            self.assertEqual(train.main(), 1)
        self.assertFalse(artifacts.exists())

    def test_explicit_conservative_training_does_not_read_receipts(self):
        artifacts = self.root / "run"
        with patch("sys.argv", ["train.py", "--conservative-history", "--artifacts-dir", str(artifacts)]), \
                patch("predict_badminton.data_sources.observations.observations_before", side_effect=AssertionError("must not read")), \
                patch("predict_badminton.data_parser.load_all_matches", return_value=([], {}, {})) as load:
            self.assertEqual(train.main(), 1)
        self.assertIsNone(load.call_args.kwargs["observed_results"])
        self.assertFalse((artifacts / "observed_result_receipts.json").exists())

    def test_default_training_directory_is_a_new_timestamped_run(self):
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        with patch("predict_badminton.data_sources.observations.OBSERVATIONS_PATH", self.db), \
                patch("sys.argv", ["train.py"]), \
                patch("predict_badminton.data_parser.load_all_matches", return_value=([], {}, {})):
            self.assertEqual(train.main(), 1)
        runs = list((self.root / "outputs/runs").iterdir())
        self.assertEqual(len(runs), 1)
        self.assertRegex(runs[0].name, r"^\d{8}T\d{6}_\d{6}Z$")
        self.assertTrue((runs[0] / "observed_result_receipts.json").exists())
        self.assertFalse((self.root / "outputs/artifacts").exists())

    def test_late_older_result_does_not_replace_the_most_recent_played_match(self):
        older = {**completed_match(), "id": "older", "matchTimeUtc": "2026-11-11 09:00:00"}
        newer = {**completed_match(), "id": "newer", "matchTimeUtc": "2026-11-11 12:00:00"}
        self.record(newer, self.when.replace(hour=13))
        self.record(older, self.when.replace(hour=14))
        matches, _ = self.load([older, newer])
        history = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
        history.build_dataframe(matches)
        self.assertEqual([m["time"].hour for m in history._player_hist["p1"]], [9, 12])
        _, _, rest = history._form(["p1"], datetime(2026, 11, 12, 10), 5)
        self.assertEqual(rest, 0)

    def test_late_oldest_result_cannot_displace_the_latest_fifty_matches(self):
        records = []
        for index in range(51):
            start = datetime(2026, 9, 1, 10)+timedelta(days=index)
            match = {**completed_match(), "id": str(index), "matchTimeUtc": str(start)}
            self.record(match, self.when if index == 0 else self.when-timedelta(hours=1))
            records.append(match)
        matches, _ = self.load(records)
        history = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
        history.build_dataframe(matches)
        times = [m["time"] for m in history._player_hist["p1"]]
        self.assertEqual(len(times), 50)
        self.assertEqual(times, [datetime(2026, 9, 1, 10)+timedelta(days=i) for i in range(1, 51)])


if __name__ == "__main__":
    unittest.main()
