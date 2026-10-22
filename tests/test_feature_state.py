from __future__ import annotations

import tempfile
import pickle
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from predict_badminton.config import FEATURE_COLS
from predict_badminton.features import FeatureBuilder
from predict_badminton.inference import InferenceBuilder, load_fixtures


def _match(match_id: str, when: datetime, winner: int) -> dict:
    return {
        "match_id": match_id,
        "match_time": when,
        "tournament_code": "example",
        "tournament_name": "Example Open",
        "tournament_tier": 5,
        "discipline": "MS",
        "round_name": "Final",
        "team1_ids": ["p1"],
        "team2_ids": ["p2"],
        "team1_country": "CAN",
        "team2_country": "USA",
        "team1_seed": None,
        "team2_seed": None,
        "winner": winner,
        "label": int(winner == 1),
    }


class FeatureStateTests(unittest.TestCase):
    def test_preview_is_read_only_for_unseen_players_and_pairs(self) -> None:
        builder = FeatureBuilder()
        builder.process(_match("known", datetime(2024, 1, 1), 1))
        before = pickle.dumps(builder.get_state())
        for discipline, team1, team2 in (("MS", ["p1"], ["new"]),
                                         ("MD", ["new1", "new2"], ["new3", "new4"])):
            fixture = {**_match("future", datetime(2024, 1, 2), 1),
                       "discipline": discipline, "team1_ids": team1, "team2_ids": team2}
            builder.preview(fixture)
            self.assertEqual(pickle.dumps(builder.get_state()), before)

    def test_configured_k_factor_changes_elo_updates(self) -> None:
        ratings = []
        for k_base in (16, 32):
            builder = FeatureBuilder(k_base=k_base)
            builder.process(_match("m1", datetime(2024, 1, 1), 1))
            ratings.append(builder.elo.team_elo(["p1"], "MS"))
        for index in (0, 1):
            self.assertAlmostEqual(ratings[1][index] - 1500, 2 * (ratings[0][index] - 1500))

    def test_past_fixtures_cannot_use_future_history(self) -> None:
        builder = FeatureBuilder()
        builder.process(_match("known", datetime(2024, 1, 2), 1))
        for when in (datetime(2024, 1, 1), datetime(2024, 1, 2)):
            fixture = pd.DataFrame([{"match_time": when}])
            with self.assertRaisesRegex(ValueError, "cutoff"):
                InferenceBuilder(builder).build_features(fixture)

    def test_fixture_times_are_normalized_to_utc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.csv"
            path.write_text("match_id,match_time,discipline,team1_player_ids,team2_player_ids\n"
                            "m1,2026-01-01T10:00:00+02:00,MS,p1,p2\n"
                            "m2,2026-01-02 10:00:00,MS,p1,p2\n")
            fixtures = load_fixtures(path)
        self.assertEqual(fixtures.iloc[0]["match_time"], pd.Timestamp("2026-01-01 08:00:00"))
        self.assertIsNone(fixtures["match_time"].dt.tz)

    def test_leaderboards_export_when_disciplines_have_no_players(self) -> None:
        builder = FeatureBuilder()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for include_match in (False, True):
                if include_match:
                    builder.process(_match("m1", datetime(2024, 1, 1), 1))
                builder.export_leaderboards(output)
                files = list(output.glob("elo_leaderboard_*.csv"))
                self.assertEqual(len(files), 9)
                for path in files:
                    self.assertGreater(len(pd.read_csv(path).columns), 0)
                self.assertEqual(len(pd.read_csv(output / "elo_leaderboard_MS.csv")), 2 if include_match else 0)

    def test_pair_leaderboards_rank_named_pairs_by_hybrid_elo(self) -> None:
        builder = FeatureBuilder(player_names={"p1": "Player, One", "p2": "Player Two",
                                               "p3": "Player Three", "p4": ""})
        builder.elo.pair_lambda = 10
        for disc in ("MD", "WD", "XD"):
            for pid, rating in (("p1", 1700), ("p2", 1900), ("p3", 2000), ("p4", 2100)):
                builder.elo._disc[pid][disc] = rating
                builder.elo._global[pid] = rating
                builder.elo._disc_count[pid][disc] = 20
            for pair, component, count in ((("p1", "p2"), 400, 5), (("p3", "p4"), 20, 20)):
                builder.elo._pair[pair][disc] = component
                builder.elo._pair_count[pair][disc] = count

        before = builder.get_state()
        with tempfile.TemporaryDirectory() as tmp:
            builder.export_leaderboards(Path(tmp))
            for disc in ("MD", "WD", "XD"):
                table = pd.read_csv(Path(tmp) / f"elo_leaderboard_pairs_{disc}.csv")
                self.assertEqual(list(table.columns),
                                 ["pair_name", "pair_key", "elo", "pair_elo_component", "pair_matches"])
                self.assertEqual(list(table["pair_key"]), ["p3|p4", "p1|p2"])
                self.assertEqual(list(table["pair_name"]),
                                 ["Player Three / p4", "Player, One / Player Two"])
                self.assertEqual(list(table["pair_elo_component"]), [20, 400])
                self.assertEqual(list(table["pair_matches"]), [20, 5])
                self.assertAlmostEqual(table.iloc[0]["elo"], 2063.33)
                self.assertAlmostEqual(table.iloc[1]["elo"], 1933.33)
        self.assertEqual(builder.get_state(), before)

    def test_pair_leaderboards_require_two_distinct_players_and_shared_history(self) -> None:
        builder = FeatureBuilder()
        for pair, count in ((("a", "b"), 1), (("a", "c"), 0),
                            (("a",), 1), (("a", "a"), 1), (("a", "b", "c"), 1)):
            builder.elo._pair[pair]["MD"] = 0
            builder.elo._pair_count[pair]["MD"] = count
        with tempfile.TemporaryDirectory() as tmp:
            builder.export_leaderboards(Path(tmp))
            table = pd.read_csv(Path(tmp) / "elo_leaderboard_pairs_MD.csv")
        self.assertEqual(list(table["pair_key"]), ["a|b"])
        self.assertEqual(table.iloc[0]["elo"], 1500)
        self.assertEqual(table.iloc[0]["pair_matches"], 1)

    def test_pair_leaderboard_ties_are_ordered_by_pair_key(self) -> None:
        builder = FeatureBuilder()
        for pair in (("c", "d"), ("a", "b")):
            builder.elo._pair[pair]["MD"] = 0
            builder.elo._pair_count[pair]["MD"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            builder.export_leaderboards(Path(tmp))
            table = pd.read_csv(Path(tmp) / "elo_leaderboard_pairs_MD.csv")
        self.assertEqual(list(table["pair_key"]), ["a|b", "c|d"])

    def test_legacy_or_incomplete_state_requires_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pkl"
            for state in ({"disc": {}}, {"state_type": "feature_builder", "version": 1}, []):
                path.write_bytes(pickle.dumps(state))
                with self.assertRaisesRegex(ValueError, "retrain"):
                    FeatureBuilder.load(path)

    def test_invalid_times_or_empty_player_lists_leave_no_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.csv"
            for timestamp, ids in (("not-a-date", "p1"), ("2026-01-01", "|")):
                path.write_text("match_id,match_time,discipline,team1_player_ids,team2_player_ids\n"
                                f"m1,{timestamp},MS,{ids},p2\n")
                with self.assertRaisesRegex(ValueError, "No valid fixture"):
                    load_fixtures(path)

    def test_duplicate_fixture_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.csv"
            path.write_text("match_id,match_time,discipline,team1_player_ids,team2_player_ids\n"
                            "m1,2026-01-01,MS,p1,p2\nm1,2026-01-01,MS,p3,p4\n")
            with self.assertRaisesRegex(ValueError, "must be unique"):
                load_fixtures(path)

    def test_persisted_state_drives_training_equivalent_inference_features(self) -> None:
        builder = FeatureBuilder()
        builder.process(_match("m1", datetime(2024, 1, 1), 1))
        builder.process(_match("m2", datetime(2024, 1, 3), 1))

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "feature_state.pkl"
            builder.save(state_path)
            restored = FeatureBuilder.load(state_path)

        fixture_time = datetime(2024, 1, 5)
        fixture = pd.DataFrame(
            [
                {
                    "match_id": "future",
                    "match_time": fixture_time,
                    "discipline": "MS",
                    "team1_player_ids": "p1",
                    "team2_player_ids": "p2",
                    "round_name": "Final",
                    "tournament_name": "Example Open",
                    "tournament_tier": "5",
                    "team1_seed": "",
                    "team2_seed": "",
                    "team1_country": "CAN",
                    "team2_country": "USA",
                }
            ]
        )
        features = InferenceBuilder(restored).build_features(fixture).iloc[0]

        self.assertEqual(list(features.index), ["match_id", "match_time", *FEATURE_COLS])
        self.assertEqual(features["form5_cnt_t1"], 2.0)
        self.assertEqual(features["form5_wr_t1"], 1.0)
        self.assertEqual(features["form5_wr_t2"], 0.0)
        self.assertEqual(features["rest_days_t1"], 2.0)
        self.assertEqual(features["h2h_wins_t1"], 2.0)
        self.assertEqual(features["h2h_total"], 2.0)


if __name__ == "__main__":
    unittest.main()
