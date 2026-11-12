import csv
from datetime import datetime
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from predict_badminton.config import FEATURE_COLS
from predict_badminton.data_parser import load_all_matches
from predict_badminton.features import FeatureBuilder
from predict_badminton.inference import InferenceBuilder
from predict_badminton.tournament_catalog import (
    CATALOG_PATH, TournamentCatalog, _FIELDS, build_catalog, official_tier,
)
from tests.test_feature_state import _match
from tests.test_observations import completed_match
import train


def catalog(*categories):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDS, lineterminator="\n")
    writer.writeheader()
    for index, category in enumerate(categories or ("HSBC BWF World Tour Super 500",), 1):
        writer.writerow({"tournament_id": str(index), "tournament_code": f"code-{index}",
                         "name": "Same tournament name", "start_date": f"{2018+index}-01-01 00:00:00",
                         "end_date": f"{2018+index}-01-06 00:00:00", "category": category})
    return TournamentCatalog(output.getvalue())


class TournamentCatalogTests(unittest.TestCase):
    def test_only_the_five_official_world_tour_levels_override_tiers(self):
        for name, tier in (("Finals", 2), ("Super 1000", 2), ("Super 750", 3), ("Super 500", 4), ("Super 300", 5)):
            self.assertEqual(official_tier(f"HSBC BWF World Tour {name}"), tier)
        for name in ("BWF Tour Super 100", "International Challenge", "Junior Super 500", "", "Unknown"):
            self.assertIsNone(official_tier(name))

    def test_lookup_uses_edition_ids_not_names(self):
        c = catalog("HSBC BWF World Tour Super 500", "HSBC BWF World Tour Super 750")
        self.assertEqual(c.tier("CODE-1"), 4)
        self.assertEqual(c.tier("", 2), 3)
        self.assertEqual(c.tier("1"), 4)
        self.assertIsNone(c.tier("Same tournament name"))
        with self.assertRaisesRegex(ValueError, "different editions"):
            c.lookup("code-1", "2")

    def test_duplicate_identifiers_and_unsupported_schema_are_rejected(self):
        c = catalog("HSBC BWF World Tour Super 500", "HSBC BWF World Tour Super 750")
        for text in (c.text.replace("2,code-2", "1,code-2"), c.text.replace("code-2", "code-1"), "name,category\nA,B\n"):
            with self.assertRaises(ValueError):
                TournamentCatalog(text)

    def test_parser_uses_official_tier_without_rewriting_raw_data(self):
        raw = {**completed_match(), "tournamentCode": "code-1"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.json"
            original = json.dumps([[raw]])
            path.write_text(original)
            mapped, _, _ = load_all_matches(Path(tmp), category_catalog=catalog())
            legacy, _, _ = load_all_matches(Path(tmp))
            self.assertEqual(path.read_text(), original)
        self.assertEqual(mapped[0]["tournament_tier"], 4)
        self.assertEqual(mapped[0]["tournament_tier_source"], "official_world_tour")
        self.assertEqual(legacy[0]["tournament_tier"], 5)

    def test_out_of_scope_categories_retain_existing_tier_behavior(self):
        raw = {**completed_match(), "tournamentCode": "code-1"}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "matches.json").write_text(json.dumps([[raw]]))
            records, _, _ = load_all_matches(Path(tmp), category_catalog=catalog("International Challenge"))
        self.assertEqual(records[0]["tournament_tier"], 5)
        self.assertEqual(records[0]["tournament_tier_source"], "name_heuristic")

    def test_official_tier_drives_features_and_elo_updates(self):
        official, legacy = FeatureBuilder(category_catalog=catalog()), FeatureBuilder()
        match = {**_match("m", datetime(2020, 1, 1), 1), "tournament_code": "code-1"}
        self.assertEqual(official.process(match)["tournament_tier"], 4)
        self.assertEqual(legacy.process(match)["tournament_tier"], 5)
        self.assertGreater(official.elo.team_elo(["p1"], "MS")[0], legacy.elo.team_elo(["p1"], "MS")[0])

    def test_catalog_snapshot_survives_save_and_external_category_changes(self):
        history = FeatureBuilder(category_catalog=catalog())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pkl"
            history.save(path)
            changed = catalog("HSBC BWF World Tour Super 750")
            restored = FeatureBuilder.load(path)
            self.assertEqual(restored.get_state()["version"], 6)
            self.assertEqual(restored.category_catalog.tier("code-1"), 4)
            self.assertNotEqual(restored.category_catalog.sha256, changed.sha256)
            model = SimpleNamespace(feature_name_=FEATURE_COLS, feature_policy_=history.model_policy,
                                    trained_through_=datetime.min)
            restored.validate_model(model)
            model.feature_policy_ = FeatureBuilder(category_catalog=changed).model_policy
            with self.assertRaisesRegex(ValueError, "policy differs"):
                restored.validate_model(model)

    def test_incomplete_new_states_and_catalogs_in_old_states_are_rejected(self):
        state = FeatureBuilder(category_catalog=catalog()).get_state()
        for change in ({"version": 5}, {"tournament_catalog": None}):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.pkl"
                path.write_bytes(pickle.dumps({**state, **change}))
                with self.assertRaisesRegex(ValueError, "unsupported feature state"):
                    FeatureBuilder.load(path)

    def test_legacy_saved_state_does_not_inherit_the_new_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pkl"
            original = FeatureBuilder()
            original.save(path)
            restored = FeatureBuilder.load(path)
        self.assertIsNone(restored.category_catalog)
        self.assertEqual(restored.get_state(), original.get_state())
        self.assertNotIn("tournament_categories", restored.model_policy)

    def test_cli_fixture_code_or_id_overrides_stale_manual_tier(self):
        history = FeatureBuilder(category_catalog=catalog())
        row = {"match_id": "future", "match_time": pd.Timestamp("2027-01-01"), "discipline": "MS",
               "team1_player_ids": "p1", "team2_player_ids": "p2", "tournament_tier": "5"}
        for identifier in ({"tournament_code": "code-1"}, {"tournament_id": "1"}):
            frame = InferenceBuilder(history).build_features(pd.DataFrame([{**row, **identifier}]))
            self.assertEqual(frame.loc[0, "tournament_tier"], 4)
        old = InferenceBuilder(FeatureBuilder()).build_features(pd.DataFrame([{**row, "tournament_code": "code-1"}]))
        self.assertEqual(old.loc[0, "tournament_tier"], 5)

    def test_training_defaults_to_catalog_and_can_reproduce_legacy_tiers(self):
        with patch("sys.argv", ["train.py"]):
            self.assertEqual(train.parse_args().category_catalog, CATALOG_PATH)
        with patch("sys.argv", ["train.py", "--legacy-tournament-tiers"]):
            self.assertIsNone(train.parse_args().category_catalog)

    def test_new_models_require_explicit_context_for_uncatalogued_fixtures(self):
        row = {"match_id": "future", "match_time": pd.Timestamp("2027-01-01"), "discipline": "MS",
               "team1_player_ids": "p1", "team2_player_ids": "p2"}
        history = FeatureBuilder(category_catalog=catalog())
        for change in ({}, {"tournament_tier": ""}, {"tournament_tier": "bad"}, {"tournament_tier": "4.5"}, {"tournament_tier": "inf"}):
            with self.assertRaisesRegex(ValueError, "tournament_tier"):
                InferenceBuilder(history).build_features(pd.DataFrame([{**row, **change}]))
        known = InferenceBuilder(history).build_features(pd.DataFrame([{**row, "tournament_code": "code-1"}]))
        self.assertEqual(known.loc[0, "tournament_tier"], 4)
        legacy = InferenceBuilder(FeatureBuilder()).build_features(pd.DataFrame([row]))
        self.assertEqual(legacy.loc[0, "tournament_tier"], 5)

    def test_builder_rejects_conflicting_official_calendar_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"id": 1, "code": "code-1", "name": "Example", "start_date": "2020-01-01",
                   "end_date": "2020-01-06", "category": "HSBC BWF World Tour Super 500"}
            for name, category in (("a", row["category"]), ("b", "HSBC BWF World Tour Super 750")):
                (root / f"api-vue-tournaments-search-{name}.json").write_text(json.dumps({"results": {"data": [{**row, "category": category}]}}))
            with patch("predict_badminton.tournament_catalog.REPO_ROOT", root):
                with self.assertRaisesRegex(ValueError, "Conflicting official"):
                    build_catalog(root, root / "catalog.csv")

    def test_missing_default_receipts_allow_equivalent_filesystem_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configured = root / "unused" / ".." / "receipts.sqlite3"
            with patch("predict_badminton.data_sources.observations.OBSERVATIONS_PATH", configured), \
                    patch("sys.argv", ["train.py", "--artifacts-dir", str(root / "run")]), \
                    patch("predict_badminton.data_parser.load_all_matches", return_value=([], {}, {})) as load:
                self.assertEqual(train.main(), 1)
            self.assertEqual(load.call_args.kwargs["observed_results"], {})
            self.assertFalse(configured.exists())


if __name__ == "__main__":
    unittest.main()
