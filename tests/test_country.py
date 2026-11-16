from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
import math
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from predict_badminton.config import COUNTRY_FEATURE_COLS, FEATURE_COLS
from predict_badminton.country import CountryContext
from predict_badminton.data_sources.observations import record_results
from predict_badminton.features import FeatureBuilder
from predict_badminton.forecast_history import refresh_history
from predict_badminton.inference import InferenceBuilder
from predict_badminton.odds import round_predictions
from predict_badminton.probabilities import PREDICTION_POLICY, predict_probabilities, swap_features
from predict_badminton.temporal import OBSERVED_AVAILABILITY_POLICY
from predict_badminton.tournament import forecast, load_bundle
from tests.test_feature_state import _match
from tests.test_observations import completed_match
from tests.test_odds import FixedModel, snapshot
from tests.test_probabilities import BiasedModel
from tests.test_temporal import historical
from tests.test_tournament import draw
from tests.test_tournament_catalog import catalog


def context():
    return CountryContext(countries=["CAN", "USA", "JPN", "ENG", "SCO"],
                          located_countries=["CAN", "USA", "JPN", "ENG", "SCO"],
                          host_codes=["example"], host_names=["example open"], vocabulary=["CAN", "USA"])


def builder():
    return FeatureBuilder(country_context=context(), result_availability_policy=OBSERVED_AVAILABILITY_POLICY)


class CountryTests(unittest.TestCase):
    def test_country_elo_is_per_discipline_uniform_k20_and_counts_are_log1p(self):
        history = context()
        match = _match("1", datetime(2026, 1, 1), 1)
        history.observe(match)
        row = history.preview(match)
        self.assertEqual(row["country_rating_t1"], 1510)
        self.assertEqual(row["country_rating_t2"], 1490)
        self.assertEqual(row["country_log_n_t1"], math.log1p(1))
        self.assertEqual(history.preview({**match, "discipline": "WS"})["country_rating_t1"], 1500)
        history = context()
        history.observe({**match, "discipline": "MD", "round_name": "Final", "tournament_tier": 1})
        self.assertEqual(history.preview({**match, "discipline": "MD"})["country_rating_t1"], 1510)

    def test_unknown_and_domestic_results_do_not_change_country_history(self):
        history = context()
        before = pickle.dumps(history.get_state())
        for code in ("CAN", "", None, "UNKNOWN"):
            history.observe({**_match("1", datetime(2026, 1, 1), 1), "team2_country": code})
        self.assertEqual(pickle.dumps(history.get_state()), before)
        row = history.preview({**_match("1", datetime(2026, 1, 1), 1), "team2_country": ""})
        self.assertTrue(math.isnan(row["country_rating_t2"]))
        self.assertIsNone(row["country_id_t2"])
        self.assertEqual(row["country_known_t2"], 0)

    def test_home_nations_keep_separate_country_identities(self):
        history = context()
        match = {**_match("1", datetime(2026, 1, 1), 1), "team1_country": "ENG", "team2_country": "SCO"}
        history.observe(match)
        row = history.preview(match)
        self.assertEqual((row["country_id_t1"], row["country_id_t2"]), ("ENG", "SCO"))
        self.assertEqual(row["country_rating_diff"], 20)

    def test_preview_is_read_only_and_host_name_lookup_normalizes_punctuation(self):
        history = context()
        before = pickle.dumps(history.get_state())
        fixture = {**_match("1", datetime(2026, 1, 1), 1), "tournament_code": "new",
                   "tournament_name": "Example - Open"}
        self.assertEqual(history.preview(fixture)["host_known"], 1)
        self.assertEqual(history.preview({**fixture, "tournament_name": "Unlisted"})["host_known"], 0)
        self.assertEqual(pickle.dumps(history.get_state()), before)

    def test_results_release_strictly_before_prediction_and_ties_are_deterministic(self):
        first = historical("1", datetime(2026, 1, 1, 10), release=datetime(2026, 1, 2))
        equal = historical("2", datetime(2026, 1, 2))
        later = historical("3", datetime(2026, 1, 2, 0, 1))
        a, b = builder(), builder()
        frame = a.build_dataframe([first, equal, later])
        pd.testing.assert_frame_equal(frame, b.build_dataframe([later, equal, first]))
        self.assertEqual(frame["country_rating_t1"].tolist(), [1500, 1500, 1510])
        self.assertEqual(frame["country_log_n_t1"].tolist(), [0, 0, math.log1p(1)])
        changed = builder().build_dataframe([first, equal, {**later, "winner": 2, "label": 0}])
        pd.testing.assert_frame_equal(frame[COUNTRY_FEATURE_COLS], changed[COUNTRY_FEATURE_COLS])
        self.assertEqual(a.get_state(), b.get_state())
        self.assertEqual(a.country_context.engine.team_disc_count(["CAN"], "MS"), 3)

    def test_country_vocabulary_is_shared_frozen_and_unknown_categories_are_missing(self):
        history = builder()
        fixture = _match("1", datetime(2026, 1, 1), 1)
        rows = [history.preview(fixture), history.preview({**fixture, "team1_country": "JPN", "team2_country": ""})]
        frame = pd.DataFrame(rows)[COUNTRY_FEATURE_COLS]
        history.encode_categories(frame)
        self.assertEqual(list(frame["country_id_t1"].cat.categories), ["CAN", "USA"])
        self.assertEqual(frame["country_id_t1"].dtype, frame["country_id_t2"].dtype)
        self.assertTrue(pd.isna(frame.iloc[1]["country_id_t1"]))
        self.assertTrue(pd.isna(frame.iloc[1]["country_id_t2"]))
        self.assertEqual(history.country_context.registry["vocabulary"], ["CAN", "USA"])

    def test_swapping_preserves_categories_context_and_complementary_probabilities(self):
        history = builder()
        fixture = _match("1", datetime(2026, 1, 1), 1)
        frame = pd.DataFrame([history.preview(fixture)])[COUNTRY_FEATURE_COLS]
        history.encode_categories(frame)
        reverse = {**fixture, "team1_ids": fixture["team2_ids"], "team2_ids": fixture["team1_ids"],
                   "team1_country": "USA", "team2_country": "CAN"}
        expected = pd.DataFrame([history.preview(reverse)])[COUNTRY_FEATURE_COLS]
        history.encode_categories(expected)
        pd.testing.assert_frame_equal(swap_features(frame), expected)
        pd.testing.assert_frame_equal(swap_features(swap_features(frame)), frame)
        model = BiasedModel(COUNTRY_FEATURE_COLS)
        np.testing.assert_allclose(predict_probabilities(model, frame) + predict_probabilities(model, expected), 1)

    def test_version7_roundtrip_and_incomplete_or_incompatible_states_are_rejected(self):
        history = builder()
        history.build_dataframe([historical("1", datetime(2026, 1, 1))])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pkl"
            history.save(path)
            restored = FeatureBuilder.load(path)
            self.assertEqual(restored.get_state(), history.get_state())
            self.assertEqual(restored.model_policy, history.model_policy)
            self.assertEqual(restored.feature_columns, COUNTRY_FEATURE_COLS)
            for change in ("missing", "old_version", "wrong_k"):
                state = deepcopy(history.get_state())
                if change == "missing":
                    del state["country_context"]
                elif change == "old_version":
                    state["version"] = 5
                else:
                    state["country_context"]["elo"]["k_base"] = 32
                path.write_bytes(pickle.dumps(state))
                with self.assertRaises(ValueError):
                    FeatureBuilder.load(path)

    def test_country_fixtures_require_explicit_country_columns_but_allow_unknown_values(self):
        history = builder()
        fixture = pd.DataFrame([{"match_id": "new", "match_time": datetime(2030, 1, 1),
                                 "discipline": "MS", "team1_player_ids": "p1", "team2_player_ids": "p2"}])
        with self.assertRaisesRegex(ValueError, "team1_country"):
            InferenceBuilder(history).build_features(fixture)
        frame = InferenceBuilder(history).build_features(fixture.assign(team1_country="CAN", team2_country=""))
        self.assertEqual(list(frame.columns[2:]), COUNTRY_FEATURE_COLS)
        self.assertTrue(pd.isna(frame.iloc[0]["country_id_t2"]))
        self.assertEqual(list(InferenceBuilder(FeatureBuilder()).build_features(fixture).columns[2:]), FEATURE_COLS)

    def test_country_state_preserves_optional_tournament_catalog(self):
        history = FeatureBuilder(country_context=context(), category_catalog=catalog())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pkl"
            history.save(path)
            restored = FeatureBuilder.load(path)
        self.assertEqual(restored.model_policy, history.model_policy)
        self.assertEqual(restored.category_catalog.tier("code-1"), 4)

    def test_refresh_rebuilds_country_ratings_with_corrections_and_retractions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, db = root / "raw", root / "observations.sqlite3"
            raw.mkdir()
            history = builder()
            before = pickle.dumps(history.get_state())
            seen = datetime(2026, 11, 11, 16, tzinfo=UTC)
            match = {**completed_match(), "matchStatus": "F"}
            for number, winner in enumerate((1, 2, 0)):
                current = {**match, "winner": winner}
                (raw / "results.json").write_text(json.dumps([[current]]))
                record_results([current], source_url="https://example.com", response_sha256=str(number),
                               observed_at=seen + timedelta(minutes=number), path=db)
                fresh = refresh_history(history, as_of=seen + timedelta(minutes=number + 1),
                                        raw_dir=raw, observations_db=db)
                rating, _ = fresh.country_context.engine.team_elo(["CAN"], "MS")
                self.assertEqual(rating, (1510, 1490, 1500)[number])
                self.assertEqual(fresh.model_policy, history.model_policy)
            self.assertEqual(pickle.dumps(history.get_state()), before)

    def test_raw_only_bundle_loads_and_rejects_unavailable_calibrators(self):
        history = builder()
        model = FixedModel(history)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history.save(root / "feature_state.pkl")
            (root / "feature_list.json").write_text(json.dumps(COUNTRY_FEATURE_COLS))
            (root / "lgbm_model.pkl").write_bytes(pickle.dumps(model))
            (root / "bundle_manifest.json").write_text(json.dumps({"version": 1, "probability_methods": ["Raw"],
                                                                 "prediction_policy": PREDICTION_POLICY}))
            bundle = load_bundle(root)
            self.assertEqual(bundle[1:3], (None, None))
            data = draw()
            for i, entry in enumerate(data["entries"]):
                entry["country"] = "CAN" if i % 2 == 0 else "USA"
            self.assertEqual(len(forecast(data, "MS", bundle, refresh=False)["table"]), 4)
            for method in ("Platt", "Isotonic"):
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    forecast(data, "MS", bundle, method, refresh=False)
            del data["entries"][0]["country"]
            with self.assertRaisesRegex(ValueError, "country"):
                forecast(data, "MS", bundle, refresh=False)
            table, _ = round_predictions(snapshot(), bundle, round_name="R32", refresh=False)
            self.assertEqual(len(table), 2)
            (root / "bundle_manifest.json").write_text(json.dumps({"version": 1, "probability_methods": ["Raw"],
                                                                 "prediction_policy": "unsupported"}))
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_bundle(root)
