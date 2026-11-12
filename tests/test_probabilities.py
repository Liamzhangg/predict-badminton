from __future__ import annotations

from datetime import datetime
import pickle
import unittest

import numpy as np
import pandas as pd

from predict_badminton.config import FEATURE_COLS, GLICKO_MODEL_FEATURE_COLS, LEGACY_FEATURE_COLS
from predict_badminton.features import FeatureBuilder
from predict_badminton.glicko import GlickoEngine
from predict_badminton.probabilities import average_directions, predict_directions, predict_probabilities, swap_features
from tests.test_feature_state import _match


class BiasedModel:
    def __init__(self, columns=FEATURE_COLS):
        self.feature_name_ = list(columns)
        self.calls = 0

    def predict_proba(self, frame):
        self.calls += 1
        value = 0.4 + 0.002*(frame["elo_t1_disc"]-1500) - 0.001*(frame["elo_t2_disc"]-1500)
        probabilities = 1 / (1 + np.exp(-value.to_numpy()))
        return np.column_stack((1-probabilities, probabilities))


class OffsetCalibrator:
    def predict(self, probabilities):
        return 0.1 + 0.8*probabilities**2


def frame(columns=FEATURE_COLS):
    values = {name: [float(i), float(i+1)] for i, name in enumerate(columns) if name != "discipline"}
    values["h2h_wr_t1"] = [0.3, 0.7]
    values["discipline"] = pd.Categorical(["MS", "MD"], categories=["MS", "WS", "MD", "WD", "XD", "OTHER"])
    return pd.DataFrame(values, index=[7, 2])[columns]


class ProbabilityTests(unittest.TestCase):
    def test_two_order_example_and_exact_ties(self):
        np.testing.assert_allclose(average_directions(np.array([[0.6, 0.3]])), [0.65])
        values = np.linspace(0, 1, 10001)
        np.testing.assert_array_equal(average_directions(np.column_stack((values, values))), np.full(len(values), 0.5))
        np.testing.assert_array_equal(average_directions(np.array([[0, 1], [1, 0]])), [0, 1])

    def test_swapping_twice_restores_every_supported_schema_and_context(self):
        for columns in (FEATURE_COLS, GLICKO_MODEL_FEATURE_COLS, LEGACY_FEATURE_COLS):
            data = frame(columns)
            before = data.copy(deep=True)
            swapped = swap_features(data)
            pd.testing.assert_frame_equal(swap_features(swapped), data, atol=1e-12)
            pd.testing.assert_frame_equal(data, before)
            for name in ("discipline", "round_importance", "tournament_tier", "same_country"):
                pd.testing.assert_series_equal(data[name], swapped[name])
            np.testing.assert_allclose(swapped["h2h_wr_t1"], 1-data["h2h_wr_t1"])
            np.testing.assert_array_equal(swapped["seed_diff"], -data["seed_diff"])

    def test_single_batch_and_complementary_predictions_for_biased_models(self):
        for columns in (FEATURE_COLS, GLICKO_MODEL_FEATURE_COLS, LEGACY_FEATURE_COLS):
            data = frame(columns)
            model = BiasedModel(columns)
            original = predict_probabilities(model, data)
            self.assertEqual(model.calls, 1)
            reverse = predict_probabilities(model, swap_features(data))
            np.testing.assert_allclose(original + reverse, 1, rtol=0, atol=1e-15)

    def test_calibration_is_directionwise_before_final_averaging(self):
        values = np.array([[0.6, 0.3]])
        calibrated = average_directions(values, OffsetCalibrator())
        np.testing.assert_allclose(calibrated, [0.608])
        self.assertNotAlmostEqual(float(calibrated[0]), float(OffsetCalibrator().predict(np.array([0.65]))[0]))
        data = frame()
        model = BiasedModel()
        a = predict_probabilities(model, data, OffsetCalibrator())
        b = predict_probabilities(model, swap_features(data), OffsetCalibrator())
        np.testing.assert_allclose(a+b, 1, rtol=0, atol=1e-15)

    def test_feature_swaps_match_native_builders_without_changing_history(self):
        for columns in (FEATURE_COLS, GLICKO_MODEL_FEATURE_COLS, LEGACY_FEATURE_COLS):
            history = FeatureBuilder()
            if columns != FEATURE_COLS:
                history.trueskill = None
                history.glicko = GlickoEngine() if columns == GLICKO_MODEL_FEATURE_COLS else None
                history._state_version = 3 if history.glicko else 1
            history.process(_match("past", datetime(2024, 1, 1), 1))
            before = pickle.dumps(history.get_state())
            for event in ("MS", "WS", "MD", "WD", "XD"):
                fixture = {**_match("future", datetime(2024, 1, 3), 1), "discipline": event,
                           "team1_seed": 1, "team2_seed": 7}
                if event in ("MD", "WD", "XD"):
                    fixture.update(team1_ids=["p1", "new1"], team2_ids=["p2", "new2"])
                reverse = dict(fixture)
                for name in ("ids", "country", "seed"):
                    reverse[f"team1_{name}"], reverse[f"team2_{name}"] = fixture[f"team2_{name}"], fixture[f"team1_{name}"]
                a = pd.DataFrame([history.preview(fixture)])[columns]
                b = pd.DataFrame([history.preview(reverse)])[columns]
                pd.testing.assert_frame_equal(swap_features(a), b, atol=1e-12)
            self.assertEqual(pickle.dumps(history.get_state()), before)

    def test_unknown_or_reordered_schemas_are_rejected(self):
        data = frame()
        for invalid in (data.iloc[:, ::-1], data.assign(unhandled_feature=0), data.drop(columns="seed_diff")):
            with self.assertRaisesRegex(ValueError, "schema"):
                swap_features(invalid)
        with self.assertRaisesRegex(ValueError, "features"):
            predict_directions(BiasedModel(LEGACY_FEATURE_COLS), data)

    def test_invalid_model_and_calibrator_outputs_are_rejected(self):
        class BadModel(BiasedModel):
            def predict_proba(self, data):
                return np.full((len(data), 2), np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_probabilities(BadModel(), frame())
        for invalid in (np.array([0.5]), np.array([[0.5, np.nan]]), np.array([[0.5, 2]])):
            with self.assertRaises(ValueError):
                average_directions(invalid)
        class BadCalibrator:
            def predict(self, data):
                return np.zeros((len(data), 1))
        with self.assertRaisesRegex(ValueError, "shape"):
            average_directions(np.array([[0.6, 0.3]]), BadCalibrator())
