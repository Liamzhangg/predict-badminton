from __future__ import annotations

import unittest
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from predict import main, parse_args
from predict_badminton.features import FeatureBuilder
from tests.test_feature_state import _match
from tests.test_tournament import FixedCalibrator, FixedModel


class PredictionArgsTests(unittest.TestCase):
    def test_cli_accepts_matching_policies_and_rejects_mixed_bundles(self):
        history = FeatureBuilder()
        history.process({**_match("known", datetime(2024, 1, 1), 1), "score_margin": 0.3})
        model = FixedModel()
        model.feature_policy_ = history.model_policy
        calibrator = FixedCalibrator()
        calibrator.feature_policy_ = history.model_policy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history.save(root / "state.pkl")
            (root / "model.pkl").write_bytes(pickle.dumps(model))
            (root / "calibrator.pkl").write_bytes(pickle.dumps(calibrator))
            (root / "fixtures.csv").write_text("match_id,match_time,discipline,team1_player_ids,team2_player_ids\n"
                                                "next,2030-01-01 10:00:00,MS,p1,p2\n")
            argv = ["predict.py", "--fixtures-csv", str(root / "fixtures.csv"),
                    "--model-path", str(root / "model.pkl"), "--state-path", str(root / "state.pkl"),
                    "--output-dir", str(root / "predictions")]
            with patch("sys.argv", argv):
                self.assertEqual(main(), 0)
            output = pd.read_csv(next((root / "predictions").glob("*.csv")))
            self.assertEqual(output["prediction_policy"].iloc[0], "two_order_average_v1")
            self.assertEqual(output["team1_win_prob"].iloc[0], 0.5)
            self.assertEqual(output["team2_win_prob"].iloc[0], 0.5)
            self.assertEqual(output["predicted_winner"].iloc[0], "even")
            model.feature_policy_ = FeatureBuilder(score_margin_weight=0).model_policy
            (root / "model.pkl").write_bytes(pickle.dumps(model))
            with patch("sys.argv", argv):
                self.assertEqual(main(), 1)
            model.feature_policy_ = history.model_policy
            (root / "model.pkl").write_bytes(pickle.dumps(model))
            calibrated = argv + ["--probabilities", "platt", "--calibrator-path", str(root / "calibrator.pkl")]
            with patch("sys.argv", calibrated):
                self.assertEqual(main(), 0)
            self.assertEqual(len(list((root / "predictions").glob("*.csv"))), 2)
            calibrator.feature_policy_ = FeatureBuilder(score_margin_weight=0).model_policy
            (root / "calibrator.pkl").write_bytes(pickle.dumps(calibrator))
            with patch("sys.argv", calibrated):
                self.assertEqual(main(), 1)

    def test_default_probabilities_are_raw(self):
        with patch("sys.argv", ["predict.py", "--fixtures-csv", "fixtures.csv"]):
            args = parse_args()
        self.assertEqual(args.probabilities, "raw")
        self.assertIsNone(args.calibrator_path)

    def test_calibration_uses_the_models_bundle_directory(self):
        for method in ("platt", "isotonic"):
            with patch("sys.argv", ["predict.py", "--fixtures-csv", "fixtures.csv",
                                    "--model-path", "outputs/runs/example/lgbm_model.pkl",
                                    "--probabilities", method]):
                args = parse_args()
            self.assertEqual(args.calibrator_path, Path(f"outputs/runs/example/{method}_calibrator.pkl"))

    def test_calibrator_cannot_be_silently_ignored(self):
        with patch("sys.argv", ["predict.py", "--fixtures-csv", "fixtures.csv",
                                "--calibrator-path", "calibrator.pkl"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
