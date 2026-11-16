from __future__ import annotations

import unittest
import pickle
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from predict import main, parse_args
from predict_badminton.features import FeatureBuilder
from predict_badminton.config import forecast_bundle_directory
from predict_badminton.data_sources.observations import record_results
from predict_badminton.temporal import OBSERVED_AVAILABILITY_POLICY
from tests.test_feature_state import _match
from tests.test_observations import completed_match
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
                    "--output-dir", str(root / "predictions"), "--frozen-history"]
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
        self.assertFalse(args.frozen_history)

    def test_calibration_uses_the_models_bundle_directory(self):
        for method in ("platt", "isotonic"):
            with patch("sys.argv", ["predict.py", "--fixtures-csv", "fixtures.csv",
                                    "--model-path", "outputs/runs/example/lgbm_model.pkl",
                                    "--probabilities", method]):
                args = parse_args()
            self.assertEqual(args.calibrator_path, Path(f"outputs/runs/example/{method}_calibrator.pkl"))
            self.assertEqual(args.state_path, Path("outputs/runs/example/feature_state.pkl"))

    def test_default_forecast_pointer_does_not_replace_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("predict_badminton.config.REPO_ROOT", root):
                self.assertEqual(forecast_bundle_directory(), root / "outputs/artifacts")
                (root / "outputs").mkdir()
                pointer = root / "outputs/forecast_bundle.json"
                pointer.write_text(json.dumps({"directory": "outputs/runs/observed"}))
                self.assertEqual(forecast_bundle_directory(), (root / "outputs/runs/observed").resolve())
                pointer.write_text(json.dumps({"directory": "../outside"}))
                with self.assertRaises(ValueError):
                    forecast_bundle_directory()

    def test_cleaned_default_changes_new_cli_selection_without_touching_frozen_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outputs").mkdir()
            previous = {"directory": "outputs/runs/previous-country"}
            cleaned = {"directory": "outputs/runs/receipt-quality-cleaned"}
            pointer = root / "outputs/forecast_bundle.json"
            pointer.write_text(json.dumps(previous))
            frozen = root / "outputs/hong-kong-forecast.json"
            frozen.write_text(json.dumps({"model_directory": previous["directory"], "probability": "0.6123456789"}))
            ledger = root / "outputs/profits.json"
            ledger.write_text('{"bets": [{"stake": "10.00", "model_probability": "0.6123456789"}]}')
            original = {path: path.read_bytes() for path in (frozen, ledger)}
            (root / "outputs/forecast_bundle.previous.json").write_text(json.dumps(previous))
            pointer.write_text(json.dumps(cleaned))
            with patch("predict_badminton.config.REPO_ROOT", root), patch("sys.argv", ["predict.py", "--fixtures-csv", "future.csv"]):
                args = parse_args()
                self.assertEqual(args.model_path, (root / cleaned["directory"] / "lgbm_model.pkl").resolve())
                self.assertEqual(args.state_path, (root / cleaned["directory"] / "feature_state.pkl").resolve())
                self.assertEqual(args.probabilities, "raw")
                self.assertFalse(args.frozen_history)
            self.assertEqual(json.loads((root / "outputs/forecast_bundle.previous.json").read_text()), previous)
            self.assertTrue(all(path.read_bytes() == content for path, content in original.items()))

    def test_cli_refreshes_by_default_and_never_overwrites_existing_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, out = root / "raw", root / "predictions"
            raw.mkdir()
            out.mkdir()
            (out / "saved.csv").write_text("Existing forecast\n")
            history = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
            history.save(root / "feature_state.pkl")
            model = FixedModel()
            model.feature_policy_ = history.model_policy
            (root / "model.pkl").write_bytes(pickle.dumps(model))
            match = {**completed_match(), "matchStatus": "F"}
            (raw / "match.json").write_text(json.dumps([[match]]))
            db = root / "observations.sqlite3"
            record_results([match], source_url="https://example.com/results", response_sha256="source",
                           observed_at=datetime(2026, 11, 11, 16, tzinfo=UTC), path=db)
            (root / "fixtures.csv").write_text("match_id,match_time,discipline,team1_player_ids,team2_player_ids\n"
                                                "next,2030-01-01 10:00:00,MS,p1,p2\n")
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            argv = ["predict.py", "--fixtures-csv", str(root / "fixtures.csv"),
                    "--model-path", str(root / "model.pkl"), "--output-dir", str(out),
                    "--raw-dir", str(raw), "--observations-db", str(db)]
            for _ in range(2):
                with patch("sys.argv", argv):
                    self.assertEqual(main(), 0)
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
            exports = list(out.glob("predictions_*.csv"))
            self.assertEqual(len(exports), 2)
            for export in exports:
                row = pd.read_csv(export).iloc[0]
                self.assertTrue(row["history_refreshed"])
                self.assertEqual(row["base_state_sha256"], hashlib.sha256(before[root / "feature_state.pkl"]).hexdigest())
                self.assertNotEqual(row["state_sha256"], row["base_state_sha256"])
                self.assertLess(pd.Timestamp(row["history_as_of"]), pd.Timestamp(row["generated_at_utc"]).tz_localize(None))

    def test_calibrator_cannot_be_silently_ignored(self):
        with patch("sys.argv", ["predict.py", "--fixtures-csv", "fixtures.csv",
                                "--calibrator-path", "calibrator.pkl"]), patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parse_args()


if __name__ == "__main__":
    unittest.main()
