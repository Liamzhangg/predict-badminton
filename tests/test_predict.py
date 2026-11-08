from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from predict import parse_args


class PredictionArgsTests(unittest.TestCase):
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
