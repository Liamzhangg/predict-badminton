from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from predict_badminton.config import FEATURE_COLS
from predict_badminton.model import (
    _chronological_validation_split,
    _default_fold_val_years,
    _train_lgbm_for_evaluation,
    chronological_holdout_split,
    refit_production_model,
    run_baselines,
    train_final_model,
)


def _feature_frame(rows: int = 60) -> pd.DataFrame:
    data = {
        column: [0.0] * rows
        for column in FEATURE_COLS
        if column != "discipline"
    }
    data["discipline"] = pd.Categorical(["MS"] * rows)
    data["label"] = [index % 2 for index in range(rows)]
    data["match_time"] = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame(data)


class ModelSplittingTests(unittest.TestCase):
    def test_baselines_work_without_optional_xgboost(self) -> None:
        df = _feature_frame()
        expected = {"log_loss": 0.5}
        with patch("predict_badminton.model._split_folds", return_value=[(df, df)]), \
                patch("predict_badminton.model.baseline_elo_lr", return_value=expected), \
                patch.dict("sys.modules", {"xgboost": None}):
            self.assertEqual(run_baselines(df), {"baseline_elo_lr": expected})

    def test_calibration_rejects_missing_or_insufficient_oof_data(self) -> None:
        df = _feature_frame()
        for predictions, labels in ((None, None), (np.full(49, 0.5), np.zeros(49)),
                                    (np.full(60, 0.5), np.zeros(59)), (np.full(60, 0.5), None)):
            with patch("predict_badminton.model._train_lgbm_for_evaluation") as train:
                with self.assertRaisesRegex(ValueError, "walk-forward"):
                    train_final_model(df, df, oof_preds=predictions, oof_labels=labels)
                train.assert_not_called()

    def test_production_refit_uses_all_rows_with_fixed_iterations(self) -> None:
        df = _feature_frame()
        with patch("predict_badminton.model.train_lgbm") as fit:
            refit_production_model(df, 7, {"n_estimators": 500, "max_depth": 4})
        self.assertEqual(len(fit.call_args.args[0]), len(df))
        self.assertEqual(len(fit.call_args.args[1]), len(df))
        self.assertEqual(fit.call_args.kwargs["params"], {"n_estimators": 7, "max_depth": 4})
        self.assertEqual(len(fit.call_args.args), 2)

    def test_default_holdout_uses_latest_year(self) -> None:
        df = pd.DataFrame(
            {
                "match_time": pd.to_datetime(
                    ["2022-01-01", "2023-01-01", "2024-01-01", "2024-02-01"]
                ),
                "label": [0, 1, 0, 1],
            }
        )
        train, test, year = chronological_holdout_split(df)

        self.assertEqual(year, 2024)
        self.assertTrue((train["match_time"].dt.year < year).all())
        self.assertTrue((test["match_time"].dt.year >= year).all())
        self.assertEqual(
            _default_fold_val_years(df, max_folds=2),
            [(2023, 2023), (2024, 2024)],
        )

    def test_early_stopping_tail_is_chronologically_isolated(self) -> None:
        df = _feature_frame()
        fit, stop = _chronological_validation_split(df)

        self.assertLess(fit["match_time"].max(), stop["match_time"].min())

    def test_evaluation_training_refits_all_rows_after_internal_stopping(self) -> None:
        df = _feature_frame()
        provisional = SimpleNamespace(best_iteration_=7)
        final_model = object()

        with patch(
            "predict_badminton.model.train_lgbm",
            side_effect=[provisional, final_model],
        ) as mocked_train:
            model, best_iteration = _train_lgbm_for_evaluation(df)

        self.assertIs(model, final_model)
        self.assertEqual(best_iteration, 7)
        self.assertEqual(mocked_train.call_count, 2)
        self.assertEqual(len(mocked_train.call_args_list[1].args[0]), len(df))
        self.assertEqual(
            mocked_train.call_args_list[1].kwargs["params"]["n_estimators"], 7
        )


if __name__ == "__main__":
    unittest.main()
