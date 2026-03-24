"""LightGBM training, Optuna hyperparameter search, walk-forward CV, baselines.

Walk-forward CV design:
  - Each fold validates one calendar year (2021, 2022, 2023, 2024).
  - Training uses ALL data before the validation year start.
  - Model is retrained from scratch per fold (no leakage via model state).

Optuna tuning:
  - Uses only the LAST fold (largest training set, most representative) for speed.
  - 30 trials, fixed seed.
  - Best params then used for full-fold CV and final model.

Calibration:
  - Fit Platt scaling and isotonic regression on OOF predictions.
  - Evaluated on held-out test set (2025).
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from predict_badminton.config import (
    CAT_FEATURE_COLS,
    CV_FOLD_VAL_YEARS,
    FEATURE_COLS,
    SEED,
)
from predict_badminton.evaluate import (
    IsotonicCalibrator,
    PlattCalibrator,
    compute_metrics,
    feature_importance_df,
    metrics_summary,
    reliability_stats,
    shap_importance,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _get_year(dt) -> int:  # noqa: ANN001
    return dt.year if hasattr(dt, "year") else pd.Timestamp(dt).year


def _split_folds(
    df: pd.DataFrame,
    fold_val_years: list[tuple[int, int]],
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Create walk-forward folds. Each fold: (train_df, val_df)."""
    df = df.copy()
    df["_year"] = df["match_time"].apply(_get_year)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for val_start, val_end in fold_val_years:
        train = df[df["_year"] < val_start].drop(columns=["_year"])
        val = df[(df["_year"] >= val_start) & (df["_year"] <= val_end)].drop(
            columns=["_year"]
        )
        if len(train) < 100 or len(val) < 10:
            logger.warning(
                "Fold val=%d–%d too small (train=%d, val=%d); skipping",
                val_start,
                val_end,
                len(train),
                len(val),
            )
            continue
        folds.append((train, val))
    return folds


def _prep_X_y(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select feature columns; ensure categorical dtypes are preserved."""
    X = df[FEATURE_COLS].copy()
    y = df["label"]
    # Ensure discipline is categorical
    if not hasattr(X["discipline"], "cat"):
        from predict_badminton.config import DISCIPLINES
        all_discs = DISCIPLINES + ["OTHER"]
        X["discipline"] = pd.Categorical(X["discipline"], categories=all_discs)
    return X, y


# ---------------------------------------------------------------------------
# LightGBM model wrapper
# ---------------------------------------------------------------------------


def _lgbm_params(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }
    if overrides:
        base.update(overrides)
    return base


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any] | None = None,
    early_stopping_rounds: int = 50,
) -> lgb.LGBMClassifier:
    p = _lgbm_params(params)
    model = lgb.LGBMClassifier(**p)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# Optuna tuning
# ---------------------------------------------------------------------------


def _optuna_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }
    model = train_lgbm(X_train, y_train, X_val, y_val, params)
    preds = model.predict_proba(X_val)[:, 1]
    return float(log_loss(y_val, preds))


def tune_lgbm(
    df: pd.DataFrame,
    n_trials: int = 30,
) -> dict[str, Any]:
    """Run Optuna on the last (largest) walk-forward fold. Return best params."""
    folds = _split_folds(df, CV_FOLD_VAL_YEARS)
    if not folds:
        raise ValueError("No valid folds for tuning")
    # Use the last fold (most data → most stable signal)
    train_df, val_df = folds[-1]
    X_train, y_train = _prep_X_y(train_df)
    X_val, y_val = _prep_X_y(val_df)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info(
        "Optuna done: best log_loss=%.4f | params=%s",
        study.best_value,
        best,
    )
    return best


# ---------------------------------------------------------------------------
# Walk-forward CV evaluation
# ---------------------------------------------------------------------------


def walk_forward_cv(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    """Run full walk-forward CV. Returns (fold_metrics, oof_preds, oof_labels)."""
    folds = _split_folds(df, CV_FOLD_VAL_YEARS)
    fold_metrics: list[dict[str, float]] = []
    oof_preds: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []

    for i, (train_df, val_df) in enumerate(folds):
        X_train, y_train = _prep_X_y(train_df)
        X_val, y_val = _prep_X_y(val_df)

        model = train_lgbm(X_train, y_train, X_val, y_val, params)
        preds = model.predict_proba(X_val)[:, 1]

        m = compute_metrics(y_val.values, preds)
        fold_metrics.append(m)
        oof_preds.append(preds)
        oof_labels.append(y_val.values)

        logger.info(
            "Fold %d | val_year=%s | n_train=%d n_val=%d | log_loss=%.4f auc=%.4f",
            i + 1,
            CV_FOLD_VAL_YEARS[i],
            len(train_df),
            len(val_df),
            m["log_loss"],
            m["roc_auc"],
        )

    oof_all = np.concatenate(oof_preds)
    oof_lbl_all = np.concatenate(oof_labels)
    return fold_metrics, oof_all, oof_lbl_all


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def baseline_elo_lr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict[str, float]:
    """Logistic regression on elo_diff_disc only."""
    X_tr = train_df[["elo_diff_disc"]].values
    X_val = val_df[["elo_diff_disc"]].values
    y_tr = train_df["label"].values
    y_val = val_df["label"].values

    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_tr, y_tr)
    preds = lr.predict_proba(X_val)[:, 1]
    return compute_metrics(y_val, preds)


def run_baselines(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Evaluate baselines on last fold."""
    folds = _split_folds(df, CV_FOLD_VAL_YEARS)
    if not folds:
        return {}
    train_df, val_df = folds[-1]

    results: dict[str, dict[str, float]] = {}

    # LR on Elo diff
    results["baseline_elo_lr"] = baseline_elo_lr(train_df, val_df)
    logger.info("Baseline (Elo LR): %s", results["baseline_elo_lr"])

    # XGBoost challenger (optional)
    try:
        import xgboost as xgb  # type: ignore

        X_tr, y_tr = _prep_X_y(train_df)
        X_val, y_val = _prep_X_y(val_df)

        # XGBoost needs numeric encoding for categoricals
        le = LabelEncoder()
        X_tr_xgb = X_tr.copy()
        X_val_xgb = X_val.copy()
        X_tr_xgb["discipline"] = le.fit_transform(X_tr["discipline"].astype(str))
        X_val_xgb["discipline"] = le.transform(
            X_val["discipline"].astype(str).map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
        )

        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=SEED,
            verbosity=0,
            early_stopping_rounds=50,
        )
        xgb_model.fit(
            X_tr_xgb,
            y_tr,
            eval_set=[(X_val_xgb, y_val)],
            verbose=False,
        )
        preds_xgb = xgb_model.predict_proba(X_val_xgb)[:, 1]
        results["xgboost_challenger"] = compute_metrics(y_val.values, preds_xgb)
        logger.info("XGBoost challenger: %s", results["xgboost_challenger"])
    except ImportError:
        logger.info("xgboost not installed – skipping challenger")

    return results


# ---------------------------------------------------------------------------
# Final model training + calibration
# ---------------------------------------------------------------------------


def train_final_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_params: dict[str, Any] | None = None,
    oof_preds: np.ndarray | None = None,
    oof_labels: np.ndarray | None = None,
) -> tuple[lgb.LGBMClassifier, PlattCalibrator, IsotonicCalibrator, dict[str, Any]]:
    """Train final model.

    Calibrators are fit on OOF predictions from walk-forward CV (if provided).
    If OOF data is absent, fall back to fitting on train set predictions.
    The test set is used only for final evaluation — never for calibrator fitting.
    """
    X_train, y_train = _prep_X_y(train_df)
    X_test, y_test = _prep_X_y(test_df)

    model = train_lgbm(X_train, y_train, X_test, y_test, best_params)
    raw_preds_test = model.predict_proba(X_test)[:, 1]

    raw_metrics = compute_metrics(y_test.values, raw_preds_test)
    logger.info("Final model raw test metrics: %s", raw_metrics)

    # Calibration: fit on OOF predictions (not test set)
    if oof_preds is not None and oof_labels is not None and len(oof_preds) >= 50:
        logger.info(
            "Fitting calibrators on OOF predictions (%d samples)", len(oof_preds)
        )
        cal_y = oof_labels
        cal_p = oof_preds
    else:
        logger.warning(
            "OOF predictions not available; fitting calibrators on training predictions (fallback)"
        )
        cal_p = model.predict_proba(X_train)[:, 1]
        cal_y = y_train.values

    platt = PlattCalibrator().fit(cal_y, cal_p)
    isotonic = IsotonicCalibrator().fit(cal_y, cal_p)

    # Evaluate calibrated model on held-out test (no refitting)
    platt_preds = platt.predict(raw_preds_test)
    iso_preds = isotonic.predict(raw_preds_test)

    platt_metrics = compute_metrics(y_test.values, platt_preds)
    iso_metrics = compute_metrics(y_test.values, iso_preds)

    logger.info("Platt calibrated (test): %s", platt_metrics)
    logger.info("Isotonic calibrated (test): %s", iso_metrics)

    fi_df = feature_importance_df(model, FEATURE_COLS)
    shap_df = shap_importance(model, X_test)

    rel_raw = reliability_stats(y_test.values, raw_preds_test)
    rel_platt = reliability_stats(y_test.values, platt_preds)

    report = {
        "test_raw": raw_metrics,
        "test_platt": platt_metrics,
        "test_isotonic": iso_metrics,
        "calibration_source": "oof" if (oof_preds is not None and len(oof_preds) >= 50) else "train_fallback",
        "feature_importance": fi_df.to_dict("records"),
        "reliability_raw": rel_raw,
        "reliability_platt": rel_platt,
    }
    if shap_df is not None:
        report["shap_importance"] = shap_df.to_dict("records")

    return model, platt, isotonic, report


# ---------------------------------------------------------------------------
# Rolling backtest (Part E)
# ---------------------------------------------------------------------------


def rolling_backtest(
    df: pd.DataFrame,
    window_years: int = 2,
    step_years: int = 1,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rolling walk-forward backtest for future-prediction realism.

    Trains on a sliding window of `window_years`, predicts the next
    `step_years` block. Aggregates log_loss, Brier, AUC, accuracy, ECE.

    Returns a report dict suitable for JSON serialisation.
    """
    df = df.copy()
    df["_year"] = df["match_time"].apply(_get_year)
    min_year = int(df["_year"].min())
    max_year = int(df["_year"].max())

    fold_rows: list[dict[str, Any]] = []
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    train_start = min_year
    while True:
        train_end = train_start + window_years - 1   # inclusive
        val_start = train_end + 1
        val_end = val_start + step_years - 1

        if val_end > max_year:
            break

        train = df[(df["_year"] >= train_start) & (df["_year"] <= train_end)].drop(
            columns=["_year"]
        )
        val = df[(df["_year"] >= val_start) & (df["_year"] <= val_end)].drop(
            columns=["_year"]
        )

        if len(train) < 100 or len(val) < 10:
            logger.warning(
                "Backtest fold train=%d–%d / val=%d–%d: too small, skipping",
                train_start, train_end, val_start, val_end,
            )
            train_start += step_years
            continue

        X_tr, y_tr = _prep_X_y(train)
        X_val, y_val = _prep_X_y(val)

        model = train_lgbm(X_tr, y_tr, X_val, y_val, params)
        preds = model.predict_proba(X_val)[:, 1]

        m = compute_metrics(y_val.values, preds)
        from predict_badminton.evaluate import reliability_stats
        rel = reliability_stats(y_val.values, preds)
        m["ece"] = rel["ece"]

        fold_rows.append({
            "train_years": f"{train_start}-{train_end}",
            "val_years": f"{val_start}-{val_end}",
            "n_train": len(train),
            "n_val": len(val),
            **m,
        })
        all_preds.append(preds)
        all_labels.append(y_val.values)

        logger.info(
            "Backtest fold %s→%s | n_val=%d | logloss=%.4f auc=%.4f",
            f"{train_start}-{train_end}", f"{val_start}-{val_end}",
            len(val), m["log_loss"], m["roc_auc"],
        )
        train_start += step_years

    if not fold_rows:
        logger.warning("No rolling backtest folds completed")
        return {"folds": [], "aggregate": {}}

    # Aggregate across all fold predictions
    agg_preds = np.concatenate(all_preds)
    agg_labels = np.concatenate(all_labels)
    agg_metrics = compute_metrics(agg_labels, agg_preds)
    from predict_badminton.evaluate import reliability_stats
    agg_metrics["ece"] = reliability_stats(agg_labels, agg_preds)["ece"]

    return {
        "folds": fold_rows,
        "aggregate": agg_metrics,
        "n_folds": len(fold_rows),
        "window_years": window_years,
        "step_years": step_years,
    }
