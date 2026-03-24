"""Metrics, calibration, and reporting utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute primary and secondary classification metrics."""
    return {
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
    }


def metrics_summary(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate fold metrics into mean ± std."""
    keys = fold_metrics[0].keys()
    summary: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """Logistic regression (Platt scaling) calibrator."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(y_prob.reshape(-1, 1), y_true)
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibrator."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(y_prob, y_true)
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        return np.clip(self._iso.predict(y_prob), 0.0, 1.0)


def reliability_stats(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Compute reliability diagram data (fraction_of_positives, mean_predicted)."""
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "ece": ece,
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def feature_importance_df(
    model: Any, feature_names: list[str]
) -> pd.DataFrame:
    """Extract gain-based feature importance from a fitted LightGBM model."""
    importances = model.booster_.feature_importance(importance_type="gain")
    df = pd.DataFrame(
        {"feature": feature_names, "importance_gain": importances}
    ).sort_values("importance_gain", ascending=False)
    return df


def shap_importance(
    model: Any, X: pd.DataFrame, max_rows: int = 5000
) -> pd.DataFrame | None:
    """Optional SHAP mean |value| per feature. Returns None if shap not installed."""
    try:
        import shap  # type: ignore

        X_sample = X.iloc[:max_rows]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1 for binary
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return pd.DataFrame(
            {"feature": X.columns.tolist(), "shap_mean_abs": mean_abs}
        ).sort_values("shap_mean_abs", ascending=False)
    except ImportError:
        logger.info("shap not installed – skipping SHAP analysis")
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def save_metrics_report(
    report: dict[str, Any], artifacts_dir: Path
) -> None:
    out = artifacts_dir / "metrics_report.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Metrics report saved → %s", out)


def print_metrics_table(
    cv_summary: dict[str, dict[str, float]],
    test_raw: dict[str, float] | None = None,
    test_platt: dict[str, float] | None = None,
    test_isotonic: dict[str, float] | None = None,
) -> None:
    rows = []
    for metric, stats in cv_summary.items():
        rows.append(
            {
                "metric": metric,
                "cv_mean": f"{stats['mean']:.4f}",
                "cv_std": f"±{stats['std']:.4f}",
                "test_raw": f"{test_raw[metric]:.4f}" if test_raw else "-",
                "test_platt": f"{test_platt[metric]:.4f}" if test_platt else "-",
                "test_isotonic": f"{test_isotonic[metric]:.4f}" if test_isotonic else "-",
            }
        )
    df = pd.DataFrame(rows).set_index("metric")
    print("\n=== METRICS TABLE ===")
    print(df.to_string())
    print()
