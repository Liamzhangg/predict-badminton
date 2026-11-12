"""Order-consistent predictions for current and complete legacy model schemas."""
from __future__ import annotations

import numpy as np
import pandas as pd

from predict_badminton.config import FEATURE_COLS, GLICKO_MODEL_FEATURE_COLS, LEGACY_FEATURE_COLS

PREDICTION_POLICY = "two_order_average_v1"

_PAIRS = [
    ("elo_t1_disc", "elo_t2_disc"), ("elo_t1_global", "elo_t2_global"),
    ("elo_matches_t1_disc", "elo_matches_t2_disc"),
    ("pair_component_t1", "pair_component_t2"), ("pair_matches_t1", "pair_matches_t2"),
    ("form5_wr_t1", "form5_wr_t2"), ("form10_wr_t1", "form10_wr_t2"),
    ("form5_cnt_t1", "form5_cnt_t2"), ("form10_cnt_t1", "form10_cnt_t2"),
    ("rest_days_t1", "rest_days_t2"), ("h2h_wins_t1", "h2h_wins_t2"),
    ("team1_seeded", "team2_seeded"),
    *[(f"{engine}_{quantity}_t1", f"{engine}_{quantity}_t2")
      for engine, quantities in (("glicko", ("rating", "rd", "volatility")),
                                 ("trueskill", ("mu", "sigma"))) for quantity in quantities],
]
_DIFFERENCES = ["elo_diff_disc", "elo_diff_global", "pair_component_diff",
                "form5_wr_diff", "form10_wr_diff", "seed_diff",
                "glicko_rating_diff", "glicko_rd_diff", "glicko_volatility_diff",
                "trueskill_mu_diff", "trueskill_sigma_diff"]
_UNCHANGED = ["elo_reliability", "pair_reliability", "h2h_total", "tournament_tier",
              "round_importance", "same_country", "discipline"]


def swap_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reverse the opponents, preserving context and categorical types."""
    if list(frame.columns) not in (FEATURE_COLS, GLICKO_MODEL_FEATURE_COLS, LEGACY_FEATURE_COLS):
        raise ValueError("Unsupported or reordered model feature schema")
    pairs = [(first, second) for first, second in _PAIRS if first in frame]
    differences = [name for name in _DIFFERENCES if name in frame]
    covered = [name for pair in pairs for name in pair] + differences + _UNCHANGED + ["h2h_wr_t1"]
    if len(covered) != len(set(covered)) or set(covered) != set(frame.columns):
        raise ValueError("Model feature schema lacks a complete opponent-swap policy")
    swapped = frame.copy()
    for first, second in pairs:
        swapped[first], swapped[second] = frame[second], frame[first]
    swapped[differences] = -frame[differences]
    swapped["h2h_wr_t1"] = 1 - frame["h2h_wr_t1"]
    return swapped


def predict_directions(model, frame: pd.DataFrame) -> np.ndarray:
    """Return columns P(team1 wins), P(team2 wins in the reversed input)."""
    if list(model.feature_name_) != list(frame.columns):
        raise ValueError("Model features differ from the prediction frame")
    both = pd.concat([frame, swap_features(frame)], ignore_index=True)
    values = np.asarray(model.predict_proba(both), dtype=float)
    if values.shape != (2 * len(frame), 2):
        raise ValueError("Expected binary probabilities for both opponent orders")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Model probabilities must be finite and between zero and one")
    return np.column_stack((values[:len(frame), 1], values[len(frame):, 1]))


def average_directions(directions: np.ndarray, calibrator=None) -> np.ndarray:
    """Calibrate directional scores when requested, then enforce symmetry."""
    values = np.asarray(directions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected two directional probabilities per match")
    if calibrator is not None:
        calibrated = np.asarray(calibrator.predict(values.ravel()), dtype=float)
        if calibrated.shape != (values.size,):
            raise ValueError("Calibrator returned an incompatible probability shape")
        values = calibrated.reshape(values.shape)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Directional probabilities must be finite and between zero and one")
    # Centering keeps equal directional scores at exactly 50%, including floating-point ties.
    return 0.5 + (values[:, 0] - values[:, 1]) / 2


def predict_probabilities(model, frame: pd.DataFrame, calibrator=None) -> np.ndarray:
    """Return order-averaged P(team1 wins); the other side is its complement."""
    return average_directions(predict_directions(model, frame), calibrator)
