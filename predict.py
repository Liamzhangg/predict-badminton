#!/usr/bin/env python3
"""Future match prediction CLI.

Usage:
  python predict.py --fixtures-csv data/upcoming_fixtures.csv
  python predict.py --fixtures-csv data/upcoming_fixtures.csv \\
      --model-path outputs/artifacts/lgbm_model.pkl \\
      --calibrator-path outputs/artifacts/isotonic_calibrator.pkl \\
      --state-path outputs/artifacts/elo_state.pkl

Output:
  outputs/predictions/predictions_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score future badminton fixtures")
    p.add_argument("--fixtures-csv", type=Path, required=True)
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/artifacts/lgbm_model.pkl"),
    )
    p.add_argument(
        "--calibrator-path",
        type=Path,
        default=Path("outputs/artifacts/isotonic_calibrator.pkl"),
    )
    p.add_argument(
        "--state-path",
        type=Path,
        default=Path("outputs/artifacts/elo_state.pkl"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ---- Validate paths ------------------------------------------------
    for attr, path in [
        ("fixtures-csv", args.fixtures_csv),
        ("model-path", args.model_path),
        ("calibrator-path", args.calibrator_path),
        ("state-path", args.state_path),
    ]:
        if not path.exists():
            logger.error("File not found: %s  (%s)", path, attr)
            return 1

    # ---- Build inference features -------------------------------------
    logger.info("Building features from fixtures …")
    from predict_badminton.inference import build_inference_features

    feat_df, raw_fixtures = build_inference_features(
        fixtures_path=args.fixtures_csv,
        elo_state_path=args.state_path,
    )
    logger.info("Feature rows: %d", len(feat_df))

    # ---- Load model and calibrator ------------------------------------
    logger.info("Loading model from %s …", args.model_path)
    with open(args.model_path, "rb") as fh:
        model = pickle.load(fh)

    logger.info("Loading calibrator from %s …", args.calibrator_path)
    with open(args.calibrator_path, "rb") as fh:
        calibrator = pickle.load(fh)

    # ---- Score ---------------------------------------------------------
    from predict_badminton.config import FEATURE_COLS, DISCIPLINES

    X = feat_df[FEATURE_COLS].copy()

    # Ensure discipline is categorical (match training encoding)
    all_discs = DISCIPLINES + ["OTHER"]
    X["discipline"] = pd.Categorical(X["discipline"], categories=all_discs)

    raw_probs = model.predict_proba(X)[:, 1]
    cal_probs = calibrator.predict(raw_probs)

    # ---- Assemble output -----------------------------------------------
    out_df = pd.DataFrame({
        "match_id": feat_df["match_id"].values,
        "match_time": feat_df["match_time"].values,
        "team1_win_prob_raw": np.round(raw_probs, 4),
        "team1_win_prob_calibrated": np.round(cal_probs, 4),
        "predicted_winner": np.where(cal_probs >= 0.5, "team1", "team2"),
        "confidence": np.round(np.abs(cal_probs - 0.5) * 2, 4),
    })

    # Merge in team IDs if available
    for col in ("team1_player_ids", "team2_player_ids", "discipline", "round_name", "tournament_name"):
        if col in raw_fixtures.columns:
            out_df[col] = raw_fixtures[col].values

    # ---- Save ----------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"predictions_{ts}.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Predictions saved → %s", out_path)

    # Print summary
    print(f"\n=== PREDICTIONS ({len(out_df)} matches) ===")
    print(out_df[["match_id", "predicted_winner", "team1_win_prob_calibrated", "confidence"]].to_string(index=False))
    print(f"\nOutput: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
