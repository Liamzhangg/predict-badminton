#!/usr/bin/env python3
"""Future match prediction CLI.

Usage:
  uv run python predict.py --fixtures-csv data/upcoming_fixtures.csv
  uv run python predict.py --fixtures-csv data/upcoming_fixtures.csv \\
      --model-path outputs/artifacts/lgbm_model.pkl \\
      --probabilities isotonic \\
      --calibrator-path outputs/artifacts/isotonic_calibrator.pkl \\
      --state-path outputs/artifacts/feature_state.pkl

Output:
  outputs/predictions/predictions_<timestamp>.csv
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

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
        "--probabilities", choices=("raw", "platt", "isotonic"), default="raw",
        help="Probability method; raw is the main model's default",
    )
    p.add_argument(
        "--calibrator-path",
        type=Path,
        help="Optional calibrator override when --probabilities is platt or isotonic",
    )
    p.add_argument(
        "--state-path",
        type=Path,
        default=Path("outputs/artifacts/feature_state.pkl"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
    )
    args = p.parse_args()
    if args.probabilities == "raw" and args.calibrator_path is not None:
        p.error("--calibrator-path requires --probabilities platt or isotonic")
    if args.probabilities != "raw" and args.calibrator_path is None:
        args.calibrator_path = args.model_path.parent / f"{args.probabilities}_calibrator.pkl"
    return args


def main() -> int:
    args = parse_args()

    # ---- Validate paths ------------------------------------------------
    for attr, path in [
        ("fixtures-csv", args.fixtures_csv),
        ("model-path", args.model_path),
        ("calibrator-path", args.calibrator_path),
        ("state-path", args.state_path),
    ]:
        if path is not None and not path.exists():
            logger.error("File not found: %s  (%s)", path, attr)
            return 1

    with open(args.model_path, "rb") as fh:
        model = pickle.load(fh)

    # ---- Build inference features -------------------------------------
    logger.info("Building features from fixtures …")
    from predict_badminton.inference import build_inference_features

    try:
        feat_df, raw_fixtures = build_inference_features(
            fixtures_path=args.fixtures_csv,
            state_path=args.state_path,
            model=model,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Feature rows: %d", len(feat_df))

    # ---- Score ---------------------------------------------------------
    from predict_badminton.config import DISCIPLINES

    columns = list(feat_df.columns[2:])
    if columns != list(model.feature_name_):
        logger.error("Model features differ from the saved history; use a matching bundle or retrain")
        return 1
    X = feat_df[columns].copy()

    # Ensure discipline is categorical (match training encoding)
    all_discs = DISCIPLINES + ["OTHER"]
    X["discipline"] = pd.Categorical(X["discipline"], categories=all_discs)

    raw_probs = model.predict_proba(X)[:, 1]
    probs = raw_probs
    if args.calibrator_path is not None:
        logger.info("Loading calibrator from %s …", args.calibrator_path)
        with open(args.calibrator_path, "rb") as fh:
            calibrator = pickle.load(fh)
        if getattr(calibrator, "feature_policy_", None) != getattr(model, "feature_policy_", None):
            logger.error("Calibrator rating policy differs from its model; use a matching bundle")
            return 1
        probs = calibrator.predict(raw_probs)

    # ---- Assemble output -----------------------------------------------
    out_df = pd.DataFrame({
        "match_id": feat_df["match_id"].values,
        "match_time": feat_df["match_time"].values,
        "team1_win_prob_raw": np.round(raw_probs, 4),
        "team1_win_prob": np.round(probs, 4),
        "probability_method": args.probabilities,
        "predicted_winner": np.where(probs >= 0.5, "team1", "team2"),
        "confidence": np.round(np.abs(probs - 0.5) * 2, 4),
    })
    if args.calibrator_path is not None:
        out_df["team1_win_prob_calibrated"] = np.round(probs, 4)

    # Merge in team IDs if available
    metadata_cols = [col for col in (
        "team1_player_ids", "team2_player_ids", "discipline", "round_name", "tournament_name"
    ) if col in raw_fixtures.columns]
    out_df = out_df.merge(raw_fixtures[["match_id", *metadata_cols]], on="match_id", validate="one_to_one")

    # ---- Save ----------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"predictions_{ts}.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Predictions saved → %s", out_path)

    # Print summary
    print(f"\n=== PREDICTIONS ({len(out_df)} matches) ===")
    print(out_df[["match_id", "predicted_winner", "team1_win_prob", "confidence"]].to_string(index=False))
    print(f"\nOutput: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
