#!/usr/bin/env python3
"""Badminton match winner prediction — training and evaluation pipeline.

Usage examples:
  # Full training run (all files, 30 Optuna trials)
  python train.py

  # Quick smoke test (100 files, 5 Optuna trials)
  python train.py --smoke-test --no-tune

  # Skip Optuna; use default LightGBM params
  python train.py --no-tune

  # Custom paths and trial count
  python train.py --data-dir data/raw_matches --artifacts-dir outputs/artifacts --n-trials 50

  # Skip rolling backtest (faster)
  python train.py --no-tune --no-backtest
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train badminton winner prediction model")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw_matches"),
        help="Directory with raw JSON tournament files",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("outputs/artifacts"),
        help="Directory to save model artifacts",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for LightGBM tuning",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick run: limit to 100 files and 5 Optuna trials",
    )
    p.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip Optuna; use default LightGBM hyperparameters",
    )
    p.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip rolling backtest (saves time)",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of JSON files loaded (for debugging)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke_test:
        args.max_files = args.max_files or 100
        args.n_trials = min(args.n_trials, 5)
        logger.info("SMOKE TEST MODE: max_files=%d, n_trials=%d", args.max_files, args.n_trials)

    artifacts_dir: Path = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and parse raw JSON
    # ------------------------------------------------------------------
    logger.info("Loading matches from %s …", args.data_dir)
    from predict_badminton.data_parser import load_all_matches

    matches, drop_counts, player_names = load_all_matches(args.data_dir, max_files=args.max_files)
    logger.info("Loaded %d matches | dropped: %s", len(matches), drop_counts)
    if len(matches) < 1000:
        logger.warning("Very few matches (%d) — results may be unreliable", len(matches))

    # ------------------------------------------------------------------
    # 2. Build feature matrix (returns builder for Elo state export)
    # ------------------------------------------------------------------
    logger.info("Building feature matrix …")
    from predict_badminton.features import build_feature_matrix

    df, feature_builder = build_feature_matrix(matches, player_names=player_names)
    logger.info("Feature matrix shape: %s", df.shape)
    logger.info("Label distribution: %.3f team1 wins", df["label"].mean())

    df.head(10000).to_csv(artifacts_dir / "features_sample.csv", index=False)
    logger.info("Feature matrix sample saved → %s/features_sample.csv", artifacts_dir)

    # ------------------------------------------------------------------
    # 2b. Export Elo leaderboards and Elo state (Part B)
    # ------------------------------------------------------------------
    logger.info("Exporting Elo leaderboards …")
    feature_builder.export_leaderboards(artifacts_dir)

    logger.info("Saving Elo state …")
    feature_builder.elo.save(artifacts_dir / "elo_state.pkl")

    # ------------------------------------------------------------------
    # 3. Train/test split: 2025 as final held-out test set
    # ------------------------------------------------------------------
    df["_year"] = df["match_time"].apply(lambda x: x.year)
    test_year = 2025
    train_df = df[df["_year"] < test_year].drop(columns=["_year"])
    test_df = df[df["_year"] >= test_year].drop(columns=["_year"])
    logger.info(
        "Train: %d rows (pre-%d) | Test: %d rows (%d+)",
        len(train_df), test_year, len(test_df), test_year,
    )

    if len(train_df) < 500:
        logger.warning(
            "Training set very small (%d) — smoke-test mode, results are illustrative only",
            len(train_df),
        )

    # ------------------------------------------------------------------
    # 4. Baseline models
    # ------------------------------------------------------------------
    logger.info("Running baselines …")
    from predict_badminton.model import run_baselines

    baseline_results = run_baselines(train_df)

    # ------------------------------------------------------------------
    # 5. Optuna hyperparameter tuning
    # ------------------------------------------------------------------
    best_params: dict | None = None
    if not args.no_tune:
        logger.info("Tuning LightGBM with Optuna (%d trials) …", args.n_trials)
        from predict_badminton.model import tune_lgbm

        best_params = tune_lgbm(train_df, n_trials=args.n_trials)
        logger.info("Best params: %s", best_params)
        with open(artifacts_dir / "best_params.json", "w") as fh:
            json.dump(best_params, fh, indent=2)
    else:
        logger.info("Skipping tuning (--no-tune)")

    # ------------------------------------------------------------------
    # 6. Walk-forward CV (produces OOF predictions for calibration)
    # ------------------------------------------------------------------
    logger.info("Running walk-forward CV …")
    from predict_badminton.model import walk_forward_cv
    from predict_badminton.evaluate import metrics_summary

    fold_metrics, oof_preds, oof_labels = walk_forward_cv(train_df, params=best_params)

    if not fold_metrics:
        logger.error("No CV folds completed — not enough data for chosen time range")
        return 1

    cv_summary = metrics_summary(fold_metrics)
    logger.info(
        "CV summary: %s",
        {k: f"{v['mean']:.4f}±{v['std']:.4f}" for k, v in cv_summary.items()},
    )

    # ------------------------------------------------------------------
    # 7. Final model (calibrated on OOF, evaluated on test)
    # ------------------------------------------------------------------
    logger.info("Training final model …")
    from predict_badminton.model import train_final_model

    if len(test_df) < 50:
        logger.warning(
            "Test set has only %d rows — calibration/test metrics may be noisy", len(test_df)
        )
        from predict_badminton.model import _split_folds
        from predict_badminton.config import CV_FOLD_VAL_YEARS
        folds = _split_folds(train_df, CV_FOLD_VAL_YEARS)
        if folds:
            _, test_df = folds[-1]
            logger.info("Using last CV fold val set as test (%d rows)", len(test_df))

    model, platt_cal, iso_cal, final_report = train_final_model(
        train_df,
        test_df,
        best_params=best_params,
        oof_preds=oof_preds,
        oof_labels=oof_labels,
    )

    # ------------------------------------------------------------------
    # 8. Rolling backtest (Part E)
    # ------------------------------------------------------------------
    backtest_report: dict = {}
    if not args.no_backtest and not args.smoke_test:
        logger.info("Running rolling backtest …")
        from predict_badminton.model import rolling_backtest

        backtest_report = rolling_backtest(df.drop(columns=["_year"], errors="ignore"), params=best_params)
        bt_path = artifacts_dir / "rolling_backtest_report.json"
        with open(bt_path, "w") as fh:
            json.dump(backtest_report, fh, indent=2, default=str)
        logger.info("Rolling backtest saved → %s", bt_path)

        # Optional fold-level CSV
        if backtest_report.get("folds"):
            pd.DataFrame(backtest_report["folds"]).to_csv(
                artifacts_dir / "rolling_backtest_folds.csv", index=False
            )
    else:
        logger.info("Skipping rolling backtest")

    # ------------------------------------------------------------------
    # 9. Save artifacts
    # ------------------------------------------------------------------
    logger.info("Saving artifacts …")

    with open(artifacts_dir / "lgbm_model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open(artifacts_dir / "platt_calibrator.pkl", "wb") as fh:
        pickle.dump(platt_cal, fh)
    with open(artifacts_dir / "isotonic_calibrator.pkl", "wb") as fh:
        pickle.dump(iso_cal, fh)

    from predict_badminton.config import FEATURE_COLS
    with open(artifacts_dir / "feature_list.json", "w") as fh:
        json.dump(FEATURE_COLS, fh, indent=2)

    fi_df = pd.DataFrame(final_report["feature_importance"])
    fi_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    full_report = {
        "cv_fold_metrics": fold_metrics,
        "cv_summary": cv_summary,
        "baselines": baseline_results,
        "best_params": best_params,
        **{k: v for k, v in final_report.items() if k != "feature_importance"},
        "drop_counts": drop_counts,
        "data_shape": list(df.shape),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "label_balance": float(df["label"].mean()),
    }
    if backtest_report:
        full_report["rolling_backtest_aggregate"] = backtest_report.get("aggregate", {})

    from predict_badminton.evaluate import save_metrics_report
    save_metrics_report(full_report, artifacts_dir)

    # ------------------------------------------------------------------
    # 10. Print summary
    # ------------------------------------------------------------------
    from predict_badminton.evaluate import print_metrics_table

    print_metrics_table(
        cv_summary,
        test_raw=final_report.get("test_raw"),
        test_platt=final_report.get("test_platt"),
        test_isotonic=final_report.get("test_isotonic"),
    )

    print("=== BEST HYPERPARAMETERS ===")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        print("  (default params used)")

    print("\n=== TOP 10 FEATURES (gain) ===")
    for _, row in fi_df.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance_gain']:.1f}")

    print("\n=== BASELINES (last fold) ===")
    for name, m in baseline_results.items():
        print(f"  {name}: log_loss={m['log_loss']:.4f}  auc={m['roc_auc']:.4f}")

    if backtest_report.get("aggregate"):
        agg = backtest_report["aggregate"]
        print(f"\n=== ROLLING BACKTEST (aggregate, {backtest_report.get('n_folds')} folds) ===")
        print(f"  log_loss={agg.get('log_loss', 'N/A'):.4f}  auc={agg.get('roc_auc', 'N/A'):.4f}  "
              f"acc={agg.get('accuracy', 'N/A'):.4f}  ece={agg.get('ece', 'N/A'):.4f}")

    print(f"\nArtifacts saved to: {artifacts_dir.resolve()}")
    print(f"Elo state:          {(artifacts_dir / 'elo_state.pkl').resolve()}")
    print(f"Leaderboards:       {artifacts_dir.resolve()}/elo_leaderboard_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
