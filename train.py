#!/usr/bin/env python3
"""Badminton match winner prediction — training and evaluation pipeline.

Usage examples:
  # Full training run (all files, 30 Optuna trials)
  uv run python train.py

  # Quick smoke test (100 files, 5 Optuna trials)
  uv run python train.py --smoke-test --no-tune

  # Skip Optuna; use default LightGBM params
  uv run python train.py --no-tune

  # Custom run and trial count
  uv run python train.py --artifacts-dir outputs/runs/my-run --n-trials 50

  # Skip rolling backtest (faster)
  uv run python train.py --no-tune --no-backtest
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    from predict_badminton.data_sources.observations import OBSERVATIONS_PATH
    from predict_badminton.tournament_catalog import CATALOG_PATH

    p = argparse.ArgumentParser(description="Train badminton winner prediction model")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bwf_matches"),
        help="Directory with raw JSON tournament files",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Directory to save model artifacts (default: a new timestamped outputs/runs/ directory)",
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
    p.add_argument(
        "--include-para",
        action="store_true",
        help="Include para matches, which are excluded by default",
    )
    p.add_argument(
        "--test-year",
        type=int,
        default=None,
        help="Held-out test start year (default: latest year in loaded data)",
    )
    p.add_argument(
        "--refit-all", action="store_true",
        help="After holdout evaluation, refit the production model on all known results",
    )
    p.add_argument("--as-of", help="UTC snapshot cutoff; default: current UTC time")
    timing = p.add_mutually_exclusive_group()
    timing.add_argument("--observations-db", type=Path, default=OBSERVATIONS_PATH,
                        help="Result receipts database (default: data/manifests/bwf_result_observations.sqlite3)")
    timing.add_argument("--conservative-history", action="store_const", const=None, dest="observations_db",
                        help="Ignore result receipts and use conservative historical timing only")
    categories = p.add_mutually_exclusive_group()
    categories.add_argument("--category-catalog", type=Path, default=CATALOG_PATH,
                            help="Official edition categories (default: data/catalogs/bwf_tournaments.csv)")
    categories.add_argument("--legacy-tournament-tiers", action="store_const", const=None, dest="category_catalog",
                            help="Reproduce name-based tournament tiers instead of applying official World Tour categories")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke_test:
        args.max_files = args.max_files or 100
        args.n_trials = min(args.n_trials, 5)
        logger.info("SMOKE TEST MODE: max_files=%d, n_trials=%d", args.max_files, args.n_trials)

    from predict_badminton.data_sources.observations import OBSERVATIONS_PATH, observations_before
    from predict_badminton.tournament_catalog import CATALOG_PATH, CATEGORY_POLICY, TournamentCatalog

    artifacts_dir = args.artifacts_dir or Path("outputs/runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    observations_db = getattr(args, "observations_db", OBSERVATIONS_PATH)
    category_path = getattr(args, "category_catalog", CATALOG_PATH)
    if observations_db is not None or category_path is not None:
        from predict_badminton.config import REPO_ROOT

        if artifacts_dir.resolve() == REPO_ROOT / "outputs/artifacts":
            logger.error("New-policy training requires a separate --artifacts-dir; active artifacts are protected")
            return 1
    try:
        category_catalog = TournamentCatalog.load(category_path) if category_path is not None else None
    except (OSError, ValueError) as exc:
        logger.error("Cannot load official tournament categories: %s", exc)
        return 1
    as_of = pd.to_datetime(getattr(args, "as_of", None) or datetime.now(timezone.utc), utc=True).tz_localize(None).to_pydatetime()
    observed_results = None
    if observations_db is not None:
        if observations_db.exists():
            try:
                observed_results = observations_before(as_of.replace(tzinfo=timezone.utc), observations_db)
            except sqlite3.Error as exc:
                logger.error("Cannot read result receipts from %s: %s", observations_db, exc)
                return 1
        elif observations_db.resolve() == OBSERVATIONS_PATH.resolve():
            observed_results = {}
            logger.info("No result receipts database yet; using conservative timing for all history")
        else:
            logger.error("Result receipts database not found: %s", observations_db)
            return 1
        logger.info("Using observation receipts for %d tracked IDs, strictly before %s", len(observed_results), as_of)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if category_catalog is not None:
        (artifacts_dir / "tournament_catalog.csv").write_text(category_catalog.text, encoding="utf-8")
    if observed_results is not None:
        with (artifacts_dir / "observed_result_receipts.json").open("w") as fh:
            json.dump(observed_results, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # 1. Load and parse raw JSON
    # ------------------------------------------------------------------
    logger.info("Loading matches from %s …", args.data_dir)
    from predict_badminton.data_parser import load_all_matches

    matches, drop_counts, player_names = load_all_matches(
        args.data_dir,
        max_files=args.max_files,
        include_para=args.include_para,
        observed_results=observed_results,
        category_catalog=category_catalog,
    )
    from predict_badminton.match_rules import standard_match_exclusion

    supported = []
    for match in matches:
        if reason := standard_match_exclusion(match):
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
        else:
            supported.append(match)
    matches = supported
    logger.info("Loaded %d matches | dropped: %s", len(matches), drop_counts)
    if not matches:
        logger.error("No standard indoor MS/WS/MD/WD/XD matches remain after filtering")
        return 1
    known = [m for m in matches if m["available_at"] < as_of]
    drop_counts["result_not_yet_available"] = len(matches)-len(known)
    matches = known
    if not matches:
        logger.error("No results eligible before %s", as_of)
        return 1
    if len(matches) < 1000:
        logger.warning("Very few matches (%d) — results may be unreliable", len(matches))

    # ------------------------------------------------------------------
    # 2. Build feature matrix (returns builder for Elo state export)
    # ------------------------------------------------------------------
    logger.info("Building feature matrix …")
    from predict_badminton.features import build_feature_matrix

    from predict_badminton.temporal import AVAILABILITY_POLICY, OBSERVED_AVAILABILITY_POLICY

    df, feature_builder = build_feature_matrix(
        matches, player_names=player_names, as_of=as_of,
        result_availability_policy=OBSERVED_AVAILABILITY_POLICY if observations_db is not None else AVAILABILITY_POLICY,
        category_catalog=category_catalog,
    )
    logger.info("Feature matrix shape: %s", df.shape)
    logger.info("Label distribution: %.3f team1 wins", df["label"].mean())

    df.head(10000).to_csv(artifacts_dir / "features_sample.csv", index=False)
    logger.info("Feature matrix sample saved → %s/features_sample.csv", artifacts_dir)

    # ------------------------------------------------------------------
    # 3. Chronological train/test split
    # ------------------------------------------------------------------
    from predict_badminton.model import chronological_holdout_split

    try:
        train_df, test_df, test_year = chronological_holdout_split(
            df,
            test_year=args.test_year,
        )
    except ValueError as exc:
        logger.error("Cannot create chronological holdout: %s", exc)
        return 1
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

    fold_metrics, oof_preds, oof_labels = walk_forward_cv(
        train_df, n_trials=None if args.no_tune else args.n_trials,
    )

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

    model, platt_cal, iso_cal, final_report = train_final_model(
        train_df,
        test_df,
        best_params=best_params,
        oof_preds=oof_preds,
        oof_labels=oof_labels,
    )
    for fitted in (model, platt_cal, iso_cal):
        fitted.feature_policy_ = feature_builder.model_policy

    # ------------------------------------------------------------------
    # 8. Rolling backtest (Part E)
    # ------------------------------------------------------------------
    backtest_report: dict = {}
    if not args.no_backtest and not args.smoke_test:
        logger.info("Running rolling backtest …")
        from predict_badminton.model import rolling_backtest

        backtest_report = rolling_backtest(df, n_trials=None if args.no_tune else args.n_trials)
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

    production_refit = None
    if args.refit_all:
        from predict_badminton.config import FEATURE_COLS
        from predict_badminton.evaluate import feature_importance_df
        from predict_badminton.model import refit_production_model

        with (artifacts_dir / "holdout_lgbm_model.pkl").open("wb") as fh:
            pickle.dump(model, fh)
        logger.info("Refitting production model on all %d known results", len(df))
        model = refit_production_model(df, final_report["best_iteration"], best_params)
        model.feature_policy_ = feature_builder.model_policy
        final_report["feature_importance"] = feature_importance_df(model, FEATURE_COLS).to_dict("records")
        production_refit = {
            "rows": len(df), "through": df["match_time"].max().isoformat(),
            "as_of": feature_builder.history_cutoff.isoformat(),
            "best_iteration": final_report["best_iteration"],
            "evaluation_model": "holdout_lgbm_model.pkl",
            "note": "Test metrics describe the pre-refit holdout model, not the all-results production model.",
        }

    # ------------------------------------------------------------------
    # 9. Save artifacts
    # ------------------------------------------------------------------
    logger.info("Saving artifacts …")
    feature_builder.export_leaderboards(artifacts_dir)
    feature_builder.save(artifacts_dir / "feature_state.pkl")

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
        "model_approach": "LightGBM + score-sensitive Elo + player-based TrueSkill",
        "feature_policy": feature_builder.model_policy,
        "history_as_of": feature_builder.history_cutoff.isoformat(),
        "availability_sources": df["availability_source"].value_counts().to_dict(),
        "tournament_categories": {"policy": CATEGORY_POLICY, "snapshot": "tournament_catalog.csv",
                                  "catalog_sha256": category_catalog.sha256,
                                  "official_world_tour_matches": sum(m.get("tournament_tier_source") == "official_world_tour" for m in matches),
                                  "heuristic_fallback_matches": sum(m.get("tournament_tier_source") != "official_world_tour" for m in matches)}
                                 if category_catalog is not None else None,
        "observed_results": {"database": str(observations_db), "receipts": "observed_result_receipts.json"}
                            if observations_db is not None else None,
        "evaluation_status": "reused_retrospective_benchmark",
        "parameter_selection": "fixed_defaults" if args.no_tune else "nested_training_only",
        "score_margin_coverage": {
            "eligible": sum(match["score_margin"] is not None for match in matches),
            "matches": len(matches),
        },
        "trueskill_parameters": feature_builder.model_policy["trueskill"],
        "cv_fold_metrics": fold_metrics,
        "cv_summary": cv_summary,
        "cv_prediction_policy": final_report["prediction_policy"],
        "baselines": baseline_results,
        "baseline_prediction_policy": "single_order",
        "best_params": best_params,
        **{k: v for k, v in final_report.items() if k != "feature_importance"},
        "drop_counts": drop_counts,
        "data_shape": list(df.shape),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "test_year": test_year,
        "holdout_purged_rows": int((df["prediction_time"].dt.year < test_year).sum())-len(train_df),
        "label_balance": float(df["label"].mean()),
        "production_refit": production_refit,
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
    print(f"Feature state:      {(artifacts_dir / 'feature_state.pkl').resolve()}")
    print(f"Leaderboards:       {artifacts_dir.resolve()}/elo_leaderboard_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
