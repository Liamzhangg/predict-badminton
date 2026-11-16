"""Verify the approved default switch without rewriting any real forecast or bet."""
from contextlib import redirect_stdout
import csv
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

import predict
from predict_badminton.config import REPO_ROOT, forecast_bundle_directory
from predict_badminton.tournament import load_bundle

OUT = Path(__file__).resolve().parent
DIRECTORY = "outputs/runs/2026-11-16-receipt-quality-cleaned"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = json.loads((OUT / "promotion-before.json").read_text())
    expected = json.loads((OUT / "candidate-verification.json").read_text())
    assert forecast_bundle_directory() == REPO_ROOT / DIRECTORY
    previous = json.loads((REPO_ROOT / "outputs/forecast_bundle.previous.json").read_text())
    assert previous == before["previous_default"]
    model, _, _, history = load_bundle()
    assert len(model.feature_name_) == len(history.feature_columns) == 55
    assert sha(forecast_bundle_directory() / "lgbm_model.pkl") == expected["model_sha256"]
    assert sha(forecast_bundle_directory() / "feature_state.pkl") == expected["state_sha256"]
    metadata = json.loads((forecast_bundle_directory() / "metrics_report.json").read_text())
    assert metadata["production_refit"]["rows"] == 302934
    assert metadata["data_quality"]["quarantine_sha256"] == sha(REPO_ROOT / "data/quarantine/bwf_matches/records.csv")
    load_bundle(REPO_ROOT / previous["directory"])
    frozen = REPO_ROOT / "outputs/rounds/hong-kong-open-2026-20261116"
    with (frozen / "fixtures.csv").open(newline="") as handle:
        original = list(csv.DictReader(handle))
    fields = ["match_id", "match_time", "discipline", "team1_player_ids", "team2_player_ids",
              "team1_country", "team2_country", "tournament_name", "tournament_tier", "round_name",
              "tournament_id", "tournament_code", "team1_seed", "team2_seed"]
    now = datetime.now(UTC)
    fixtures = []
    for event in ("MS", "WS", "MD", "WD", "XD"):
        source = next(row for row in original if row["discipline"] == event)
        row = {field: source.get(field, "") for field in fields}
        row.update(match_id="synthetic-promotion-" + event, match_time=(now + timedelta(days=2)).isoformat())
        fixtures.append(row)
    fixtures.append({**fixtures[0], "match_id": "synthetic-promotion-unknown-country", "team1_country": "", "team2_country": ""})
    with tempfile.TemporaryDirectory(prefix="badminton-promotion-") as temporary:
        root = Path(temporary)
        path = root / "synthetic-fixtures.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(fixtures)
        with patch("sys.argv", ["predict.py", "--fixtures-csv", str(path), "--output-dir", str(root / "predictions")]), redirect_stdout(io.StringIO()):
            assert predict.main() == 0
        exported = pd.read_csv(next((root / "predictions").glob("*.csv")))
        assert len(exported) == 6 and set(exported["discipline"]) == {"MS", "WS", "MD", "WD", "XD"}
        assert exported["model_sha256"].eq(expected["model_sha256"]).all()
        assert exported["base_state_sha256"].eq(expected["state_sha256"]).all()
        assert exported["probability_method"].eq("raw").all() and exported["history_refreshed"].all()
        assert exported["team1_win_prob"].between(0, 1).all()
        np.testing.assert_allclose(exported["team1_win_prob"] + exported["team2_win_prob"], 1, rtol=0, atol=1e-12)
    changed = [path for path, digest in before["frozen_sha256"].items() if not (REPO_ROOT / path).is_file() or sha(REPO_ROOT / path) != digest]
    assert not changed, changed
    comparison = json.loads((OUT / "comparison.json").read_text())["primary_quarterly"]["all_2026"]
    record = {"promoted_at_utc": datetime.now(UTC).isoformat(), "authorization": "Explicit user request after reviewing the cleaned-model comparison",
              "default_directory": DIRECTORY, "previous_directory": previous["directory"],
              "default_pointer": "outputs/forecast_bundle.json", "rollback_pointer": "outputs/forecast_bundle.previous.json",
              "older_observed_backup_unchanged": True, "model_sha256": expected["model_sha256"], "state_sha256": expected["state_sha256"],
              "training_rows": 302934, "features": 55, "default_bundle_loader_verified": True,
              "synthetic_cli_predictions": 6, "synthetic_predictions_removed": True, "default_cli_used_cleaned_model": True,
              "fresh_history_verified": True, "all_disciplines_and_unknown_countries_verified": True,
              "frozen_files_checked": len(before["frozen_sha256"]), "frozen_files_changed": changed,
              "hong_kong_forecasts_odds_and_bets_unchanged": True, "existing_model_binaries_unchanged": True,
              "hong_kong_forecast_sha256": sha(frozen / "forecast.json"), "ledger_sha256": sha(REPO_ROOT / "outputs/profits.json"),
              "comparison_stage_default_unchanged": True, "promotion_is_separate_approved_action": True,
              "historical_replay": {"n": comparison["n"], "current_control_accuracy": comparison["current_recipe"]["mean"]["accuracy"],
                                    "cleaned_accuracy": comparison["cleaned_recipe"]["mean"]["accuracy"],
                                    "kind": "reused historical recipe replay, not all-data artifact evaluation or proof of a performance gain"}}
    with (OUT / "promotion.json").open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
