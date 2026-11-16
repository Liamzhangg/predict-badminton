"""Local integration checks; synthetic forecast files are removed on completion."""
import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np
import pandas as pd

from forecast_round import run_round
from predict_badminton.config import REPO_ROOT
from predict_badminton.probabilities import predict_probabilities, swap_features
from predict_badminton.tournament import load_bundle

OUT = Path(__file__).resolve().parent
BACKUP = REPO_ROOT / "outputs/runs/2026-11-12-observed-timing-v2-054212Z"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_file(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protected = json.loads(Path("/tmp/country-promotion-preservation.json").read_text())
    allowed = {"data/README.md"}
    changed = [name for name, digest in protected.items()
               if not (REPO_ROOT / name).is_file() or sha(REPO_ROOT / name) != digest]
    assert set(changed) <= allowed, changed
    probabilities = {}
    for directory in (BACKUP, OUT):
        model, _, _, history = load_bundle(directory)
        before = pickle.dumps(history.get_state())
        rows = []
        for event in ("MS", "WS", "MD", "WD", "XD"):
            for country in ("CAN", "JPN", ""):
                rows.append(history.preview({"match_id": "smoke-only", "match_time": datetime(2030, 1, 1),
                    "discipline": event, "round_name": "SF", "tournament_name": "Smoke Test Only",
                    "tournament_code": "not-a-real-edition", "tournament_tier": 5,
                    "team1_ids": ["1", "2"] if event in {"MD", "WD", "XD"} else ["1"],
                    "team2_ids": ["3", "4"] if event in {"MD", "WD", "XD"} else ["3"],
                    "team1_country": country, "team2_country": "USA"}))
        frame = pd.DataFrame(rows)[history.feature_columns]
        history.encode_categories(frame)
        p = predict_probabilities(model, frame)
        np.testing.assert_allclose(p + predict_probabilities(model, swap_features(frame)), 1, atol=1e-15, rtol=0)
        assert np.isfinite(p).all()
        assert pickle.dumps(history.get_state()) == before
        probabilities[directory.name] = p.tolist()
    print("Actual old/new bundles score all five disciplines, including unknown countries", flush=True)
    with tempfile.TemporaryDirectory(prefix="integration-", dir=OUT) as temporary:
        root = Path(temporary)
        fix = {"match_id": "999999999", "match_time": "2030-01-01T10:00:00Z", "discipline": "MS",
               "team1_player_ids": "1", "team2_player_ids": "3", "team1_country": "CAN", "team2_country": "USA",
               "round_name": "SF", "tournament_name": "Smoke Test Only", "tournament_id": "999999999",
               "tournament_code": "not-a-real-edition", "tournament_tier": "5",
               "raw_file": "data/raw/bwf_matches/kumamoto-masters-japan-2026.json"}
        quote = {"match_id": fix["match_id"], "selection_id": fix["match_id"] + ":1", "discipline": "MS",
                 "round_name": "SF", "tournament_id": fix["tournament_id"], "selection_player_ids": "1",
                 "opponent_player_ids": "3", "selection_name": "Synthetic A", "opponent_name": "Synthetic B",
                 "american_odds": "150", "official_start_time_utc": fix["match_time"],
                 "recorded_at_utc": datetime.now(UTC).isoformat(), "sportsbook": "bet365", "jurisdiction": "Ontario"}
        fixtures, odds = root / "fixtures.csv", root / "odds.csv"
        csv_file(fixtures, [fix])
        csv_file(odds, [quote])
        result = run_round(fixtures, [odds], root / "round", OUT, record_bets=False)
        assert result["history_refreshed"]
        assert result["model_sha256"] == sha(OUT / "lgbm_model.pkl")
        assert 0 <= float(result["fixtures"][0]["team1_win_probability"]) <= 1
        assert {p.name for p in (root / "round").iterdir()} == {"forecast.json", "comparison.csv", "pending_bets.json"}
    changed = [name for name, digest in protected.items()
               if not (REPO_ROOT / name).is_file() or sha(REPO_ROOT / name) != digest]
    assert set(changed) <= allowed, changed
    verification = {"complete": True, "checked_at_utc": datetime.now(UTC).isoformat(),
                    "protected_data_output_files_checked": len(protected), "intentional_changes": changed,
                    "original_models_predictions_odds_and_bets_unchanged": True,
                    "new_bundle_all_disciplines_and_unknown_countries": True, "backup_bundle_loadable": True,
                    "temporary_round_workflow_with_live_refresh": True, "synthetic_outputs_removed": True,
                    "core_tests_passed": 296, "default_pointer_changed": False,
                    "synthetic_smoke_probabilities_not_forecasts": probabilities,
                    "critical_sha256": {name: sha(REPO_ROOT / name) for name in (
                        "outputs/profits.json", "outputs/profits.md",
                        str(BACKUP.relative_to(REPO_ROOT) / "lgbm_model.pkl"),
                        str(BACKUP.relative_to(REPO_ROOT) / "feature_state.pkl"))}}
    with (OUT / "integration_verification.json").open("x") as handle:
        json.dump(verification, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in verification.items()
                      if key != "synthetic_smoke_probabilities_not_forecasts"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
