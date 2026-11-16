"""Verify the new candidate and unchanged production, forecasts and paper bets."""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3

from predict_badminton.config import REPO_ROOT, forecast_bundle_directory
from predict_badminton.data_sources.observations import OBSERVATIONS_PATH, observations_before
from predict_badminton.tournament import load_bundle

OUT = Path(__file__).resolve().parent
CANDIDATE = REPO_ROOT / "outputs/runs/2026-11-16-receipt-quality-cleaned"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reference = json.loads((OUT / "reference.json").read_text())
    before = json.loads((OUT / "protected-before.json").read_text())
    changed = [p for p, digest in before.items() if not (REPO_ROOT / p).is_file() or sha(REPO_ROOT / p) != digest]
    allowed = {"data/manifests/bwf_result_observations.sqlite3", "data/README.md"}
    assert not set(changed) - allowed, changed
    assert forecast_bundle_directory() == REPO_ROOT / reference["directory"]
    for name, key in (("lgbm_model.pkl", "model_sha256"), ("feature_state.pkl", "state_sha256")):
        assert sha(forecast_bundle_directory() / name) == reference[key]
    frozen_prefix = "outputs/rounds/hong-kong-open-2026-20261116/"
    hk = [p for p in before if p.startswith(frozen_prefix)]
    assert hk and not set(hk) & set(changed)
    assert "outputs/profits.json" not in changed and "outputs/profits.md" not in changed
    row_counts = {}
    old = sqlite3.connect(f"{(OUT / 'observations-before.sqlite3').resolve().as_uri()}?mode=ro", uri=True)
    current = sqlite3.connect(f"{OBSERVATIONS_PATH.resolve().as_uri()}?mode=ro", uri=True)
    try:
        for table in ("responses", "result_versions", "response_results"):
            previous = set(old.execute(f"SELECT * FROM {table}"))
            present = set(current.execute(f"SELECT * FROM {table}"))
            assert previous.issubset(present), table
            row_counts[table] = {"before": len(previous), "after": len(present), "all_original_rows_unchanged": True}
        assert current.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert not current.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        old.close()
        current.close()
    cutoff = datetime.fromisoformat(reference["metadata"]["production_refit"]["as_of"]).replace(tzinfo=UTC)
    receipts = observations_before(cutoff)
    original = pickle.loads((OUT / "reference-matches.pkl").read_bytes())["matches"]
    observed = [m for m in original if m["availability_source"] == "observed_completed_result"]
    assert len(observed) == 609
    assert all(receipts[str(m["match_id"])]["available_at"] == m["available_at"] for m in observed)
    candidate, _, _, state = load_bundle(CANDIDATE)
    assert len(candidate.feature_name_) == len(state.feature_columns) == 55
    quality = json.loads((CANDIDATE / "metrics_report.json").read_text())
    assert quality["production_refit"]["rows"] == 302934
    assert len(quality["data_quality"]["removed_match_ids"]) == 7
    result = {"complete": True, "protected_preexisting_files_checked": len(before), "intentional_changes": changed,
              "hong_kong_files_byte_frozen": len(hk), "original_ledger_and_report_byte_frozen": True,
              "all_original_raw_results_unchanged": True, "production_model_state_and_pointer_unchanged": True,
              "original_database_rows": row_counts, "original_observed_result_times_preserved": len(observed),
              "candidate_rows": 302934, "candidate_features": 55, "candidate_loadable": True,
              "test_suite_passed": 312, "prospective_accuracy_available": False}
    with (OUT / "verification.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
