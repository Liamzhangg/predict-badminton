"""Verify the actual artifact after restoring its original data-vintage cutoff."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import pickle

import run as r


def main():
    out, root = r.OUT, r.ROOT
    path = out / "report.json"
    report = json.loads(path.read_text())
    assert report["complete"], "Finish the primary replay before this diagnostic"
    assert not (out / "artifact-recheck.json").exists(), "Preserve previous diagnostics"
    reference = json.loads((out / "production-reference.json").read_text())
    directory = root / reference["directory"]
    assert r.p.s.e.forecast_bundle_directory() == directory
    assert r.p.s.e.sha(directory / "lgbm_model.pkl") == reference["model_sha256"]
    assert r.p.s.e.sha(directory / "feature_state.pkl") == reference["state_sha256"]
    metadata = json.loads((directory / "metrics_report.json").read_text())
    production = pickle.loads((directory / "lgbm_model.pkl").read_bytes())
    base = r.p.s.e.FeatureBuilder.load(directory / "feature_state.pkl")
    snapshot_cutoff = r.p.s.e.datetime.fromisoformat(metadata["production_refit"]["as_of"])
    locked = r.p.s.e.datetime.now(r.p.s.e.UTC).isoformat()
    r.p.s.save("artifact-recheck-protocol.json", {
        "locked_at_utc": locked,
        "reason": "Original saved receipt roster does not list later-ingested older matches, so it cannot by itself restore the old raw-file inventory.",
        "repair": "Use the existing historical observation database strictly before the original production snapshot. Later-known IDs are unobserved at that snapshot and excluded, without using winners to select records.",
        "verification": "Require original training-row count and numerical/booster reproduction of the actual saved model before candidate scoring. Same 500-tree model parameters and seed as production. Only strictly post-snapshot targets are evaluated.",
        "primary_gate": "Unchanged; this small retrospective diagnostic is not the primary replay or fresh forward validation.",
        "snapshot_cutoff": snapshot_cutoff.isoformat(),
    })
    historical = r.p.s.e.observations_before(snapshot_cutoff.replace(tzinfo=r.p.s.e.UTC), r.p.s.e.OBSERVATIONS_PATH)
    staging = out / "artifact-training-observation-snapshot"
    staging.mkdir(exist_ok=False)
    (staging / "observed_result_receipts.json").write_text(json.dumps(historical, indent=2, default=str)+"\n")
    frozen = r.p.s.e.datetime.fromisoformat(report["cutoff"])
    receipts = r.p.s.e.observations_before(frozen, r.p.s.e.OBSERVATIONS_PATH)
    matches, _, names = r.p.s.e.load_all_matches(root / "data/raw/bwf_matches", observed_results=receipts,
                                                category_catalog=base.category_catalog)
    matches = r.eligible(matches, frozen.replace(tzinfo=None))
    assert len(matches) == report["matches"]
    prefix = [m for m in matches if m["prediction_time"] < snapshot_cutoff and m["available_at"] < snapshot_cutoff]
    assert len(prefix) == metadata["production_refit"]["rows"]
    prefix_digest = r.cohort_digest(prefix)
    print(f"Restoring actual artifact comparison from verified {len(prefix):,}-row observation roster", flush=True)
    history = r.make_history(base, names)
    frame = history.build_dataframe(matches, as_of=frozen.replace(tzinfo=None))
    locations = r.p.s.e.geography()
    frame = frame.merge(r.p.s.context_frame(matches, locations), on="fixture_id", validate="one_to_one", how="left")
    del matches, prefix, historical, receipts, names, history
    gc.collect()
    result = r.artifact_check(frame, base, production, metadata, staging, prefix_digest, locations)
    result.update(saved_receipts_used=False, historical_observation_snapshot_used=True,
                  receipt_source="Historical observation database strictly before original production snapshot",
                  initial_saved_receipt_reconstruction=report["actual_artifact"])
    r.p.s.save("artifact-recheck.json", result)
    primary_fields = ("annual", "quarterly", "validation", "annual_2026", "rolling_2026", "summaries", "paired", "gate")
    before = r.hashlib.sha256(json.dumps({name: report[name] for name in primary_fields}, sort_keys=True).encode()).hexdigest()
    report["actual_artifact"] = result
    report["artifact_recheck_primary_fields_sha256"] = before
    report["protected_files_unchanged"] = r.p.s.verify(json.loads((out / "protected-before.json").read_text()))
    after = r.hashlib.sha256(json.dumps({name: report[name] for name in primary_fields}, sort_keys=True).encode()).hexdigest()
    assert before == after
    r.p.s.save("report.json", report)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
