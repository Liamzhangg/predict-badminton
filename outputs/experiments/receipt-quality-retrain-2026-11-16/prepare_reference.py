"""Freeze the pre-fix production cohort and exact cached receipt evidence."""
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
import pickle
import sqlite3
import subprocess

from predict_badminton.config import REPO_ROOT, forecast_bundle_directory
from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import OBSERVATIONS_PATH, observations_before
from predict_badminton.features import EloEngine, FeatureBuilder
from predict_badminton.match_rules import standard_match_exclusion
from predict_badminton.tournament import load_bundle
from predict_badminton.trueskill import TrueSkillEngine

OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")


def main():
    logging.basicConfig(level=logging.WARNING)
    assert not (OUT / "reference-frame.pkl").exists()
    protected = {p for base in (REPO_ROOT / "data", REPO_ROOT / "outputs") for p in base.rglob("*")
                 if p.is_file() and not p.is_relative_to(OUT)}
    save("protected-before.json", {str(p.relative_to(REPO_ROOT)): sha(p) for p in sorted(protected)})
    directory = forecast_bundle_directory()
    model, _, _, base = load_bundle(directory)
    metadata = json.loads((directory / "metrics_report.json").read_text())
    cutoff = datetime.fromisoformat(metadata["production_refit"]["as_of"])
    receipts = observations_before(cutoff.replace(tzinfo=UTC))
    matches, drops, names = load_all_matches(REPO_ROOT / "data/raw/bwf_matches", observed_results=receipts,
                                            category_catalog=base.category_catalog)
    matches = [m for m in matches if not standard_match_exclusion(m) and m.get("match_status") in (None, "", "F")
               and m["available_at"] < cutoff and m["match_time"] < cutoff]
    cohort = sorted((m["fixture_id"], m["prediction_time"].isoformat(), m["available_at"].isoformat(), m["winner"]) for m in matches)
    assert len(matches) == metadata["production_refit"]["rows"]
    assert hashlib.sha256(json.dumps(cohort).encode()).hexdigest() == metadata["cohort_sha256"]
    with (OUT / "reference-matches.pkl").open("xb") as handle:
        pickle.dump({"matches": matches, "names": names}, handle)
    history = FeatureBuilder(player_names=names, result_availability_policy=base.model_policy["result_availability"],
                             category_catalog=base.category_catalog, country_context=base.country_context.empty_copy())
    history.elo = EloEngine(**base.model_policy["elo"])
    history.trueskill = TrueSkillEngine(**base.trueskill.parameters)
    print(f"Freezing causal pre-fix reference features for {len(matches):,} rows", flush=True)
    frame = history.build_dataframe(matches, as_of=cutoff)
    frame.to_pickle(OUT / "reference-frame.pkl")
    save("reference.json", {"directory": str(directory.relative_to(REPO_ROOT)), "metadata": metadata,
                            "parameters": model.get_params(), "model_sha256": sha(directory / "lgbm_model.pkl"),
                            "state_sha256": sha(directory / "feature_state.pkl"), "drops": drops,
                            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()})
    source = sqlite3.connect(f"{OBSERVATIONS_PATH.resolve().as_uri()}?mode=ro", uri=True)
    with sqlite3.connect(OUT / "observations-before.sqlite3") as backup:
        source.backup(backup)
    wanted = {row[0] for row in source.execute("SELECT DISTINCT response_sha256 FROM responses")}
    source.close()
    found = {}
    print(f"Checking cached response bytes for {len(wanted)} receipt hashes", flush=True)
    for path in sorted((REPO_ROOT / "data/source_cache/bwf").rglob("*")):
        if not path.is_file() or path.suffix not in (".json", ".html"):
            continue
        digest = sha(path)
        if digest in wanted:
            found.setdefault(digest, str(path.relative_to(REPO_ROOT)))
        if set(found) == wanted:
            break
    save("completion-source-index.json", {"found": found, "missing_response_hashes": sorted(wanted - set(found))})
    print(f"Reference frozen; exact cached bodies found for {len(found)}/{len(wanted)} hashes", flush=True)


if __name__ == "__main__":
    main()
