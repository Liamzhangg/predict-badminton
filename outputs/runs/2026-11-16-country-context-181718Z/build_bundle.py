"""One-time, audited country-bundle refit. Does not select or replace a bundle."""
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from predict_badminton.config import COUNTRY_CONTEXT_COLS, COUNTRY_FEATURE_COLS, REPO_ROOT
from predict_badminton.country import CountryContext
from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import OBSERVATIONS_PATH, observations_before
from predict_badminton.features import EloEngine, FeatureBuilder
from predict_badminton.match_rules import standard_match_exclusion
from predict_badminton.probabilities import PREDICTION_POLICY, predict_probabilities, swap_features
from predict_badminton.temporal import OBSERVED_AVAILABILITY_POLICY
from predict_badminton.tournament import load_bundle
from predict_badminton.trueskill import TrueSkillEngine

OUT = Path(__file__).resolve().parent
BACKUP = REPO_ROOT / "outputs/runs/2026-11-12-observed-timing-v2-054212Z"
CUTOFF = datetime.fromisoformat("2026-11-16T18:17:18+00:00")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")


def main():
    started = time.perf_counter()
    logging.basicConfig(level=logging.WARNING)
    assert not (OUT / "lgbm_model.pkl").exists(), "Never overwrite a fitted bundle"
    base_model, _, _, base = load_bundle(BACKUP)
    # Compile the exact cached reference registry once; forecasts never import experiments.
    travel = module("travel_reference", REPO_ROOT / "outputs/experiments/travel-distance-2026-11-15/experiment.py")
    travel.save = lambda *args: None
    locations = travel.geography()
    coordinates, codes, hosts, named_hosts = locations
    assert all(value in coordinates for value in hosts.values())
    assert all(value in coordinates for value in named_hosts.values())
    receipts = observations_before(CUTOFF, OBSERVATIONS_PATH)
    matches, drops, names = load_all_matches(REPO_ROOT / "data/raw/bwf_matches", observed_results=receipts,
                                            category_catalog=base.category_catalog)
    assert not drops.get("file_error")
    cutoff = CUTOFF.replace(tzinfo=None)
    matches = [m for m in matches if not standard_match_exclusion(m)
               and m.get("match_status") in (None, "", "F")
               and m["available_at"] < cutoff and m["match_time"] < cutoff]
    vocabulary = sorted({str(m.get(f"team{side}_country") or "").strip().upper()
                         for m in matches for side in (1, 2)} & set(codes))
    country = CountryContext(countries=codes, located_countries=[key for key, value in codes.items() if value in coordinates],
                             host_codes=hosts, host_names=named_hosts, vocabulary=vocabulary)
    save("country_registry.json", country.registry)
    history = FeatureBuilder(player_names=names, result_availability_policy=OBSERVED_AVAILABILITY_POLICY,
                             category_catalog=base.category_catalog, country_context=country)
    history.elo = EloEngine(**base.model_policy["elo"])
    history.trueskill = TrueSkillEngine(**base.trueskill.parameters)
    print(f"Building causal 55-feature matrix for {len(matches):,} matches", flush=True)
    frame = history.build_dataframe(matches, as_of=cutoff)
    reference = module("country_reference", REPO_ROOT / "outputs/experiments/home-distance-controls-2026-11-15/study.py")
    expected = reference.context_frame(matches, locations).set_index("fixture_id").loc[frame["fixture_id"], COUNTRY_CONTEXT_COLS]
    actual = frame.set_index("fixture_id")[COUNTRY_CONTEXT_COLS]
    for name in COUNTRY_CONTEXT_COLS:
        if name.startswith("country_id_"):
            assert actual[name].astype(object).fillna("").tolist() == expected[name].fillna("").tolist(), name
        else:
            np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=0, equal_nan=True, err_msg=name)
    print("All country features exactly match the tested research replay", flush=True)
    params = {**base_model.get_params(), "n_jobs": 4}
    model = LGBMClassifier(**params)
    model.fit(frame[COUNTRY_FEATURE_COLS], frame["label"])
    model.feature_policy_ = history.model_policy
    model.trained_through_ = frame["available_at"].max().to_pydatetime()
    history.validate_model(model)
    with (OUT / "lgbm_model.pkl").open("xb") as handle:
        pickle.dump(model, handle)
    history.save(OUT / "feature_state.pkl")
    save("feature_list.json", COUNTRY_FEATURE_COLS)
    save("bundle_manifest.json", {"version": 1, "probability_methods": ["Raw"], "prediction_policy": PREDICTION_POLICY})
    save("observed_result_receipts.json", receipts)
    cohort = sorted((m["fixture_id"], m["prediction_time"].isoformat(), m["available_at"].isoformat(), m["winner"]) for m in matches)
    report = {"bundle": "country_elo_identity_counts", "backup_directory": str(BACKUP.relative_to(REPO_ROOT)),
              "production_refit": {"rows": len(frame), "as_of": cutoff.isoformat(),
                                   "through": model.trained_through_.isoformat(), "best_iteration": params["n_estimators"]},
              "feature_policy": history.model_policy, "model_parameters": params,
              "cohort_sha256": hashlib.sha256(json.dumps(cohort).encode()).hexdigest(),
              "country_feature_parity": {"rows": len(frame), "features": len(COUNTRY_CONTEXT_COLS), "exact": True},
              "evaluation": {"source": "outputs/experiments/country-vs-production-promotion-2026-11-16/report.json",
                             "kind": "prior five-seed historical quarterly replay, not evaluation of this all-data refit",
                             "n": 44179, "accuracy": 0.7774146087507638, "log_loss": 0.4549719888395691,
                             "brier": 0.15013093329450433, "prospective_forward_validation": False},
              "no_home_advantage_or_distance": True, "probability_method": "Raw", "prediction_policy": PREDICTION_POLICY,
              "elapsed_seconds": time.perf_counter() - started}
    save("metrics_report.json", report)
    loaded, _, _, restored = load_bundle(OUT)
    fixture = {**matches[-1], "match_id": "promotion-smoke-only", "match_time": cutoff + pd.Timedelta(days=1)}
    sample = pd.DataFrame([restored.preview(fixture)])[COUNTRY_FEATURE_COLS]
    restored.encode_categories(sample)
    probability = predict_probabilities(loaded, sample)
    np.testing.assert_allclose(probability + predict_probabilities(loaded, swap_features(sample)), 1, rtol=0, atol=1e-15)
    save("build_verification.json", {"complete": True, "reloaded_bundle": True,
                                    "country_features_exact": True, "side_order_complementary": True,
                                    "default_pointer_changed": False, "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
