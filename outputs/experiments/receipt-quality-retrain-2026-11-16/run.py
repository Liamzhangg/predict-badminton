"""Paired cleaned-data replay and isolated refit; never change frozen bets."""
from datetime import UTC, datetime
import gc
import hashlib
import json
import logging
from pathlib import Path
import pickle
import re
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from predict_badminton.config import REPO_ROOT
from predict_badminton.country import CountryContext
from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import observations_before
from predict_badminton.evaluate import compute_metrics, reliability_stats
from predict_badminton.features import EloEngine, FeatureBuilder
from predict_badminton.inference import build_inference_features
from predict_badminton.match_rules import standard_match_exclusion
from predict_badminton.model import _chronological_validation_split, train_lgbm
from predict_badminton.probabilities import PREDICTION_POLICY, predict_probabilities
from predict_badminton.tournament import load_bundle
from predict_badminton.tournament_catalog import TournamentCatalog
from predict_badminton.trueskill import TrueSkillEngine

OUT = Path(__file__).resolve().parent
CANDIDATE = REPO_ROOT / "outputs/runs/2026-11-16-receipt-quality-cleaned"
IDENTITIES = ["country_id_t1", "country_id_t2"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value, directory=OUT):
    with (directory / name).open("x") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")


def history_for(base, names, vocabulary=None):
    registry = base.country_context.registry
    if vocabulary is not None:
        registry = {**registry, "vocabulary": vocabulary}
    history = FeatureBuilder(player_names=names, result_availability_policy=base.model_policy["result_availability"],
                             category_catalog=base.category_catalog, country_context=CountryContext(**registry))
    history.elo = EloEngine(**base.model_policy["elo"])
    history.trueskill = TrueSkillEngine(**base.trueskill.parameters)
    return history


def vocabulary(rows):
    return sorted(set(rows[IDENTITIES].astype(object).stack().dropna()))


def matrix(rows, columns, categories):
    value = rows[columns].copy()
    for name in IDENTITIES:
        raw = value[name].astype(object)
        value[name] = pd.Categorical(raw.where(raw.isin(categories)), categories=categories)
    return value


def fit_window(frame, targets, boundary, columns, parameters, seed, variant):
    stem = f"replay-{seed}-{boundary:%Y%m%d}-{variant}"
    array_path = OUT / f"{stem}.npz"
    metadata_path = OUT / f"{stem}.json"
    if array_path.is_file() and metadata_path.is_file():
        saved = np.load(array_path)
        assert saved["fixture_id"].tolist() == targets["fixture_id"].tolist()
        return saved["probability"], json.loads(metadata_path.read_text())
    began = time.perf_counter()
    train = frame.loc[(frame["prediction_time"] < boundary) & (frame["available_at"] < boundary)]
    assert not set(train["fixture_id"]) & set(targets["fixture_id"])
    fitting, stopping = _chronological_validation_split(train)
    params = {**parameters, "random_state": seed, "n_jobs": 4}
    categories = vocabulary(fitting)
    provisional = train_lgbm(matrix(fitting, columns, categories), fitting["label"],
                            matrix(stopping, columns, categories), stopping["label"], params)
    trees = int(provisional.best_iteration_ or params["n_estimators"])
    categories = vocabulary(train)
    model = train_lgbm(matrix(train, columns, categories), train["label"], params={**params, "n_estimators": trees})
    probability = predict_probabilities(model, matrix(targets, columns, categories))
    metadata = {"seed": seed, "variant": variant, "training_boundary": boundary.isoformat(),
                "train_n": len(train), "trees": trees, "target_n": len(targets),
                "fit_max_prediction_time": fitting["prediction_time"].max(),
                "fit_max_available_at": fitting["available_at"].max(),
                "stop_min_prediction_time": stopping["prediction_time"].min(),
                "seconds": time.perf_counter() - began}
    np.savez_compressed(array_path, fixture_id=targets["fixture_id"].to_numpy(dtype=str), probability=probability)
    save(metadata_path.name, metadata)
    print(f"{stem}: train={len(train):,} trees={trees} scored={len(targets):,} ({metadata['seconds']:.1f}s)", flush=True)
    del model, provisional, train, fitting, stopping
    gc.collect()
    return probability, metadata


def summarize(targets, probabilities, masks):
    labels = targets["label"].to_numpy()
    clusters = targets["tournament_code"].astype(str) + ":" + targets["prediction_time"].dt.year.astype(str)
    result = {}
    for number, (name, mask) in enumerate(masks.items()):
        truth = labels[mask]
        per_variant = {}
        for variant in ("current_recipe", "cleaned_recipe"):
            per_seed = [{**compute_metrics(truth, p[mask]), "ece": reliability_stats(truth, p[mask])["ece"]}
                        for p in probabilities[variant]]
            per_variant[variant] = {"mean": {key: float(np.mean([item[key] for item in per_seed])) for key in per_seed[0]},
                                    "per_seed": per_seed}
        current = probabilities["current_recipe"][:, mask]
        candidate = probabilities["cleaned_recipe"][:, mask]
        paired = np.stack((((candidate >= .5) == truth).astype(float) - ((current >= .5) == truth).astype(float),
                           -(truth * np.log(candidate) + (1-truth) * np.log1p(-candidate))
                           + truth * np.log(current) + (1-truth) * np.log1p(-current),
                           (candidate-truth)**2 - (current-truth)**2), axis=2)
        codes, unique = pd.factorize(clusters[mask], sort=True)
        counts = np.bincount(codes)
        sums = np.zeros((len(candidate), len(unique), 3))
        for seed in range(len(candidate)):
            for metric in range(3):
                sums[seed, :, metric] = np.bincount(codes, weights=paired[seed, :, metric])
        random = np.random.default_rng(20261116 + number)
        draws = []
        for _ in range(5000):
            sampled_clusters = random.integers(0, len(unique), len(unique))
            sampled_seeds = random.integers(0, len(candidate), len(candidate))
            draws.append(sums[sampled_seeds][:, sampled_clusters].sum(axis=1).mean(axis=0) / counts[sampled_clusters].sum())
        intervals = np.quantile(draws, [.025, .975], axis=0)
        result[name] = {"n": len(truth), "tournament_clusters": len(unique), **per_variant,
                        "cleaned_minus_current": {metric: {"mean": float(paired[:, :, i].mean()),
                                                           "crossed_ci95": intervals[:, i].tolist()}
                                                  for i, metric in enumerate(("accuracy", "log_loss", "brier_score"))}}
    return result


def main():
    logging.basicConfig(level=logging.WARNING)
    protocol = json.loads((OUT / "protocol.json").read_text())
    reference = json.loads((OUT / "reference.json").read_text())
    directory = REPO_ROOT / reference["directory"]
    production, _, _, base = load_bundle(directory)
    assert sha(directory / "lgbm_model.pkl") == reference["model_sha256"]
    cutoff = datetime.fromisoformat(protocol["as_of_utc"]).replace(tzinfo=None)
    old = pickle.loads((OUT / "reference-matches.pkl").read_bytes())
    current_frame = pd.read_pickle(OUT / "reference-frame.pkl")
    receipts = observations_before(cutoff.replace(tzinfo=UTC))
    matches, drops, names = load_all_matches(REPO_ROOT / "data/raw/bwf_matches", observed_results=receipts,
                                            category_catalog=base.category_catalog)
    matches = [m for m in matches if not standard_match_exclusion(m) and m.get("match_status") in (None, "", "F")
               and m["available_at"] < cutoff and m["match_time"] < cutoff]
    by_id = {m["fixture_id"]: m for m in matches}
    removed = [m for m in old["matches"] if m["fixture_id"] not in by_id]
    assert len(removed) == 7 and len(matches) == 302934
    fields = ("winner", "available_at", "prediction_time", "score_margin")
    assert all(all(m[key] == by_id[m["fixture_id"]][key] for key in fields)
               for m in old["matches"] if m["fixture_id"] in by_id)
    verification = {"reference_rows": len(old["matches"]), "cleaned_rows": len(matches), "drops": drops,
                    "removed_match_ids": sorted(str(m["match_id"]) for m in removed),
                    "retained_labels_times_and_score_margins_identical": True,
                    "receipt_snapshot": "Explicit final status required and fingerprinted; legacy receipts require exact cached-response proof",
                    "quarantine_sha256": sha(REPO_ROOT / "data/quarantine/bwf_matches/records.csv")}
    if not (OUT / "cohort-verification.json").exists():
        save("cohort-verification.json", verification)
    history = history_for(base, names)
    if (OUT / "cleaned-frame.pkl").exists():
        cleaned_frame = pd.read_pickle(OUT / "cleaned-frame.pkl")
    else:
        print(f"Building cleaned causal features for {len(matches):,} rows", flush=True)
        cleaned_frame = history.build_dataframe(matches, as_of=cutoff)
        cleaned_frame.to_pickle(OUT / "cleaned-frame.pkl")
    common = current_frame.loc[(current_frame["prediction_time"] >= pd.Timestamp("2026-01-01"))
                               & current_frame["fixture_id"].isin(by_id)].copy().reset_index(drop=True)
    clean_targets = cleaned_frame.set_index("fixture_id", drop=False).loc[common["fixture_id"]].reset_index(drop=True)
    assert common["label"].tolist() == clean_targets["label"].tolist()
    assert len(common) == 44188
    catalog = TournamentCatalog.load()
    common["audit_world_tour"] = [bool((item := catalog.lookup(m.get("tournament_code"), m.get("tournament_id")))
                                           and "world tour" in item["category"].casefold()) for m in (by_id[key] for key in common["fixture_id"])]
    common["audit_junior"] = [bool(re.search(r"junior|\bu[ -]?(?:1[13579]|2[13])\b", m["tournament_name"] + " " + m["event_name"], re.I)
                                        or re.match(r"^(BS|GS|BD|GD)(?:$|[ -])", m["event_name"])) for m in (by_id[key] for key in common["fixture_id"])]
    common["audit_round"] = [by_id[key]["round_name"] for key in common["fixture_id"]]
    rolling = {name: np.full((len(protocol["seeds"]), len(common)), np.nan) for name in protocol["variants"]}
    annual = {name: np.full_like(value, np.nan) for name, value in rolling.items()}
    boundaries = [pd.Timestamp(value) for value in ("2026-01-01", "2026-04-01", "2026-07-01", "2026-10-01", cutoff.isoformat())]
    for window, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        mask = ((common["prediction_time"] >= start) & (common["prediction_time"] < end)).to_numpy()
        for seed_index, seed in enumerate(protocol["seeds"]):
            for variant, frame, targets in (("current_recipe", current_frame, common), ("cleaned_recipe", cleaned_frame, clean_targets)):
                subset = targets if window == 0 else targets.loc[mask]
                probability, _ = fit_window(frame, subset, start, base.feature_columns, reference["parameters"], seed, variant)
                rolling[variant][seed_index, mask] = probability[mask] if window == 0 else probability
                if window == 0:
                    annual[variant][seed_index] = probability
    assert all(np.isfinite(value).all() for value in [*rolling.values(), *annual.values()])
    world = common["audit_world_tour"].to_numpy() & ~common["audit_junior"].to_numpy()
    masks = {"all_2026": np.ones(len(common), dtype=bool), "senior_world_tour": world,
             "world_tour_R32": world & common["audit_round"].eq("R32").to_numpy(),
             "junior_tagged": common["audit_junior"].to_numpy(),
             **{event: common["discipline"].eq(event).to_numpy() for event in ("MS", "WS", "MD", "WD", "XD")}}
    results = {"protocol": protocol, "cohort": verification, "primary_quarterly": summarize(common, rolling, masks),
               "secondary_annual": summarize(common, annual, masks), "prospective_validation": False,
               "production_pointer_changed": False}
    save("comparison.json", results)
    export = common[["fixture_id", "match_id", "prediction_time", "tournament_code", "discipline", "label"]].copy()
    for variant, values in rolling.items():
        for i, seed in enumerate(protocol["seeds"]):
            export[f"{variant}_{seed}"] = values[i]
    export.to_csv(OUT / "paired-predictions.csv", index=False)
    print("Paired replay complete:", json.dumps(results["primary_quarterly"]["all_2026"]), flush=True)
    del current_frame, cleaned_frame, clean_targets, rolling, annual, export, old
    gc.collect()
    assert not CANDIDATE.exists(), "Never overwrite a trained candidate bundle"
    CANDIDATE.mkdir()
    categories = sorted({str(m.get(f"team{side}_country") or "").strip().upper() for m in matches for side in (1, 2)}
                        & base.country_context.countries)
    history = history_for(base, names, categories)
    frame = history.build_dataframe(matches, as_of=cutoff)
    model = LGBMClassifier(**reference["parameters"])
    model.fit(frame[base.feature_columns], frame["label"])
    model.feature_policy_ = history.model_policy
    model.trained_through_ = frame["available_at"].max().to_pydatetime()
    history.validate_model(model)
    with (CANDIDATE / "lgbm_model.pkl").open("xb") as handle:
        pickle.dump(model, handle)
    history.save(CANDIDATE / "feature_state.pkl")
    save("feature_list.json", base.feature_columns, CANDIDATE)
    save("bundle_manifest.json", {"version": 1, "probability_methods": ["Raw"], "prediction_policy": PREDICTION_POLICY}, CANDIDATE)
    cohort = sorted((m["fixture_id"], m["prediction_time"].isoformat(), m["available_at"].isoformat(), m["winner"]) for m in matches)
    save("metrics_report.json", {"bundle": "country_receipt_quality_cleaned", "reference_directory": reference["directory"],
                                "production_refit": {"rows": len(frame), "as_of": cutoff.isoformat(), "through": model.trained_through_.isoformat(),
                                                     "best_iteration": model.n_estimators}, "model_parameters": model.get_params(),
                                "feature_policy": history.model_policy, "cohort_sha256": hashlib.sha256(json.dumps(cohort).encode()).hexdigest(),
                                "data_quality": verification, "evaluation": {"kind": "paired reused historical recipe replay, not evaluation of this all-data refit",
                                    "source": str(OUT.relative_to(REPO_ROOT) / "comparison.json"), "prospective_forward_validation": False},
                                "probability_method": "Raw", "prediction_policy": PREDICTION_POLICY}, CANDIDATE)
    loaded, _, _, restored = load_bundle(CANDIDATE)
    fixture = {**matches[-1], "match_id": "cleanup-smoke-only", "match_time": cutoff + pd.Timedelta(days=1)}
    smoke = pd.DataFrame([restored.preview(fixture)])[base.feature_columns]
    restored.encode_categories(smoke)
    assert np.isfinite(predict_probabilities(loaded, smoke)).all()
    del frame, model, history, matches, names
    gc.collect()
    frozen = REPO_ROOT / "outputs/rounds/hong-kong-open-2026-20261116"
    hk = json.loads((frozen / "forecast.json").read_text())
    features, fixtures = build_inference_features(frozen / "fixtures.csv", CANDIDATE / "feature_state.pkl", model=loaded,
                                                refresh=True, as_of=datetime.fromisoformat(hk["forecast_cutoff_utc"]))
    new_p = predict_probabilities(loaded, features[restored.feature_columns])
    current_clean_p = predict_probabilities(production, features[base.feature_columns])
    frozen_p = {row["match_id"]: float(row["team1_win_probability"]) for row in hk["fixtures"]}
    shadow = pd.DataFrame({"match_id": fixtures["match_id"].astype(str), "current_frozen_probability": [frozen_p[str(mid)] for mid in fixtures["match_id"]],
                           "current_weights_clean_history_probability": current_clean_p, "candidate_clean_probability": new_p})
    shadow["candidate_minus_frozen_pp"] = (shadow["candidate_clean_probability"] - shadow["current_frozen_probability"]) * 100
    shadow.to_csv(OUT / "hong-kong-shadow-probabilities.csv", index=False)
    save("candidate-verification.json", {"candidate_directory": str(CANDIDATE.relative_to(REPO_ROOT)), "model_sha256": sha(CANDIDATE / "lgbm_model.pkl"),
                                         "state_sha256": sha(CANDIDATE / "feature_state.pkl"), "roundtrip_load": True, "frozen_bet_edits": False,
                                         "shadow_fixtures": len(shadow), "max_shadow_change_pp": float(shadow["candidate_minus_frozen_pp"].abs().max()),
                                         "median_shadow_change_pp": float(shadow["candidate_minus_frozen_pp"].abs().median()),
                                         "production_pointer_changed": False})
    print("Separate candidate trained, reloaded and shadow-compared; no frozen bets changed", flush=True)


if __name__ == "__main__":
    main()
