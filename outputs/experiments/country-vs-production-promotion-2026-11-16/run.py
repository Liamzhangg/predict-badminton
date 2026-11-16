"""Country candidate versus the active production recipe, without promotion."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
HELPER = ROOT / "outputs/experiments/country-home-distance-promotion-2026-11-16/check.py"
spec = importlib.util.spec_from_file_location("paired_promotion", HELPER)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
CONTROLS = p.PREVIOUS
VARIANTS = {"production_recipe": [], "country_candidate": list(p.s.VARIANTS["country_only"])}
REFERENCES = {"production_recipe": "raw_control", "country_candidate": "country_strength"}
p.OUT = OUT
p.s.OUT = OUT
p.s.e.OUT = OUT
p.VARIANTS = VARIANTS
p.s.VARIANTS = VARIANTS
BEHAVIOR_FIELDS = ("fixture_id", "match_id", "match_time", "prediction_time", "available_at", "discipline",
                   "team1_ids", "team2_ids", "team1_country", "team2_country", "winner", "tournament_tier",
                   "round_name", "team1_seed", "team2_seed", "score_margin")


def cohort_digest(matches):
    digest = hashlib.sha256()
    for match in sorted(matches, key=lambda m: m["fixture_id"]):
        digest.update(json.dumps({name: match.get(name) for name in BEHAVIOR_FIELDS},
                                 sort_keys=True, default=str, allow_nan=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def training_params(current, seed):
    return {**current, "random_state": seed, "n_jobs": 4}


def post_training_rows(frame, cutoff):
    return frame.loc[frame["prediction_time"] > pd.Timestamp(cutoff)]


def fit_window(frame, start, end, variant, seed, current_params):
    began = time.perf_counter()
    train, test = p.split_window(frame, start, end)
    fitting, stopping = p.s.e._chronological_validation_split(train)
    first, second = p.s.matrices(fitting, stopping, variant)
    params = training_params(current_params, seed)
    provisional = p.s.e.train_lgbm(first, fitting["label"], second, stopping["label"], params)
    trees = int(provisional.best_iteration_ or current_params["n_estimators"])
    first, second = p.s.matrices(train, test, variant)
    model = p.s.e.train_lgbm(first, train["label"], params={**params, "n_estimators": trees})
    probability = p.s.predict(model, second, variant)
    result = {**p.metrics(test, probability), "train_n": len(train), "trees": trees,
              "seconds": time.perf_counter()-began}
    print(f"seed={seed} {start}:{end} {variant}: n={len(test)} accuracy={result['accuracy']:.6f} loss={result['log_loss']:.9f}", flush=True)
    del first, second, provisional, model
    gc.collect()
    return result, probability, test.index.to_numpy()


def make_history(base, names):
    history = p.s.e.FeatureBuilder(player_names=names, result_availability_policy=p.s.e.OBSERVED_AVAILABILITY_POLICY,
                                   category_catalog=base.category_catalog)
    history.elo = p.s.e.EloEngine(**base.model_policy["elo"])
    history.trueskill = p.s.e.TrueSkillEngine(**base.trueskill.parameters)
    assert history.model_policy == base.model_policy
    return history


def eligible(matches, cutoff):
    return [m for m in matches if not p.s.e.standard_match_exclusion(m)
            and m.get("match_status") in (None, "", "F") and m["available_at"] < cutoff and m["match_time"] < cutoff]


def artifact_check(frame, base, production, metadata, directory, prefix_digest, locations):
    cutoff = pd.Timestamp(metadata["production_refit"]["as_of"])
    saved = json.loads((directory / "observed_result_receipts.json").read_text())
    saved = {mid: None if row is None else {**row, "available_at": p.s.e.datetime.fromisoformat(row["available_at"])}
             for mid, row in saved.items()}
    matches, drops, names = p.s.e.load_all_matches(ROOT / "data/raw/bwf_matches", observed_results=saved,
                                                  category_catalog=base.category_catalog)
    matches = eligible(matches, cutoff.to_pydatetime())
    test = post_training_rows(frame, cutoff)
    result = {"evaluation_type": "Retrospective post-training slice for actual saved production weights; not prospective candidate forecasts",
              "snapshot_cutoff": cutoff.isoformat(), "production_training_rows": metadata["production_refit"]["rows"],
              "reconstructed_training_rows": len(matches), "post_snapshot_matches": len(test),
              "prediction_policy": "raw two-order average, features refreshed strictly before each target",
              "saved_receipts_used": True, "drops": drops}
    if len(matches) != metadata["production_refit"]["rows"] or test.empty:
        result.update(status="not_verified", reason="Training row count differs or no post-snapshot matches")
        return result
    reconstructed_digest = cohort_digest(matches)
    result["saved_receipt_training_digest"] = reconstructed_digest
    result["matches_primary_replay_prefix"] = reconstructed_digest == prefix_digest
    if reconstructed_digest == prefix_digest:
        train = frame.loc[(frame["prediction_time"] < cutoff) & (frame["available_at"] < cutoff)]
    else:
        history = make_history(base, names)
        train = history.build_dataframe(matches, as_of=cutoff.to_pydatetime())
        train = train.merge(p.s.context_frame(matches, locations), on="fixture_id", validate="one_to_one", how="left")
        del history
    assert len(train) == len(matches) and not set(train["fixture_id"]) & set(test["fixture_id"])
    del matches, names, saved
    gc.collect()
    first, second = p.s.matrices(train, test, "production_recipe")
    params = production.get_params()
    print(f"Checking actual production artifact on {len(test)} strictly later matches", flush=True)
    mirror = p.s.e.train_lgbm(first, train["label"], params=params)
    actual_probability = p.s.predict(production, second, "production_recipe")
    mirror_probability = p.s.predict(mirror, second, "production_recipe")
    model_hash = lambda m: hashlib.sha256(json.dumps(m.booster_.dump_model(), sort_keys=True, default=str).encode()).hexdigest()
    same_trees = model_hash(production) == model_hash(mirror)
    maximum_difference = float(np.max(np.abs(actual_probability-mirror_probability)))
    if not same_trees and maximum_difference <= 1e-10:
        maximum_difference = max(maximum_difference, float(np.max(np.abs(
            p.s.predict(production, first, "production_recipe")-p.s.predict(mirror, first, "production_recipe")))))
    result.update(identical_booster_dump=same_trees, mirror_maximum_probability_difference=maximum_difference,
                  production_metrics=p.metrics(test, actual_probability))
    if not same_trees and maximum_difference > 1e-10:
        result.update(status="not_verified", reason="Reconstructed production fit does not reproduce saved model predictions")
        return result
    first, second = p.s.matrices(train, test, "country_candidate")
    candidate = p.s.e.train_lgbm(first, train["label"], params=params)
    probability = p.s.predict(candidate, second, "country_candidate")
    paired, evidence = p.paired_check(test, [actual_probability], [probability], seeds=(params["random_state"],))
    p.s.save("paired-clusters-actual-artifact.json", evidence)
    result.update(status="verified_retrospective_comparison", candidate_metrics=p.metrics(test, probability),
                  paired=paired, original_tree_count=params["n_estimators"],
                  note="Both fits use the original production training snapshot and parameters. This small, already-seen later slice is diagnostic, not a fresh independent gate.")
    print(json.dumps({"artifact_status": result["status"], "n": len(test),
                      "production": result["production_metrics"], "candidate": result["candidate_metrics"]}, default=str), flush=True)
    return result


def main():
    logging.basicConfig(level=logging.WARNING)
    began = time.perf_counter()
    assert not (OUT / "report.json").exists(), "Do not overwrite earlier research"
    old = json.loads((CONTROLS / "report.json").read_text())
    cutoff = p.s.e.datetime.fromisoformat(old["cutoff"])
    snapshot = p.s.protect()
    directory = p.s.e.forecast_bundle_directory()
    production = pickle.loads((directory / "lgbm_model.pkl").read_bytes())
    metadata = json.loads((directory / "metrics_report.json").read_text())
    base = p.s.e.FeatureBuilder.load(directory / "feature_state.pkl")
    assert list(production.feature_name_) == p.s.e.FEATURE_COLS
    assert production.feature_policy_ == base.model_policy
    assert metadata["prediction_policy"] == "two_order_average_v1"
    assert pd.Timestamp(metadata["production_refit"]["as_of"]) >= pd.Timestamp(production.trained_through_)
    locked = p.s.e.datetime.now(p.s.e.UTC).isoformat()
    reference = {"directory": str(directory.relative_to(ROOT)), "model_sha256": p.s.e.sha(directory / "lgbm_model.pkl"),
                 "state_sha256": p.s.e.sha(directory / "feature_state.pkl"), "active_feature_count": len(production.feature_name_),
                 "parameters": production.get_params(), "trained_through": str(production.trained_through_),
                 "production_refit": metadata["production_refit"], "model_policy": base.model_policy}
    p.s.save("production-reference.json", reference)
    p.s.save("forward-audit.json", p.forward_audit(locked))
    p.s.save("protocol.json", {
        "locked_at_utc": locked, "frozen_cutoff": cutoff.isoformat(), "seeds": p.SEEDS, "variants": VARIANTS,
        "primary": "Production 44-feature recipe versus unchanged 55-feature country candidate; pooled quarterly-retraining 2026 comparison",
        "feature_difference": "Country Elo/discipline and log international-history counts (side values/differences), two training-only country identities, and the same three location-coverage flags as earlier country studies. No home/distance features.",
        "training": "Both use active production model hyperparameters; only random seed and execution thread count vary in replay. Availability-purged early stopping then refit earlier available rows. Raw two-order probabilities, no calibrator change.",
        "evaluation": "Annual 2024-2025 validation and 2026 reused benchmark; quarterly retraining January/April/July/October 2026 on earlier prediction times and labels available strictly before boundary. Same fixtures; Q1 reuses identical January fit.",
        "artifact_diagnostic": "Never score the current all-results artifact on historical training matches. Reconstruct its original training snapshot from saved receipts, verify row count and numerical/booster reproduction, then compare against a country candidate trained on that same snapshot/parameters using only later target prediction times.",
        "statistics": "5000 paired tournament-edition/year resamples; crossed intervals also resample the five seeds without multiplying match count. Per-seed averages, not ensembles. Artifact diagnostic has one original seed and few later events.",
        "fixed_gates": {"validation_accuracy_delta": ">=0", "validation_log_loss_delta": "<=0", "annual_2026_accuracy_delta": ">=0",
                        "rolling_accuracy_crossed_ci95_lower": ">0", "rolling_log_loss_crossed_ci95_upper": "<=0", "rolling_brier_mean_delta": "<=0",
                        "prospective_forward": "Not currently available for the candidate; required before declaring fully ready for promotion."},
        "scope": "Research-only check. No production/default pointer, saved forecast, odds or bet edits. Reused historical outcomes, not fresh forward validation. No model export.",
        "research_helper_sha256": p.s.e.sha(HELPER), "production_reference": reference,
    })
    for name in ("countries.html", "countryInfo.txt"):
        shutil.copy2(CONTROLS / name, OUT / name)
    receipts = p.s.e.observations_before(cutoff, p.s.e.OBSERVATIONS_PATH)
    matches, drops, names = p.s.e.load_all_matches(ROOT / "data/raw/bwf_matches", observed_results=receipts,
                                                  category_catalog=base.category_catalog)
    matches = eligible(matches, cutoff.replace(tzinfo=None))
    assert len(matches) == old["matches"]
    production_cutoff = pd.Timestamp(metadata["production_refit"]["as_of"]).to_pydatetime()
    prefix_digest = cohort_digest([m for m in matches if m["prediction_time"] < production_cutoff and m["available_at"] < production_cutoff])
    print(f"Building identical frozen cohort: {len(matches):,} matches", flush=True)
    history = make_history(base, names)
    frame = history.build_dataframe(matches, as_of=cutoff.replace(tzinfo=None))
    locations = p.s.e.geography()
    frame = frame.merge(p.s.context_frame(matches, locations), on="fixture_id", validate="one_to_one", how="left")
    assert len(frame) == len(matches) and not frame["host_known"].isna().any()
    del matches, names, receipts, history
    gc.collect()
    report = {"cutoff": cutoff.isoformat(), "matches": len(frame), "seeds": p.SEEDS, "drops": drops,
              "annual": {}, "quarterly": {}, "validation": {}, "annual_2026": {}, "rolling_2026": {}}
    report["actual_artifact"] = artifact_check(frame, base, production, metadata, directory, prefix_digest, locations)
    p.s.save("report.json", report)
    predictions = {mode: {variant: [] for variant in VARIANTS} for mode in ("validation", "annual_2026", "rolling_2026")}
    indices_by_mode = {}
    for seed in p.SEEDS:
        report["annual"][str(seed)], report["quarterly"][str(seed)] = {}, {}
        probabilities_by_year, indices_by_year = {}, {}
        for year in (2024, 2025, 2026):
            report["annual"][str(seed)][str(year)] = {}
            probabilities_by_year[year] = {}
            for variant in VARIANTS:
                result, probability, idx = fit_window(frame, f"{year}-01-01", f"{year+1}-01-01", variant, seed, production.get_params())
                report["annual"][str(seed)][str(year)][variant] = result
                probabilities_by_year[year][variant], indices_by_year[year] = probability, idx
                if seed in (42, 7, 2026):
                    expected = old["validation"][str(seed)][REFERENCES[variant]][str(year)] if year < 2026 else old["holdout"][str(seed)][REFERENCES[variant]]
                    for key in ("n", "train_n", "trees", "accuracy", "log_loss"):
                        assert abs(result[key]-expected[key]) < 1e-12, (seed, year, variant, key)
                p.s.save("report.json", report)
        val_idx = np.concatenate([indices_by_year[year] for year in (2024, 2025)])
        indices_by_mode["validation"], indices_by_mode["annual_2026"] = val_idx, indices_by_year[2026]
        report["validation"][str(seed)], report["annual_2026"][str(seed)], report["rolling_2026"][str(seed)] = {}, {}, {}
        for variant in VARIANTS:
            probability = np.concatenate([probabilities_by_year[year][variant] for year in (2024, 2025)])
            predictions["validation"][variant].append(probability)
            report["validation"][str(seed)][variant] = p.metrics(frame.loc[val_idx], probability)
            probability = probabilities_by_year[2026][variant]
            predictions["annual_2026"][variant].append(probability)
            report["annual_2026"][str(seed)][variant] = p.metrics(frame.loc[indices_by_year[2026]], probability)
        rolling, rolling_indices = {variant: [] for variant in VARIANTS}, []
        for quarter, (start, end) in enumerate(p.QUARTERS, 1):
            report["quarterly"][str(seed)][str(quarter)] = {}
            for variant in VARIANTS:
                if quarter == 1:
                    mask = frame.loc[indices_by_year[2026], "prediction_time"].to_numpy() < np.datetime64(end)
                    idx = indices_by_year[2026][mask]
                    probability = probabilities_by_year[2026][variant][mask]
                    result = {**p.metrics(frame.loc[idx], probability), "reused_identical_january_fit": True}
                else:
                    result, probability, idx = fit_window(frame, start, end, variant, seed, production.get_params())
                report["quarterly"][str(seed)][str(quarter)][variant] = result
                rolling[variant].append(probability)
                if variant == "production_recipe":
                    rolling_indices.extend(idx)
                p.s.save("report.json", report)
        rolling_idx = np.asarray(rolling_indices)
        assert np.array_equal(rolling_idx, indices_by_year[2026])
        indices_by_mode["rolling_2026"] = rolling_idx
        for variant in VARIANTS:
            probability = np.concatenate(rolling[variant])
            predictions["rolling_2026"][variant].append(probability)
            report["rolling_2026"][str(seed)][variant] = p.metrics(frame.loc[rolling_idx], probability)
        p.s.save("report.json", report)
    report["summaries"], report["paired"] = {}, {}
    for mode in predictions:
        report["summaries"][mode] = p.aggregate(report[mode])
        paired, evidence = p.paired_check(frame.loc[indices_by_mode[mode]], predictions[mode]["production_recipe"], predictions[mode]["country_candidate"])
        report["paired"][mode] = paired
        p.s.save(f"paired-clusters-{mode}.json", evidence)
    report["gate"] = p.promotion_gate(report["paired"]["validation"], report["paired"]["annual_2026"], report["paired"]["rolling_2026"])
    report.update(complete=True, anchors_reproduced=True, elapsed_seconds=time.perf_counter()-began,
                  protected_files_unchanged=p.s.verify(snapshot))
    p.s.save("report.json", report)
    lines = ["# Country Candidate vs Current Production", "",
             "Main comparison: current production's 44-feature recipe retrained at matched historical cutoffs, versus the unchanged 55-feature country candidate. "
             "Five seeds; quarterly retraining is primary. Scores are per-seed means, not an ensemble. "
             "The actual saved artifact is never scored on its old training matches; its separate post-snapshot check is below. No production change.", "",
             "| Evaluation | Bundle | Accuracy | Log Loss | Brier |", "| --- | --- | ---: | ---: | ---: |"]
    for mode, variants in report["summaries"].items():
        for variant, row in variants.items():
            lines.append(f"| {mode} | {variant} | {row['accuracy']['mean']:.3%} | {row['log_loss']['mean']:.6f} | {row['brier_score']['mean']:.6f} |")
    lines.extend(["", "## Paired Changes", "", "Candidate minus production recipe, crossed seed/tournament 95% intervals.", "",
                  "| Evaluation | Accuracy Change (pp) / 95% Interval | Log-Loss Change / 95% Interval |", "| --- | --- | --- |"])
    for mode, row in report["paired"].items():
        a, loss = row["accuracy"], row["log_loss"]
        ci, lc = a["crossed_seed_cluster_ci95"], loss["crossed_seed_cluster_ci95"]
        lines.append(f"| {mode} | {100*a['mean']:+.3f} / [{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] | {loss['mean']:+.6f} / [{lc[0]:+.6f}, {lc[1]:+.6f}] |")
    artifact = report["actual_artifact"]
    lines.extend(["", "## Actual Saved Artifact", "", f"Status: `{artifact['status']}`. Post-snapshot matches: {artifact['post_snapshot_matches']}. "
                  "This is a small retrospective diagnostic, not prospective forward evidence.", ""])
    if artifact["status"] == "verified_retrospective_comparison":
        for name in ("production_metrics", "candidate_metrics"):
            value = artifact[name]
            lines.append(f"- {name}: accuracy {value['accuracy']:.3%}, log loss {value['log_loss']:.6f}.")
    lines.extend(["", "## Gates", ""])
    for name, passed in report["gate"]["historical_checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'NOT MET'}")
    lines.extend([f"- Historical gate passed: {report['gate']['historical_gate_passed']}",
                  "- Prospective paired forward gate: NOT AVAILABLE. Candidate forecasts were not frozen before results.",
                  "- Production/default bundle and saved predictions/bets: unchanged. No candidate model exported or promoted.", ""])
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "seconds": report["elapsed_seconds"], "summaries": report["summaries"],
                      "paired": report["paired"], "actual_artifact": report["actual_artifact"], "gate": report["gate"],
                      "protected_files_unchanged": report["protected_files_unchanged"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
