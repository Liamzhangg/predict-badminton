"""Paired, research-only promotion check; never switch the production bundle."""
from __future__ import annotations

import gc
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import pandas as pd

from predict_badminton.evaluate import reliability_stats

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "outputs/experiments/home-distance-controls-2026-11-15"
spec = importlib.util.spec_from_file_location("country_controls", PREVIOUS / "study.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
SEEDS = (42, 7, 2026, 13, 99)
REFERENCES = {"country_only": "country_strength", "home_distance": "all"}
VARIANTS = {name: list(s.VARIANTS[reference]) for name, reference in REFERENCES.items()}
QUARTERS = (("2026-01-01", "2026-04-01"), ("2026-04-01", "2026-07-01"),
            ("2026-07-01", "2026-10-01"), ("2026-10-01", "2027-01-01"))
METRICS = ("accuracy", "log_loss", "brier_score", "roc_auc", "ece")
DELTA_METRICS = ("accuracy", "log_loss", "brier_score")
# Configure only this process's imported research helpers, not their source files.
s.OUT = OUT
s.e.OUT = OUT
s.SEEDS = SEEDS
s.VARIANTS = VARIANTS


def split_window(frame, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    train = frame.loc[(frame["prediction_time"] < start) & (frame["available_at"] < start)]
    test = frame.loc[(frame["prediction_time"] >= start) & (frame["prediction_time"] < end)]
    assert not set(train["fixture_id"]) & set(test["fixture_id"])
    return train, test


def metrics(test, probability):
    labels = test["label"].to_numpy()
    return {"n": len(test), **s.e.compute_metrics(labels, probability),
            "ece": reliability_stats(labels, probability)["ece"]}


def fit_window(frame, start, end, variant, seed):
    began = time.perf_counter()
    train, test = split_window(frame, start, end)
    fitting, stopping = s.e._chronological_validation_split(train)
    first, second = s.matrices(fitting, stopping, variant)
    params = {"n_jobs": 4, "random_state": seed}
    provisional = s.e.train_lgbm(first, fitting["label"], second, stopping["label"], params)
    trees = int(provisional.best_iteration_ or 500)
    first, second = s.matrices(train, test, variant)
    model = s.e.train_lgbm(first, train["label"], params={**params, "n_estimators": trees})
    probability = s.predict(model, second, variant)
    result = {**metrics(test, probability), "train_n": len(train), "trees": trees,
              "seconds": time.perf_counter()-began}
    print(f"seed={seed} {start}:{end} {variant}: n={len(test)} accuracy={result['accuracy']:.6f} loss={result['log_loss']:.9f}", flush=True)
    del first, second, provisional, model
    gc.collect()
    return result, probability, test.index.to_numpy()


def aggregate(rows):
    return {variant: {"n": rows[str(SEEDS[0])][variant]["n"], **{
        name: {"mean": float(np.mean([rows[str(seed)][variant][name] for seed in SEEDS])),
               "min": float(min(rows[str(seed)][variant][name] for seed in SEEDS)),
               "max": float(max(rows[str(seed)][variant][name] for seed in SEEDS))}
        for name in METRICS}} for variant in VARIANTS}


def paired_check(test, control, candidate, seeds=SEEDS, repeats=5000):
    """Resample paired tournament editions, with a second seed-resampling check."""
    y = test["label"].to_numpy()
    control, candidate = np.asarray(control), np.asarray(candidate)
    assert control.shape == candidate.shape == (len(seeds), len(test))
    loss = lambda p: -(y*np.log(np.clip(p, 1e-15, 1-1e-15)) + (1-y)*np.log(np.clip(1-p, 1e-15, 1-1e-15)))
    delta = np.stack(((((candidate >= .5) == y).astype(float)-((control >= .5) == y).astype(float)),
                      loss(candidate)-loss(control), (candidate-y)**2-(control-y)**2), axis=2)
    groups = test["tournament_code"].fillna("").astype(str)
    groups = groups.where(groups != "", test["tournament_name"].astype(str)) + ":" + test["prediction_time"].dt.year.astype(str)
    names, inverse = np.unique(groups.to_numpy(), return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.asarray([[np.bincount(inverse, weights=delta[seed, :, metric], minlength=len(names))
                        for metric in range(3)] for seed in range(len(seeds))]).transpose(0, 2, 1)
    rng = np.random.default_rng(73016)
    conditional, crossed = [], []
    for offset in range(0, repeats, 250):
        size = min(250, repeats-offset)
        draw = rng.integers(0, len(names), (size, len(names)))
        denominator = counts[draw].sum(axis=1)[:, None]
        conditional.append(sums.mean(axis=0)[draw].sum(axis=1)/denominator)
        weights = rng.multinomial(len(seeds), np.full(len(seeds), 1/len(seeds)), size=size)/len(seeds)
        seed_means = np.einsum("bs,scm->bcm", weights, sums)
        crossed.append(seed_means[np.arange(size)[:, None], draw].sum(axis=1)/denominator)
    conditional, crossed = np.concatenate(conditional), np.concatenate(crossed)
    mean = delta.mean(axis=(0, 1))
    result = {"n": len(test), "seeds": list(seeds), "clusters": len(names), "replicates": repeats,
              "unit": "Unique match; seed-average paired differences, not stacked seed observations",
              **{name: {"mean": float(mean[column]),
                         "seed_values": {str(seed): float(delta[index, :, column].mean()) for index, seed in enumerate(seeds)},
                         "cluster_ci95": np.quantile(conditional[:, column], [.025, .975]).tolist(),
                         "crossed_seed_cluster_ci95": np.quantile(crossed[:, column], [.025, .975]).tolist()}
                 for column, name in enumerate(DELTA_METRICS)}}
    evidence = {"metrics": DELTA_METRICS, "seeds": list(seeds), "clusters": names.tolist(),
                "match_counts": counts.tolist(), "paired_metric_sums_by_seed_cluster": sums.tolist()}
    return result, evidence


def promotion_gate(validation, annual, rolling, forward_available=False):
    checks = {
        "validation_accuracy_not_worse": validation["accuracy"]["mean"] >= 0,
        "validation_log_loss_not_worse": validation["log_loss"]["mean"] <= 0,
        "annual_2026_accuracy_not_worse": annual["accuracy"]["mean"] >= 0,
        "rolling_accuracy_crossed_ci_strictly_positive": rolling["accuracy"]["crossed_seed_cluster_ci95"][0] > 0,
        "rolling_log_loss_crossed_ci_not_worse": rolling["log_loss"]["crossed_seed_cluster_ci95"][1] <= 0,
        "rolling_brier_mean_not_worse": rolling["brier_score"]["mean"] <= 0,
    }
    return {"historical_checks": checks, "historical_gate_passed": all(checks.values()),
            "prospective_forward_gate_passed": forward_available,
            "evidence_ready_for_promotion": all(checks.values()) and forward_available,
            "production_changed": False}


def forward_audit(locked_at):
    db = s.e.OBSERVATIONS_PATH.resolve()
    connection = sqlite3.connect(db.as_uri()+"?mode=ro", uri=True)
    try:
        latest = connection.execute("SELECT MAX(observed_at) FROM responses").fetchone()[0]
    finally:
        connection.close()
    prior_bundles = {name: [] for name in VARIANTS}
    for path in (ROOT / "outputs").rglob("feature_list.json"):
        schema = json.loads(path.read_text())
        for name in VARIANTS:
            if schema == s.e.FEATURE_COLS + VARIANTS[name]:
                prior_bundles[name].append(str(path.relative_to(ROOT)))
    forecasts = []
    for path in (ROOT / "outputs").rglob("forecast.json"):
        record = json.loads(path.read_text())
        if record.get("prediction_generated_at_utc"):
            forecasts.append({"file": str(path.relative_to(ROOT)), "generated_at_utc": record["prediction_generated_at_utc"],
                              "model_directory": record.get("model_directory")})
    return {"locked_at_utc": locked_at, "latest_local_result_observation": latest,
            "prior_bundles_matching_exact_candidate_schemas": prior_bundles,
            "existing_timestamped_forecast_records": forecasts,
            "prospective_forward_gate_passed": False,
            "reason": "No verified pre-result predictions for these two exact candidates. Completed local results and all replay scores are retrospective; future unseen matches are required.",
            "scope": "Local repository audit only; no result fetching or edits to existing forecasts."}


def main():
    logging.basicConfig(level=logging.WARNING)
    began = time.perf_counter()
    assert not (OUT / "report.json").exists(), "Keep previous research immutable"
    old = json.loads((PREVIOUS / "report.json").read_text())
    cutoff = s.e.datetime.fromisoformat(old["cutoff"])
    snapshot = s.protect()
    locked = s.e.datetime.now(s.e.UTC).isoformat()
    audit = forward_audit(locked)
    s.save("forward-audit.json", audit)
    for name in ("countries.html", "countryInfo.txt"):
        shutil.copy2(PREVIOUS / name, OUT / name)
    directory = s.e.forecast_bundle_directory()
    base = s.e.FeatureBuilder.load(directory / "feature_state.pkl")
    s.save("protocol.json", {
        "locked_at_utc": locked, "cutoff": cutoff.isoformat(), "seeds": SEEDS, "variants": VARIANTS,
        "comparison": "country Elo + represented-country identity + log international-history counts, versus exactly that bundle with home indicators and home-country log-distance",
        "primary": "Pooled quarterly-retraining 2026 accuracy; identical matches in both candidates. Raw two-order probability policy, same as current round forecasts.",
        "annual": "Earlier-year refits scored on 2024, 2025, 2026; first two pooled as validation, 2026 reused benchmark.",
        "rolling": "Retrain at January/April/July/October 2026 boundaries, using only earlier prediction times and labels available strictly before boundary. Score each next quarter; Q1 reuses identical January-trained annual predictions.",
        "training": "Same fixed LightGBM defaults, availability-purged inner early stopping, refit available training rows. Country categories fit separately on each fitting/training split; unseen categories missing. Base/country history is updated only from results available before each target prediction.",
        "statistics": "5000 paired tournament-edition/year bootstrap draws. Crossed intervals additionally resample the five seeds, sharing tournament draws. Matches are not multiplied by seed count. ECE uses fixed 10 uniform bins and is descriptive only.",
        "gates_fixed_before_scoring": {"validation_accuracy_delta": ">=0", "validation_log_loss_delta": "<=0", "annual_2026_accuracy_delta": ">=0",
            "rolling_accuracy_crossed_ci95_lower": ">0", "rolling_log_loss_crossed_ci95_upper": "<=0", "rolling_brier_mean_delta": "<=0",
            "prospective_forward": "Required but not currently available; no candidate can be declared production-ready from these reused outcomes alone."},
        "evaluation_status": "Reused retrospective benchmark, including rolling replay. New seeds are robustness checks, not new test data. No tuning, model export, production switch, or saved-bet/prediction changes.",
        "base_bundle": str(directory.relative_to(ROOT)), "base_policy": base.model_policy,
        "research_helper_sha256": s.e.sha(PREVIOUS / "study.py"),
    })
    assert base.model_policy["result_availability"] == s.e.OBSERVED_AVAILABILITY_POLICY
    receipts = s.e.observations_before(cutoff, s.e.OBSERVATIONS_PATH)
    matches, drops, names = s.e.load_all_matches(ROOT / "data/raw/bwf_matches", observed_results=receipts,
                                                category_catalog=base.category_catalog)
    stamp = cutoff.replace(tzinfo=None)
    matches = [m for m in matches if not s.e.standard_match_exclusion(m) and m.get("match_status") in (None, "", "F")
               and m["available_at"] < stamp and m["match_time"] < stamp]
    assert len(matches) == old["matches"]
    history = s.e.FeatureBuilder(player_names=names, result_availability_policy=s.e.OBSERVED_AVAILABILITY_POLICY,
                                 category_catalog=base.category_catalog)
    history.elo = s.e.EloEngine(**base.model_policy["elo"])
    history.trueskill = s.e.TrueSkillEngine(**base.trueskill.parameters)
    assert history.model_policy == base.model_policy
    print(f"Building identical frozen cohort: {len(matches):,} matches", flush=True)
    frame = history.build_dataframe(matches, as_of=stamp)
    frame = frame.merge(s.context_frame(matches, s.e.geography()), on="fixture_id", validate="one_to_one", how="left")
    assert len(frame) == len(matches) and not frame["host_known"].isna().any()
    del matches, history, base, receipts
    gc.collect()
    report = {"cutoff": cutoff.isoformat(), "matches": len(frame), "seeds": SEEDS, "drops": drops,
              "annual": {}, "quarterly": {}, "validation": {}, "annual_2026": {}, "rolling_2026": {}}
    predictions = {name: {variant: [] for variant in VARIANTS} for name in ("validation", "annual_2026", "rolling_2026")}
    evaluation_indices = {}
    for seed in SEEDS:
        annual_probabilities, annual_indices = {}, {}
        report["annual"][str(seed)], report["quarterly"][str(seed)] = {}, {}
        for year in (2024, 2025, 2026):
            annual_probabilities[year] = {}
            report["annual"][str(seed)][str(year)] = {}
            for variant in VARIANTS:
                result, probability, idx = fit_window(frame, f"{year}-01-01", f"{year+1}-01-01", variant, seed)
                report["annual"][str(seed)][str(year)][variant] = result
                annual_probabilities[year][variant], annual_indices[year] = probability, idx
                if seed in (42, 7, 2026):
                    reference = old["validation"][str(seed)][REFERENCES[variant]][str(year)] if year < 2026 else old["holdout"][str(seed)][REFERENCES[variant]]
                    for key in ("n", "train_n", "trees", "accuracy", "log_loss"):
                        assert abs(result[key]-reference[key]) < 1e-12, (seed, year, variant, key)
                s.save("report.json", report)
        validation_idx = np.concatenate([annual_indices[year] for year in (2024, 2025)])
        evaluation_indices["validation"], evaluation_indices["annual_2026"] = validation_idx, annual_indices[2026]
        report["validation"][str(seed)], report["annual_2026"][str(seed)], report["rolling_2026"][str(seed)] = {}, {}, {}
        for variant in VARIANTS:
            probability = np.concatenate([annual_probabilities[year][variant] for year in (2024, 2025)])
            predictions["validation"][variant].append(probability)
            report["validation"][str(seed)][variant] = metrics(frame.loc[validation_idx], probability)
            predictions["annual_2026"][variant].append(annual_probabilities[2026][variant])
            report["annual_2026"][str(seed)][variant] = metrics(frame.loc[annual_indices[2026]], annual_probabilities[2026][variant])
        rolling_parts, rolling_indices = {variant: [] for variant in VARIANTS}, []
        for quarter, (start, end) in enumerate(QUARTERS, 1):
            report["quarterly"][str(seed)][str(quarter)] = {}
            for variant in VARIANTS:
                if quarter == 1:
                    mask = frame.loc[annual_indices[2026], "prediction_time"].to_numpy() < np.datetime64(end)
                    idx = annual_indices[2026][mask]
                    probability = annual_probabilities[2026][variant][mask]
                    result = {**metrics(frame.loc[idx], probability), "reused_identical_january_fit": True}
                else:
                    result, probability, idx = fit_window(frame, start, end, variant, seed)
                report["quarterly"][str(seed)][str(quarter)][variant] = result
                rolling_parts[variant].append(probability)
                if variant == "country_only":
                    rolling_indices.extend(idx)
                s.save("report.json", report)
        rolling_idx = np.asarray(rolling_indices)
        assert np.array_equal(rolling_idx, annual_indices[2026]), "Rolling replay changed the evaluation cohort"
        evaluation_indices["rolling_2026"] = rolling_idx
        for variant in VARIANTS:
            probability = np.concatenate(rolling_parts[variant])
            predictions["rolling_2026"][variant].append(probability)
            report["rolling_2026"][str(seed)][variant] = metrics(frame.loc[rolling_idx], probability)
        s.save("report.json", report)
    report["summaries"], report["paired"] = {}, {}
    for mode in predictions:
        report["summaries"][mode] = aggregate(report[mode])
        result, evidence = paired_check(frame.loc[evaluation_indices[mode]], predictions[mode]["country_only"], predictions[mode]["home_distance"])
        report["paired"][mode] = result
        s.save(f"paired-clusters-{mode}.json", evidence)
    report["gate"] = promotion_gate(report["paired"]["validation"], report["paired"]["annual_2026"], report["paired"]["rolling_2026"])
    report.update(complete=True, anchors_reproduced=True, elapsed_seconds=time.perf_counter()-began,
                  protected_files_unchanged=s.verify(snapshot))
    s.save("report.json", report)
    lines = ["# Country vs Country + Home Advantage + Distance", "",
             "Five fixed seeds, matched frozen cohort, raw two-order probabilities. "
             "Quarterly 2026 retraining is the primary replay; annual scores replicate the earlier experiment. "
             "All outcomes are reused historical results, not prospective forward validation. Nothing was promoted.", "",
             "| Evaluation | Bundle | Accuracy | Log Loss | Brier | ECE (10 bins) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for mode, variants in report["summaries"].items():
        for variant, row in variants.items():
            lines.append(f"| {mode} | {variant} | {row['accuracy']['mean']:.3%} | {row['log_loss']['mean']:.6f} | {row['brier_score']['mean']:.6f} | {row['ece']['mean']:.6f} |")
    lines.extend(["", "## Paired Differences", "",
                  "Expanded minus country-only; accuracy is in percentage points. Crossed 95% intervals resample tournament editions and training seeds.", "",
                  "| Evaluation | Accuracy Change / 95% Interval | Log-Loss Change / 95% Interval |", "| --- | --- | --- |"])
    for mode, paired in report["paired"].items():
        accuracy, loss = paired["accuracy"], paired["log_loss"]
        a, b = accuracy["crossed_seed_cluster_ci95"], loss["crossed_seed_cluster_ci95"]
        lines.append(f"| {mode} | {100*accuracy['mean']:+.3f} / [{100*a[0]:+.3f}, {100*a[1]:+.3f}] | {loss['mean']:+.6f} / [{b[0]:+.6f}, {b[1]:+.6f}] |")
    lines.extend(["", "## Fixed Gates", ""])
    for name, passed in report["gate"]["historical_checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'NOT MET'}")
    lines.extend(["- Prospective forward gate: NOT AVAILABLE. No verified pre-result records for these exact candidates.",
                  f"- Historical gate passed: {report['gate']['historical_gate_passed']}",
                  "- Evidence ready for production promotion: False. Production was not changed.", "",
                  "No model bundles, feature matrices or per-match probability arrays retained. "
                  "Compact per-cluster paired sums preserve the uncertainty calculation. Protocol, source audit and detailed scores are alongside this report.", ""])
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "seconds": report["elapsed_seconds"], "summaries": report["summaries"],
                      "paired": report["paired"], "gate": report["gate"],
                      "protected_files_unchanged": report["protected_files_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
