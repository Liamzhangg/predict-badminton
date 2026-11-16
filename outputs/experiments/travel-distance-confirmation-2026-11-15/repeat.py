"""Repeat the frozen travel screening across seeds without promoting models."""
from __future__ import annotations

import gc
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "outputs/experiments/travel-distance-2026-11-15"
SEEDS = (42, 7, 2026)
VARIANTS = ("control", "geography_14d", "home_only", "travel_7d", "travel_14d", "travel_21d", "travel_28d")
spec = importlib.util.spec_from_file_location("travel_screening", ORIGINAL / "experiment.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
e.OUT = OUT


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def protect():
    paths = {p for base in (ROOT / "data", ROOT / "outputs") for p in base.rglob("*")
             if p.is_file() and not p.is_relative_to(OUT)}
    paths.update(ROOT / p for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines())
    paths.add(ROOT / "EXPERIMENTS.md")
    snapshot = {str(p.relative_to(ROOT)): e.sha(p) for p in sorted(paths) if p.is_file()}
    save("protected-before.json", snapshot)
    return snapshot


def verify(snapshot):
    changed = [name for name, digest in snapshot.items() if not (ROOT / name).is_file() or e.sha(ROOT / name) != digest]
    assert not changed, changed
    return len(snapshot)


def fit(frame, year, variant, seed):
    boundary = pd.Timestamp(year, 1, 1)
    train = frame.loc[(frame["prediction_time"] < boundary) & (frame["available_at"] < boundary)]
    test = frame.loc[frame["prediction_time"].dt.year == year]
    assert not set(train["fixture_id"]) & set(test["fixture_id"])
    fitting, stopping = e._chronological_validation_split(train)
    params = {"n_jobs": 4, "random_state": seed}
    provisional = e.train_lgbm(e.inputs(fitting, variant), fitting["label"],
                               e.inputs(stopping, variant), stopping["label"], params)
    trees = int(provisional.best_iteration_ or 500)
    model = e.train_lgbm(e.inputs(train, variant), train["label"], params={**params, "n_estimators": trees})
    probability = e.predict(model, test, variant)
    metrics = {"n": len(test), "train_n": len(train), "trees": trees,
               **e.compute_metrics(test["label"].to_numpy(), probability)}
    print(f"seed={seed} year={year} {variant}: accuracy={metrics['accuracy']:.6f} loss={metrics['log_loss']:.9f}", flush=True)
    del provisional, model
    gc.collect()
    return metrics, probability, test.index


def aggregate(results):
    summary = {}
    for variant in VARIANTS:
        rows = [results[str(seed)][variant] for seed in SEEDS]
        summary[variant] = {"n": rows[0]["n"], **{
            key: {"mean": float(np.mean([r[key] for r in rows])),
                  "min": float(min(r[key] for r in rows)), "max": float(max(r[key] for r in rows))}
            for key in ("accuracy", "log_loss")}}
    return summary


def main():
    started = time.perf_counter()
    logging.basicConfig(level=logging.WARNING)
    assert not (OUT / "report.json").exists(), "Keep previous research outputs immutable"
    original = json.loads((ORIGINAL / "report.json").read_text())
    cutoff = e.datetime.fromisoformat(original["cutoff"])
    snapshot = protect()
    for name in ("countries.html", "countryInfo.txt"):
        shutil.copy2(ORIGINAL / name, OUT / name)
    directory = e.forecast_bundle_directory()
    base = e.FeatureBuilder.load(directory / "feature_state.pkl")
    assert base.model_policy["result_availability"] == e.OBSERVED_AVAILABILITY_POLICY
    save("protocol.json", {
        "cutoff": cutoff.isoformat(), "seeds": SEEDS, "variants": VARIANTS,
        "source_experiment": str(ORIGINAL.relative_to(ROOT)),
        "source_script_sha256": e.sha(ORIGINAL / "experiment.py"),
        "base_bundle": str(directory.relative_to(ROOT)), "base_policy": base.model_policy,
        "primary": "14-day distance versus matched control; other fixed cutoffs are sensitivity checks.",
        "selection": "Lowest mean per-seed pooled 2024-2025 log loss; record selection before scoring 2026.",
        "evaluation": "Reused retrospective 2026 benchmark, not a fresh holdout. Fixed LightGBM defaults, strict result availability, two-order probabilities. No automatic promotion.",
        "origin": "Last earlier observed completed tournament country across disciplines; after gap >7/14/21/28 days use federation-home country. Retain inbound trip across rounds. Unknown locations remain missing; doubles average partners.",
        "distance": "Great-circle km between cached country representative points, not actual flights/residences/venues. Geography-only and always-home controls distinguish geographic proxy signal from itineraries.",
    })
    receipts = e.observations_before(cutoff, e.OBSERVATIONS_PATH)
    matches, drops, names = e.load_all_matches(ROOT / "data/raw/bwf_matches", observed_results=receipts,
                                              category_catalog=base.category_catalog)
    stamp = cutoff.replace(tzinfo=None)
    matches = [m for m in matches if not e.standard_match_exclusion(m) and m.get("match_status") in (None, "", "F")
               and m["available_at"] < stamp and m["match_time"] < stamp]
    assert len(matches) == original["matches"], "Frozen screening cohort changed"
    history = e.FeatureBuilder(player_names=names, result_availability_policy=e.OBSERVED_AVAILABILITY_POLICY,
                               category_catalog=base.category_catalog)
    history.elo = e.EloEngine(**base.model_policy["elo"])
    history.trueskill = e.TrueSkillEngine(**base.trueskill.parameters)
    assert history.model_policy == base.model_policy
    print(f"Rebuilding frozen cohort: {len(matches):,} matches", flush=True)
    frame = history.build_dataframe(matches, as_of=stamp)
    travel = e.travel_frame(matches, e.geography())
    frame = frame.merge(travel, on="fixture_id", validate="one_to_one", how="left")
    assert len(frame) == len(matches) and not frame["host_known"].isna().any()
    result = {"cutoff": cutoff.isoformat(), "matches": len(matches), "drops": drops, "seeds": SEEDS,
              "validation": {}, "pooled_validation": {}, "holdout": {}, "comparisons": {},
              "coverage": {"host_known": float(frame["host_known"].mean()),
                           "both_distances_known_14d": float((frame["d14_km_t1"].notna() & frame["d14_km_t2"].notna()).mean())}}
    del matches, history, base, travel, receipts
    gc.collect()
    for seed in SEEDS:
        pooled, indices = {v: [] for v in VARIANTS}, []
        result["validation"][str(seed)] = {}
        for year in (2024, 2025):
            for variant in VARIANTS:
                metrics, probability, idx = fit(frame, year, variant, seed)
                result["validation"][str(seed)].setdefault(variant, {})[str(year)] = metrics
                pooled[variant].append(probability)
                if variant == "control": indices.extend(idx.tolist())
                if seed == 42:
                    expected = original["validation"][variant][str(year)]
                    for metric in ("accuracy", "log_loss"):
                        assert abs(metrics[metric] - expected[metric]) < 1e-12, (variant, year, metric)
            save("report.json", result)
        labels = frame.loc[indices, "label"].to_numpy()
        result["pooled_validation"][str(seed)] = {
            variant: {"n": len(labels), **e.compute_metrics(labels, np.concatenate(parts))}
            for variant, parts in pooled.items()}
    result["validation_summary"] = aggregate(result["pooled_validation"])
    ordered = sorted(VARIANTS, key=lambda v: result["validation_summary"][v]["log_loss"]["mean"])
    result["validation_winner"] = ordered[0]
    result["cutoff_winner"] = next(v for v in ordered if v.startswith("travel_"))
    save("selection-before-scoring-2026.json", {"locked_at_utc": e.datetime.now(e.UTC).isoformat(),
         "winner": result["validation_winner"], "cutoff_winner": result["cutoff_winner"],
         "scores": result["validation_summary"], "reused_benchmark": True})
    for seed in SEEDS:
        predictions = {}
        result["holdout"][str(seed)], result["comparisons"][str(seed)] = {}, {}
        for variant in VARIANTS:
            metrics, probability, indices = fit(frame, 2026, variant, seed)
            result["holdout"][str(seed)][variant] = metrics
            predictions[variant] = probability
            if seed == 42 and variant in original["holdout"]:
                for metric in ("accuracy", "log_loss"):
                    assert abs(metrics[metric] - original["holdout"][variant][metric]) < 1e-12
        test = frame.loc[indices]
        for variant in VARIANTS[1:]:
            result["comparisons"][str(seed)][variant] = e.paired_interval(test, predictions["control"], predictions[variant])
        result.setdefault("distance_beyond_geography_14d", {})[str(seed)] = e.paired_interval(
            test, predictions["geography_14d"], predictions["travel_14d"])
        result.setdefault("correlation_14d", {})[str(seed)] = e.correlations(test, predictions["control"])
        save("report.json", result)
    result["holdout_summary"] = aggregate(result["holdout"])
    result.update(complete=True, seed42_reproduced=True, elapsed_seconds=time.perf_counter()-started,
                  protected_files_unchanged=verify(snapshot))
    save("report.json", result)
    lines = ["# Travel Distance Repeat Check", "",
             "Three seeds (42, 7, 2026), identical frozen screening cohort and observed-v2 timing. "
             "Threshold selection uses pooled 2024-2025 results. The reused 2026 benchmark is descriptive, not new independent evidence. "
             "No production changes or model files saved.", "",
             "Country-level great-circle estimates use the previous observed completed tournament until the cutoff, then the federation-home proxy. "
             "These are not actual player journeys. Inbound distance persists across rounds; missing locations stay missing.", "",
             "Each score is the mean of three separately scored seeds, not an ensemble score. Lower log loss is better.", "",
             "| Setting | Validation Accuracy | Validation Loss | 2026 Accuracy | 2026 Loss | Loss Change vs Control |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    control = result["holdout_summary"]["control"]["log_loss"]["mean"]
    for variant in VARIANTS:
        val, held = result["validation_summary"][variant], result["holdout_summary"][variant]
        lines.append(f"| {variant} | {val['accuracy']['mean']:.3%} | {val['log_loss']['mean']:.6f} | "
                     f"{held['accuracy']['mean']:.3%} | {held['log_loss']['mean']:.6f} | {held['log_loss']['mean']-control:+.6f} |")
    lines.extend(["", f"Validation winner: `{result['validation_winner']}`. Best cutoff on validation: `{result['cutoff_winner']}`.", "",
                  "Per-seed paired differences and tournament-year bootstrap intervals are in report.json. "
                  "The original seed-42 scores were reproduced exactly. All pre-existing files passed SHA-256 checks.", ""])
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "seconds": result["elapsed_seconds"],
                      "validation_winner": result["validation_winner"], "cutoff_winner": result["cutoff_winner"],
                      "validation": result["validation_summary"], "holdout": result["holdout_summary"],
                      "protected_files_unchanged": result["protected_files_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
