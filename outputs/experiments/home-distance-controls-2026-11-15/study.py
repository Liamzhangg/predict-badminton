"""Isolated home/country/distance ablation; never promote models or edit bets."""
from __future__ import annotations

import gc
import heapq
import importlib.util
from itertools import groupby
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "outputs/experiments/travel-distance-2026-11-15"
spec = importlib.util.spec_from_file_location("travel_screening", ORIGINAL / "experiment.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
e.OUT = OUT
SEEDS = (42, 7, 2026)
SIDES = ("t1", "t2", "diff")
LOCATION = ["host_known", "country_known_t1", "country_known_t2"]
HOME = [f"home_{side}" for side in SIDES]
STRENGTH = [f"country_{quantity}_{side}" for quantity in ("rating", "log_n") for side in SIDES]
IDENTITY = ["country_id_t1", "country_id_t2"]
DISTANCE = [f"home_log_km_{side}" for side in SIDES]
VARIANTS = {
    "raw_control": [], "baseline": LOCATION, "home": LOCATION + HOME,
    "country_strength": LOCATION + STRENGTH + IDENTITY,
    "home_country_strength": LOCATION + HOME + STRENGTH + IDENTITY,
    "distance": LOCATION + DISTANCE, "home_distance": LOCATION + HOME + DISTANCE,
    "country_strength_distance": LOCATION + STRENGTH + IDENTITY + DISTANCE,
    "all": LOCATION + HOME + STRENGTH + IDENTITY + DISTANCE,
}
CONTRASTS = {
    "home_alone": ("baseline", "home"), "country_strength_alone": ("baseline", "country_strength"),
    "distance_alone": ("baseline", "distance"),
    "home_beyond_country_strength": ("country_strength", "home_country_strength"),
    "country_strength_beyond_home": ("home", "home_country_strength"),
    "distance_beyond_home": ("home", "home_distance"),
    "distance_beyond_country_strength": ("country_strength", "country_strength_distance"),
    "distance_beyond_both": ("home_country_strength", "all"),
}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def protect():
    files = {p for base in (ROOT / "data", ROOT / "outputs") for p in base.rglob("*")
             if p.is_file() and not p.is_relative_to(OUT)}
    files.update(ROOT / p for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines())
    files.add(ROOT / "EXPERIMENTS.md")
    snapshot = {str(p.relative_to(ROOT)): e.sha(p) for p in sorted(files) if p.is_file()}
    save("protected-before.json", snapshot)
    shutil.copy2(ROOT / "EXPERIMENTS.md", OUT / "EXPERIMENTS-before.md")
    return snapshot


def verify(snapshot):
    changed = [name for name, digest in snapshot.items() if not (ROOT / name).is_file() or e.sha(ROOT / name) != digest]
    assert not changed, changed
    return len(snapshot)


class CountryStrength:
    def __init__(self, codes):
        self.codes = codes
        self.engine = e.EloEngine(k_base=20, alpha_player=1, score_margin_weight=0)

    def country(self, value):
        code = str(value or "").strip().upper()
        return code if code in self.codes else None

    def preview(self, country, discipline):
        if country is None:
            return float("nan"), float("nan")
        rating, _ = self.engine.team_elo([country], discipline)
        return rating, math.log1p(self.engine.team_disc_count([country], discipline))

    def observe(self, match):
        first, second = [self.country(match.get(f"team{side}_country")) for side in (1, 2)]
        if first is None or second is None or first == second:
            return
        # Country IDs stand in for players; neutral round/tier gives uniform K=20.
        self.engine.update([first], [second], match["discipline"], "", 6, match["winner"])


def context_frame(matches, locations):
    coordinates, codes, hosts, named_hosts = locations
    history, pending, rows = CountryStrength(codes), [], []
    ordered = sorted(matches, key=lambda m: (m["prediction_time"], str(m["match_id"]), m["fixture_id"]))
    for stamp, group in groupby(ordered, key=lambda m: m["prediction_time"]):
        while pending and pending[0][0] < stamp:
            _, _, completed = heapq.heappop(pending)
            history.observe(completed)
        for match in group:
            edition = str(match.get("tournament_code") or e.normalized(match["tournament_name"])).casefold()
            host = hosts.get(edition) or named_hosts.get(e.normalized(match["tournament_name"]))
            row = {"fixture_id": match["fixture_id"], "host_known": float(host in coordinates)}
            values = {key: [] for key in ("home", "country_rating", "country_log_n", "home_log_km")}
            for side in (1, 2):
                country = history.country(match.get(f"team{side}_country"))
                origin = codes.get(country)
                row[f"country_id_t{side}"] = country
                row[f"country_known_t{side}"] = float(origin in coordinates)
                known = origin in coordinates and host in coordinates
                values["home"].append(float(origin == host) if known else float("nan"))
                values["home_log_km"].append(math.log1p(e.distance(origin, host, coordinates)) if known else float("nan"))
                rating, count = history.preview(country, match["discipline"])
                values["country_rating"].append(rating)
                values["country_log_n"].append(count)
            for name, (first, second) in values.items():
                row.update({f"{name}_t1": first, f"{name}_t2": second, f"{name}_diff": first-second})
            rows.append(row)
            key = (match["match_time"], str(match["match_id"]), match["fixture_id"])
            heapq.heappush(pending, (match["available_at"], key, match))
    return pd.DataFrame(rows)


def matrices(train, test, variant):
    columns = e.FEATURE_COLS + VARIANTS[variant]
    first, second = train[columns].copy(), test[columns].copy()
    if "country_id_t1" in columns:
        categories = sorted(set(train[IDENTITY].stack().dropna()))
        dtype = pd.CategoricalDtype(categories=categories)
        for name in IDENTITY:
            first[name] = pd.Categorical(first[name].where(first[name].isin(categories)), dtype=dtype)
            second[name] = pd.Categorical(second[name].where(second[name].isin(categories)), dtype=dtype)
    return first, second


def reverse(frame, variant):
    result = e.swap_features(frame[e.FEATURE_COLS])
    for name in VARIANTS[variant]:
        if name == "host_known":
            result[name] = frame[name]
        elif name.endswith("_diff"):
            result[name] = -frame[name]
        else:
            result[name] = frame[name[:-1] + ("2" if name.endswith("1") else "1")]
    return result


def predict(model, frame, variant):
    both = pd.concat([frame, reverse(frame, variant)], ignore_index=True)
    probabilities = model.predict_proba(both)[:, 1]
    return e.average_directions(np.column_stack((probabilities[:len(frame)], probabilities[len(frame):])))


def fit(frame, year, variant, seed):
    boundary = pd.Timestamp(year, 1, 1)
    train = frame.loc[(frame["prediction_time"] < boundary) & (frame["available_at"] < boundary)]
    test = frame.loc[frame["prediction_time"].dt.year == year]
    assert not set(train["fixture_id"]) & set(test["fixture_id"])
    fitting, stopping = e._chronological_validation_split(train)
    first, second = matrices(fitting, stopping, variant)
    params = {"n_jobs": 4, "random_state": seed}
    provisional = e.train_lgbm(first, fitting["label"], second, stopping["label"], params)
    trees = int(provisional.best_iteration_ or 500)
    first, second = matrices(train, test, variant)
    model = e.train_lgbm(first, train["label"], params={**params, "n_estimators": trees})
    probability = predict(model, second, variant)
    metrics = {"n": len(test), "train_n": len(train), "trees": trees,
               **e.compute_metrics(test["label"].to_numpy(), probability)}
    print(f"seed={seed} year={year} {variant}: accuracy={metrics['accuracy']:.6f} loss={metrics['log_loss']:.9f}", flush=True)
    del provisional, model, first, second
    gc.collect()
    return metrics, probability, test.index


def aggregate(results):
    return {variant: {"n": results[str(SEEDS[0])][variant]["n"], **{
        key: {"mean": float(np.mean([results[str(seed)][variant][key] for seed in SEEDS])),
              "min": float(min(results[str(seed)][variant][key] for seed in SEEDS)),
              "max": float(max(results[str(seed)][variant][key] for seed in SEEDS))}
        for key in ("accuracy", "log_loss")}} for variant in VARIANTS}


def main():
    started = time.perf_counter()
    logging.basicConfig(level=logging.WARNING)
    assert not (OUT / "report.json").exists(), "Do not overwrite a previous study"
    original = json.loads((ORIGINAL / "report.json").read_text())
    repeated = json.loads((ROOT / "outputs/experiments/travel-distance-confirmation-2026-11-15/report.json").read_text())
    cutoff = e.datetime.fromisoformat(original["cutoff"])
    snapshot = protect()
    for name in ("countries.html", "countryInfo.txt"):
        shutil.copy2(ORIGINAL / name, OUT / name)
    directory = e.forecast_bundle_directory()
    base = e.FeatureBuilder.load(directory / "feature_state.pkl")
    assert base.model_policy["result_availability"] == e.OBSERVED_AVAILABILITY_POLICY
    save("protocol.json", {
        "started_at_utc": e.datetime.now(e.UTC).isoformat(), "frozen_cutoff": cutoff.isoformat(),
        "seeds": SEEDS, "variants": VARIANTS, "contrasts": CONTRASTS,
        "primary": "all minus home_country_strength: does home-country distance add beyond home advantage and country strength?",
        "home": "1 when represented-country geographic code matches host, 0 otherwise; missing when location unknown. UK subdivisions use GB, as in the earlier distance study.",
        "country_strength": "Uniform K=20 country Elo, initial 1500, separate disciplines, no decay or score weighting; skip domestic/unknown-country matches. Update only when available_at < target prediction_time, deterministic available-time order. Include log1p count and represented-country categories fit only on the training split; ENG/SCO/WAL remain separate country-rating identities.",
        "distance": "log1p country representative-point km, each side and difference; represented country always used, not previous tournament. Team-country proxy, including doubles, not verified individual residences or actual journeys.",
        "coverage": "Same host-known and represented-country-location-known flags in every factorial variant. Raw 44-feature control is an extra reproducibility anchor.",
        "timing": "Matching observed-v2 historical replay; training labels available before each year boundary; availability-purged inner early stopping then earlier-year refit; two-order averaged probabilities.",
        "evaluation": "2024-2025 validation plus reused retrospective 2026 benchmark; three seeds are not independent test sets. No tuning of feature definitions or automatic promotion. Fixed LightGBM defaults.",
        "base_bundle": str(directory.relative_to(ROOT)), "base_policy": base.model_policy,
        "source_script_sha256": e.sha(ORIGINAL / "experiment.py"),
    })
    receipts = e.observations_before(cutoff, e.OBSERVATIONS_PATH)
    matches, drops, names = e.load_all_matches(ROOT / "data/raw/bwf_matches", observed_results=receipts,
                                              category_catalog=base.category_catalog)
    stamp = cutoff.replace(tzinfo=None)
    matches = [m for m in matches if not e.standard_match_exclusion(m) and m.get("match_status") in (None, "", "F")
               and m["available_at"] < stamp and m["match_time"] < stamp]
    assert len(matches) == original["matches"], "Frozen cohort changed"
    history = e.FeatureBuilder(player_names=names, result_availability_policy=e.OBSERVED_AVAILABILITY_POLICY,
                               category_catalog=base.category_catalog)
    history.elo = e.EloEngine(**base.model_policy["elo"])
    history.trueskill = e.TrueSkillEngine(**base.trueskill.parameters)
    assert history.model_policy == base.model_policy
    print(f"Building frozen matched cohort: {len(matches):,} matches", flush=True)
    frame = history.build_dataframe(matches, as_of=stamp)
    context = context_frame(matches, e.geography())
    frame = frame.merge(context, on="fixture_id", validate="one_to_one", how="left")
    assert len(frame) == len(matches) and not frame["host_known"].isna().any()
    result = {"matches": len(matches), "cutoff": cutoff.isoformat(), "seeds": SEEDS, "drops": drops,
              "validation": {}, "pooled_validation": {}, "holdout": {}, "contrasts": {},
              "coverage": {"host_known": float(frame["host_known"].mean()),
                           "both_distances_known": float((frame["home_log_km_t1"].notna() & frame["home_log_km_t2"].notna()).mean())}}
    del matches, history, base, context, receipts
    gc.collect()
    for seed in SEEDS:
        pooled, indices = {name: [] for name in VARIANTS}, []
        result["validation"][str(seed)] = {}
        for year in (2024, 2025):
            for variant in VARIANTS:
                metrics, probability, idx = fit(frame, year, variant, seed)
                result["validation"][str(seed)].setdefault(variant, {})[str(year)] = metrics
                pooled[variant].append(probability)
                if variant == "raw_control":
                    indices.extend(idx.tolist())
                    expected = repeated["validation"][str(seed)]["control"][str(year)]
                    for key in ("accuracy", "log_loss"):
                        assert abs(metrics[key] - expected[key]) < 1e-12
            save("report.json", result)
        labels = frame.loc[indices, "label"].to_numpy()
        probabilities = {name: np.concatenate(parts) for name, parts in pooled.items()}
        result["pooled_validation"][str(seed)] = {
            name: {"n": len(labels), **e.compute_metrics(labels, probability)} for name, probability in probabilities.items()}
        result.setdefault("validation_primary", {})[str(seed)] = e.paired_interval(
            frame.loc[indices], probabilities["home_country_strength"], probabilities["all"])
    result["validation_summary"] = aggregate(result["pooled_validation"])
    save("validation-before-2026.json", {"locked_at_utc": e.datetime.now(e.UTC).isoformat(),
         "primary_fixed_before_run": CONTRASTS["distance_beyond_both"], "scores": result["validation_summary"],
         "reused_benchmark": True})
    for seed in SEEDS:
        probabilities = {}
        result["holdout"][str(seed)] = {}
        for variant in VARIANTS:
            metrics, probability, indices = fit(frame, 2026, variant, seed)
            result["holdout"][str(seed)][variant], probabilities[variant] = metrics, probability
            if variant == "raw_control":
                for key in ("accuracy", "log_loss"):
                    assert abs(metrics[key] - repeated["holdout"][str(seed)]["control"][key]) < 1e-12
        result["contrasts"][str(seed)] = {name: e.paired_interval(frame.loc[indices], probabilities[control], probabilities[candidate])
                                             for name, (control, candidate) in CONTRASTS.items()}
        save("report.json", result)
    result["holdout_summary"] = aggregate(result["holdout"])
    result.update(complete=True, raw_controls_reproduced=True, elapsed_seconds=time.perf_counter()-started,
                  protected_files_unchanged=verify(snapshot))
    save("report.json", result)
    lines = ["# Home Advantage, Country Strength and Home-Country Distance", "",
             "Three seeds, identical frozen cohort and observed-v2 timing. All factorial models include the same location-coverage flags. "
             "Country strength includes point-in-time country Elo plus training-only country identities. "
             "Distance is a country-level geographic proxy, not actual travel. Reused 2026 benchmark; no production promotion.", "",
             "Scores below are mean per-seed metrics, not ensemble scores. Lower log loss is better.", "",
             "| Variant | Validation Accuracy | Validation Loss | 2026 Accuracy | 2026 Loss |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        val, held = result["validation_summary"][variant], result["holdout_summary"][variant]
        lines.append(f"| {variant} | {val['accuracy']['mean']:.3%} | {val['log_loss']['mean']:.6f} | "
                     f"{held['accuracy']['mean']:.3%} | {held['log_loss']['mean']:.6f} |")
    lines.extend(["", "## Distance Beyond Both Controls", "",
                  "| Seed | 2026 Loss Change | 95% Tournament-Cluster Interval |", "| --- | ---: | --- |"])
    for seed in SEEDS:
        change = result["contrasts"][str(seed)]["distance_beyond_both"]["loss"]
        low, high = change["ci95"]
        lines.append(f"| {seed} | {change['mean']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
    lines.extend(["", "Definitions, formulas, split policies, per-seed metrics and all paired contrasts are in protocol.json/report.json. "
                  "All original controls reproduced exactly. All protected pre-existing files remained unchanged. "
                  "No models, feature matrices or per-match prediction arrays retained.", ""])
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "seconds": result["elapsed_seconds"],
                      "validation": result["validation_summary"], "holdout": result["holdout_summary"],
                      "primary": {str(seed): result["contrasts"][str(seed)]["distance_beyond_both"] for seed in SEEDS},
                      "protected_files_unchanged": result["protected_files_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
