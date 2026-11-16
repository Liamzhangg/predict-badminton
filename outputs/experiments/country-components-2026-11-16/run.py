"""Separate country Elo, identity and history counts in a frozen research run."""
from __future__ import annotations

import gc
import importlib.util
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "outputs/experiments/home-distance-controls-2026-11-15"
spec = importlib.util.spec_from_file_location("country_controls", PREVIOUS / "study.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
ORIGINAL_COMBINED = list(s.VARIANTS["country_strength"])
RATINGS = [f"country_rating_{side}" for side in s.SIDES]
COUNTS = [f"country_log_n_{side}" for side in s.SIDES]
VARIANTS = {
    "baseline": s.LOCATION,
    "counts_only": s.LOCATION + COUNTS,
    "elo_only": s.LOCATION + RATINGS,
    "identity_only": s.LOCATION + s.IDENTITY,
    "elo_identity": s.LOCATION + RATINGS + s.IDENTITY,
    "original_combined": ORIGINAL_COMBINED,
}
CONTRASTS = {
    "elo_alone": ("baseline", "elo_only"),
    "identity_alone": ("baseline", "identity_only"),
    "counts_alone": ("baseline", "counts_only"),
    "identity_beyond_elo": ("elo_only", "elo_identity"),
    "elo_beyond_identity": ("identity_only", "elo_identity"),
    "counts_beyond_both": ("elo_identity", "original_combined"),
    "original_combined_vs_baseline": ("baseline", "original_combined"),
}
# Configure the imported research helpers in this process; do not edit their files.
s.OUT = OUT
s.e.OUT = OUT
s.VARIANTS = VARIANTS


def comparisons(frame, probabilities):
    return {name: s.e.paired_interval(frame, probabilities[control], probabilities[candidate])
            for name, (control, candidate) in CONTRASTS.items()}


def check_anchor(metrics, previous, variant):
    anchor = {"baseline": "baseline", "original_combined": "country_strength"}.get(variant)
    if anchor:
        for key in ("n", "train_n", "trees", "accuracy", "log_loss"):
            assert abs(metrics[key] - previous[anchor][key]) < 1e-12, (variant, key)


def main():
    logging.basicConfig(level=logging.WARNING)
    started = time.perf_counter()
    assert not (OUT / "report.json").exists(), "Do not overwrite a previous study"
    previous = json.loads((PREVIOUS / "report.json").read_text())
    cutoff = s.e.datetime.fromisoformat(previous["cutoff"])
    snapshot = s.protect()
    for name in ("countries.html", "countryInfo.txt"):
        shutil.copy2(PREVIOUS / name, OUT / name)
    directory = s.e.forecast_bundle_directory()
    base = s.e.FeatureBuilder.load(directory / "feature_state.pkl")
    assert base.model_policy["result_availability"] == s.e.OBSERVED_AVAILABILITY_POLICY
    s.save("protocol.json", {
        "started_at_utc": s.e.datetime.now(s.e.UTC).isoformat(), "frozen_cutoff": cutoff.isoformat(),
        "seeds": s.SEEDS, "variants": VARIANTS, "contrasts": CONTRASTS,
        "primary": ["identity_beyond_elo", "elo_beyond_identity"],
        "rating": "Country Elo only: side ratings and difference, no counts/identity. Same uniform K=20, initial 1500, per-discipline, past-available international results only, as previous study.",
        "identity": "Represented-country categorical labels only, shared vocabulary across sides fit on each training split; unseen test categories become missing. No ratings or counts in identity_only.",
        "counts": "Separate log1p counts of earlier available international matches, both sides and difference. Counts-only and counts-beyond-both prevent attributing activity effects to Elo or identity.",
        "fixed_context": "Same location-coverage flags in every variant. Home indicators and distance excluded throughout. Original combined column order preserved for exact reproducibility.",
        "timing": "Observed-v2 replay, available_at strictly before target time; train labels available before year boundary; availability-purged internal early stopping then earlier-year refit; fixed LightGBM defaults, two-order probabilities.",
        "evaluation": "Pooled 2024-2025 validation and reused retrospective 2026 benchmark; no fresh independent holdout, tuning, model promotion or saved forecast/bet updates. Three seeds are not independent test sets.",
        "base_bundle": str(directory.relative_to(ROOT)), "base_policy": base.model_policy,
        "research_helper": str((PREVIOUS / "study.py").relative_to(ROOT)),
        "research_helper_sha256": s.e.sha(PREVIOUS / "study.py"),
    })
    receipts = s.e.observations_before(cutoff, s.e.OBSERVATIONS_PATH)
    matches, drops, names = s.e.load_all_matches(ROOT / "data/raw/bwf_matches", observed_results=receipts,
                                                category_catalog=base.category_catalog)
    stamp = cutoff.replace(tzinfo=None)
    matches = [m for m in matches if not s.e.standard_match_exclusion(m) and m.get("match_status") in (None, "", "F")
               and m["available_at"] < stamp and m["match_time"] < stamp]
    assert len(matches) == previous["matches"], "Frozen cohort changed"
    history = s.e.FeatureBuilder(player_names=names, result_availability_policy=s.e.OBSERVED_AVAILABILITY_POLICY,
                                 category_catalog=base.category_catalog)
    history.elo = s.e.EloEngine(**base.model_policy["elo"])
    history.trueskill = s.e.TrueSkillEngine(**base.trueskill.parameters)
    assert history.model_policy == base.model_policy
    print(f"Building identical frozen cohort: {len(matches):,} matches", flush=True)
    frame = history.build_dataframe(matches, as_of=stamp)
    context = s.context_frame(matches, s.e.geography())
    frame = frame.merge(context, on="fixture_id", validate="one_to_one", how="left")
    assert len(frame) == len(matches) and not frame["host_known"].isna().any()
    result = {"matches": len(matches), "cutoff": cutoff.isoformat(), "seeds": s.SEEDS,
              "drops": drops, "variants": VARIANTS, "validation": {}, "pooled_validation": {},
              "validation_contrasts": {}, "holdout": {}, "holdout_contrasts": {}}
    del matches, history, base, context, receipts
    gc.collect()
    for seed in s.SEEDS:
        result["validation"][str(seed)] = {}
        pooled, indices = {name: [] for name in VARIANTS}, []
        for year in (2024, 2025):
            for variant in VARIANTS:
                metrics, probability, idx = s.fit(frame, year, variant, seed)
                result["validation"][str(seed)].setdefault(variant, {})[str(year)] = metrics
                check_anchor(metrics, {name: years[str(year)] for name, years in previous["validation"][str(seed)].items()}, variant)
                pooled[variant].append(probability)
                if variant == "baseline":
                    indices.extend(idx.tolist())
            s.save("report.json", result)
        labels = frame.loc[indices, "label"].to_numpy()
        probabilities = {name: np.concatenate(parts) for name, parts in pooled.items()}
        result["pooled_validation"][str(seed)] = {
            name: {"n": len(labels), **s.e.compute_metrics(labels, probability)} for name, probability in probabilities.items()}
        result["validation_contrasts"][str(seed)] = comparisons(frame.loc[indices], probabilities)
        s.save("report.json", result)
    result["validation_summary"] = s.aggregate(result["pooled_validation"])
    s.save("validation-before-2026.json", {"locked_at_utc": s.e.datetime.now(s.e.UTC).isoformat(),
           "fixed_primary": ["identity_beyond_elo", "elo_beyond_identity"],
           "summary": result["validation_summary"], "reused_benchmark": True})
    for seed in s.SEEDS:
        probabilities = {}
        result["holdout"][str(seed)] = {}
        for variant in VARIANTS:
            metrics, probability, indices = s.fit(frame, 2026, variant, seed)
            check_anchor(metrics, previous["holdout"][str(seed)], variant)
            result["holdout"][str(seed)][variant], probabilities[variant] = metrics, probability
        result["holdout_contrasts"][str(seed)] = comparisons(frame.loc[indices], probabilities)
        s.save("report.json", result)
    result["holdout_summary"] = s.aggregate(result["holdout"])
    result.update(complete=True, anchors_reproduced=True, elapsed_seconds=time.perf_counter()-started,
                  protected_files_unchanged=s.verify(snapshot))
    s.save("report.json", result)
    lines = ["# Country Elo, Identity and History Counts", "",
             "Same frozen cohort, three seeds and observed-v2 timing as the earlier country-control study. "
             "Home advantage and distance are off throughout; location-coverage flags are held constant. "
             "Counts are separate from Elo ratings. Country categories are training-only. "
             "The reused 2026 benchmark is retrospective, not independent confirmation. No production changes.", "",
             "Mean per-seed metrics, not ensemble scores. Lower log loss is better.", "",
             "| Setting | Validation Accuracy | Validation Loss | 2026 Accuracy | 2026 Loss |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        val, held = result["validation_summary"][variant], result["holdout_summary"][variant]
        lines.append(f"| {variant} | {val['accuracy']['mean']:.3%} | {val['log_loss']['mean']:.6f} | "
                     f"{held['accuracy']['mean']:.3%} | {held['log_loss']['mean']:.6f} |")
    lines.extend(["", "Protocol, exact features, fold/seed scores and paired tournament-cluster intervals are in protocol.json/report.json. "
                  "Baseline and original-combined anchors reproduced exactly in every seed/year. "
                  "No models, feature matrices or per-match prediction arrays were retained.", ""])
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "seconds": result["elapsed_seconds"],
                      "validation": result["validation_summary"], "holdout": result["holdout_summary"],
                      "primary": {str(seed): {name: result["holdout_contrasts"][str(seed)][name]
                                               for name in ("identity_beyond_elo", "elo_beyond_identity", "counts_beyond_both")}
                                  for seed in s.SEEDS},
                      "protected_files_unchanged": result["protected_files_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
