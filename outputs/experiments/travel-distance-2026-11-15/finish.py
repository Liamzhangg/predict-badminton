"""Describe the completed travel screening run; never promote or settle bets."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import experiment as e

OUT = Path(__file__).resolve().parent
ROOT = Path.cwd()
report = json.loads((OUT / "report.json").read_text())
assert report["complete"]
data = pd.read_csv(OUT / "paired-holdout.csv.gz")
data["prediction_time"] = pd.to_datetime(data["prediction_time"])
residual = data["label"].to_numpy() - data["p_control"].to_numpy()
gap = data["d14_km_t1"].to_numpy() - data["d14_km_t2"].to_numpy()
known = np.isfinite(gap)
cluster = data["tournament_code"].fillna("").astype(str)
cluster = cluster.where(cluster != "", data["tournament_name"].astype(str))
cluster = cluster + ":" + data["prediction_time"].dt.year.astype(str)


def correlation(mask):
    x, y = gap[mask], residual[mask]
    frame = pd.DataFrame({"cluster": cluster[mask].to_numpy(), "n": 1, "x": x, "y": y,
                          "xx": x*x, "yy": y*y, "xy": x*y})
    totals = frame.groupby("cluster")[["n", "x", "y", "xx", "yy", "xy"]].sum().to_numpy()
    rng = np.random.default_rng(42)
    sample = totals[rng.integers(0, len(totals), (2000, len(totals)))].sum(axis=1)
    n, sx, sy, sxx, syy, sxy = sample.T
    corr = (sxy-sx*sy/n)/np.sqrt((sxx-sx*sx/n)*(syy-sy*sy/n))
    return {"n": int(mask.sum()), "tournament_year_clusters": len(totals),
            "pearson": float(np.corrcoef(x, y)[0, 1]), "pearson_cluster_ci95": np.quantile(corr, [.025, .975]).tolist(),
            "spearman_descriptive": float(pd.Series(x).rank().corr(pd.Series(y).rank())),
            "note": "Travel-gap association with baseline win residual; exploratory correlation, not a causal fatigue estimate."}


report["correlation_with_cluster_intervals"] = {
    "all_known": correlation(known),
    "first_entry_known": correlation(known & data["d14_first_entry"].to_numpy()),
}
bins = []
for name, low, high in (("at_least_3000km_closer", -np.inf, -3000), ("500_to_3000km_closer", -3000, -500),
                         ("within_500km", -500, 500), ("500_to_3000km_farther", 500, 3000),
                         ("at_least_3000km_farther", 3000, np.inf)):
    mask = known & (gap >= low) & (gap < high)
    bins.append({"team1_distance_gap": name, "n": int(mask.sum()),
                 "observed_win_rate": float(data.loc[mask, "label"].mean()),
                 "baseline_expected_win_rate": float(data.loc[mask, "p_control"].mean()),
                 "mean_residual": float(residual[mask].mean())})
report["distance_gap_bins_14d"] = bins
report["known_distance_subset"] = {}
for name in report["holdout"]:
    probability = data[f"p_{name}"].to_numpy()
    report["known_distance_subset"][name] = {"n": int(known.sum()), **e.compute_metrics(data.loc[known, "label"].to_numpy(), probability[known])}
if report["selected_travel_before_2026"] == "home_only":
    report["interpretation"] = "Validation preferred always using the federation-home proxy, not the previous-tournament itinerary. Any predictive improvement may encode geography/home advantage or country-strength differences; it does not demonstrate that longer real journeys worsen play."
report["protected_files_unchanged"] = e.verify()
e.save("report.json", report)


def score(metrics):
    return "--" if metrics is None else f"{100*metrics['accuracy']:.3f}% / {metrics['log_loss']:.6f}"


labels = {"control": "Existing 44-feature control", "geography_14d": "Home/origin/missingness flags only",
          "home_only": "Always start from federation-home country", "travel_7d": "Fallback after >7 days",
          "travel_14d": "Fallback after >14 days", "travel_21d": "Fallback after >21 days",
          "travel_28d": "Fallback after >28 days", "travel_14d_monotone": "14 days; force own longer distance to lower win chance"}
base_val = report["pooled_validation"]["control"]
base_test = report["holdout"]["control"]
lines = ["", "## Travel Distance (Country Proxy)", "",
    "Separate observed-v2 timing study, seed 42; same matches and fixed LightGBM settings. Validation is pooled 2024-2025; the fallback was locked before scoring the reused 2026 benchmark. No production bundle, default pointer or saved bets changed.", "",
    "Estimate great-circle km from the last known completed tournament's country, falling back to the current federation/home-country proxy after >7/14/21/28 days without play or with no prior history. A home-only control ignores the prior tournament. Preserve inbound distance across rounds of a continuous tournament; doubles average partner estimates. Only earlier available results update previous locations. Unknown countries remain missing.", "",
    "These are country representative points, not actual flights, residences, cities or venue coordinates. England/Scotland/Wales use GB. Host-country metadata covers " + f"{report['coverage']['host_known']:.2%}" + "; both 14-day distances are known for " + f"{report['coverage']['both_distances_known_14d']:.2%}" + " of all study matches. Geography-only flags control for home status, origin fallback and missingness.", "",
    "Coordinates: [Google country reference](https://developers.google.com/public-data/docs/canonical/countries_csv); country-code reference: [GeoNames](https://download.geonames.org/export/dump/countryInfo.txt). Local protocol, original sources, scripts, tests, matched predictions and detailed metrics: `outputs/experiments/travel-distance-2026-11-15/`.", "",
    f"Matched counts: validation {base_val['n']:,}; 2026 holdout {base_test['n']:,}. Score cells are accuracy / log loss. Unselected holdout variants were not scored.", "",
    "| Setting | Validation Accuracy / Loss | 2026 Accuracy / Loss | Helped? |", "| --- | ---: | ---: | --- |"]
for name in report["variants"]:
    validation = report["pooled_validation"][name]
    test = report["holdout"].get(name)
    reference = base_test if test else base_val
    metric = test or validation
    difference = metric["log_loss"] - reference["log_loss"]
    helped = "Control" if name == "control" else ("Lower loss" if difference < -1e-6 else "Higher loss" if difference > 1e-6 else "Essentially unchanged")
    if test is None: helped = "Validation only: " + helped
    if name == report["selected_travel_before_2026"]: helped += "; selected on validation"
    lines.append(f"| `{name}`: {labels[name]} | {score(validation)} | {score(test)} | {helped} |")
lines.extend(["", "Paired 2026 differences versus the control (candidate minus control; 2,000 tournament-year bootstrap resamples):", "",
              "| Candidate | Accuracy Change (pp) | Log-Loss Change | 95% Cluster Interval, Loss |", "| --- | ---: | ---: | --- |"])
for name, change in report["comparisons"].items():
    low, high = change["loss"]["ci95"]
    lines.append(f"| `{name}` | {change['accuracy']['mean']*100:+.3f} | {change['loss']['mean']:+.6f} | [{low:+.6f}, {high:+.6f}] |")
change = report["distance_beyond_geography_14d"]["loss"]
lines.extend(["", f"14-day distance beyond geography-only flags: loss change {change['mean']:+.6f}, cluster interval [{change['ci95'][0]:+.6f}, {change['ci95'][1]:+.6f}].", ""])
for name, value in report["correlation_with_cluster_intervals"].items():
    low, high = value["pearson_cluster_ci95"]
    lines.append(f"- {name}: Pearson correlation between extra km and baseline win residual {value['pearson']:+.4f}, 95% tournament-cluster interval [{low:+.4f}, {high:+.4f}], n={value['n']:,}. Negative would support the hypothesized direction; this is not causation.")
lines.extend(["", report.get("interpretation", "All results are exploratory; no feature is promoted automatically."),
              "The reused historical holdout and geographic proxies do not establish a durable real-travel effect or profitable bets. Production still has no travel feature.", ""])
document = ROOT / "EXPERIMENTS.md"
previous = (OUT / "EXPERIMENTS-before.md").read_bytes()
assert document.read_bytes() == previous, "Do not overwrite another edit to the summary"
document.write_bytes(previous + "\n".join(lines).encode())
e.verify()
(OUT / "summary.md").write_text("# Travel Distance Screening\n" + "\n".join(lines))
print(json.dumps({"holdout": report["holdout"], "comparisons": report["comparisons"],
                  "correlations": report["correlation_with_cluster_intervals"], "bins": bins}, indent=2))
