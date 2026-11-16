"""Isolated, availability-aware country-distance screening experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import gc
import hashlib
import heapq
from html.parser import HTMLParser
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import time
import unicodedata
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from predict_badminton.config import FEATURE_COLS, forecast_bundle_directory
from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import observations_before, OBSERVATIONS_PATH
from predict_badminton.evaluate import compute_metrics
from predict_badminton.features import FeatureBuilder, EloEngine
from predict_badminton.match_rules import standard_match_exclusion
from predict_badminton.model import _chronological_validation_split, train_lgbm
from predict_badminton.probabilities import swap_features, average_directions
from predict_badminton.temporal import OBSERVED_AVAILABILITY_POLICY
from predict_badminton.trueskill import TrueSkillEngine

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
DAYS = (0, 7, 14, 21, 28)
GEO = [f"travel_{quantity}_{side}" for quantity in ("home_origin", "origin_missing", "at_home")
       for side in ("t1", "t2", "diff")]
DISTANCE = [f"travel_log_km_{side}" for side in ("t1", "t2", "diff")]
VARIANTS = {"control": (None, False, False), "geography_14d": (14, False, False),
            "home_only": (0, True, False), **{f"travel_{days}d": (days, True, False) for days in DAYS if days},
            "travel_14d_monotone": (14, True, True)}
GOOGLE = "https://developers.google.com/public-data/docs/canonical/countries_csv"
GEONAMES = "https://download.geonames.org/export/dump/countryInfo.txt"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(value):
    value = unicodedata.normalize("NFKD", str(value)).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.table, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table": self.table = []
        elif tag == "tr" and self.table is not None: self.row = []
        elif tag in ("td", "th") and self.row is not None: self.cell = []

    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip()); self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row); self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table); self.table = None


def prepare():
    sources = []
    for url, filename in ((GOOGLE, "countries.html"), (GEONAMES, "countryInfo.txt")):
        request = Request(url, headers={"User-Agent": "predict-badminton research"})
        with urlopen(request, timeout=60) as response:
            body = response.read()
        path = OUT / filename
        path.write_bytes(body)
        sources.append({"url": url, "file": filename, "sha256": sha(path), "retrieved_at_utc": datetime.now(UTC).isoformat()})
    save("geography-sources.json", sources)
    print("Saved two primary geographic reference sources; no match results fetched.", flush=True)


def geography():
    parser = Tables()
    parser.feed((OUT / "countries.html").read_text())
    table = next(t for t in parser.tables if t and t[0][:4] == ["country", "latitude", "longitude", "name"])
    coordinates, names, codes = {}, {}, {}
    for row in table[1:]:
        if len(row) != 4 or not row[1] or not row[2]: continue
        try: point = (float(row[1]), float(row[2]))
        except ValueError: continue
        assert -90 <= point[0] <= 90 and -180 <= point[1] <= 180
        coordinates[row[0]] = point
        names[normalized(row[3])] = row[0]
        codes[row[0]] = row[0]
    for line in (OUT / "countryInfo.txt").read_text().splitlines():
        if not line or line.startswith("#"): continue
        fields = line.split("\t")
        if len(fields) >= 5 and fields[0] in coordinates:
            codes[fields[1]] = fields[0]
            names[normalized(fields[4])] = fields[0]
    aliases = {"England": "GB", "Scotland": "GB", "Wales": "GB", "Northern Ireland": "GB",
               "Chinese Taipei": "TW", "Taipei": "TW", "Hong Kong China": "HK", "Hong Kong SAR": "HK",
               "Macau China": "MO", "Macau SAR": "MO", "Korea": "KR", "Korea Republic of": "KR",
               "Republic of Korea": "KR", "Vietnam": "VN", "Czechia": "CZ", "Turkiye": "TR",
               "Trinidad & Tobago": "TT", "IR Iran": "IR", "Iran Islamic Republic of": "IR",
               "United States of America": "US", "U S A": "US"}
    names.update({normalized(k): v for k, v in aliases.items()})
    candidates, host_codes, host_names = defaultdict(set), defaultdict(set), defaultdict(set)
    cache = ROOT / "data/source_cache/bwf"
    paths = sorted(set(cache.rglob("tournaments.json")) | set(cache.rglob("calendar.json"))
                   | set(cache.rglob("api-vue-tournaments-search-*.json")))
    unknown = Counter()
    for path in paths:
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else (payload.get("results") or {}).get("data", [])
        for row in rows:
            if not isinstance(row, dict) or not row.get("country"): continue
            country = names.get(normalized(row["country"]))
            if country is None: unknown[row["country"]] += 1; continue
            flag = Path(urlparse(str(row.get("flag_url") or "")).path).stem.upper()
            if flag: candidates[flag].add(country)
            for field in ("code", "id"):
                if row.get(field) is not None: host_codes[str(row[field]).strip().casefold()].add(country)
            host_names[normalized(row.get("name", ""))].add(country)
    # Official calendar flag/name matches disambiguate BWF/IOC codes (e.g. BRN=Bahrain).
    for code, options in candidates.items():
        if len(options) == 1: codes[code] = next(iter(options))
        else: codes.pop(code, None)
    codes.update({"ENG": "GB", "SCO": "GB", "WAL": "GB", "NIR": "GB"})
    unique = lambda mapping: {k: next(iter(v)) for k, v in mapping.items() if k and len(v) == 1}
    hosts, named_hosts = unique(host_codes), unique(host_names)
    save("geography-coverage.json", {"country_points": len(coordinates), "calendar_sources": len(paths),
         "unique_edition_codes": len(hosts), "unresolved_calendar_countries": unknown,
         "country_aliases": aliases, "resolved_bwf_codes": codes,
         "limitations": "Country representative points, not venue/city/airport coordinates. England/Scotland/Wales use GB. Federation is a home-country proxy, not verified residence. No unknown location is replaced by zero distance."})
    return coordinates, codes, hosts, named_hosts


def distance(a, b, coordinates):
    if a not in coordinates or b not in coordinates: return float("nan")
    lat1, lon1 = map(math.radians, coordinates[a]); lat2, lon2 = map(math.radians, coordinates[b])
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


class Travel:
    def __init__(self, coordinates, codes):
        self.coordinates, self.codes = coordinates, codes
        self.previous = {}
        self.trips = {days: {} for days in DAYS}

    def preview(self, match, host, days):
        stamp, edition = match["prediction_time"], match["edition"]
        teams, changes = [], []
        for side in (1, 2):
            players = []
            home = self.codes.get(str(match.get(f"team{side}_country") or "").upper())
            for pid in match[f"team{side}_ids"]:
                trip = self.trips[days].get(pid)
                if trip and trip["edition"] == edition and (stamp-trip["last_target"]).total_seconds() <= days*86400:
                    current = dict(trip)
                else:
                    prior = self.previous.get(pid)
                    if prior and prior["time"] >= stamp: prior = None
                    gap = (stamp-prior["time"]).total_seconds()/86400 if prior else float("inf")
                    assert gap >= 0
                    home_origin = prior is None or gap > days
                    origin = home if home_origin else prior["host"]
                    current = {"edition": edition, "km": distance(origin, host, self.coordinates),
                               "home_origin": float(home_origin), "origin_missing": float(origin is None),
                               "at_home": float(host == home) if host is not None and home is not None else float("nan"),
                               "first_target": stamp}
                current["last_target"] = stamp
                players.append(current)
                changes.append((pid, current))
            km = [p["km"] for p in players]
            teams.append({"km": float(np.mean(km)) if all(math.isfinite(x) for x in km) else float("nan"),
                          **{key: float(np.mean([p[key] for p in players])) for key in ("home_origin", "origin_missing", "at_home")},
                          "first_entry": all(p["first_target"] == stamp for p in players)})
        row = {}
        for key in ("home_origin", "origin_missing", "at_home", "log_km"):
            values = [math.log1p(team["km"]) if key == "log_km" else team[key] for team in teams]
            row.update({f"travel_{key}_t1": values[0], f"travel_{key}_t2": values[1], f"travel_{key}_diff": values[0]-values[1]})
        row.update(km_t1=teams[0]["km"], km_t2=teams[1]["km"], first_entry=teams[0]["first_entry"] and teams[1]["first_entry"])
        return row, changes

    def commit_targets(self, changes, days):
        by_player = defaultdict(list)
        for pid, trip in changes: by_player[pid].append(trip)
        for pid, trips in by_player.items():
            if len({t["edition"] for t in trips}) == 1: self.trips[days][pid] = trips[0]
            else: self.trips[days].pop(pid, None)

    def observe(self, match, host):
        for pid in set(match["team1_ids"] + match["team2_ids"]):
            previous = self.previous.get(pid)
            current = {"time": match["match_time"], "host": host, "edition": match["edition"]}
            if previous is None or current["time"] > previous["time"]: self.previous[pid] = current
            elif current["time"] == previous["time"] and current["edition"] != previous["edition"]:
                self.previous[pid] = {**current, "host": None}


def travel_frame(matches, locations):
    coordinates, codes, hosts, named_hosts = locations
    history, pending, rows = Travel(coordinates, codes), [], []
    ordered = sorted(matches, key=lambda m: (m["prediction_time"], str(m["match_id"]), m["fixture_id"]))
    from itertools import groupby
    for stamp, group in groupby(ordered, key=lambda m: m["prediction_time"]):
        while pending and pending[0][0] < stamp:
            _, _, match, host = heapq.heappop(pending)
            history.observe(match, host)
        commits = defaultdict(list)
        for source in group:
            match = dict(source)
            match["edition"] = str(match.get("tournament_code") or normalized(match["tournament_name"])).casefold()
            host = hosts.get(match["edition"]) or named_hosts.get(normalized(match["tournament_name"]))
            row = {"fixture_id": match["fixture_id"], "host_known": host is not None}
            for days in DAYS:
                values, changes = history.preview(match, host, days)
                row.update({f"d{days}_{k}": v for k, v in values.items()})
                commits[days].extend(changes)
            rows.append(row)
            if len(rows) % 50000 == 0: print(f"Travel inputs: {len(rows):,}/{len(matches):,}", flush=True)
            key = (match["match_time"], str(match["match_id"]), match["fixture_id"])
            heapq.heappush(pending, (match["available_at"], key, match, host))
        for days, changes in commits.items(): history.commit_targets(changes, days)
    return pd.DataFrame(rows)


def columns(variant):
    days, with_distance, _ = VARIANTS[variant]
    return FEATURE_COLS + ([] if days is None else GEO) + (DISTANCE if with_distance else [])


def inputs(frame, variant):
    days, _, _ = VARIANTS[variant]
    result = frame[FEATURE_COLS].copy()
    for name in columns(variant)[len(FEATURE_COLS):]: result[name] = frame[f"d{days}_{name}"]
    return result


def predict(model, frame, variant):
    original = inputs(frame, variant)
    reverse = swap_features(original[FEATURE_COLS])
    for name in columns(variant)[len(FEATURE_COLS):]:
        if name.endswith("_diff"): reverse[name] = -original[name]
        else: reverse[name] = original[name[:-1] + ("2" if name.endswith("1") else "1")]
    scores = model.predict_proba(pd.concat([original, reverse], ignore_index=True))[:, 1]
    return average_directions(np.column_stack((scores[:len(frame)], scores[len(frame):])))


def fit_score(frame, year, variant):
    start = time.perf_counter()
    boundary = pd.Timestamp(year, 1, 1)
    train = frame.loc[(frame["prediction_time"] < boundary) & (frame["available_at"] < boundary)]
    test = frame.loc[frame["prediction_time"].dt.year == year]
    fit, stop = _chronological_validation_split(train)
    params = {"n_jobs": 4, "random_state": 42}
    if VARIANTS[variant][2]: params["monotone_constraints"] = [0]*(len(columns(variant))-3) + [-1, 1, -1]
    provisional = train_lgbm(inputs(fit, variant), fit["label"], inputs(stop, variant), stop["label"], params)
    trees = int(provisional.best_iteration_ or 500)
    model = train_lgbm(inputs(train, variant), train["label"], params={**params, "n_estimators": trees})
    probabilities = predict(model, test, variant)
    result = {"n": len(test), "train_n": len(train), "trees": trees, "seconds": time.perf_counter()-start,
              **compute_metrics(test["label"].to_numpy(), probabilities)}
    print(f"{year} {variant}: n={len(test)} acc={result['accuracy']:.6f} loss={result['log_loss']:.9f} trees={trees} ({result['seconds']:.1f}s)", flush=True)
    del provisional, model
    gc.collect()
    return result, probabilities, test.index


def paired_interval(test, control, candidate, repeats=2000):
    y = test["label"].to_numpy()
    loss = lambda p: -(y*np.log(np.clip(p, 1e-15, 1-1e-15))+(1-y)*np.log(np.clip(1-p, 1e-15, 1-1e-15)))
    group = test["tournament_code"].fillna("").astype(str)
    group = group.where(group != "", test["tournament_name"].astype(str)) + ":" + test["prediction_time"].dt.year.astype(str)
    delta = pd.DataFrame({"cluster": group, "n": 1, "loss": loss(candidate)-loss(control),
                          "accuracy": ((candidate >= .5) == y).astype(float)-((control >= .5) == y).astype(float)}, index=test.index)
    totals = delta.groupby("cluster")[["n", "loss", "accuracy"]].sum().to_numpy()
    rng = np.random.default_rng(42)
    sampled = totals[rng.integers(0, len(totals), (repeats, len(totals)))].sum(axis=1)
    return {"clusters": len(totals), "replicates": repeats,
            **{key: {"mean": float(delta[key].mean()), "ci95": np.quantile(sampled[:, i]/sampled[:, 0], [.025, .975]).tolist()}
               for i, key in ((1, "loss"), (2, "accuracy"))}}


def correlations(test, probability):
    a, b = test["d14_km_t1"].to_numpy(), test["d14_km_t2"].to_numpy()
    known = np.isfinite(a) & np.isfinite(b)
    residual = test["label"].to_numpy() - probability
    records = {}
    for name, mask in (("all_known", known), ("first_entry_known", known & test["d14_first_entry"].to_numpy())):
        x, r = a[mask]-b[mask], residual[mask]
        records[name] = {"n": int(mask.sum()), "pearson_distance_gap_vs_baseline_residual": float(np.corrcoef(x, r)[0, 1]),
                         "residual_slope_per_1000km": float(np.polyfit(x/1000, r, 1)[0])}
    return records


def protection():
    files = {p for base in (ROOT / "data", ROOT / "outputs") for p in base.rglob("*")
             if p.is_file() and not p.is_relative_to(OUT)}
    files.update(ROOT / p for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines())
    files.add(ROOT / ".cache/tmp/experiment-cleanup.py")
    save("protected-before.json", {str(p.relative_to(ROOT)): sha(p) for p in sorted(files) if p.is_file()})
    (OUT / "EXPERIMENTS-before.md").write_bytes((ROOT / "EXPERIMENTS.md").read_bytes())


def verify():
    frozen = json.loads((OUT / "protected-before.json").read_text())
    changed = [name for name, expected in frozen.items() if not (ROOT/name).is_file() or sha(ROOT/name) != expected]
    assert not changed, changed
    assert (ROOT/"EXPERIMENTS.md").read_bytes().startswith((OUT/"EXPERIMENTS-before.md").read_bytes())
    return len(frozen)


def run():
    logging.basicConfig(level=logging.WARNING)
    start = time.perf_counter()
    cutoff = datetime.now(UTC)
    directory = forecast_bundle_directory()
    base = FeatureBuilder.load(directory / "feature_state.pkl")
    assert base.model_policy["result_availability"] == OBSERVED_AVAILABILITY_POLICY
    protection()
    save("protocol.json", {"started_at_utc": cutoff.isoformat(), "variants": VARIANTS, "validation_years": [2024, 2025],
         "threshold_selection": "Minimum pooled 2024-2025 log loss among predeclared travel variants, before scoring 2026; no retuning or promotion.",
         "seed": 42, "probabilities": "raw two-order average", "model_parameters": "Existing fixed LightGBM defaults; early stopping on availability-purged training tail, then refit earlier-year rows only.",
         "base_bundle": str(directory.relative_to(ROOT)), "base_policy": base.model_policy,
         "origin_policy": "Last observed completed event location across disciplines; federation home proxy when no history or gap > threshold. Retain inbound trip features within a continuous edition; tied-time targets preview before committing label-free trip metadata. Results release only at available_at < prediction_time. Unknown locations stay missing.",
         "distance_policy": "Great-circle km between Google country representative points, not actual flights, residence or venue distance. Doubles use mean partner km, missing if either partner is unknown. Home/coverage controls distinguish geography flags from distance. Monotone variant constrains longer own distance to lower chance.",
         "evaluation_status": "Reused retrospective 2026 benchmark; not prospective evidence or sportsbook profitability."})
    receipts = observations_before(cutoff, OBSERVATIONS_PATH)
    matches, drops, names = load_all_matches(ROOT / "data/raw/bwf_matches", observed_results=receipts, category_catalog=base.category_catalog)
    stamp = cutoff.replace(tzinfo=None)
    matches = [m for m in matches if not standard_match_exclusion(m) and m.get("match_status") in (None, "", "F")
               and m["available_at"] < stamp and m["match_time"] < stamp]
    history = FeatureBuilder(player_names=names, result_availability_policy=OBSERVED_AVAILABILITY_POLICY, category_catalog=base.category_catalog)
    history.elo = EloEngine(**base.model_policy["elo"])
    history.trueskill = TrueSkillEngine(**base.trueskill.parameters)
    assert history.model_policy == base.model_policy
    print(f"Building matching base features for {len(matches):,} matches", flush=True)
    frame = history.build_dataframe(matches, as_of=stamp)
    assert frame["fixture_id"].is_unique
    travel = travel_frame(matches, geography())
    frame = frame.merge(travel, on="fixture_id", validate="one_to_one", how="left")
    assert len(frame) == len(matches) and not frame["host_known"].isna().any()
    result = {"cutoff": cutoff.isoformat(), "matches": len(matches), "drops": drops, "variants": VARIANTS,
              "coverage": {"host_known": float(frame["host_known"].mean()),
                           "both_distances_known_14d": float((frame["d14_km_t1"].notna() & frame["d14_km_t2"].notna()).mean()),
                           "home_origin_mean_14d": float(frame[["d14_travel_home_origin_t1", "d14_travel_home_origin_t2"]].to_numpy().mean())},
              "validation": {}, "pooled_validation": {}, "holdout": {}, "comparisons": {}}
    print("Coverage:", result["coverage"], flush=True)
    del base, history, matches, travel
    gc.collect()
    pooled, all_indices = defaultdict(list), []
    for year in (2024, 2025):
        for variant in VARIANTS:
            metrics, probability, indices = fit_score(frame, year, variant)
            result["validation"].setdefault(variant, {})[str(year)] = metrics
            pooled[variant].append(probability)
            if variant == "control": all_indices.extend(indices.tolist())
        save("report.json", result)
    y = frame.loc[all_indices, "label"].to_numpy()
    for variant, parts in pooled.items(): result["pooled_validation"][variant] = {"n": len(y), **compute_metrics(y, np.concatenate(parts))}
    travel_variants = [v for v in VARIANTS if VARIANTS[v][1]]
    selected = min(travel_variants, key=lambda v: result["pooled_validation"][v]["log_loss"])
    overall = min(VARIANTS, key=lambda v: result["pooled_validation"][v]["log_loss"])
    result["selected_travel_before_2026"], result["overall_validation_winner"] = selected, overall
    save("selection-before-2026.json", {"selected_travel": selected, "overall_winner": overall,
         "scores": result["pooled_validation"], "locked_at_utc": datetime.now(UTC).isoformat()})
    print("Locked selection before 2026:", selected, "overall:", overall, flush=True)
    predictions = {}
    for variant in dict.fromkeys(("control", "geography_14d", "travel_14d", selected)):
        metrics, probability, indices = fit_score(frame, 2026, variant)
        result["holdout"][variant], predictions[variant] = metrics, probability
        save("report.json", result)
    test = frame.loc[indices]
    for variant, probability in predictions.items():
        if variant != "control": result["comparisons"][variant] = paired_interval(test, predictions["control"], probability)
    if "travel_14d" in predictions:
        result["distance_beyond_geography_14d"] = paired_interval(test, predictions["geography_14d"], predictions["travel_14d"])
    result["correlation_14d"] = correlations(test, predictions["control"])
    export = test[["match_id", "fixture_id", "tournament_code", "tournament_name", "discipline", "prediction_time", "label",
                   "d14_km_t1", "d14_km_t2", "d14_first_entry"]].copy()
    for name, probability in predictions.items(): export[f"p_{name}"] = probability
    export.to_csv(OUT/"paired-holdout.csv.gz", index=False, compression="gzip")
    result.update(complete=True, elapsed_seconds=time.perf_counter()-start, protected_files_unchanged=verify())
    save("report.json", result)
    print(json.dumps({"selected": selected, "validation_winner": overall, "elapsed_seconds": result["elapsed_seconds"],
                      "holdout": result["holdout"], "comparisons": result["comparisons"], "correlation": result["correlation_14d"]}, indent=2), flush=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("mode", choices=("prepare", "run", "verify"))
    mode = args.parse_args().mode
    if mode == "prepare": prepare()
    elif mode == "run": run()
    else: print("Protected files unchanged:", verify())
