"""Read-only leakage/data audit with isolated retrospective diagnostic fits."""
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
import gc
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata

import numpy as np
import pandas as pd

from predict_badminton.config import FEATURE_COLS, REPO_ROOT, forecast_bundle_directory
from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import observations_before, record_results
from predict_badminton.evaluate import compute_metrics, reliability_stats
from predict_badminton.features import EloEngine, FeatureBuilder
from predict_badminton.match_rules import point_margin, standard_match_exclusion
from predict_badminton.model import _chronological_validation_split, train_lgbm
from predict_badminton.probabilities import predict_probabilities
from predict_badminton.temporal import OBSERVED_AVAILABILITY_POLICY
from predict_badminton.tournament import load_bundle
from predict_badminton.tournament_catalog import TournamentCatalog
from predict_badminton.trueskill import TrueSkillEngine

OUT = Path(__file__).resolve().parent
RAW = REPO_ROOT / "data/raw/bwf_matches"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")


def normalized(value):
    value = str(value).replace("æ", "ae").replace("ø", "o")
    value = unicodedata.normalize("NFKD", value).casefold()
    return frozenset(re.findall(r"[a-z0-9]+", "".join(c for c in value if not unicodedata.combining(c))))


def protect():
    files = {p for base in (REPO_ROOT / "data", REPO_ROOT / "outputs") for p in base.rglob("*")
             if p.is_file() and not p.is_relative_to(OUT)}
    files.update(REPO_ROOT / p for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines())
    files.update((REPO_ROOT / "predict_badminton").rglob("*.py"))
    files.add(REPO_ROOT / "EXPERIMENTS.md")
    snapshot = {str(p.relative_to(REPO_ROOT)): sha(p) for p in sorted(files) if p.is_file()}
    save("protected-before.json", snapshot)
    return snapshot


def raw_checks(matches, receipts):
    ids = {str(mid) for m in matches for mid in m.get("source_match_ids", [m["match_id"]])}
    counts, examples, player_names, genders = Counter(), defaultdict(list), defaultdict(Counter), defaultdict(Counter)
    for path in sorted(RAW.glob("*.json")):
        for rnd in (r for r in json.loads(path.read_text()) if isinstance(r,list)):
            for m in rnd:
                if not isinstance(m, dict):
                    continue
                mid = str(m.get("id"))
                if mid not in ids:
                    continue
                counts["source_rows_for_eligible_ids"] += 1
                disc = str(m.get("eventName") or "")
                for side in (1, 2):
                    for p in (m.get(f"team{side}") or {}).get("players", []):
                        pid = str(p.get("id"))
                        name = p.get("nameDisplay") or (str(p.get("firstName") or "") + " " + str(p.get("lastName") or ""))
                        if name.strip(): player_names[pid][name] += 1
                        if disc in {"MS", "WS"}: genders[pid][disc] += 1
                reasons = []
                winner = m.get("winner")
                if isinstance(winner, bool): reasons.append("boolean_winner")
                if m.get("matchStatus") not in (None, "", "F"): reasons.append("nonfinal_status_source")
                if point_margin(m) is None: counts["missing_or_nonstandard_point_margin_source"] += 1
                games = [(g.get("home"), g.get("away")) for g in (m.get("score") or []) if isinstance(g, dict)
                         and (g.get("home") or g.get("away"))]
                if len(games) in (2, 3) and all(isinstance(v, (int,float)) and not isinstance(v,bool) for g in games for v in g):
                    def valid(pair):
                        high, low = max(pair), min(pair)
                        return high > low and ((high == 21 and low <= 19) or (22 <= high <= 29 and high-low == 2)
                                               or (high == 30 and low in (28,29)))
                    if all(valid(pair) for pair in games):
                        wins = Counter(1 if a>b else 2 for a,b in games)
                        if wins.get(winner, 0) != 2 or len(games) == 3 and (games[0][0]>games[0][1]) == (games[1][0]>games[1][1]):
                            reasons.append("normal_21point_score_contradicts_winner_or_match_end")
                receipt = receipts.get(mid)
                start, duration = m.get("matchTimeUtc"), m.get("duration")
                if receipt and start and isinstance(duration, (int,float)) and not isinstance(duration,bool) and 0<duration<=300:
                    if receipt["available_at"] < datetime.fromisoformat(start) + timedelta(minutes=duration):
                        reasons.append("observed_before_recorded_end")
                for reason in reasons:
                    counts[reason] += 1
                    if len(examples[reason]) < 15:
                        examples[reason].append({"path":str(path.relative_to(REPO_ROOT)),"id":mid,"event":disc,
                                                "winner":winner,"score":games,"status":m.get("matchStatus")})
    incompatible = []
    for pid, names in player_names.items():
        common = [(name,n) for name,n in names.most_common() if n>=2 and len(normalized(name))>=2]
        for i,(first,n1) in enumerate(common):
            for second,n2 in common[i+1:]:
                if not normalized(first).intersection(normalized(second)):
                    incompatible.append({"player_id":pid,"first":first,"first_rows":n1,"second":second,"second_rows":n2})
                    break
            else:
                continue
            break
    mixed = [{"player_id":pid,"MS":v["MS"],"WS":v["WS"],"names":player_names[pid].most_common(3)}
             for pid,v in genders.items() if v["MS"]>=2 and v["WS"]>=2]
    result = {"counts":counts,"examples":examples,"player_ids_with_multiple_disjoint_frequent_names":incompatible,
              "player_ids_with_both_MS_and_WS":mixed,
              "note":"Raw checks count source copies of eligible IDs; identity flags require manual confirmation and are not automatically declared errors."}
    save("raw-data-checks.json", result)
    print("Raw source checks:",dict(counts),"identity flags",len(incompatible),"mixed MS/WS IDs",len(mixed),flush=True)


def leakage_probe():
    from tests.test_observations import completed_match
    from predict_badminton.data_parser import parse_file
    stamp = datetime(2026,1,1,10,tzinfo=UTC)
    match = {**completed_match(),"matchTimeUtc":"2026-01-01 10:00:00","matchStatus":"L","duration":60}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        root = Path(tmp)
        db = root/"observations.sqlite3"
        record_results([match],source_url="https://example.test/provisional",response_sha256="test",observed_at=stamp+timedelta(minutes=5),path=db)
        (root/"raw.json").write_text(json.dumps([[{**match,"matchStatus":"F"}]]))
        rows = parse_file(root/"raw.json",{},observed_results=observations_before(stamp+timedelta(minutes=10),db))
        accepted = bool(rows) and rows[0].get("_observation_error") is None
        result = {"live_winner_can_create_receipt":bool(rows),"later_final_same_version_accepted_at_earlier_cutoff":accepted,
                  "release":rows[0]["available_at"] if rows else None,"recorded_end":(stamp+timedelta(hours=1)).replace(tzinfo=None),
                  "explanation":"matchStatus is not required or fingerprinted by the receipt snapshot. The final-status source can reuse an earlier provisional-winner receipt."}
    save("provisional-receipt-probe.json",result)


def main():
    logging.basicConfig(level=logging.WARNING)
    protected = json.loads((OUT/"protected-before.json").read_text()) if (OUT/"protected-before.json").exists() else protect()
    directory = forecast_bundle_directory()
    production,_,_,base = load_bundle(directory)
    metadata = json.loads((directory/"metrics_report.json").read_text())
    cutoff = datetime.fromisoformat(metadata["production_refit"]["as_of"]).replace(tzinfo=UTC)
    receipts = observations_before(cutoff)
    print("Loading the exact active training cohort",flush=True)
    all_matches,drops,names = load_all_matches(RAW,observed_results=receipts,category_catalog=base.category_catalog)
    matches = [m for m in all_matches if not standard_match_exclusion(m) and m.get("match_status") in (None,"","F")
               and m["available_at"]<cutoff.replace(tzinfo=None) and m["match_time"]<cutoff.replace(tzinfo=None)]
    assert len(matches)==metadata["production_refit"]["rows"]
    cohort = sorted((m["fixture_id"],m["prediction_time"].isoformat(),m["available_at"].isoformat(),m["winner"]) for m in matches)
    assert hashlib.sha256(json.dumps(cohort).encode()).hexdigest()==metadata["cohort_sha256"]
    del cohort,all_matches
    sources = Counter(m["availability_source"] for m in matches)
    forbidden = set(base.feature_columns)&{"label","winner","score_margin","match_id","fixture_id","available_at","duration","match_time"}
    assert not forbidden
    duplicate = defaultdict(list)
    catalog = TournamentCatalog.load()
    categories, junior = {}, {}
    for m in matches:
        edition = catalog.lookup(m.get("tournament_code"),m.get("tournament_id"))
        categories[m["fixture_id"]] = edition["category"] if edition else ""
        junior[m["fixture_id"]] = bool(re.search(r"junior|\bu[ -]?(?:1[13579]|2[13])\b",m["tournament_name"]+" "+m["event_name"],re.I)
                                       or re.match(r"^(BS|GS|BD|GD)(?:$|[ -])",m["event_name"]))
        if m["is_team_match"] or m["round_name"] not in {"R128","R64","R32","R16","QF","SF","Final","Qual. R32","Qual. R16","Qual. QF","Qual. SF"}: continue
        code = edition["tournament_code"] if edition else m["tournament_code"]
        teams = tuple(sorted((tuple(sorted(m["team1_ids"])),tuple(sorted(m["team2_ids"])))))
        duplicate[(code,m["discipline"],m["round_name"],teams)].append(m)
    suspect = [[{k:m[k] for k in ("match_id","fixture_id","match_time","prediction_time","tournament_name","round_name","team1_ids","team2_ids","winner")} for m in group]
               for group in duplicate.values() if len(group)>1]
    save("suspect-repeat-fixtures.json",suspect)
    summary = {"training_rows":len(matches),"cohort_reproduced_exactly":True,"drops":drops,"availability_sources":sources,
               "year_counts":Counter(m["prediction_time"].year for m in matches),"discipline_counts":Counter(m["discipline"] for m in matches),
               "label_mean":float(np.mean([m["label"] for m in matches])),"junior_tagged_rows":sum(junior.values()),
               "suspect_same_edition_round_opponents_groups":len(suspect),"direct_outcome_features":list(forbidden),
               "actual_feature_policy":base.model_policy,"audit_cutoff_utc":cutoff.isoformat()}
    save("data-summary.json",summary)
    print("Cohort verified:",len(matches),"availability",dict(sources),"suspect repeats",len(suspect),flush=True)
    raw_checks(matches,receipts)
    leakage_probe()
    history = FeatureBuilder(player_names=names,result_availability_policy=OBSERVED_AVAILABILITY_POLICY,
                             category_catalog=base.category_catalog,country_context=base.country_context.empty_copy())
    history.elo = EloEngine(**base.model_policy["elo"])
    history.trueskill = TrueSkillEngine(**base.trueskill.parameters)
    print("Building causal features for isolated historical checks",flush=True)
    frame = history.build_dataframe(matches,as_of=cutoff.replace(tzinfo=None))
    assert history.model_policy==base.model_policy
    frame["audit_category"] = frame["fixture_id"].map(categories)
    frame["audit_junior"] = frame["fixture_id"].map(junior)
    del matches,names,receipts,history
    gc.collect()
    boundary = pd.Timestamp("2026-01-01")
    train = frame.loc[(frame["prediction_time"]<boundary)&(frame["available_at"]<boundary)]
    test = frame.loc[frame["prediction_time"]>=boundary].copy()
    assert not set(train["fixture_id"])&set(test["fixture_id"])
    result = {"kind":"Retrospective reused-benchmark diagnostic, not a new holdout", "test_rows":len(test),"train_rows":len(train),"models":{}}
    probabilities = {}
    for variant,columns in (("baseline",FEATURE_COLS),("country",base.feature_columns)):
        fit,stop = _chronological_validation_split(train)
        vocabulary = sorted(set(fit[["country_id_t1","country_id_t2"]].stack().dropna().astype(str)))
        def matrix(rows,vocab):
            value = rows[columns].copy()
            if variant=="country":
                for name in ("country_id_t1","country_id_t2"):
                    value[name] = pd.Categorical(value[name].where(value[name].isin(vocab)),categories=vocab)
            return value
        params = {**production.get_params(),"n_jobs":4}
        provisional = train_lgbm(matrix(fit,vocabulary),fit["label"],matrix(stop,vocabulary),stop["label"],params)
        trees = int(provisional.best_iteration_ or params["n_estimators"])
        vocabulary = sorted(set(train[["country_id_t1","country_id_t2"]].stack().dropna().astype(str)))
        model = train_lgbm(matrix(train,vocabulary),train["label"],params={**params,"n_estimators":trees})
        probability = predict_probabilities(model,matrix(test,vocabulary))
        probabilities[variant] = probability
        test[f"audit_p_{variant}"] = probability
        groups = {"all_2026":np.ones(len(test),dtype=bool),"senior_world_tour":test["audit_category"].str.contains("World Tour",case=False).to_numpy()&~test["audit_junior"].to_numpy(),
                  "junior_tagged":test["audit_junior"].to_numpy(),"R32":test["round_importance"].eq(5).to_numpy(),
                  **{d:test["discipline"].eq(d).to_numpy() for d in ("MS","WS","MD","WD","XD")}}
        groups["world_tour_R32"] = groups["senior_world_tour"]&groups["R32"]
        metrics = {}
        for name,mask in groups.items():
            subset = test.loc[mask]
            if len(subset)>=20 and subset["label"].nunique()==2:
                p = probability[mask]
                metrics[name] = {"n":len(subset),**compute_metrics(subset["label"].to_numpy(),p),"ece":reliability_stats(subset["label"].to_numpy(),p)["ece"]}
        training_p = predict_probabilities(model,matrix(train,vocabulary))
        result["models"][variant] = {"trees":trees,"training_in_sample_not_validation":compute_metrics(train["label"].to_numpy(),training_p),"subgroups":metrics}
        print(variant,"train",result["models"][variant]["training_in_sample_not_validation"],"test",metrics,flush=True)
        del provisional,model,training_p
        gc.collect()
    result["label_permutation_control"] = compute_metrics(np.random.default_rng(15616).permutation(test["label"].to_numpy()),probabilities["country"])
    bins = []
    for low,high in ((0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.00001)):
        mask = (probabilities["country"]>=low)&(probabilities["country"]<high)
        bins.append({"low":low,"high":min(high,1),"n":int(mask.sum()),"mean_p":float(probabilities["country"][mask].mean()),"actual_team1_win_rate":float(test.loc[mask,"label"].mean())})
    result["country_reliability_bins"] = bins
    save("retrospective-diagnostics.json",result)
    test[["fixture_id","match_id","prediction_time","tournament_code","tournament_name","discipline","label","audit_category","audit_junior","audit_p_baseline","audit_p_country"]].to_csv(OUT/"diagnostic-predictions.csv",index=False)
    changed = [p for p,h in protected.items() if not(REPO_ROOT/p).is_file() or sha(REPO_ROOT/p)!=h]
    assert not changed,changed
    save("preservation-verification.json",{"checked_files":len(protected),"changed":changed,"production_forecasts_bets_unchanged":True})
    print("Audit diagnostics complete; protected files unchanged",flush=True)


if __name__=="__main__":
    main()
