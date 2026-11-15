#!/usr/bin/env python3
"""Forecast an upcoming round and draft $10 positive-EV paper bets."""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import sys

import settle_bets
from predict_badminton.config import REPO_ROOT, forecast_bundle_directory
from predict_badminton.inference import build_inference_features
from predict_badminton.match_rules import standard_match_exclusion
from predict_badminton.probabilities import PREDICTION_POLICY, predict_probabilities
from predict_badminton.tournament import load_bundle

FIXTURE_FIELDS = {
    "match_id", "match_time", "discipline", "team1_player_ids", "team2_player_ids",
    "round_name", "tournament_name", "tournament_code", "tournament_id", "raw_file",
}
QUOTE_FIELDS = {
    "match_id", "selection_id", "discipline", "round_name", "tournament_id",
    "selection_player_ids", "opponent_player_ids", "selection_name", "opponent_name",
    "american_odds", "official_start_time_utc", "recorded_at_utc", "sportsbook", "jurisdiction",
}


def utc(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Input times must include a UTC offset")
    return stamp.astimezone(UTC)


def source_name(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_table(path: Path, required: set[str]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))
    if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
        raise ValueError(f"Duplicate columns in {path}")
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(reader.fieldnames or []))}")
    rows = list(reader)
    if not rows or any(None in row or any(not (row.get(key) or "").strip() for key in required) for row in rows):
        raise ValueError(f"Empty or incomplete rows in {path}")
    return rows


def load_inputs(fixtures_path: Path, odds_paths: list[Path], cutoff: datetime) -> tuple[dict, list[dict]]:
    fixtures = {}
    for row in read_table(fixtures_path, FIXTURE_FIELDS):
        mid = row["match_id"]
        teams = [row[f"team{side}_player_ids"].split("|") for side in (1, 2)]
        if (mid in fixtures or not mid.isdigit() or int(mid) <= 0
                or row["round_name"] not in {"R128", "R64", "R32", "R16", "QF", "SF", "Final"}
                or any(not pid.isdigit() or int(pid) <= 0 for team in teams for pid in team)
                or standard_match_exclusion({**row, "team1_ids": teams[0], "team2_ids": teams[1]})):
            raise ValueError(f"Invalid or duplicate fixture: {mid}")
        if utc(row["match_time"]) <= cutoff:
            raise ValueError(f"Fixture is not upcoming: {mid}")
        if (row.get("winner", "0") not in ("", "0") or row.get("score_status", "0") not in ("", "0")
                or row.get("status", "Normal").casefold() not in {"", "n", "normal", "scheduled"}):
            raise ValueError(f"Fixture is not an unplayed normal match: {mid}")
        raw = Path(row["raw_file"])
        if raw.is_absolute() or not (REPO_ROOT / raw).resolve().is_relative_to((REPO_ROOT / "data/raw/bwf_matches").resolve()):
            raise ValueError("raw_file must be a repository-relative path inside data/raw/bwf_matches")
        fixtures[mid] = row
    if len({row["round_name"] for row in fixtures.values()}) != 1:
        raise ValueError("Supply one knockout round per run; multiple tournaments are allowed")

    quotes, seen = [], set()
    for path in odds_paths:
        for row in read_table(path, QUOTE_FIELDS):
            fix = fixtures.get(row["match_id"])
            if fix is None:
                raise ValueError(f"Quote has no fixture: {row['match_id']}")
            teams = [row[field].split("|") for field in ("selection_player_ids", "opponent_player_ids")]
            if standard_match_exclusion({**row, "team1_ids": teams[0], "team2_ids": teams[1]}):
                raise ValueError(f"Invalid quoted participants: {row['selection_id']}")
            selected, opponent = map(set, teams)
            sides = [side for side in (1, 2) if selected == set(fix[f"team{side}_player_ids"].split("|"))
                     and opponent == set(fix[f"team{3-side}_player_ids"].split("|"))]
            if (len(sides) != 1 or row["selection_id"] != f"{row['match_id']}:{sides[0]}"
                    or any(row[key] != fix[key] for key in ("tournament_id", "round_name", "discipline"))
                    or utc(row["official_start_time_utc"]) != utc(fix["match_time"])):
                raise ValueError(f"Quote identity or start time differs from fixture: {row['selection_id']}")
            odds = int(row["american_odds"])
            if abs(odds) < 100 or row["selection_id"] in seen:
                raise ValueError(f"Invalid odds or duplicate quote: {row['selection_id']}")
            if row["sportsbook"].casefold() != "bet365" or row["jurisdiction"].casefold() != "ontario":
                raise ValueError("The current paper ledger supports bet365 Ontario prices only")
            if utc(row["recorded_at_utc"]) > cutoff or (row.get("captured_at_utc") and utc(row["captured_at_utc"]) > cutoff):
                raise ValueError("Quote timestamps cannot follow the forecast cutoff")
            seen.add(row["selection_id"])
            quotes.append({**row, "american_odds": odds, "quote_source": source_name(path)})
    return fixtures, quotes


def pending_ledger(selections: list[dict], fixtures: dict, metadata: dict, output: Path) -> dict:
    batches = {}
    for row in selections:
        if not row["theoretical_positive_ev"]:
            continue
        fix = fixtures[row["match_id"]]
        key = (fix["tournament_code"].upper(), fix["round_name"], utc(fix["match_time"]).date().isoformat())
        if key not in batches:
            batches[key] = {**metadata, "name": fix["tournament_name"], "tournament_name": fix["tournament_name"],
                            "tournament_code": key[0], "round": key[1], "date": key[2], "raw_file": fix["raw_file"],
                            "source": source_name(output / "forecast.json"), "bets": [],
                            "note": "Prospective $10 paper bets at saved prices, not actual wagers. Odds and full-precision probabilities are frozen; continued price availability is unverified."}
        batch = batches[key]
        if batch["raw_file"] != fix["raw_file"] or batch["tournament_name"] != fix["tournament_name"]:
            raise ValueError("Conflicting tournament metadata in fixtures")
        batch["bets"].append({
            "match_id": row["match_id"], "selection_id": row["selection_id"], "discipline": row["discipline"],
            "selection": row["selection_name"], "opponent": row["opponent_name"],
            "selection_player_ids": row["selection_player_ids"], "opponent_player_ids": row["opponent_player_ids"],
            "american_odds": row["american_odds"], "model_probability": row["model_probability"],
            "model_ev_percent": row["model_ev_percent"], "odds_source": row["quote_source"],
            "stake": "10.00", "status": "pending", "profit": None, "returned": None, "score": "",
        })
    ledger = {"currency": "CAD", "batches": list(batches.values())}
    settle_bets.validate(ledger)
    return ledger


def merge_pending(existing: dict, pending: dict) -> dict:
    """Append only new fixtures, including when source IDs or side order changed."""
    settle_bets.validate(existing)
    if existing["currency"] != pending["currency"]:
        raise ValueError("Paper-bet currencies differ")
    seen_ids, seen_teams = set(), set()
    for batch in [*existing["batches"], *pending["batches"]]:
        for bet in batch["bets"]:
            mid = (batch["tournament_code"].upper(), str(bet["match_id"]))
            teams = tuple(sorted(tuple(sorted(bet[key].split("|")))
                                 for key in ("selection_player_ids", "opponent_player_ids")))
            fixture = (mid[0], batch["round"], bet["discipline"], teams)
            if mid in seen_ids or fixture in seen_teams:
                raise ValueError(f"Paper bet already recorded for this fixture: {mid}")
            seen_ids.add(mid)
            seen_teams.add(fixture)
    combined = {**existing, "batches": [*existing["batches"], *pending["batches"]]}
    settle_bets.validate(combined)
    return combined


def run_round(fixtures_path: Path, odds_paths: list[Path], output: Path, model_dir: Path,
              *, record_bets: bool = False) -> dict:
    output = output.resolve()
    if not output.is_relative_to((REPO_ROOT / "outputs").resolve()) or output.exists():
        raise ValueError("Output must be a new directory inside outputs; existing runs are never overwritten")
    cutoff = datetime.now(UTC)
    paths = [fixtures_path, *odds_paths, model_dir / "lgbm_model.pkl", model_dir / "feature_state.pkl"]
    hashes = {source_name(path): checksum(path) for path in paths}
    fixtures, quotes = load_inputs(fixtures_path, odds_paths, cutoff)
    model, _, _, history = load_bundle(model_dir)
    features, loaded = build_inference_features(
        fixtures_path, model_dir / "feature_state.pkl", model=model, refresh=True, as_of=cutoff,
        raw_dir=REPO_ROOT / "data/raw/bwf_matches",
        observations_db=REPO_ROOT / "data/manifests/bwf_result_observations.sqlite3",
    )
    if set(loaded["match_id"].astype(str)) != set(fixtures) or len(features) != len(fixtures):
        raise ValueError("Inference dropped fixture rows; no round will be saved")
    probabilities = dict(zip(features["match_id"].astype(str),
                             predict_probabilities(model, features[history.feature_columns]), strict=True))
    generated = datetime.now(UTC)
    metadata = {"prediction_generated_at_utc": generated.isoformat(), "forecast_cutoff_utc": cutoff.isoformat(),
                "history_as_of": features.attrs["history_as_of"], "history_refreshed": features.attrs["history_refreshed"],
                "result_availability_policy": features.attrs["result_availability_policy"],
                "prediction_policy": PREDICTION_POLICY, "probability_method": "raw",
                "model_sha256": hashes[source_name(model_dir / "lgbm_model.pkl")],
                "base_state_sha256": hashes[source_name(model_dir / "feature_state.pkl")],
                "state_sha256": features.attrs["state_sha256"]}
    selections = []
    for quote in quotes:
        first = float(probabilities[quote["match_id"]])
        probability = repr(first if quote["selection_id"].endswith(":1") else 1 - first)
        odds = quote["american_odds"]
        decimal = 1 + (Decimal(odds) / 100 if odds > 0 else Decimal(100) / -odds)
        ev = Decimal(probability) * decimal - 1
        selections.append({**quote, **metadata, "model_probability": probability, "decimal_odds": str(decimal),
                           "break_even_probability": str(1 / decimal), "model_ev_percent": str(ev * 100),
                           "theoretical_positive_ev": ev > 0, "expected_profit_10": str(ev * 10)})
    forecast = {**metadata, "input_sha256": hashes,
                "fixtures": [{**fix, "team1_win_probability": repr(float(probabilities[mid])),
                              "team2_win_probability": repr(1 - float(probabilities[mid]))} for mid, fix in fixtures.items()],
                "selections": selections}
    pending = pending_ledger(selections, fixtures, metadata, output)
    ledger_path = REPO_ROOT / "outputs/profits.json"
    original = ledger_path.read_bytes() if record_bets else None
    combined = merge_pending(json.loads(original), pending) if record_bets else None
    report = settle_bets.render_report(combined) if combined is not None else None
    if any(checksum(path) != hashes[source_name(path)] for path in paths):
        raise ValueError("An input or model changed during forecasting; no round will be saved")
    if any(utc(fix["match_time"]) <= datetime.now(UTC) for fix in fixtures.values()):
        raise ValueError("A fixture started while forecasting; no round will be saved")
    if record_bets and ledger_path.read_bytes() != original:
        raise ValueError("The profit ledger changed during forecasting; retry without overwriting it")
    output.mkdir(parents=True)
    for name, value in (("forecast.json", forecast), ("pending_bets.json", pending)):
        with (output / name).open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in selections for key in row))
    with (output / "comparison.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selections)
    if combined is not None and pending["batches"]:
        settle_bets.write_if_changed(ledger_path, json.dumps(combined, ensure_ascii=False, indent=2) + "\n")
        settle_bets.write_if_changed(REPO_ROOT / "outputs/profits.md", report)
    return forecast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-csv", type=Path, required=True)
    parser.add_argument("--odds-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, help="New run directory (default: outputs/rounds/<UTC timestamp>)")
    parser.add_argument("--model-dir", type=Path, help="Trusted local bundle; defaults to the selected forecast bundle")
    parser.add_argument("--record-bets", action="store_true", help="Append new pending $10 bets to profits.json; reject already recorded fixtures")
    args = parser.parse_args(argv)
    try:
        output = args.output_dir or REPO_ROOT / "outputs/rounds" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        forecast = run_round(args.fixtures_csv, args.odds_csv, output, args.model_dir or forecast_bundle_directory(),
                             record_bets=args.record_bets)
        positives = [row for row in forecast["selections"] if row["theoretical_positive_ev"]]
        print(f"Saved {len(forecast['fixtures'])} matches, {len(forecast['selections'])} prices, {len(positives)} positive-EV $10 paper bets: {output.resolve()}")
        for row in positives:
            print(f"{row['selection_name']} | {row['american_odds']:+d} | {Decimal(row['model_probability']):.1%} | EV {Decimal(row['model_ev_percent']):+.2f}%")
        if args.record_bets:
            print("New paper bets recorded." if positives else "No positive-EV bets to record; profit ledger unchanged.")
        else:
            print("Draft only. Existing profits.json and profits.md were not changed.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation, csv.Error) as exc:
        print(f"Cannot forecast round: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
