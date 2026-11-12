#!/usr/bin/env python3
"""Audit normalized BWF data without modifying raw source files."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import UTC, datetime
import json
import sys
from pathlib import Path

from predict_badminton.data_parser import build_data_report, load_all_matches
from predict_badminton.config import DISCIPLINES, REPO_ROOT
from predict_badminton.tournament_catalog import CATALOG_PATH, TournamentCatalog, build_catalog, official_tier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bwf_matches"),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--include-para",
        action="store_true",
        help="Include para matches, which are excluded by default",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Report all parsed records without match deduplication",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--world-tour", action="store_true", help="Audit official World Tour coverage by event and round")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/source_cache/bwf")
    parser.add_argument("--refresh-catalog", action="store_true", help="Rebuild categories from saved calendar responses, without fetching")
    parser.add_argument("--as-of", help="World Tour audit cutoff with UTC offset; default: current UTC time")
    args = parser.parse_args()
    if (args.refresh_catalog or args.as_of) and not args.world_tour:
        parser.error("--refresh-catalog and --as-of require --world-tour")
    if args.output is None:
        args.output = Path("data/processed/world_tour_audit.json" if args.world_tour else "data/processed/data_audit.json")
    return args


def world_tour_audit(raw_dir: Path, catalog: TournamentCatalog, cache_dir: Path, as_of: datetime,
                     observations_db: Path | None = None) -> tuple[dict, list[dict], list[dict]]:
    """Compare local records with compatible cached official results, not an invented match quota."""
    from predict_badminton.data_parser import normalize_discipline
    from predict_badminton.data_sources.bwf import find_rounds_or_matches
    from predict_badminton.data_sources.bwf_schema import flatten_matches
    from predict_badminton.data_sources.observations import OBSERVATIONS_PATH, observations_before
    from predict_badminton.match_rules import standard_match_exclusion, unplayed_status

    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("Audit cutoff requires a UTC offset")
    cutoff = as_of.astimezone(UTC)
    day = cutoff.date().isoformat()
    editions = {row["tournament_id"]: row for row in catalog.rows
                if official_tier(row["category"]) is not None and row["start_date"][:10] <= day}
    raw_keys, slots, raw_ids = set(), set(), {}
    paths_by_edition = defaultdict(set)
    unplayed_finals = defaultdict(set)

    def key_for(match):
        edition = catalog.lookup(match.get("tournamentCode"), match.get("sourceTournamentId"))
        if edition is None or edition["tournament_id"] not in editions:
            return None
        teams = [sorted(str(p["id"]) for p in (match.get(side) or {}).get("players", []) if p.get("id"))
                 for side in ("team1", "team2")]
        discipline = normalize_discipline(match.get("eventName") or match.get("matchTypeValue") or "")
        if standard_match_exclusion({"discipline": discipline, "team1_ids": teams[0], "team2_ids": teams[1]}):
            return None
        return (edition["tournament_id"], discipline, str(match.get("roundName") or "Unknown Round"),
                tuple(sorted(tuple(team) for team in teams)))

    for path in sorted(raw_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for match in find_rounds_or_matches(data):
            key = key_for(match)
            if key is None:
                continue
            paths_by_edition[key[0]].add(str(path))
            explicit_no_match = str(match.get("scoreStatusValue") or "").strip().casefold() == "no match"
            if key[2] == "Final" and (explicit_no_match or unplayed_status(match)):
                unplayed_finals[key[0]].add(key[1])
                slots.add(key)
            if match.get("winner") not in (1, 2):
                continue
            slots.add(key)
            if not unplayed_status(match):
                raw_keys.add(key)
                if match.get("id") is not None:
                    raw_ids[str(match["id"])] = key

    published, source_paths, cache_errors = {}, defaultdict(set), 0
    for path in sorted(cache_dir.glob("**/api-*")):
        if not path.is_file() or not path.name.startswith(("api-tournaments-day-matches-", "api-vue-tournament-draw-data-")):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            found = find_rounds_or_matches(payload)
            if not found and isinstance(payload, dict):
                found = find_rounds_or_matches(payload.get("results"))
            records = flatten_matches(found)
        except (ValueError, UnicodeDecodeError):
            cache_errors += 1
            continue
        for match in records:
            if isinstance(match.get("winner"), bool) or match.get("winner") not in (1, 2) or unplayed_status(match):
                continue
            key = key_for(match)
            if key is None:
                continue
            same_id = raw_ids.get(str(match.get("id")))
            if same_id is not None and same_id[0] == key[0]:
                key = same_id
            published.setdefault(key, {"match_id": str(match.get("id") or ""), "source_cache_path": str(path)})
            source_paths[key[0]].add(str(path))

    db = observations_db or OBSERVATIONS_PATH
    receipts = observations_before(cutoff, db) if db.exists() else {}
    matches, drops, _ = load_all_matches(raw_dir, observed_results=receipts, category_catalog=catalog)
    training_keys, eligible_keys, tier_changes = set(), set(), Counter()
    from predict_badminton.data_parser import infer_tournament_tier

    for match in matches:
        edition = catalog.lookup(match["tournament_code"], match["tournament_id"])
        if edition is None or edition["tournament_id"] not in editions or standard_match_exclusion(match):
            continue
        key = (edition["tournament_id"], match["discipline"], match["round_name"] or "Unknown Round",
               tuple(sorted((tuple(sorted(match["team1_ids"])), tuple(sorted(match["team2_ids"]))))))
        training_keys.add(key)
        if match["available_at"] < cutoff.replace(tzinfo=None):
            eligible_keys.add(key)
        if match["tournament_tier"] != infer_tournament_tier(match["tournament_name"]):
            tier_changes[key[0]] += 1

    missing = set(published) - raw_keys
    unusable = raw_keys - training_keys
    counts = {name: Counter(key[:3] for key in values) for name, values in
              (("raw_played", raw_keys), ("cached_published", published), ("training_ready", training_keys),
               ("eligible_at_cutoff", eligible_keys), ("missing_cached_records", missing), ("unusable_raw_matches", unusable))}
    rounds = []
    for tid, discipline, rnd in sorted(set().union(*(counter.keys() for counter in counts.values()))):
        rounds.append({"tournament_id": tid, "name": editions[tid]["name"], "discipline": discipline, "round_name": rnd,
                       **{name: counter[(tid, discipline, rnd)] for name, counter in counts.items()}})
    totals = {name: Counter(key[0] for key in values) for name, values in
              (("raw_played", raw_keys), ("cached_published", published), ("training_ready", training_keys),
               ("eligible_at_cutoff", eligible_keys), ("missing_cached_records", missing), ("unusable_raw_matches", unusable))}
    finals = defaultdict(set)
    unknown_rounds = Counter(key[0] for key in raw_keys if key[2] == "Unknown Round")
    for tid, discipline, rnd, _teams in slots:
        if rnd == "Final":
            finals[tid].add(discipline)
    tournaments = []
    for tid, edition in sorted(editions.items(), key=lambda item: (item[1]["start_date"], item[0])):
        name_status = edition["name"].casefold()
        not_played = edition["status"] in {"cancelled", "postponed"} or "cancelled" in name_status or "postponed" in name_status
        if not totals["raw_played"][tid]:
            status = "cancelled_or_postponed" if not_played else "no_usable_results"
        elif edition["end_date"][:10] >= day:
            status = "in_progress"
        elif totals["missing_cached_records"][tid]:
            status = "cached_result_gaps"
        elif totals["unusable_raw_matches"][tid]:
            status = "training_data_gaps"
        elif unknown_rounds[tid]:
            status = "round_metadata_needs_review"
        elif finals[tid] != set(DISCIPLINES):
            status = "final_or_discipline_needs_review"
        elif not totals["cached_published"][tid]:
            status = "raw_present_no_cached_comparison"
        else:
            status = "cached_sources_match"
        tournaments.append({"tournament_id": tid, "tournament_code": edition["tournament_code"], "name": edition["name"],
                            "category": edition["category"], "start_date": edition["start_date"], "end_date": edition["end_date"],
                            "coverage_status": status, "final_disciplines": "|".join(sorted(finals[tid])),
                            "unplayed_final_disciplines": "|".join(sorted(unplayed_finals[tid])),
                            "unknown_round_matches": unknown_rounds[tid],
                            **{name: counter[tid] for name, counter in totals.items()},
                            "tier_corrected_matches": tier_changes[tid], "raw_paths": "|".join(sorted(paths_by_edition[tid])),
                            "cached_source_files": len(source_paths[tid]), "source_url": edition["source_url"]})
    gaps = [{"tournament_id": key[0], "name": editions[key[0]]["name"], "discipline": key[1], "round_name": key[2],
             **published[key]} for key in sorted(missing)]
    report = {"as_of": cutoff.isoformat(), "catalog_sha256": catalog.sha256, "scope": "Super 300/500/750/1000 and World Tour Finals",
              "calendar_editions": len(editions), "coverage_statuses": dict(Counter(t["coverage_status"] for t in tournaments)),
              "raw_played_matches": len(raw_keys), "cached_published_matches": len(published), "training_ready_matches": len(training_keys),
              "eligible_matches": len(eligible_keys), "missing_cached_records": len(missing), "tier_corrected_matches": sum(tier_changes.values()),
              "unusable_raw_matches": len(unusable),
              "unknown_round_matches": sum(unknown_rounds.values()),
              "uncategorized_world_tour_url_editions": [{"tournament_id": row["tournament_id"], "name": row["name"]}
                  for row in catalog.rows if not row["category"] and "bwfworldtour.bwfbadminton.com" in row["source_url"]
                  and row["start_date"][:10] <= day],
              "unreadable_or_nonmatch_cached_responses": cache_errors, "parser_drops": drops, "tournaments": tournaments,
              "limits": ["Agreement with saved sources is not proof every played match was published or downloaded.",
                         "No live fetch was performed. Cancelled/postponed listings and the ongoing tournament are not missing completed tournaments.",
                         "This inventories current saved sources; the cutoff controls eligibility/status, not a reconstruction of historical downloads.",
                         "Historical publication times and unclassifiable legacy draw records remain uncertain."]}
    return report, rounds, gaps


def write_world_tour_report(report: dict, rounds: list[dict], gaps: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for suffix, rows, fields in (("tournaments", report["tournaments"], list(report["tournaments"][0]) if report["tournaments"] else ["tournament_id"]),
                                 ("rounds", rounds, list(rounds[0]) if rounds else ["tournament_id"]),
                                 ("gaps", gaps, ["tournament_id", "name", "discipline", "round_name", "match_id", "source_cache_path"])):
        with output.with_name(f"{output.stem}_{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# World Tour Coverage Audit", "", f"Cutoff: `{report['as_of']}`. Scope: {report['scope']}.", "",
             "| Coverage Status | Editions |", "| --- | ---: |"]
    lines.extend(f"| {status} | {count} |" for status, count in report["coverage_statuses"].items())
    lines += ["", f"Raw played matches: **{report['raw_played_matches']:,}**. Eligible training matches: **{report['eligible_matches']:,}**.",
              f"Cached published records absent from raw: **{report['missing_cached_records']:,}**. Matches receiving corrected tiers: **{report['tier_corrected_matches']:,}**.",
              "", "## Needs Review", "", "| Year | Tournament | Status |", "| --- | --- | --- |"]
    lines.extend(f"| {t['start_date'][:4]} | {t['name']} | {t['coverage_status']} |" for t in report["tournaments"]
                 if t["coverage_status"] not in {"cached_sources_match", "cancelled_or_postponed", "in_progress"})
    lines += ["", "## Limits", "", *[f"- {note}" for note in report["limits"]], "",
              "## Explicitly Unplayed Finals", ""]
    lines.extend(f"- {t['name']}: {t['unplayed_final_disciplines']} marked walkover/withdrawn/cancelled/No Match in the raw source; no played result was invented."
                 for t in report["tournaments"] if t["unplayed_final_disciplines"])
    lines += ["",
              "Per-tournament, per-discipline/round counts and exact cached gaps are in the accompanying CSVs.", ""]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.world_tour:
        if args.max_files is not None or args.include_para or args.no_deduplicate:
            raise ValueError("World Tour coverage requires the complete, deduplicated non-para dataset")
        catalog = build_catalog(args.cache_dir, args.catalog) if args.refresh_catalog else TournamentCatalog.load(args.catalog)
        cutoff = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
        report, rounds, gaps = world_tour_audit(args.data_dir, catalog, args.cache_dir, cutoff)
        write_world_tour_report(report, rounds, gaps, args.output)
        print(json.dumps({key: value for key, value in report.items() if key != "tournaments"}, indent=2))
        return 0
    files = sorted(args.data_dir.glob("*.json"))
    if args.max_files is not None:
        files = files[: args.max_files]

    matches, drop_counts, _ = load_all_matches(
        args.data_dir,
        max_files=args.max_files,
        include_para=args.include_para,
        deduplicate=not args.no_deduplicate,
    )
    report = build_data_report(
        matches,
        drop_counts,
        files_scanned=len(files),
        include_para=args.include_para,
        deduplicate=not args.no_deduplicate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nAudit written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
