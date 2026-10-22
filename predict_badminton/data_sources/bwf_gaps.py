"""Retry an earlier collection's empty entries using additional official draw views."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode, urlparse

from predict_badminton.config import RAW_JSON_DIR
from predict_badminton.data_sources.bwf import MANIFEST_PATH, is_para_tournament, iter_raw_matches, parse_match_time, slugify
from predict_badminton.data_sources.bwf_backfill import API_BASE, _save_json, backfill_matches


def _priority(tournament: dict) -> tuple:
    label = (tournament["name"] + " " + str(tournament.get("category") or "")).casefold()
    junior = any(word in label for word in ("junior", "u17", "u19", "u15", "u13"))
    major = any(word in label for word in ("world tour", "continental", "sea games", "african games", "european", "pan am"))
    return (2 if junior else 0 if major else 1, tournament["start_date"], tournament["id"])


def recover_gaps(
    source_report: Path, *, today: date, transport: str = "chrome", delay_seconds: float = 1.0,
    raw_dir: Path = RAW_JSON_DIR, manifest_path: Path = MANIFEST_PATH,
) -> dict:
    previous = json.loads(source_report.read_text(encoding="utf-8"))
    calendar = json.loads((source_report.parent / "tournaments.json").read_text(encoding="utf-8"))
    ids = {entry["tournament_id"] for entry in previous["tournaments"] if entry["status"] == "empty"}
    tournaments = sorted((entry for entry in calendar if entry["id"] in ids), key=_priority)
    if {entry["id"] for entry in tournaments} != ids:
        raise ValueError("Some source gaps are missing from the saved official calendar")
    if any(is_para_tournament(entry["name"]) for entry in tournaments):
        raise ValueError("Gap recovery is restricted to non-para tournaments")
    start = date.fromisoformat(previous["start_date"])
    end = date.fromisoformat(previous["collected_through"])
    run_dir = source_report.parent.parent / f"gap-recovery-{start}-{end}"
    if source_report.resolve() == (run_dir / "report.json").resolve():
        raise ValueError("Pass the original backfill report, not the gap-recovery report")

    # Reuse exact-URL responses already fetched after these tournaments ended.
    # Only small day/list responses for the selected gaps are copied; no raw data moves.
    cache = run_dir / "responses"
    cache.mkdir(parents=True, exist_ok=True)
    for tournament in tournaments:
        day = parse_match_time(tournament["start_date"]).date()
        last_day = parse_match_time(tournament["end_date"]).date()
        requests = []
        while day <= min(last_day, end):
            requests.append(("tournaments/day-matches", {
                "tournamentCode": tournament["code"], "date": day.isoformat(), "order": 2, "court": 0,
            }))
            day += timedelta(days=1)
        requests.append(("vue-tournament-draws", {"tmtTab": "draw", "tmtId": tournament["id"], "tmtType": 1}))
        for endpoint, params in requests:
            url = API_BASE + endpoint + "?" + urlencode(params)
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            name = f"{slugify(urlparse(url).path)}-{digest}"
            for suffix in (".json", ".html"):
                source = source_report.parent / "responses" / (name + suffix)
                destination = cache / source.name
                if source.exists() and not destination.exists():
                    shutil.copy2(source, destination)
    report = backfill_matches(start, end, today=today, transport=transport, delay_seconds=delay_seconds,
                             raw_dir=raw_dir, manifest_path=manifest_path, tournaments=tournaments, run_dir=run_dir)
    write_gap_report(report, source_report)
    return report


def write_gap_report(report: dict, source_report: Path) -> None:
    """Keep archival recovery separate from usable matches and calendar corrections."""
    run_dir = Path(report["report_path"]).parent
    evidence_path = run_dir / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else []
    evidence_by_id = {entry["tournament_id"]: entry for entry in evidence}
    previous = json.loads(source_report.read_text(encoding="utf-8"))
    previous_by_id = {entry["tournament_id"]: entry for entry in previous["tournaments"]}
    rows = []
    for entry in report["tournaments"]:
        records = []
        if entry.get("raw_path"):
            payload = json.loads(Path(entry["raw_path"]).read_text(encoding="utf-8"))
            records = list(iter_raw_matches(payload))
        scored = sum(match.get("winner") in (1, 2) and bool(match.get("score")) and all(
            any(player.get("id") for player in (match.get(key) or {}).get("players", []))
            for key in ("team1", "team2")) for match in records)
        assessment = ("training_ready" if entry.get("training_matches") else "scored_results_missing_times" if scored
                      else "team_results_only" if any(match.get("score") for match in records)
                      else "draw_only_no_scores" if records else "unresolved")
        note = evidence_by_id.get(entry["tournament_id"], {})
        if note.get("assessment") == "covered_by_existing_events":
            if not note.get("covered_by_ids") or any(not previous_by_id.get(identifier, {}).get("training_matches")
                                                     for identifier in note["covered_by_ids"]):
                raise ValueError("Calendar coverage evidence must reference existing training-ready tournament data")
        if not records:
            assessment = note.get("assessment", assessment)
        row = {
            "tournament_id": entry["tournament_id"], "year": entry["start_date"][:4], "name": entry["name"],
            "assessment": assessment, "raw_records": len(records), "scored_player_matches": scored,
            "training_matches": entry.get("training_matches", 0), "source_url": entry["source_url"],
            "evidence_urls": " | ".join(note.get("urls", [])), "covered_by_ids": "|".join(map(str, note.get("covered_by_ids", []))),
            "notes": note.get("notes", ""), "raw_path": entry.get("raw_path", ""),
        }
        rows.append(row)
        entry.update(assessment=assessment, scored_player_matches=scored)
    report["gap_investigation"] = {
        "source_report": str(source_report), "assessments": dict(Counter(row["assessment"] for row in rows)),
        "scored_player_matches": sum(row["scored_player_matches"] for row in rows),
        "training_matches": sum(row["training_matches"] for row in rows),
    }
    with open(run_dir / "gap_investigation.csv", "w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["tournament_id", "assessment"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# BWF Gap Investigation", "", f"Checked {len(rows)} previously empty non-para calendar entries; senior championships first.",
             f"Window: {report['start_date']} through {report['collected_through']}.", "",
             "## Results", "", *[f"- {key}: {value}" for key, value in report["gap_investigation"]["assessments"].items()], "",
             f"Scored matches with player IDs: {report['gap_investigation']['scored_player_matches']}.",
             f"Training-ready matches in recovered files: {report['gap_investigation']['training_matches']}.",
             "Raw records include byes, unplayed draw slots, and team-level summaries; do not count every raw record as a played match.",
             "Missing times and IDs were not invented. Archived results do not establish complete tournament coverage.", "",
             "## Archived Source Records", "", "| Year | Tournament | Raw Records | Scored Player Matches | Training Ready |",
             "| --- | --- | ---: | ---: | ---: |"]
    lines.extend(f"| {row['year']} | {row['name'].replace('|', '/')} | {row['raw_records']} | {row['scored_player_matches']} | {row['training_matches']} |"
                 for row in rows if row["raw_records"])
    lines.extend(["", "## Official Evidence And Calendar Corrections", ""])
    for row in rows:
        if row["evidence_urls"]:
            lines.extend([f"### {row['name']}", "", f"Assessment: {row['assessment']}.", row["notes"],
                          *[f"- {url}" for url in row["evidence_urls"].split(" | ")], ""])
    lines.extend(["## Files And Remaining Work", "", "gap_investigation.csv lists every checked entry, including unresolved sources and possible duplicates.",
                  "evidence.json retains manually reviewed official-source findings. official-sources/ retains downloaded evidence.",
                  "report.json retains fetch outcomes; summary.md and changes.csv compare raw data against the pre-recovery baseline.",
                  "Empty results, broken links and access challenges do not prove that no matches were played.",
                  "Full recovery needs dated, player-identified match records from an official source. No login or access challenge was bypassed.", ""])
    (run_dir / "gap_investigation.md").write_text("\n".join(lines), encoding="utf-8")
    _save_json(Path(report["report_path"]), report)
