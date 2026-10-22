"""Collect public BWF calendar entries and their daily match results."""
from __future__ import annotations

import hashlib
import csv
import json
import logging
import shutil
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from predict_badminton.config import RAW_JSON_DIR
from predict_badminton.data_parser import is_para_match, parse_file
from predict_badminton.data_sources.bwf import (
    MANIFEST_PATH,
    SOURCE_CACHE_DIR,
    FetchResult,
    fetch_url,
    group_matches_by_round,
    is_para_tournament,
    iter_raw_matches,
    parse_match_time,
    slugify,
    upsert_manifest_rows,
)
from predict_badminton.data_sources.bwf_schema import flatten_matches

logger = logging.getLogger(__name__)
API_BASE = "https://extranet-lv.bwfbadminton.com/api/"


class SourceFetchError(RuntimeError):
    def __init__(self, result: FetchResult) -> None:
        self.result = result
        super().__init__(f"{result.status}: {result.url} (HTTP {result.status_code}) {result.message}")


def _api_json(
    endpoint: str, params: dict[str, Any], cache_dir: Path,
    transport: str, delay_seconds: float, *, refresh: bool = False,
    completed_on: date | None = None,
) -> Any:
    url = API_BASE + endpoint + "?" + urlencode(params)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    cached = cache_dir / f"{slugify(urlparse(url).path)}-{digest}.json"
    for candidate in (cached, cached.with_suffix(".html")):
        if candidate.exists() and not refresh and (
            completed_on is None or datetime.fromtimestamp(candidate.stat().st_mtime, UTC).date() > completed_on
        ):
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Refreshing invalid cached JSON: %s", candidate)
                break
    time.sleep(delay_seconds)
    result = fetch_url(url, cache_dir, transport=transport)
    if result.status != "cached" or result.cache_path is None:
        raise SourceFetchError(result)
    try:
        return json.loads(result.cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected JSON from {url}") from exc


def discover_calendar(
    start: date, end: date, cache_dir: Path, transport: str, delay_seconds: float,
    *, refresh: bool = False,
) -> list[dict[str, Any]]:
    tournaments: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        payload = _api_json("vue-tournaments-search", {
            "startDate": (start - timedelta(days=1)).isoformat(),
            "endDate": (end + timedelta(days=1)).isoformat(),
            "discipline": "Badminton", "page": page, "perPage": 100,
            "drawCount": 1, "activeTab": 1,
        }, cache_dir, transport, delay_seconds, refresh=refresh)
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, dict) or not isinstance(results.get("data"), list):
            raise ValueError("Unexpected BWF calendar response structure")
        if int(results.get("current_page", 0)) != page:
            raise ValueError("BWF calendar returned the wrong page")
        for tournament in results["data"]:
            if is_para_tournament(str(tournament.get("name", ""))):
                continue
            if not tournament.get("id"):
                raise ValueError("BWF calendar entry has no tournament ID")
            if tournament["start_date"][:10] <= end.isoformat() and tournament["end_date"][:10] >= start.isoformat():
                tournaments[str(tournament["id"])] = tournament
        if page >= int(results["last_page"]):
            break
        if not results["data"]:
            raise ValueError("BWF calendar pagination ended before the last page")
        page += 1
    return sorted(tournaments.values(), key=lambda t: (t["start_date"], t["id"]))


def _draw_matches(
    tournament: dict[str, Any], cache_dir: Path, transport: str, delay_seconds: float,
    *, refresh: bool = False,
) -> list[dict[str, Any]]:
    # The official individual and team views do not always expose the same draws.
    for tournament_type in (1, 0):
        matches = _draw_view_matches(tournament, tournament_type, cache_dir, transport, delay_seconds, refresh=refresh)
        if matches:
            return matches
    return []


def _draw_view_matches(
    tournament: dict[str, Any], tournament_type: int, cache_dir: Path,
    transport: str, delay_seconds: float, *, refresh: bool = False,
) -> list[dict[str, Any]]:
    params = {"tmtTab": "draw", "tmtId": tournament["id"], "tmtType": tournament_type}
    payload = _api_json("vue-tournament-draws", params, cache_dir, transport, delay_seconds, refresh=refresh,
                        completed_on=parse_match_time(tournament["end_date"]).date())
    if not isinstance(payload, dict):
        raise ValueError("Unexpected BWF draws response structure")
    draws = payload.get("results") or []
    if not isinstance(draws, list):
        raise ValueError("Expected a list of BWF draws")
    matches = []
    for draw in draws:
        draw_type = int(draw["type"])
        # BWF's frontend sends every non-round-robin type to this endpoint.
        endpoint = "vue-tournament-draw-data-round-robin" if draw_type == 1 else "vue-tournament-draw-data"
        data = _api_json(endpoint, {**params, "drawId": draw["value"], "isPara": "false"}, cache_dir, transport, delay_seconds,
                         refresh=refresh, completed_on=parse_match_time(tournament["end_date"]).date())
        if not isinstance(data, dict):
            raise ValueError("Unexpected BWF draw-data response structure")
        records = data.get("results") if draw_type == 1 else data.get("matches")
        if not records and draw_type != 1 and data.get("results"):
            # Older knockout responses only expose bracket cells, with draw-local codes.
            cells = data["results"]
            if cells == "No draw data - possible exception":
                logger.warning("BWF reports no knockout data for tournament %s, draw %s", tournament["id"], draw["value"])
                continue
            if not isinstance(cells, dict) or any(not isinstance(cell, dict) or not isinstance(cell.get("match"), dict)
                                                  for cell in cells.values()):
                raise ValueError("Unexpected legacy BWF bracket structure")
            records = []
            for cell in cells.values():
                match = dict(cell["match"])
                source_code = str(match.get("tournamentCode", ""))
                if source_code not in {str(tournament["id"]), str(tournament["code"])}:
                    raise ValueError("Legacy draw contains a different tournament")
                match.update(tournamentCode=tournament["code"], tournamentName=tournament["name"],
                             sourceTournamentId=tournament["id"], sourceDrawId=draw["value"],
                             sourceUrl=API_BASE + endpoint + "?" + urlencode({**params, "drawId": draw["value"], "isPara": "false"}))
                records.append(match)
        matches.extend(flatten_matches(records or []))
    return matches


def _identity(match: dict[str, Any]) -> str:
    match_id = match.get("id")
    if match_id is not None and str(match_id).strip():
        return "id:" + str(match_id)
    if match.get("code"):
        scope = str(match["sourceDrawId"]) + ":" if "sourceDrawId" in match else ""
        return "code:" + str(match.get("tournamentCode", "")).casefold() + ":" + scope + str(match["code"]).casefold()
    teams = [[str(p["id"]) for p in (match.get(key) or {}).get("players", []) if p.get("id")]
             for key in ("team1", "team2")]
    stamp = match.get("matchTimeUtc") or match.get("matchTime")
    if stamp and all(teams):
        signature = [match.get("tournamentCode") or match.get("tournamentName"),
                     stamp, match.get("eventName"), match.get("roundName"), teams]
        return "signature:" + hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    return "content:" + hashlib.sha256(json.dumps(match, sort_keys=True).encode()).hexdigest()


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _merge_match(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    # A thinner or not-yet-scored response must not erase an existing result.
    if old.get("winner") in (1, 2) and new.get("winner") not in (1, 2):
        return old
    merged = dict(old)
    for key, value in new.items():
        if value in (None, "", [], {}):
            continue
        if key == "players" and isinstance(value, list) and isinstance(old.get(key), list):
            players = {str(p.get("id")): p for p in old[key]}
            merged[key] = [_merge_match(players.get(str(p.get("id")), {}), p) for p in value]
        elif isinstance(old.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_match(old[key], value)
        else:
            merged[key] = value
    return merged


def write_change_report(report: dict[str, Any], run_dir: Path, raw_dir: Path) -> None:
    """Compare with the original baseline, including changes made before a resume."""
    baseline = json.loads((run_dir / "baseline_files.json").read_text(encoding="utf-8"))
    files = sorted(raw_dir.glob("*.json"))
    rows = []
    for path in files:
        if baseline.get(path.name) == hashlib.sha256(path.read_bytes()).hexdigest():
            continue
        old = {}
        if path.name in baseline:
            backup = run_dir / "backups" / path.name
            if not backup.exists() or hashlib.sha256(backup.read_bytes()).hexdigest() != baseline[path.name]:
                raise ValueError(f"Original backup does not match baseline: {path.name}")
            old = {_identity(m): m for m in iter_raw_matches(json.loads(backup.read_text(encoding="utf-8")))}
        current = {_identity(m): m for m in iter_raw_matches(json.loads(path.read_text(encoding="utf-8")))}
        added = current.keys() - old.keys()
        if old.keys() - current.keys():
            raise ValueError(f"Stored match records disappeared from {path.name}")
        records = list(current.values())
        dates = [stamp for m in records if (stamp := parse_match_time(m.get("matchTimeUtc") or m.get("matchTime")))]
        entry = next((t for t in report["tournaments"] if t.get("raw_path") == str(path)), {})
        year = entry.get("start_date", "")[:4] or (str(min(dates).year) if dates else "unknown")
        rows.append({
            "year": year, "tournament": entry.get("name") or next((m.get("tournamentName") for m in records if m.get("tournamentName")), path.stem),
            "new_file": path.name not in baseline, "new_matches": len(added),
            "updated_matches": sum(current[key] != value for key, value in old.items()),
            "newly_finished_matches": sum(value.get("winner") not in (1, 2) and current[key].get("winner") in (1, 2)
                                          for key, value in old.items()),
            "raw_path": str(path),
        })
    missing = sorted(set(baseline) - {path.name for path in files})
    if missing:
        raise ValueError(f"Baseline raw files disappeared: {missing}")
    rows.sort(key=lambda row: (row["year"], row["tournament"]))
    report["changes"] = {"new_files": sum(row["new_file"] for row in rows),
        "changed_existing_files": sum(not row["new_file"] for row in rows),
        "new_matches": sum(row["new_matches"] for row in rows),
        "updated_matches": sum(row["updated_matches"] for row in rows), "tournaments": rows}
    report["storage"] = {"raw_files": len(files), "logical_bytes": sum(p.stat().st_size for p in files),
                         "allocated_bytes": sum(p.stat().st_blocks * 512 for p in files)}
    with open(run_dir / "changes.csv", "w", newline="", encoding="utf-8") as handle:
        fields = ["year", "tournament", "new_file", "new_matches", "updated_matches", "newly_finished_matches", "raw_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# BWF Collection Report", "", f"Window: {report['start_date']} through {report['collected_through']}.",
             f"Status: {report['status']}. Calendar entries: {report.get('discovered_tournaments', 0)}.", "",
             f"New match records: {report['changes']['new_matches']:,}. New files: {report['changes']['new_files']}.",
             "Raw records can include unplayed bracket entries and duplicate source representations; they are not a count of unique completed matches.",
             "Years below use the tournament's calendar start year, which can differ from the season year in its name.",
             f"Raw files: {len(files):,}. Logical bytes: {report['storage']['logical_bytes']:,}. Allocated bytes: {report['storage']['allocated_bytes']:,}.",
             "", "## Tournaments With Added Match Records", "",
             "| Year | Tournament | New Matches | New File |", "| --- | --- | ---: | --- |"]
    lines.extend(f"| {r['year']} | {r['tournament'].replace('|', '/')} | {r['new_matches']} | {'yes' if r['new_file'] else 'no'} |"
                 for r in rows if r["new_matches"])
    lines.extend(["", "## Unavailable Or Incomplete Sources", "", "| Year | Tournament | Status |", "| --- | --- | --- |"])
    lines.extend(f"| {t.get('start_date', '')[:4]} | {t['name'].replace('|', '/')} | {t['status']} |"
                 for t in report["tournaments"] if t["status"] in {"empty", "failed", "blocked", "partial"})
    lines.extend(["", "## Files With No Training-Ready Records", "",
                  "Raw source records are retained even when player IDs, times, or finished results are unavailable.", "",
                  "| Year | Tournament | Parser Exclusions |", "| --- | --- | --- |"])
    lines.extend(f"| {t.get('start_date', '')[:4]} | {t['name'].replace('|', '/')} | {json.dumps(t.get('parser_drops', {}), sort_keys=True)} |"
                 for t in report["tournaments"] if t.get("saved_matches") and t.get("training_matches") == 0)
    lines.extend(["", "Existing records were preserved. Changed original files have checksum-verified backups in backups/.",
                  "Updates may be metadata changes, not only score changes. changes.csv lists every changed tournament file.",
                  "Source coverage does not guarantee that BWF published every played match.", ""])
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    report["summary_path"] = str(run_dir / "summary.md")


def backfill_matches(
    start: date, end: date, *, today: date,
    transport: str = "chrome", delay_seconds: float = 1.0,
    raw_dir: Path = RAW_JSON_DIR, cache_root: Path = SOURCE_CACHE_DIR,
    manifest_path: Path = MANIFEST_PATH,
    refresh: bool = False,
    tournaments: list[dict[str, Any]] | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    if start > end or start > today or delay_seconds < 0:
        raise ValueError("Require start <= end, start <= today, and a nonnegative delay")
    effective_end = min(end, today)
    run_dir = run_dir or cache_root / f"backfill-{start}-{end}"
    report_path = run_dir / "report.json"
    baseline_path = run_dir / "baseline_files.json"
    if not baseline_path.exists():
        _save_json(baseline_path, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(raw_dir.glob("*.json"))})
    previous = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    if previous:
        attempts = run_dir / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, attempts / f"report-{datetime.now(UTC):%Y%m%dT%H%M%S.%fZ}.json")
    previous_results = {t["tournament_id"]: t for t in previous.get("tournaments", [])}
    report: dict[str, Any] = {
        "start_date": start.isoformat(), "requested_end_date": end.isoformat(),
        "collected_through": effective_end.isoformat(), "transport": transport,
        "report_path": str(report_path), "status": "running", "tournaments": [],
    }
    _save_json(report_path, report)
    try:
        if tournaments is None:
            tournaments = discover_calendar(start, effective_end, run_dir / "responses", transport, delay_seconds, refresh=True)
            tournaments.sort(key=lambda t: t["start_date"], reverse=True)
    except (SourceFetchError, ValueError) as exc:
        report.update(status="failed", message=str(exc))
        _save_json(report_path, report)
        raise
    _save_json(run_dir / "tournaments.json", tournaments)
    report["discovered_tournaments"] = len(tournaments)
    logger.info("Discovered %d non-para tournaments for %s to %s", len(tournaments), start, effective_end)
    owners: dict[str, Path] = {}
    paths_by_code: dict[str, list[Path]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        for match in iter_raw_matches(json.loads(path.read_text(encoding="utf-8"))):
            owners[_identity(match)] = path
            code = str(match.get("tournamentCode") or "").casefold()
            if code and path not in paths_by_code.setdefault(code, []):
                paths_by_code[code].append(path)
    for index, tournament in enumerate(tournaments, 1):
        name = tournament["name"]
        result: dict[str, Any] = {
            "tournament_id": tournament["id"], "name": name, "source_url": tournament["url"],
            "status": "running", "new_matches": 0, "updated_matches": 0, "days_fetched": 0,
            "start_date": tournament["start_date"], "end_date": tournament["end_date"],
        }
        prior = previous_results.get(tournament["id"], {})
        if prior.get("raw_path") and Path(prior["raw_path"]).exists():
            for key in ("raw_path", "training_matches", "parser_drops"):
                if key in prior:
                    result[key] = prior[key]
            result["saved_matches"] = prior.get("saved_matches", prior.get("new_matches", 0))
        report["tournaments"].append(result)
        try:
            tournament_start = parse_match_time(tournament["start_date"])
            tournament_end = parse_match_time(tournament["end_date"])
            if tournament_start is None or tournament_end is None or tournament_end < tournament_start:
                raise ValueError("Invalid tournament dates in calendar")
            status = tournament.get("status") or {}
            if (str(status.get("status")) in {"198", "199"}
                    or status.get("code") in {"cancelled", "postponed"}
                    or "cancelled" in name.casefold() or "postponed" in name.casefold()):
                result.update(status="skipped", message=str(status.get("label") or "cancelled/postponed"))
            elif not tournament.get("code"):
                result.update(status="empty", message="No public results code in the official calendar")
            else:
                matches: list[dict[str, Any]] = []
                local_seen = set()
                day = max(tournament_start.date(), start - timedelta(days=1))
                last_day = min(tournament_end.date(), effective_end + timedelta(days=1), today)
                while day <= last_day:
                    try:
                        payload = _api_json("tournaments/day-matches", {
                            "tournamentCode": tournament["code"], "date": day.isoformat(),
                            "order": 2, "court": 0,
                        }, run_dir / "responses", transport, delay_seconds, refresh=refresh,
                           completed_on=tournament_end.date())
                    except SourceFetchError as exc:
                        if exc.result.status != "failed" or not 500 <= (exc.result.status_code or 0) < 600:
                            raise
                        result["daily_error"] = str(exc)
                        break
                    for match in flatten_matches(payload):
                        if str(match.get("tournamentCode")).casefold() != str(tournament["code"]).casefold():
                            raise ValueError("Match response contains a different tournament code")
                        if is_para_match(match):
                            continue
                        match_time = parse_match_time(match.get("matchTimeUtc") or match.get("matchTime"))
                        if match_time is not None and not start <= match_time.date() <= effective_end:
                            continue
                        if match_time is None and not start <= day <= effective_end:
                            continue
                        identity = _identity(match)
                        if identity not in local_seen:
                            local_seen.add(identity)
                            matches.append(match)
                    result["days_fetched"] += 1
                    day += timedelta(days=1)
                if (not matches or result.get("daily_error")) and tournament_start.date() >= start and tournament_end.date() <= effective_end:
                    result["source"] = "draws_fallback"
                    for match in _draw_matches(tournament, run_dir / "responses", transport, delay_seconds, refresh=refresh):
                        if str(match.get("tournamentCode")).casefold() != str(tournament["code"]).casefold():
                            raise ValueError("Draw response contains a different tournament code")
                        if is_para_match(match):
                            continue
                        match_time = parse_match_time(match.get("matchTimeUtc") or match.get("matchTime"))
                        if match_time and not start <= match_time.date() <= effective_end:
                            continue
                        identity = _identity(match)
                        if identity not in local_seen:
                            local_seen.add(identity)
                            matches.append(match)
                result["source_matches"] = len(matches)
                code = str(tournament["code"]).casefold()
                slug = slugify(urlparse(tournament["url"]).path.rstrip("/").split("/")[-1])
                known_paths = paths_by_code.get(code, [])
                target = known_paths[0] if known_paths else raw_dir / f"{slug}.json"
                if target.exists() and target not in known_paths:
                    existing_codes = {str(m.get("tournamentCode") or "").casefold()
                                      for m in iter_raw_matches(json.loads(target.read_text(encoding="utf-8")))}
                    if existing_codes - {code, ""}:
                        target = raw_dir / f"{slug}-{tournament['id']}.json"
                by_path: dict[Path, list[dict[str, Any]]] = {}
                for match in matches:
                    path = owners.get(_identity(match), target)
                    if path not in known_paths and path != target:
                        raise ValueError(f"Match ID collision with another tournament in {path.name}")
                    by_path.setdefault(path, []).append(match)
                training_matches = saved_matches = 0
                all_drops: Counter = Counter()
                changed_paths = []
                for path, incoming in by_path.items():
                    original = list(iter_raw_matches(json.loads(path.read_text(encoding="utf-8")))) if path.exists() else []
                    combined = {_identity(m): m for m in original}
                    changed = 0
                    for match in incoming:
                        identity = _identity(match)
                        if identity not in combined:
                            result["new_matches"] += 1
                            combined[identity] = match
                            changed += 1
                        else:
                            merged = _merge_match(combined[identity], match)
                            if merged != combined[identity]:
                                result["updated_matches"] += 1
                                combined[identity] = merged
                                changed += 1
                    staged = run_dir / "staging" / path.name
                    _save_json(staged, group_matches_by_round(list(combined.values())))
                    drops: dict[str, int] = {}
                    parsed = parse_file(staged, drops)
                    if drops.get("file_error") or drops.get("para_match"):
                        raise ValueError(f"Staged file failed validation: {drops}")
                    training_matches += len(parsed)
                    saved_matches += len(combined)
                    all_drops.update(drops)
                    if changed:
                        if path.exists():
                            backup = run_dir / "backups" / path.name
                            backup.parent.mkdir(parents=True, exist_ok=True)
                            if not backup.exists():
                                shutil.copy2(path, backup)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        staged.replace(path)
                        changed_paths.append(str(path))
                    else:
                        staged.unlink()
                    for identity in combined:
                        owners[identity] = path
                    if path not in paths_by_code.setdefault(code, []):
                        paths_by_code[code].append(path)
                if matches:
                    result.update(training_matches=training_matches, parser_drops=dict(all_drops),
                                  raw_path=str(target), saved_matches=saved_matches, changed_paths=changed_paths)
                    result["status"] = "updated" if result["updated_matches"] else "imported" if result["new_matches"] else "verified"
                    if tournament_end.date() > today or result.get("daily_error"):
                        result["status"] = "partial"
                else:
                    result["status"] = "empty"
        except SourceFetchError as exc:
            result.update(status=exc.result.status, message=str(exc), blocked_url=exc.result.url)
            # Network-wide failures must not be mislabeled as hundreds of bad tournaments.
            if exc.result.status == "blocked":
                report["status"] = "blocked"
            elif exc.result.status_code is None:
                report["status"] = "network_error"
        except (ValueError, OSError) as exc:
            result.update(status="failed", message=str(exc))
        upsert_manifest_rows([{
            **result, "message": json.dumps({k: v for k, v in result.items() if k not in {"name", "source_url", "raw_path"}}),
        }], manifest_path)
        report["summary"] = dict(Counter(t["status"] for t in report["tournaments"]))
        report["new_matches"] = sum(t["new_matches"] for t in report["tournaments"] if t["status"] in {"imported", "updated", "partial"})
        report["updated_matches"] = sum(t["updated_matches"] for t in report["tournaments"] if t["status"] in {"updated", "partial"})
        report["total_saved_matches"] = sum(t.get("saved_matches", 0) for t in report["tournaments"])
        _save_json(report_path, report)
        logger.info("[%d/%d] %s: %s (%d new matches)", index, len(tournaments), name, result["status"], result["new_matches"])
        if report["status"] in {"blocked", "network_error"}:
            break
    if report["status"] not in {"blocked", "network_error"}:
        report["status"] = "complete" if not report.get("summary", {}).get("failed") else "incomplete"
    report["unattempted_tournaments"] = len(tournaments) - len(report["tournaments"])
    by_id = {t["id"]: t for t in tournaments}
    report["unavailable_tournaments"] = [t["name"] for t in report["tournaments"] if t["status"] == "empty"
        and by_id[t["tournament_id"]]["end_date"][:10] >= start.isoformat()
        and by_id[t["tournament_id"]]["start_date"][:10] <= effective_end.isoformat()]
    if (report["unavailable_tournaments"] or report.get("summary", {}).get("partial")) and report["status"] == "complete":
        report["status"] = "incomplete"
    write_change_report(report, run_dir, raw_dir)
    _save_json(report_path, report)
    return report
