"""Load and normalise raw BWF JSON tournament files into flat match records.

Each JSON file is a list of rounds, each round is a list of match objects.
We flatten to individual matches, parse timestamps, and extract player IDs.

Drop policy (tracked in counts dict returned by load_all_matches):
  - winner not in {1, 2}: cancelled / in-progress
  - official walkover status: no played match for training or Elo
  - no parseable match time: cannot order chronologically
  - empty player IDs for either team: cannot compute Elo
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from predict_badminton.match_rules import point_margin, unplayed_status
from predict_badminton.temporal import match_timing
from predict_badminton.tournament_catalog import TournamentCatalog, official_tier

logger = logging.getLogger(__name__)

_DISC_PREFIX_RE = re.compile(r"^(MS|WS|MD|WD|XD|BS|GS|BD|GD)(?:$|[ -]|U\d+\b)")
_JUNIOR_DISCIPLINES = {"BS": "MS", "GS": "WS", "BD": "MD", "GD": "WD"}
_DISC_NAMES = {
    "men's singles": "MS", "women's singles": "WS",
    "men's doubles": "MD", "women's doubles": "WD", "mixed doubles": "XD",
    "boy's singles": "MS", "girl's singles": "WS",
    "boy's doubles": "MD", "girl's doubles": "WD",
}

_PARA_TOURNAMENT_RE = re.compile(
    r"\b(?:para(?:\s*-\s*|\s+)?badminton|paralympic|para\s+games?)\b",
    re.IGNORECASE,
)
_PARA_CLASS_RE = re.compile(
    r"\b(?:WH\s*[12]|SL\s*[34]|SU\s*5|SH\s*6)\b",
    re.IGNORECASE,
)

_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


def normalize_discipline(event_name: str) -> str:
    """Map raw eventName → one of MS / WS / MD / WD / XD / OTHER."""
    if not event_name:
        return "OTHER"
    en = event_name.strip().replace("\u2019", "'")
    if discipline := _DISC_NAMES.get(en.casefold()):
        return discipline
    if prefix := _DISC_PREFIX_RE.match(en.upper()):
        disc = prefix.group(1)
        return _JUNIOR_DISCIPLINES.get(disc, disc)
    return "OTHER"


def infer_tournament_tier(name: str) -> int:
    """Heuristic tournament tier 1 (elite) → 6 (lowest) from name keywords."""
    n = name.lower()
    if any(
        kw in n
        for kw in ["olympic", "world championship", "thomas cup", "uber cup", "sudirman"]
    ):
        return 1
    if "super 1000" in n or "bwf world tour finals" in n or "all england" in n:
        return 2
    if "super 750" in n:
        return 3
    if "super 500" in n:
        return 4
    if "super 300" in n or "super series" in n:
        return 5
    if any(
        kw in n
        for kw in [
            "international series",
            "international challenge",
            "grand prix",
            "international circuit",
        ]
    ):
        return 6
    return 5  # default


def is_para_match(match: dict[str, Any]) -> bool:
    """Return whether tournament or event metadata identifies para badminton."""
    tournament_name = str(match.get("tournamentName") or "")
    event_name = str(match.get("eventName") or "")
    return bool(
        _PARA_TOURNAMENT_RE.search(tournament_name)
        or _PARA_CLASS_RE.search(event_name)
    )


def _parse_time(raw: str) -> datetime | None:
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            continue
    return None


def _extract_players(
    team: dict[str, Any],
    name_map: dict[str, str] | None = None,
) -> list[str]:
    players = team.get("players") or []
    ids = []
    for p in players:
        if not p.get("id"):
            continue
        pid = str(p["id"])
        ids.append(pid)
        if name_map is not None:
            name = p.get("nameDisplay") or p.get("firstName", "") + " " + p.get("lastName", "")
            name_map[pid] = name.strip()
    return ids


def _iter_matches(data: Any) -> Iterator[dict[str, Any]]:
    """Yield raw match dicts from a parsed JSON file (list-of-rounds format)."""
    if not isinstance(data, list):
        return
    for rnd in data:
        if not isinstance(rnd, list):
            continue
        for match in rnd:
            if isinstance(match, dict):
                yield match


def parse_file(
    filepath: Path,
    drop_counts: dict[str, int],
    name_map: dict[str, str] | None = None,
    *,
    include_para: bool = False,
    observed_results: dict[str, dict[str, Any] | None] | None = None,
    category_catalog: TournamentCatalog | None = None,
) -> list[dict[str, Any]]:
    """Parse one JSON file; return list of normalised match dicts.

    If name_map is provided it is updated in-place with {player_id: display_name}.
    Observation rejections stay marked until load_all_matches resolves fixture aliases.
    """
    records: list[dict[str, Any]] = []
    try:
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.warning("Cannot load %s: %s", filepath.name, exc)
        drop_counts["file_error"] = drop_counts.get("file_error", 0) + 1
        return records

    for match in _iter_matches(data):
        if not include_para and is_para_match(match):
            drop_counts["para_match"] = drop_counts.get("para_match", 0) + 1
            continue

        match_id = str(match.get("id", "")).strip()
        tracked = observed_results is not None and match_id in observed_results
        winner = match.get("winner")
        if winner not in (1, 2) and not tracked:
            drop_counts["bad_winner"] = drop_counts.get("bad_winner", 0) + 1
            continue
        if not tracked and (status := unplayed_status(match)):
            drop_counts[status] = drop_counts.get(status, 0) + 1
            continue

        # Prefer UTC timestamp; fall back to local matchTime
        raw_time = match.get("matchTimeUtc") or match.get("matchTime") or ""
        mt = _parse_time(raw_time) if raw_time else None
        if mt is None:
            drop_counts["no_time"] = drop_counts.get("no_time", 0) + 1
            continue

        team1 = match.get("team1") or {}
        team2 = match.get("team2") or {}
        ids1 = _extract_players(team1, name_map)
        ids2 = _extract_players(team2, name_map)
        if not ids1 or not ids2:
            drop_counts["no_player_ids"] = drop_counts.get("no_player_ids", 0) + 1
            continue

        disc = normalize_discipline(match.get("eventName") or "")
        t_name = match.get("tournamentName") or ""
        category = category_catalog.lookup(match.get("tournamentCode"), match.get("sourceTournamentId")) if category_catalog else None
        tier = official_tier(category["category"]) if category else None
        timing = match_timing(
            mt, utc=bool(match.get("matchTimeUtc")), duration=match.get("duration"),
            coarse=(len(raw_time.strip()) == 10 or mt.time() == datetime.min.time()
                    or match.get("timePrecision") == "date"),
        )
        observation_error = None
        if tracked:
            from predict_badminton.data_sources.observations import result_version

            receipt = observed_results[match_id]
            version = result_version(match)
            if receipt is None or version is None or version[0] != receipt["result_sha256"]:
                observation_error = "unobserved_or_conflicting_result_version"
            elif receipt["available_at"] <= timing["prediction_time"]:
                observation_error = "observation_before_match"
            else:
                timing.update(available_at=receipt["available_at"], availability_source="observed_completed_result")
        teams = tuple(sorted((tuple(sorted(ids1)), tuple(sorted(ids2)))))

        records.append(
            {
                "match_id": match.get("id"),
                "match_time": mt,
                "tournament_code": match.get("tournamentCode") or "",
                "tournament_name": t_name,
                "tournament_tier": tier if tier is not None else infer_tournament_tier(t_name),
                "tournament_id": category["tournament_id"] if category else "",
                "tournament_category": category["category"] if category else "",
                "tournament_tier_source": "official_world_tour" if tier is not None else "name_heuristic",
                "event_name": match.get("eventName") or "",
                "discipline": disc,
                "round_name": match.get("roundName") or "",
                "location_name": match.get("locationName") or "",
                "is_team_match": bool(match.get("isTeamMatch")),
                "team1_ids": ids1,
                "team2_ids": ids2,
                "team1_country": team1.get("countryCode") or "",
                "team2_country": team2.get("countryCode") or "",
                "team1_seed": match.get("team1seed"),
                "team2_seed": match.get("team2seed"),
                "winner": int(winner) if winner in (1, 2) else None,
                "label": int(winner == 1) if winner in (1, 2) else None,
                "score_margin": point_margin(match),
                **timing,
                "_observation_error": observation_error,
                "_has_utc_time": bool(match.get("matchTimeUtc")),
                "_fixture_key": (
                    str(match.get("tournamentCode") or t_name).strip().casefold(),
                    match.get("matchTime") or raw_time, match.get("eventName"),
                    match.get("roundName"), teams, match.get("matchOrder"), match.get("matchTypeNo"),
                ),
            }
        )
    return records


def _match_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    match_id = record.get("match_id")
    if match_id is not None and str(match_id).strip():
        return ("id", str(match_id).strip())
    if "_fixture_key" in record:
        return ("signature", record["_fixture_key"])

    teams = tuple(
        sorted(
            (
                tuple(sorted(record["team1_ids"])),
                tuple(sorted(record["team2_ids"])),
            )
        )
    )
    tournament = record.get("tournament_code") or record.get("tournament_name") or ""
    return (
        "signature",
        str(tournament).strip().casefold(),
        record["match_time"].isoformat(),
        record["discipline"],
        str(record.get("round_name") or "").strip().casefold(),
        teams,
    )


def _deduplicate_matches(
    records: list[dict[str, Any]],
    drop_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Join ID/fixture aliases; exclude conflicting outcomes instead of choosing one."""
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple, int] = {}
    for index, record in enumerate(records):
        for key in (_match_identity(record), ("fixture", record["_fixture_key"])):
            if key in seen:
                parents[find(index)] = find(seen[key])
            seen[key] = index
    groups: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)

    unique = []
    for group in groups.values():
        # Keep rejected copies until alias grouping so an untracked copy cannot restore them.
        errors = sorted({m["_observation_error"] for m in group if m.get("_observation_error")})
        if errors:
            drop_counts[errors[0]] = drop_counts.get(errors[0], 0) + len(group)
            continue
        teams = {tuple(sorted((tuple(sorted(m["team1_ids"])), tuple(sorted(m["team2_ids"]))))) for m in group}
        winners = {tuple(sorted(m[f"team{m['winner']}_ids"])) for m in group}
        if len(teams) != 1 or len(winners) != 1:
            drop_counts["conflicting_result"] = drop_counts.get("conflicting_result", 0) + len(group)
            continue
        group.sort(key=lambda m: (m["availability_source"] != "observed_completed_result",
                                  not m["_has_utc_time"], m["match_time"], str(m["match_id"])))
        record = group[0]
        preferred = [m for m in group if m["_has_utc_time"] == record["_has_utc_time"]
                     and (record["availability_source"] != "observed_completed_result"
                          or m["availability_source"] == "observed_completed_result")]
        record["prediction_time"] = min(m["prediction_time"] for m in preferred)
        record["available_at"] = max(m["available_at"] for m in preferred)
        if len({m["score_margin"] for m in group}) > 1:
            record["score_margin"] = None
        if len(group) > 1:
            identities = {_match_identity(m) for m in group}
            if len({m["_has_utc_time"] for m in group}) > 1:
                reason = "duplicate_source_alias"
            elif len(identities) == 1:
                reason = "duplicate_match_id" if next(iter(identities))[0] == "id" else "duplicate_signature"
            else:
                reason = "duplicate_fixture"
            drop_counts[reason] = drop_counts.get(reason, 0) + len(group)-1
        unique.append(record)
    return unique


def load_all_matches(
    raw_dir: Path,
    max_files: int | None = None,
    *,
    include_para: bool = False,
    deduplicate: bool = True,
    observed_results: dict[str, dict[str, Any] | None] | None = None,
    category_catalog: TournamentCatalog | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str]]:
    """Load all JSON files; return (sorted match list, drop counts, player name map).

    player name map: {player_id: display_name}
    """
    files = sorted(raw_dir.glob("*.json"))
    if max_files is not None:
        files = files[:max_files]

    all_records: list[dict[str, Any]] = []
    drop_counts: dict[str, int] = {}
    name_map: dict[str, str] = {}

    for fp in files:
        records = parse_file(
            fp,
            drop_counts,
            name_map,
            include_para=include_para,
            observed_results=observed_results,
            category_catalog=category_catalog,
        )
        all_records.extend(records)

    if deduplicate:
        all_records = _deduplicate_matches(all_records, drop_counts)

    valid_records = []
    for record in all_records:
        if reason := record.pop("_observation_error", None):
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
            continue
        record["fixture_id"] = hashlib.sha256(json.dumps(record["_fixture_key"]).encode()).hexdigest()
        record.pop("_fixture_key", None)
        record.pop("_has_utc_time", None)
        valid_records.append(record)
    all_records = valid_records

    all_records.sort(key=lambda m: (m["prediction_time"], str(m["match_id"])))

    logger.info(
        "Loaded %d matches from %d files | drops: %s | players with names: %d",
        len(all_records),
        len(files),
        drop_counts,
        len(name_map),
    )
    return all_records, drop_counts, name_map


def build_data_report(
    matches: list[dict[str, Any]],
    drop_counts: dict[str, int],
    *,
    files_scanned: int,
    include_para: bool,
    deduplicate: bool,
) -> dict[str, Any]:
    """Build a JSON-serialisable summary of normalized training records."""
    years = Counter(match["match_time"].year for match in matches)
    disciplines = Counter(match["discipline"] for match in matches)
    tournaments = {
        match.get("tournament_code") or match.get("tournament_name")
        for match in matches
        if match.get("tournament_code") or match.get("tournament_name")
    }
    return {
        "files_scanned": files_scanned,
        "matches_included": len(matches),
        "date_min": min(m["match_time"] for m in matches).isoformat() if matches else None,
        "date_max": max(m["match_time"] for m in matches).isoformat() if matches else None,
        "tournaments": len(tournaments),
        "matches_by_year": dict(sorted(years.items())),
        "matches_by_discipline": dict(sorted(disciplines.items())),
        "label_balance_team1_win": (
            sum(match["label"] for match in matches) / len(matches)
            if matches
            else None
        ),
        "excluded": dict(sorted(drop_counts.items())),
        "policy": {
            "include_para": include_para,
            "deduplicate": deduplicate,
        },
    }
