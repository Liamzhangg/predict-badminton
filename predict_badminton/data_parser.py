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

import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

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
) -> list[dict[str, Any]]:
    """Parse one JSON file; return list of normalised match dicts.

    If name_map is provided it is updated in-place with {player_id: display_name}.
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

        winner = match.get("winner")
        if winner not in (1, 2):
            drop_counts["bad_winner"] = drop_counts.get("bad_winner", 0) + 1
            continue
        if match.get("scoreStatus") == 1 or str(match.get("scoreStatusValue", "")).casefold() == "walkover":
            drop_counts["walkover"] = drop_counts.get("walkover", 0) + 1
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

        records.append(
            {
                "match_id": match.get("id"),
                "match_time": mt,
                "tournament_code": match.get("tournamentCode") or "",
                "tournament_name": t_name,
                "tournament_tier": infer_tournament_tier(t_name),
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
                "winner": int(winner),  # 1 or 2
                "label": int(winner == 1),  # target: 1 if team1 wins
                "_has_utc_time": bool(match.get("matchTimeUtc")),
                "_source_alias": (
                    match.get("tournamentCode") or t_name, match.get("matchTime"),
                    match.get("eventName"), match.get("roundName"), tuple(ids1), tuple(ids2),
                    match.get("matchOrder"), match.get("matchTypeNo"), winner,
                    json.dumps(match.get("score"), sort_keys=True),
                ) if match.get("matchTime") else None,
            }
        )
    return records


def _match_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    match_id = record.get("match_id")
    if match_id is not None and str(match_id).strip():
        return ("id", str(match_id).strip())

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
    """Keep the first deterministic occurrence of each match identity."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    aliases: dict[tuple[Any, ...], int] = {}
    for record in records:
        identity = _match_identity(record)
        if identity in seen:
            reason = "duplicate_match_id" if identity[0] == "id" else "duplicate_signature"
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
            continue
        seen.add(identity)
        alias = record.get("_source_alias")
        if alias in aliases:
            index = aliases[alias]
            # Some BWF copies have new IDs and UTC clocks but identical local results.
            if record["_has_utc_time"] != unique[index]["_has_utc_time"]:
                drop_counts["duplicate_source_alias"] = drop_counts.get("duplicate_source_alias", 0) + 1
                if record["_has_utc_time"]:
                    unique[index] = record
                continue
        if alias is not None:
            aliases[alias] = len(unique)
        unique.append(record)
    return unique


def load_all_matches(
    raw_dir: Path,
    max_files: int | None = None,
    *,
    include_para: bool = False,
    deduplicate: bool = True,
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
        )
        all_records.extend(records)

    if deduplicate:
        all_records = _deduplicate_matches(all_records, drop_counts)

    for record in all_records:
        record.pop("_source_alias", None)
        record.pop("_has_utc_time", None)

    # Stable chronological sort (Python sort is stable)
    all_records.sort(key=lambda x: x["match_time"])

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
        "date_min": matches[0]["match_time"].isoformat() if matches else None,
        "date_max": matches[-1]["match_time"].isoformat() if matches else None,
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
