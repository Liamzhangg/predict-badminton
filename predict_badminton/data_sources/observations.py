"""Fresh result receipts, explicit invalidations, and availability snapshots."""
from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sqlite3
from typing import Any

from predict_badminton.config import REPO_ROOT
from predict_badminton.data_parser import is_para_match, normalize_discipline
from predict_badminton.match_rules import standard_match_exclusion, unplayed_status

OBSERVATIONS_PATH = REPO_ROOT / "data/manifests/bwf_result_observations.sqlite3"

_RESULT_FIELDS = (
    "tournamentCode", "tournamentName", "eventName", "roundName", "locationName",
    "matchTimeUtc", "matchTime", "timePrecision", "duration", "winner", "scoreStatus",
    "scoreStatusValue", "team1seed", "team2seed", "isTeamMatch", "teamTieId",
    "matchOrder", "matchTypeNo",
)


def _utc_stamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Observation times must have a UTC offset")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _invalidation_reason(match: dict[str, Any]) -> str | None:
    if status := unplayed_status(match):
        return status
    if not isinstance(match.get("winner"), bool) and match.get("winner") in (0, "0"):
        return "winner_reset"
    return None


def _snapshot(match: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(match.get("winner"), bool) or match.get("winner") not in (1, 2):
        return None
    if match.get("id") is None or not str(match["id"]).strip() or is_para_match(match):
        return None
    if _invalidation_reason(match):
        return None
    result = {field: match.get(field) for field in _RESULT_FIELDS}
    result["id"] = str(match["id"]).strip()
    teams = []
    for side in ("team1", "team2"):
        source = match.get(side) or {}
        ids = sorted(str(p["id"]) for p in source.get("players", []) if p.get("id"))
        teams.append(ids)
        result[side] = {"countryCode": source.get("countryCode"), "players": [{"id": pid} for pid in ids]}
    if standard_match_exclusion({"discipline": normalize_discipline(str(result["eventName"] or "")),
                                 "event_name": result["eventName"], "tournament_name": result["tournamentName"],
                                 "team1_ids": teams[0], "team2_ids": teams[1]}):
        return None
    score = match.get("score") or []
    if not isinstance(score, list) or any(not isinstance(game, dict) for game in score):
        return None
    result["score"] = [{"home": game.get("home"), "away": game.get("away")} for game in score]
    return result


def record_results(
    matches: list[dict[str, Any]], *, source_url: str, response_sha256: str,
    observed_at: datetime, path: Path = OBSERVATIONS_PATH,
) -> tuple[int, int]:
    """Record completed versions and explicit invalidations of previously tracked IDs."""
    stamp = _utc_stamp(observed_at)
    versions = {}
    invalidations = {}
    for match in matches:
        version = result_version(match)
        if version is not None:
            fingerprint, payload = version
            versions[(str(match["id"]).strip(), fingerprint)] = payload
        elif match.get("id") is not None and str(match["id"]).strip() and (reason := _invalidation_reason(match)):
            invalidations[str(match["id"]).strip()] = reason
    if not versions and (not invalidations or not path.exists()):
        return 0, 0
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = None
    try:
        connection = sqlite3.connect(path)
        with connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("""CREATE TABLE IF NOT EXISTS responses (
                response_id INTEGER PRIMARY KEY, observed_at TEXT NOT NULL,
                source_url TEXT NOT NULL, response_sha256 TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS result_versions (
                match_id TEXT NOT NULL, result_sha256 TEXT NOT NULL,
                first_seen_at TEXT NOT NULL, result_json TEXT NOT NULL,
                PRIMARY KEY (match_id, result_sha256))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS response_results (
                response_id INTEGER NOT NULL REFERENCES responses(response_id),
                match_id TEXT NOT NULL, result_sha256 TEXT NOT NULL,
                PRIMARY KEY (response_id, match_id, result_sha256),
                FOREIGN KEY (match_id, result_sha256) REFERENCES result_versions(match_id, result_sha256))""")
            batch_ids = {mid for mid, _ in versions}
            invalidations = {mid: reason for mid, reason in invalidations.items()
                             if mid in batch_ids or connection.execute(
                                 "SELECT 1 FROM result_versions WHERE match_id=? LIMIT 1", (mid,)).fetchone()}
            if not versions and not invalidations:
                return 0, 0
            response_id = connection.execute(
                "INSERT INTO responses (observed_at,source_url,response_sha256) VALUES (?,?,?)",
                (stamp, source_url, response_sha256),
            ).lastrowid
            added = 0
            for (match_id, fingerprint), payload in versions.items():
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO result_versions VALUES (?,?,?,?)", (match_id, fingerprint, stamp, payload),
                ).rowcount
                added += inserted
                if not inserted:
                    # Concurrent requests may finish recording in a different order.
                    connection.execute("UPDATE result_versions SET first_seen_at=MIN(first_seen_at,?) WHERE match_id=? AND result_sha256=?",
                                       (stamp, match_id, fingerprint))
                connection.execute("INSERT INTO response_results VALUES (?,?,?)", (response_id, match_id, fingerprint))
            if invalidations:
                connection.execute("""CREATE TABLE IF NOT EXISTS result_invalidations (
                    response_id INTEGER NOT NULL REFERENCES responses(response_id),
                    match_id TEXT NOT NULL, reason TEXT NOT NULL,
                    PRIMARY KEY (response_id, match_id))""")
                connection.executemany("INSERT INTO result_invalidations VALUES (?,?,?)",
                                       [(response_id, mid, reason) for mid, reason in invalidations.items()])
    except sqlite3.Error as exc:
        raise OSError(f"Could not record result observations in {path}: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    return len(versions), added


def result_version(match: dict[str, Any]) -> tuple[str, str] | None:
    """Return the exact normalized version hash and its canonical JSON snapshot."""
    snapshot = _snapshot(match)
    if snapshot is None:
        return None
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest(), payload


def observations_before(cutoff: datetime, path: Path = OBSERVATIONS_PATH) -> dict[str, dict[str, Any] | None]:
    """Freeze known versions, including correction/reversion boundaries, strictly before cutoff.

    None means a tracked ID was not yet observed, conflicting, or explicitly invalidated.
    Missing entries and absent/null winner fields do not establish a source retraction.
    """
    stamp = _utc_stamp(cutoff)
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        known = {mid: None for (mid,) in connection.execute("SELECT DISTINCT match_id FROM result_versions")}
        query = """SELECT rr.match_id,r.observed_at,rr.result_sha256,r.source_url,r.response_sha256,r.response_id
            FROM response_results rr JOIN responses r USING(response_id)
            WHERE r.observed_at < ?"""
        parameters = [stamp]
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='result_invalidations' AND type='table'").fetchone():
            query += """ UNION ALL SELECT ri.match_id,r.observed_at,NULL,r.source_url,r.response_sha256,r.response_id
                FROM result_invalidations ri JOIN responses r USING(response_id) WHERE r.observed_at < ?"""
            parameters.append(stamp)
        rows = connection.execute(query + " ORDER BY 1,2,6", parameters).fetchall()
    finally:
        connection.close()
    for (mid, observed), receipt_rows in groupby(rows, key=lambda row: row[:2]):
        receipts = list(receipt_rows)
        fingerprints = {row[2] for row in receipts}
        if len(fingerprints) != 1 or None in fingerprints:
            known[mid] = None
            continue
        fingerprint = receipts[0][2]
        previous = known[mid]
        if previous is None or previous["result_sha256"] != fingerprint:
            known[mid] = {"result_sha256": fingerprint,
                          "available_at": datetime.fromisoformat(observed).astimezone(UTC).replace(tzinfo=None),
                          "source_url": receipts[0][3], "response_sha256": receipts[0][4]}
    return known


def observation_status(path: Path = OBSERVATIONS_PATH) -> dict[str, Any]:
    result = {"path": str(path), "responses": 0, "matches": 0, "result_versions": 0,
              "result_invalidations": 0,
              "first_observed_at": None, "last_observed_at": None}
    if not path.exists():
        return result
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        result["responses"], result["first_observed_at"], result["last_observed_at"] = connection.execute(
            "SELECT COUNT(*),MIN(observed_at),MAX(observed_at) FROM responses").fetchone()
        result["matches"], result["result_versions"] = connection.execute(
            "SELECT COUNT(DISTINCT match_id),COUNT(*) FROM result_versions").fetchone()
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='result_invalidations' AND type='table'").fetchone():
            result["result_invalidations"] = connection.execute("SELECT COUNT(*) FROM result_invalidations").fetchone()[0]
    finally:
        connection.close()
    return result
