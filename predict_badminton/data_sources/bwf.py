"""BWF data fetching, manual import, and normalization utilities.

Supports standard HTTP and Chrome-compatible request fingerprints for public
sources. Responses that still contain challenges are recorded without caching.
"""
from __future__ import annotations

import csv
import atexit
import hashlib
import html.parser
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from predict_badminton.config import RAW_JSON_DIR, REPO_ROOT
from predict_badminton.data_parser import load_all_matches, parse_file
from predict_badminton.data_sources.bwf_schema import flatten_matches, indexed_list

logger = logging.getLogger(__name__)

SOURCE_CACHE_DIR = REPO_ROOT / "data" / "source_cache" / "bwf"
MANIFEST_PATH = REPO_ROOT / "data" / "manifests" / "bwf_fetch_manifest.csv"

MANIFEST_FIELDS = [
    "tournament_id",
    "slug",
    "name",
    "source_url",
    "status",
    "start_date",
    "end_date",
    "cache_path",
    "raw_path",
    "message",
    "updated_at",
]

_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)

_BLOCKED_MARKERS = (
    "checking your browser",
    "attention required",
    "captcha",
    "cf_chl_opt",
    "cf-chl-",
    "just a moment",
    "turnstile",
    "access denied",
    "rate limited",
)

_PARA_MARKERS = (
    "para-badminton",
    "para badminton",
    "paralympic",
    "classification",
    "wh1",
    "wh2",
    "sl3",
    "sl4",
    "su5",
    "sh6",
)


@dataclass
class FetchResult:
    url: str
    status: str
    status_code: int | None = None
    cache_path: Path | None = None
    message: str = ""


@dataclass
class ImportResult:
    source_path: Path
    raw_path: Path
    matches: int
    slug: str
    tournament_name: str


def ensure_data_dirs() -> None:
    SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_match_time(raw: Any) -> datetime | None:
    if raw in (None, ""):
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(str(raw).strip(), fmt)
        except ValueError:
            continue
    return None


def is_para_tournament(value: str) -> bool:
    text = value.lower()
    return any(marker in text for marker in _PARA_MARKERS)


def slugify(value: str, fallback: str = "tournament") -> str:
    text = urllib.parse.unquote(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or fallback


def latest_non_para_match_date(raw_dir: Path = RAW_JSON_DIR) -> date | None:
    latest: datetime | None = None
    for fp in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for match in iter_raw_matches(data):
            tournament_name = str(match.get("tournamentName") or fp.stem)
            if is_para_tournament(tournament_name) or is_para_tournament(fp.stem):
                continue
            mt = parse_match_time(match.get("matchTimeUtc") or match.get("matchTime"))
            if mt and (latest is None or mt > latest):
                latest = mt
    return latest.date() if latest else None


def default_start_date(raw_dir: Path = RAW_JSON_DIR) -> date:
    earliest: date | None = None
    for path in sorted(raw_dir.glob("*.json")):
        for match in iter_raw_matches(json.loads(path.read_text(encoding="utf-8"))):
            if is_para_tournament(str(match.get("tournamentName") or path.stem)):
                continue
            stamp = parse_match_time(match.get("matchTimeUtc") or match.get("matchTime"))
            if stamp and (earliest is None or stamp.date() < earliest):
                earliest = stamp.date()
    return date(earliest.year, 1, 1) if earliest else date(2018, 1, 1)


def iter_raw_matches(data: Any) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, list):
                matches.extend(m for m in item if isinstance(m, dict))
            elif looks_like_match(item):
                matches.append(item)
    elif isinstance(data, dict):
        for value in data.values():
            matches.extend(iter_raw_matches(value))
    return matches


def looks_like_match(value: Any) -> bool:
    return isinstance(value, dict) and {"team1", "team2"}.issubset(value)


def count_round_matches(rounds: list[list[dict[str, Any]]]) -> int:
    return sum(len(round_item) for round_item in rounds)


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_manifest(rows: list[dict[str, Any]], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for row in rows:
        normalized.append({field: str(row.get(field, "")) for field in MANIFEST_FIELDS})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    temporary.replace(path)


def upsert_manifest_rows(new_rows: list[dict[str, Any]], path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    existing = load_manifest(path)
    by_key = {
        row.get("source_url") or row.get("slug") or row.get("tournament_id"): row
        for row in existing
    }
    for row in new_rows:
        key = str(row.get("source_url") or row.get("slug") or row.get("tournament_id"))
        merged = dict(by_key.get(key, {}))
        merged.update(row)
        merged["updated_at"] = utc_now_iso()
        by_key[key] = merged
    rows = list(by_key.values())
    write_manifest(rows, path)
    return rows


class ScriptCollector(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_script = False
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._in_script = tag.lower() == "script"

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.scripts.append(data)


@lru_cache(maxsize=1)
def _chrome_session():
    from curl_cffi.requests import Session

    session = Session(impersonate="chrome")
    atexit.register(session.close)
    return session


def fetch_url(
    url: str,
    cache_dir: Path = SOURCE_CACHE_DIR,
    timeout: float = 30.0,
    transport: str = "standard",
) -> FetchResult:
    if transport == "chrome":
        try:
            from curl_cffi import requests
        except ImportError as exc:
            raise RuntimeError("Chrome transport requires: uv sync --extra fetch") from exc
        try:
            response = _chrome_session().get(url, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            return FetchResult(url=url, status="failed", message=str(exc))
        status_code = response.status_code
        body = response.content
        headers = {k.lower(): v for k, v in response.headers.items()}
    elif transport == "standard":
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "predict-badminton/0.1 (+https://github.com/Liamzhangg/predict-badminton)",
                "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status_code = int(response.status)
                body = response.read()
                headers = {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:
            body = exc.read()
            status_code = int(exc.code)
            headers = {k.lower(): v for k, v in exc.headers.items()}
        except urllib.error.URLError as exc:
            return FetchResult(url=url, status="failed", message=str(exc.reason))
        except (TimeoutError, OSError) as exc:
            return FetchResult(url=url, status="failed", message=str(exc))
    else:
        raise ValueError(f"Unknown HTTP transport: {transport}")

    if is_blocked_response(status_code, headers, body):
        return FetchResult(url=url, status="blocked", status_code=status_code, message="blocked or challenged")
    if not 200 <= status_code < 300:
        return FetchResult(url=url, status="failed", status_code=status_code, message=f"HTTP {status_code}")

    suffix = ".json" if "json" in headers.get("content-type", "") else ".html"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"{slugify(urllib.parse.urlparse(url).path, 'source')}-{digest}{suffix}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return FetchResult(url=url, status="cached", status_code=status_code, cache_path=path)


def is_blocked_response(status_code: int | None, headers: dict[str, str], body: bytes) -> bool:
    if status_code in {401, 403, 429}:
        return True
    if headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    body_text = body[:20000].decode("utf-8", errors="ignore").lower()
    if "json" in headers.get("content-type", "").lower():
        return False
    return any(marker in body_text for marker in _BLOCKED_MARKERS)


def fetch_manifest_rows(
    urls: list[str],
    manifest_path: Path = MANIFEST_PATH,
    delay_seconds: float = 1.0,
    force: bool = False,
    transport: str = "standard",
) -> list[FetchResult]:
    rows = load_manifest(manifest_path)
    manifest_urls = [
        row.get("source_url", "")
        for row in rows
        if row.get("source_url") and (force or row.get("status") in {"", "discovered", "failed"})
    ]
    target_urls = urls or manifest_urls
    results: list[FetchResult] = []
    updates: list[dict[str, Any]] = []
    for url in target_urls:
        result = fetch_url(url, transport=transport)
        results.append(result)
        updates.append(
            {
                "source_url": url,
                "status": result.status,
                "cache_path": result.cache_path or "",
                "message": result.message,
            }
        )
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    if updates:
        upsert_manifest_rows(updates, manifest_path)
    return results


def import_source_file(
    source_path: Path,
    output_dir: Path = RAW_JSON_DIR,
    slug: str | None = None,
    force: bool = False,
) -> ImportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds, tournament_name = normalize_source_file(source_path)
    if not rounds or count_round_matches(rounds) == 0:
        raise ValueError(f"No matches found in {source_path}")

    resolved_slug = slugify(slug or tournament_name or source_path.stem)
    raw_path = output_dir / f"{resolved_slug}.json"
    if raw_path.exists() and not force:
        raise FileExistsError(f"{raw_path} already exists; pass --force to overwrite")

    temporary = raw_path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as fh:
        json.dump(rounds, fh, ensure_ascii=False, indent=2)
    try:
        drops: dict[str, int] = {}
        parsed = parse_file(temporary, drops)
        if drops.get("file_error"):
            raise ValueError(f"Invalid normalized source file: {drops}")
        if drops:
            logger.warning("Imported source has %d training matches; exclusions: %s", len(parsed), drops)
        temporary.replace(raw_path)
    finally:
        temporary.unlink(missing_ok=True)

    result = ImportResult(
        source_path=source_path,
        raw_path=raw_path,
        matches=count_round_matches(rounds),
        slug=resolved_slug,
        tournament_name=tournament_name,
    )
    upsert_manifest_rows(
        [
            {
                "slug": resolved_slug,
                "name": tournament_name,
                "status": "imported",
                "cache_path": source_path,
                "raw_path": raw_path,
                "message": f"imported {result.matches} matches",
            }
        ]
    )
    return result


def normalize_source_file(source_path: Path) -> tuple[list[list[dict[str, Any]]], str]:
    suffix = source_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        return normalize_json_data(data, source_path.stem)
    if suffix == ".csv":
        return normalize_csv_data(source_path)
    if suffix in {".html", ".htm"}:
        return normalize_html_data(source_path)
    raise ValueError(f"Unsupported source file type: {source_path.suffix}")


def normalize_json_data(data: Any, fallback_name: str) -> tuple[list[list[dict[str, Any]]], str]:
    found = find_rounds_or_matches(data)
    if not found:
        raise ValueError("Could not find compatible team1/team2 match objects in JSON")
    rounds = group_matches_by_round(flatten_matches(found))
    tournament_name = first_tournament_name(rounds) or fallback_name
    return rounds, tournament_name


def find_rounds_or_matches(value: Any) -> Any | None:
    if looks_like_match(value):
        return [value]
    value = indexed_list(value)
    if isinstance(value, dict):
        # A knockout response also has UI wrappers under results; matches is canonical.
        if "matches" in value:
            return find_rounds_or_matches(value["matches"])
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return []
    return [match for child in children for match in find_rounds_or_matches(child)]


def group_matches_by_round(matches: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for match in matches:
        round_name = str(match.get("roundName") or match.get("round_name") or "Unknown Round")
        grouped.setdefault(round_name, []).append(match)
    return list(grouped.values())


def normalize_csv_data(source_path: Path) -> tuple[list[list[dict[str, Any]]], str]:
    with open(source_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    matches = [csv_row_to_match(row) for row in rows]
    matches = [match for match in matches if match is not None]
    if not matches:
        raise ValueError("No compatible match rows found in CSV")
    return group_matches_by_round(matches), first_match_tournament_name(matches) or source_path.stem


def csv_row_to_match(row: dict[str, str]) -> dict[str, Any] | None:
    lowered = {k.lower().strip(): v for k, v in row.items() if k}

    def pick(*names: str) -> str:
        for name in names:
            value = lowered.get(name.lower())
            if value not in (None, ""):
                return value
        return ""

    ids1 = split_ids(pick("team1_player_ids", "team1_ids", "player1_ids", "team1"))
    ids2 = split_ids(pick("team2_player_ids", "team2_ids", "player2_ids", "team2"))
    if not ids1 or not ids2:
        return None

    winner_raw = pick("winner", "winning_team", "winner_team")
    winner = parse_winner(winner_raw)
    return {
        "id": pick("match_id", "id"),
        "tournamentCode": pick("tournament_code", "tournamentcode"),
        "tournamentName": pick("tournament_name", "tournamentname"),
        "isTeamMatch": parse_bool(pick("is_team_match", "isteammatch")),
        "winner": winner,
        "team1": {
            "countryCode": pick("team1_country", "team1country"),
            "players": [{"id": pid, "nameDisplay": ""} for pid in ids1],
        },
        "team2": {
            "countryCode": pick("team2_country", "team2country"),
            "players": [{"id": pid, "nameDisplay": ""} for pid in ids2],
        },
        "team1seed": pick("team1_seed", "team1seed") or None,
        "team2seed": pick("team2_seed", "team2seed") or None,
        "matchTime": pick("match_time", "matchtime", "datetime"),
        "matchTimeUtc": pick("match_time_utc", "matchtimeutc"),
        "eventName": pick("event_name", "eventname", "discipline"),
        "roundName": pick("round_name", "roundname"),
        "locationName": pick("location_name", "locationname"),
        "score": [],
    }


def split_ids(raw: str) -> list[str]:
    return [part.strip() for part in re.split(r"[|,;/]", str(raw)) if part.strip()]


def parse_winner(raw: str) -> int | None:
    value = str(raw).strip().lower()
    if value in {"1", "team1", "home"}:
        return 1
    if value in {"2", "team2", "away"}:
        return 2
    return None


def parse_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def normalize_html_data(source_path: Path) -> tuple[list[list[dict[str, Any]]], str]:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    candidates = extract_json_candidates_from_html(text)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        try:
            return normalize_json_data(data, source_path.stem)
        except ValueError:
            continue
    raise ValueError("No embedded JSON match data found in HTML")


def extract_json_candidates_from_html(text: str) -> list[str]:
    parser = ScriptCollector()
    parser.feed(text)
    # Only parse actual JSON script contents, not every brace in JavaScript/HTML.
    return [script.strip() for script in parser.scripts if script.lstrip().startswith(("[", "{"))]


def first_tournament_name(rounds: list[list[dict[str, Any]]]) -> str:
    for match in iter_raw_matches(rounds):
        name = match.get("tournamentName") or match.get("tournament_name")
        if name:
            return str(name)
    return ""


def first_match_tournament_name(matches: list[dict[str, Any]]) -> str:
    for match in matches:
        name = match.get("tournamentName") or match.get("tournament_name")
        if name:
            return str(name)
    return ""


def validate_raw_dir(raw_dir: Path = RAW_JSON_DIR, max_files: int | None = None) -> dict[str, Any]:
    matches, drop_counts, player_names = load_all_matches(raw_dir, max_files=max_files)
    return {
        "raw_dir": str(raw_dir),
        "matches": len(matches),
        "drops": drop_counts,
        "players_with_names": len(player_names),
    }
