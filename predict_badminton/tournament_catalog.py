"""Edition-specific official categories; only World Tour tiers override legacy guesses."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from predict_badminton.config import REPO_ROOT

CATALOG_PATH = REPO_ROOT / "data/catalogs/bwf_tournaments.csv"
CATEGORY_POLICY = "official_world_tour_v1"
_TIERS = {
    "hsbc bwf world tour finals": 2,
    "hsbc bwf world tour super 1000": 2,
    "hsbc bwf world tour super 750": 3,
    "hsbc bwf world tour super 500": 4,
    "hsbc bwf world tour super 300": 5,
}
_FIELDS = ["tournament_id", "tournament_code", "name", "start_date", "end_date", "category",
           "status", "source_url", "source_cache_path", "source_sha256"]


def official_tier(category: str) -> int | None:
    return _TIERS.get(" ".join(category.casefold().split()))


class TournamentCatalog:
    """A frozen CSV snapshot shared by training and inference, never name-matched."""

    def __init__(self, text: str) -> None:
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames != _FIELDS:
            raise ValueError("Unsupported tournament catalog columns")
        self.rows = list(reader)
        self._by_id: dict[str, dict[str, str]] = {}
        self._by_code: dict[str, dict[str, str]] = {}
        for row in self.rows:
            if None in row or any(row[field] is None for field in _FIELDS):
                raise ValueError("Incomplete tournament catalog row")
            tid, code = row["tournament_id"].strip(), row["tournament_code"].strip().casefold()
            if not tid or tid in self._by_id or code and code in self._by_code:
                raise ValueError("Tournament catalog IDs/codes must identify unique editions")
            self._by_id[tid] = row
            if code:
                self._by_code[code] = row
        self.text = text
        self.sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> TournamentCatalog:
        return cls(path.read_text(encoding="utf-8"))

    def lookup(self, code: Any = "", tournament_id: Any = "") -> dict[str, str] | None:
        by_code = self._by_code.get(str(code or "").strip().casefold())
        by_id = self._by_id.get(str(tournament_id or "").strip())
        # Some older BWF records put the calendar ID in tournamentCode.
        by_id = by_id or self._by_id.get(str(code or "").strip())
        if by_code and by_id and by_code["tournament_id"] != by_id["tournament_id"]:
            raise ValueError("Tournament ID and code refer to different editions")
        return by_code or by_id

    def tier(self, code: Any = "", tournament_id: Any = "") -> int | None:
        row = self.lookup(code, tournament_id)
        return official_tier(row["category"]) if row else None

    def get_state(self) -> dict[str, Any]:
        return {"version": 1, "csv": self.text}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TournamentCatalog:
        if not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("csv"), str):
            raise ValueError("Unsupported tournament catalog snapshot")
        return cls(state["csv"])


def build_catalog(cache_dir: Path, output: Path = CATALOG_PATH) -> TournamentCatalog:
    """Build from saved official calendar JSON only; conflicting edition facts fail closed."""
    candidates: dict[str, list[tuple[Path, dict]]] = {}
    paths = sorted(cache_dir.glob("**/api-vue-tournaments-search-*.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results", {}).get("data", []):
            if row.get("id"):
                candidates.setdefault(str(row["id"]), []).append((path, row))
    if not candidates:
        raise ValueError("No saved official BWF calendar entries found")
    rows = []
    for tid, versions in candidates.items():
        categories = {r["category"].strip() for _, r in versions if r.get("category")}
        codes = {str(r["code"]).strip().casefold() for _, r in versions if r.get("code")}
        if len(categories) > 1 or len(codes) > 1:
            raise ValueError(f"Conflicting official category/code for tournament {tid}")
        # mtime selects a saved response, not a historical publication timestamp.
        path, row = max(versions, key=lambda item: (bool(item[1].get("category")), item[0].stat().st_mtime_ns, str(item[0])))
        rows.append({"tournament_id": tid, "tournament_code": str(row.get("code") or ""),
                     "name": row["name"], "start_date": row["start_date"], "end_date": row["end_date"],
                     "category": row.get("category") or "", "status": (row.get("status") or {}).get("code", ""),
                     "source_url": row.get("url") or "", "source_cache_path": str(path.resolve().relative_to(REPO_ROOT)),
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    rows.sort(key=lambda row: (row["start_date"], row["tournament_id"]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    catalog = TournamentCatalog(buffer.getvalue())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(catalog.text, encoding="utf-8")
    return catalog
