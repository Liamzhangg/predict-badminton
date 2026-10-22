"""Normalize the match arrays shared by BWF downloads and manual imports."""
from __future__ import annotations

from typing import Any

from predict_badminton.data_parser import normalize_discipline


def indexed_list(value: Any) -> Any:
    # PHP may encode sparse arrays as numeric-keyed objects.
    if isinstance(value, dict) and all(str(key).isdigit() for key in value):
        return [value[key] for key in sorted(value, key=lambda key: int(key))]
    return value


def flatten_matches(payload: Any) -> list[dict[str, Any]]:
    payload = indexed_list(payload)
    if not isinstance(payload, list):
        raise ValueError("Expected a list of BWF matches")
    flat: list[dict[str, Any]] = []
    for original in payload:
        if not isinstance(original, dict) or not {"team1", "team2"}.issubset(original):
            raise ValueError("Unexpected BWF match structure")
        match = dict(original)
        for key in ("team1", "team2"):
            team = match.get(key)
            if team is not None and not isinstance(team, dict):
                raise ValueError(f"Expected a BWF team object for {key}")
            if isinstance(team, dict) and "players" in team:
                players = indexed_list(team["players"] or [])
                if not isinstance(players, list) or any(not isinstance(p, dict) for p in players):
                    raise ValueError("Expected a list of BWF player objects")
                match[key] = {**team, "players": players}
        if "score" in match:
            match["score"] = indexed_list(match["score"])
        children = indexed_list(match.get("matches") or [])
        if "matches" in match:
            match["matches"] = children
        if children:
            for child in flatten_matches(children):
                for key in ("tournamentCode", "tournamentName", "roundName"):
                    if not child.get(key):
                        child[key] = match.get(key)
                child["teamTieId"] = match.get("id")
                child["isTeamMatch"] = True
                flat.append(child)
        else:
            discipline = normalize_discipline(str(match.get("matchTypeValue") or ""))
            if discipline != "OTHER" and normalize_discipline(str(match.get("eventName") or "")) != discipline:
                match["sourceEventName"] = match.get("eventName")
                match["eventName"] = discipline
            flat.append(match)
    return flat
