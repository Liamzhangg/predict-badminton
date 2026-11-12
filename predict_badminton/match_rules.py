"""Standard indoor match eligibility and conservative score-margin parsing."""
from __future__ import annotations

import math
import re
from typing import Any

from predict_badminton.config import DISCIPLINES, DOUBLES_DISCIPLINES

_AIR_EVENT = re.compile(r"\b(?:air\s*badminton|anoc\s+wbg|world\s+beach\s+games)\b", re.IGNORECASE)


def unplayed_status(match: dict[str, Any]) -> str | None:
    """Recognized statuses that must not count as a played result."""
    status = str(match.get("scoreStatusValue") or "").strip().casefold()
    if match.get("scoreStatus") in (1, "1") or status == "walkover":
        return "walkover"
    return status if status in {"withdrawn", "withdrawal", "cancelled", "canceled"} else None


def standard_match_exclusion(match: dict[str, Any]) -> str | None:
    """Return why a normalized match/fixture is outside the production scope."""
    discipline = match.get("discipline")
    if discipline not in DISCIPLINES:
        return "unsupported_discipline"
    if _AIR_EVENT.search(str(match.get("tournament_name") or "") + " " + str(match.get("event_name") or "")):
        return "airbadminton"
    count = 2 if discipline in DOUBLES_DISCIPLINES else 1
    teams = [match.get(side) or [] for side in ("team1_ids", "team2_ids")]
    if any(len(ids) != count or len(set(ids)) != count for ids in teams) or set(teams[0]).intersection(teams[1]):
        return "invalid_team_format"
    return None


def point_margin(match: dict[str, Any]) -> float | None:
    """Winner-oriented margin for a complete, normal 21-point best-of-three match."""
    status = str(match.get("scoreStatusValue") or "").casefold()
    if status not in ("", "normal") or match.get("scoreStatus") not in (None, 0, "0"):
        return None
    scores = match.get("score")
    if not isinstance(scores, list) or not scores:
        return None
    games: list[tuple[int, int]] = []
    trailing_empty = False
    for game in scores:
        if not isinstance(game, dict):
            return None
        pair = (game.get("home"), game.get("away"))
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or v != int(v) or v < 0 for v in pair):
            return None
        a, b = map(int, pair)
        if a == b == 0:
            trailing_empty = True
            continue
        if trailing_empty:
            return None
        high, low = max(a, b), min(a, b)
        if not ((high == 21 and low <= 19) or (22 <= high <= 29 and high-low == 2)
                or (high == 30 and low in (28, 29))):
            return None
        games.append((a, b))
    if len(games) not in (2, 3):
        return None
    winners = [1 if a > b else 2 for a, b in games]
    winner = match.get("winner")
    if winner not in (1, 2) or winners.count(winner) != 2:
        return None
    if len(games) == 3 and winners[0] == winners[1]:
        return None
    home, away = sum(a for a, _ in games), sum(b for _, b in games)
    won, lost = (home, away) if winner == 1 else (away, home)
    return (won-lost) / (won+lost)


def score_multiplier(margin: float | None, weight: float, reference: float) -> float:
    """Bound score evidence; unavailable/unsafe margins retain the baseline K."""
    if (margin is None or isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not math.isfinite(margin) or not -1 <= margin <= 1):
        return 1.0
    return min(1+weight, max(1-weight, 1 + weight*(margin-reference)/reference))
