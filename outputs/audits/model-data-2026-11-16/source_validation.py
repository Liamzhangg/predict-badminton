"""Supplement the read-only audit with score and recent-event checks."""
from collections import Counter, defaultdict
from datetime import UTC, datetime
import json
import math

from audit import OUT, RAW, REPO_ROOT, save
from predict_badminton.config import forecast_bundle_directory
from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import observations_before
from predict_badminton.match_rules import standard_match_exclusion
from predict_badminton.tournament import load_bundle


def main():
    directory = forecast_bundle_directory()
    _, _, _, history = load_bundle(directory)
    metadata = json.loads((directory / "metrics_report.json").read_text())
    cutoff = datetime.fromisoformat(metadata["production_refit"]["as_of"])
    matches, _, _ = load_all_matches(
        RAW, observed_results=observations_before(cutoff.replace(tzinfo=UTC)),
        category_catalog=history.category_catalog,
    )
    matches = [m for m in matches if not standard_match_exclusion(m)
               and m.get("match_status") in (None, "", "F")
               and m["available_at"] < cutoff and m["match_time"] < cutoff]
    ids = {str(mid) for m in matches for mid in m.get("source_match_ids", [m["match_id"]])}
    canonical = {str(mid): m for m in matches for mid in m.get("source_match_ids", [m["match_id"]])}
    counts, examples = Counter(), defaultdict(list)
    for path in sorted(RAW.glob("*.json")):
        for group in json.loads(path.read_text()):
            if not isinstance(group, list):
                continue
            for match in group:
                if not isinstance(match, dict) or str(match.get("id")) not in ids:
                    continue
                status = str(match.get("scoreStatusValue") or "").casefold()
                if status not in ("", "normal") or match.get("scoreStatus") not in (None, 0, "0"):
                    counts["non_normal_score_status_source_rows"] += 1
                    continue
                games = []
                malformed = False
                for game in match.get("score") or []:
                    pair = (game.get("home"), game.get("away")) if isinstance(game, dict) else (None, None)
                    if any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) or v < 0 or v != int(v) for v in pair):
                        malformed = True
                        break
                    if pair != (0, 0):
                        games.append(pair)
                def valid(pair):
                    high, low = max(pair), min(pair)
                    return ((high == 21 and low <= 19) or (22 <= high <= 29 and high-low == 2)
                            or (high == 30 and low in (28, 29)))
                if malformed or len(games) not in (2, 3) or not all(valid(pair) for pair in games):
                    counts["normal_status_missing_or_nonstandard_21point_sequence"] += 1
                    continue
                winners = [1 if a > b else 2 for a, b in games]
                if len(games) == 3 and winners[0] == winners[1]:
                    reason = "normal_status_extra_game_after_2_0"
                elif max(winners.count(1), winners.count(2)) != 2:
                    reason = "normal_status_incomplete_best_of_three"
                elif winners.count(match["winner"]) != 2:
                    reason = "normal_status_valid_sequence_winner_conflict"
                else:
                    counts["normal_status_valid_sequence_agrees_with_winner"] += 1
                    continue
                counts[reason] += 1
                if len(examples[reason]) < 10:
                    selected = canonical[str(match["id"])]
                    examples[reason].append({"path": str(path.relative_to(REPO_ROOT)), "id": match["id"],
                                             "score": games, "winner": match["winner"],
                                             "canonical_match_id": selected["match_id"],
                                             "canonical_winner": selected["winner"],
                                             "canonical_score_margin": selected["score_margin"]})
    save("refined-score-checks.json", {
        "counts": counts, "examples": examples,
        "note": "Source-copy counts, not unique canonical fixtures. Nonstandard sequences and retirements do not establish wrong winner labels.",
    })
    recent = {}
    for name in ("kumamoto-masters-japan-2026", "victor-kaohsiung-masters-2026"):
        raw = [m for group in json.loads((RAW / f"{name}.json").read_text()) if isinstance(group, list)
               for m in group if isinstance(m, dict)]
        raw_ids = {str(m["id"]) for m in raw}
        eligible = [m for m in matches if str(m["match_id"]) in raw_ids]
        finals = [{"match_id": m["match_id"], "discipline": m["discipline"], "available_at": m["available_at"]}
                  for m in eligible if m["round_name"] == "Final"]
        recent[name] = {"raw_rows": len(raw), "eligible_training_rows": len(eligible),
                        "raw_completed_non_walkover_winner_ids_missing_from_cohort": [str(m["id"]) for m in raw
                            if m.get("matchStatus") == "F" and m.get("winner") in (1, 2)
                            and m.get("scoreStatus") not in (1, "1") and str(m["id"]) not in ids],
                        "finals": finals}
    forecast = json.loads((REPO_ROOT / "outputs/rounds/hong-kong-open-2026-20261116/forecast.json").read_text())
    targets = {str(m["match_id"]) for m in forecast["fixtures"]}
    save("recent-events-and-targets.json", {
        "production_refit": metadata["production_refit"], "recent_events": recent,
        "hong_kong_target_count": len(targets), "hong_kong_targets_already_in_training": sorted(targets & ids),
        "note": "Local source/cohort completeness only; not an independent new crawl of official results.",
    })
    print(json.dumps({"score_counts": counts, "recent_events": recent,
                      "hong_kong_target_training_overlap": sorted(targets & ids)}, indent=2, default=str))


if __name__ == "__main__":
    main()
