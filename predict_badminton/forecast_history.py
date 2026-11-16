"""Read-only, point-in-time rating refresh for new forecasts."""
from datetime import UTC, datetime
import hashlib
import logging
from pathlib import Path
import pickle
import sqlite3

from predict_badminton.config import RAW_JSON_DIR
from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import OBSERVATIONS_PATH, observations_before
from predict_badminton.features import EloEngine, FeatureBuilder
from predict_badminton.match_rules import standard_match_exclusion
from predict_badminton.temporal import OBSERVED_AVAILABILITY_POLICY
from predict_badminton.trueskill import TrueSkillEngine

logger = logging.getLogger(__name__)


def history_sha256(history: FeatureBuilder) -> str:
    return hashlib.sha256(pickle.dumps(history.get_state())).hexdigest()


def refresh_history(
    history: FeatureBuilder, *, as_of: datetime | None = None,
    raw_dir: Path = RAW_JSON_DIR, observations_db: Path = OBSERVATIONS_PATH,
    target_match_ids: set[str] | frozenset[str] = frozenset(),
) -> FeatureBuilder:
    """Rebuild matching ratings using results known strictly before one UTC cutoff.

    Never fetch, retrain, mutate the supplied state, or write source/model files.
    Rebuilding also removes retracted results and replaces corrected versions.
    """
    cutoff = as_of or datetime.now(UTC)
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("Forecast cutoff must have a UTC offset")
    cutoff = cutoff.astimezone(UTC).replace(tzinfo=None)
    if history.model_policy.get("result_availability") != OBSERVED_AVAILABILITY_POLICY:
        raise ValueError("Live refresh requires a matching observed-v2 model bundle; use frozen history for older bundles")
    if not observations_db.is_file():
        raise ValueError("Result observations database is missing; fetch completed results before forecasting")
    if not raw_dir.is_dir() or not any(raw_dir.glob("*.json")):
        raise ValueError("No raw match files are available for a history refresh")
    try:
        receipts = observations_before(cutoff.replace(tzinfo=UTC), observations_db)
    except sqlite3.Error as exc:
        raise ValueError(f"Cannot read result observations: {exc}") from exc
    matches, drops, names = load_all_matches(raw_dir, observed_results=receipts,
                                           category_catalog=history.category_catalog)
    if drops.get("file_error"):
        raise ValueError("Cannot refresh history from unreadable raw match files")
    eligible = [m for m in matches if not standard_match_exclusion(m)
                and m.get("match_status") in (None, "", "F")
                and m["available_at"] < cutoff and m["match_time"] < cutoff]
    if any(target_match_ids.intersection(m.get("source_match_ids", [str(m["match_id"])])) for m in eligible):
        raise ValueError("A completed target match is already known; it cannot be forecast as upcoming")
    fresh = FeatureBuilder(player_names=names, result_availability_policy=OBSERVED_AVAILABILITY_POLICY,
                           category_catalog=history.category_catalog,
                           country_context=history.country_context.empty_copy() if history.country_context is not None else None)
    fresh.elo = EloEngine(**history.model_policy["elo"])
    fresh.trueskill = TrueSkillEngine(**history.trueskill.parameters)
    if fresh.model_policy != history.model_policy:
        raise ValueError("Refreshed rating policy differs from the saved model bundle")
    fresh.replay_results(eligible, as_of=cutoff)
    logger.info("Refreshed %d results (%d observed) strictly before %s UTC", len(eligible),
                sum(m.get("availability_source") == "observed_completed_result" for m in eligible), cutoff)
    return fresh
