"""Conservative historical result eligibility, not verified publication times."""
from datetime import datetime, timedelta
import math

AVAILABILITY_POLICY = "conservative_day_history_v1"
LEGACY_OBSERVED_AVAILABILITY_POLICY = "observed_results_with_conservative_history_v1"
OBSERVED_AVAILABILITY_POLICY = "observed_results_with_conservative_history_v2"
OBSERVED_AVAILABILITY_POLICIES = (LEGACY_OBSERVED_AVAILABILITY_POLICY, OBSERVED_AVAILABILITY_POLICY)


def match_timing(start: datetime, *, utc: bool, duration: object, coarse: bool) -> dict:
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    if not utc:
        # UTC+14 is the earliest possible instant for an unzoned local date.
        prediction_time = day - timedelta(hours=14)
        available_at = day + timedelta(days=2)
        source = "unverified_local_two_day_embargo"
    elif coarse or isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 300:
        prediction_time = day if coarse else start
        available_at = day + timedelta(days=2)
        source = "utc_uncertain_two_day_embargo"
    else:
        prediction_time = start
        end = start + timedelta(minutes=duration)
        available_at = end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        source = "utc_day_after_recorded_end"
    return {"prediction_time": prediction_time, "available_at": available_at, "availability_source": source}
