"""Fixed-round predictions and model-implied moneyline prices."""
from __future__ import annotations

import math

import pandas as pd

from predict_badminton.config import DISCIPLINES
from predict_badminton.probabilities import PREDICTION_POLICY, predict_probabilities


def american_to_decimal(odds: float) -> float:
    if not math.isfinite(odds) or abs(odds) < 100:
        raise ValueError("American odds must be at most -100 or at least +100")
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def fair_american(probability: float) -> float:
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("Fair odds require a probability strictly between zero and one")
    if probability <= 0.5:
        return 100 * (1 - probability) / probability
    return -100 * probability / (1 - probability)


def expected_return(probability: float, odds: float) -> float:
    """Expected net profit per unit staked, assuming the probability is correct."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be between zero and one")
    return probability * american_to_decimal(odds) - 1


def positive_ev_from(probability: float) -> int:
    """Smallest valid whole-number American price strictly above break-even."""
    price = math.ceil(fair_american(probability))
    if -100 < price < 100:
        price = 100
    if expected_return(probability, price) <= 1e-12:
        price += 1
        if -100 < price < 100:
            price = 100
    return price


def first_round_predictions(snapshot: dict, bundle: tuple) -> tuple[pd.DataFrame, str]:
    """Score named, unplayed R32 opponents using fixed, matching model history."""
    return round_predictions(snapshot, bundle, round_name="R32")


def round_predictions(snapshot: dict, bundle: tuple, *, round_name: str) -> tuple[pd.DataFrame, str]:
    """Score one confirmed, unplayed round across all five standard disciplines."""
    if round_name not in {"R128", "R64", "R32", "R16", "QF", "SF", "Final"}:
        raise ValueError("Unsupported knockout round")
    model, _, _, history = bundle
    history.validate_model(model)
    cutoff = history.history_cutoff
    fixtures, rows, seen = [], [], set()
    for fixture in snapshot["matches"]:
        if fixture["match_id"] in seen:
            raise ValueError("Round match IDs must be unique")
        seen.add(fixture["match_id"])
        if (fixture["round_name"] != round_name or fixture.get("winner", 0) != 0
                or fixture.get("score_status") in (1, "1")
                or str(fixture.get("status") or "").casefold() == "walkover"):
            raise ValueError(f"Only unplayed {round_name} matches without walkovers can be scored")
        stamp = pd.to_datetime(fixture["match_time"], utc=True).tz_localize(None).to_pydatetime()
        if pd.isna(stamp) or stamp <= cutoff:
            raise ValueError("These fixtures predate the saved history cutoff; past matches cannot be reforecast with newer history")
        if any(not fixture.get(f"team{side}_name") or not fixture.get(f"team{side}_ids")
               for side in (1, 2)):
            raise ValueError("Both opponents need names and verified player IDs")
        if any(not str(player).isdigit() or int(player) <= 0
               for side in (1, 2) for player in fixture[f"team{side}_ids"]):
            raise ValueError("BWF player IDs must be positive numeric IDs, not qualifier placeholders")
        match = {**fixture, "match_time": stamp, "tournament_name": snapshot["name"],
                 "tournament_code": snapshot["tournament_code"],
                 "tournament_tier": snapshot["tournament_tier"]}
        rows.append(history.preview(match))
        fixtures.append(fixture)
    if not rows:
        raise ValueError("No confirmed unplayed matches are available")
    frame = pd.DataFrame(rows)[history.feature_columns]
    frame["discipline"] = pd.Categorical(frame["discipline"], categories=DISCIPLINES + ["OTHER"])
    probabilities = predict_probabilities(model, frame)
    selections = []
    for fixture, probability in zip(fixtures, probabilities, strict=True):
        for side, chance in ((1, float(probability)), (2, float(1 - probability))):
            selections.append({
                "selection_id": f"{fixture['match_id']}:{side}", "match_id": fixture["match_id"],
                "event": fixture["discipline"], "match": fixture["slot"],
                "selection": fixture[f"team{side}_name"], "opponent": fixture[f"team{3-side}_name"],
                "probability": chance, "fair_american": fair_american(chance),
                "prediction_policy": PREDICTION_POLICY,
                "history_as_of": cutoff.isoformat(),
                "result_availability_policy": history.model_policy.get("result_availability", "legacy_start_order"),
                "positive_ev_from": positive_ev_from(chance),
                "recorded_matches": history.elo.team_disc_count(fixture[f"team{side}_ids"], fixture["discipline"]),
                "forecast_time_utc": fixture["match_time"], "time_estimated": fixture["time_estimated"],
            })
    return pd.DataFrame(selections), cutoff.isoformat()
