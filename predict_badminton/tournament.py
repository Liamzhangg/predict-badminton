"""Fixed-history forecasts for complete, single-elimination tournament draws."""
from __future__ import annotations

import json
import math
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from predict_badminton.config import DISCIPLINES, forecast_bundle_directory
from predict_badminton.features import FeatureBuilder
from predict_badminton.forecast_history import history_sha256, refresh_history
from predict_badminton.probabilities import PREDICTION_POLICY, predict_probabilities


def load_bundle(directory: Path | None = None) -> tuple:
    """Load only a trusted local model bundle, never an uploaded pickle."""
    directory = directory if directory is not None else forecast_bundle_directory()
    history = FeatureBuilder.load(directory / "feature_state.pkl")
    columns = json.loads((directory / "feature_list.json").read_text())
    if columns != history.feature_columns:
        raise ValueError("Model feature list differs from its saved history; retrain first")
    objects = []
    manifest_path = directory / "bundle_manifest.json"
    raw_only = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("version") != 1 or manifest.get("probability_methods") != ["Raw"]
                or manifest.get("prediction_policy") != PREDICTION_POLICY):
            raise ValueError("Unsupported bundle manifest")
        raw_only = True
    for name in ("lgbm_model.pkl", "platt_calibrator.pkl", "isotonic_calibrator.pkl"):
        if raw_only and name != "lgbm_model.pkl":
            objects.append(None)
            continue
        with (directory / name).open("rb") as handle:
            objects.append(pickle.load(handle))
    if list(objects[0].feature_name_) != columns:
        raise ValueError("Model feature list differs from its saved model; retrain first")
    history.validate_model(objects[0])
    if any(getattr(calibrator, "feature_policy_", None) != getattr(objects[0], "feature_policy_", None)
           for calibrator in objects[1:] if calibrator is not None):
        raise ValueError("Calibrator rating policy differs from its saved model; use a matching bundle")
    for calibrator in objects[1:]:
        if calibrator is not None:
            history.validate_training_cutoff(calibrator)
    return (*objects, history)


def validate_draw(draw: dict, discipline: str) -> None:
    entries, rounds = draw["entries"], draw["rounds"]
    count = len(entries)
    if discipline not in DISCIPLINES or count < 2 or count & (count - 1):
        raise ValueError("A complete power-of-two singles or doubles draw is required")
    if len(rounds) != int(math.log2(count)):
        raise ValueError("Draw rounds do not match its entry count")
    players, identifiers = set(), set()
    for entry in entries:
        ids = [str(player["id"]) for player in entry["players"] if player.get("id")]
        expected = 2 if discipline in {"MD", "WD", "XD"} else 1
        if len(ids) != expected or len(set(ids)) != expected or players.intersection(ids):
            raise ValueError("Each entry needs distinct, verified player IDs")
        if not entry.get("name") or entry["id"] in identifiers:
            raise ValueError("Entry names and unique entry IDs are required")
        players.update(ids)
        identifiers.add(entry["id"])
    for level, rnd in enumerate(rounds):
        size = count >> level
        expected = f"R{size}" if size > 8 else {8: "QF", 4: "SF", 2: "Final"}[size]
        if rnd["name"] != expected or len(rnd["matches"]) != size // 2:
            raise ValueError("Round names or match slots are inconsistent")
        for match in rnd["matches"]:
            datetime.fromisoformat(match["time_utc"])


def propagate(count: int, probabilities: list[float]) -> tuple[np.ndarray, list[dict]]:
    """Combine all routes exactly; probabilities follow round/slot/left/right order."""
    levels = int(math.log2(count)) if count > 0 else 0
    if count < 2 or count & (count - 1) or len(probabilities) != count * (count - 1) // 2:
        raise ValueError("Expected one probability for each possible bracket matchup")
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Match probabilities must be finite and between zero and one")
    advance = np.zeros((count, levels + 1))
    advance[:, 0] = 1
    matchups, cursor = [], 0
    for level in range(levels):
        width = 2 ** (level + 1)
        for start in range(0, count, width):
            for left in range(start, start + width // 2):
                for right in range(start + width // 2, start + width):
                    probability = float(values[cursor])
                    meeting = float(advance[left, level] * advance[right, level])
                    advance[left, level + 1] += meeting * probability
                    advance[right, level + 1] += meeting * (1 - probability)
                    matchups.append({"level": level, "slot": start // width, "left": left,
                                     "right": right, "win": probability, "meeting": meeting})
                    cursor += 1
    return advance, matchups


def forecast(draw: dict, discipline: str, bundle: tuple, method: str = "Raw", *,
             refresh: bool = True, as_of: datetime | None = None) -> dict:
    validate_draw(draw, discipline)
    model, platt, isotonic, history = bundle
    if method not in {"Raw", "Platt", "Isotonic"}:
        raise ValueError("Unknown probability method")
    calibrator = None if method == "Raw" else platt if method == "Platt" else isotonic
    if method != "Raw" and calibrator is None:
        raise ValueError(f"{method} probabilities are unavailable in this model bundle")
    if history.country_context is not None and any("country" not in entry for entry in draw["entries"]):
        raise ValueError("Country-model draw entries require country; use an empty value when unknown")
    if refresh:
        stamp = as_of or datetime.now(UTC)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("Forecast cutoff must have a UTC offset")
        if any(pd.to_datetime(m["time_utc"], utc=True).to_pydatetime() <= stamp
               for rnd in draw["rounds"] for m in rnd["matches"]):
            raise ValueError("Live forecast times must follow the forecast cutoff; use frozen history for archival replay")
        history.validate_model(model)
        history = refresh_history(history, as_of=stamp)
        history.validate_model(model)
        for calibrator in (platt, isotonic):
            if calibrator is not None:
                history.validate_training_cutoff(calibrator)
    cutoff = history.history_cutoff
    entries, rounds = draw["entries"], draw["rounds"]
    rows = []
    for level, rnd in enumerate(rounds):
        width = 2 ** (level + 1)
        for slot, match in enumerate(rnd["matches"]):
            stamp = pd.to_datetime(match["time_utc"], utc=True).tz_localize(None).to_pydatetime()
            if stamp <= cutoff:
                raise ValueError("Forecast times must follow the saved history cutoff")
            start = slot * width
            for left in range(start, start + width // 2):
                for right in range(start + width // 2, start + width):
                    a, b = entries[left], entries[right]
                    rows.append(history.preview({
                        "match_id": f"{level}-{slot}-{left}-{right}", "match_time": stamp,
                        "discipline": discipline, "round_name": rnd["name"],
                        "tournament_name": draw["tournament_name"], "tournament_code": draw["tournament_code"],
                        "tournament_tier": draw["tournament_tier"],
                        "team1_ids": [str(p["id"]) for p in a["players"]],
                        "team2_ids": [str(p["id"]) for p in b["players"]],
                        "team1_seed": a.get("seed"), "team2_seed": b.get("seed"),
                        "team1_country": a.get("country", ""), "team2_country": b.get("country", ""),
                    }))
    frame = pd.DataFrame(rows)[history.feature_columns]
    history.encode_categories(frame)
    probabilities = predict_probabilities(model, frame, calibrator)
    if refresh and as_of is None and any(pd.to_datetime(m["time_utc"], utc=True) <= datetime.now(UTC)
                                        for rnd in rounds for m in rnd["matches"]):
        raise ValueError("A fixture started while refreshing history; no live forecast is available")
    advance, matchups = propagate(len(entries), probabilities)
    labels = [rnd["name"] for rnd in rounds[1:]] + ["Title"]
    records = []
    for index, entry in enumerate(entries):
        ids = [str(p["id"]) for p in entry["players"]]
        records.append({"Entry": entry["name"], "Country": entry.get("country", ""),
                        "Seed": entry.get("seed"), "Elo": round(history.elo.team_elo(ids, discipline)[0], 1),
                        "History": history.elo.team_disc_count(ids, discipline),
                        **{label: float(advance[index, level + 1] * 100) for level, label in enumerate(labels)}})
    return {"advance": advance, "matchups": matchups, "table": pd.DataFrame(records),
            "cutoff": cutoff.isoformat(), "method": method, "prediction_policy": PREDICTION_POLICY,
            "history_refreshed": refresh, "state_sha256": history_sha256(history),
            "result_availability_policy": history.model_policy.get("result_availability", "legacy_start_order")}
