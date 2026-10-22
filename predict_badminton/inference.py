"""Inference module: generate training-compatible features for future fixtures.

Loads persisted training feature state and produces FEATURE_COLS-aligned rows
(no labels) that can be fed directly to the saved LightGBM model.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from predict_badminton.config import (
    DISCIPLINES,
    FEATURE_COLS,
)
from predict_badminton.features import FeatureBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

_REQUIRED_COLS = {
    "match_id", "match_time", "discipline",
    "team1_player_ids", "team2_player_ids",
}


def load_fixtures(path: Path) -> pd.DataFrame:
    """Load and validate a fixtures CSV.

    Drops rows with missing required fields and logs reasons.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    missing_cols = _REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Fixture CSV missing required columns: {missing_cols}")

    n_before = len(df)
    drop_reasons: dict[str, int] = defaultdict(int)

    # Drop rows with empty required fields
    for col in _REQUIRED_COLS:
        bad = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if bad.any():
            drop_reasons[f"missing_{col}"] += int(bad.sum())
            df = df[~bad]

    # Validate discipline
    valid_discs = set(DISCIPLINES) | {"OTHER"}
    bad_disc = ~df["discipline"].isin(valid_discs)
    if bad_disc.any():
        drop_reasons["unknown_discipline"] += int(bad_disc.sum())
        df = df[~bad_disc]

    n_after = len(df)
    if n_before != n_after:
        logger.warning(
            "Dropped %d fixture rows: %s", n_before - n_after, dict(drop_reasons)
        )

    if len(df) == 0:
        raise ValueError("No valid fixture rows after validation")

    # Parse match_time
    df["match_time"] = pd.to_datetime(df["match_time"], errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
    bad_time = df["match_time"].isna()
    if bad_time.any():
        logger.warning("Dropping %d fixtures with unparseable match_time", bad_time.sum())
        df = df[~bad_time]

    for col in ("team1_player_ids", "team2_player_ids"):
        valid_ids = df[col].map(lambda value: bool(_parse_player_ids(value))).astype(bool)
        df = df.loc[valid_ids].copy()
    if df.empty:
        raise ValueError("No valid fixture rows after validation")
    if df["match_id"].duplicated().any():
        raise ValueError("Fixture match_id values must be unique")

    df = df.sort_values("match_time").reset_index(drop=True)
    logger.info("Loaded %d valid fixture rows from %s", len(df), path)
    return df


def _parse_player_ids(raw: str) -> list[str]:
    """Parse pipe-delimited player IDs; return cleaned list."""
    return [p.strip() for p in str(raw).split("|") if p.strip()]


# ---------------------------------------------------------------------------
# Feature generation for fixtures
# ---------------------------------------------------------------------------


class InferenceBuilder:
    """Generate fixture features from persisted training history."""

    def __init__(self, feature_builder: FeatureBuilder) -> None:
        self.feature_builder = feature_builder

    def build_features(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with exactly FEATURE_COLS columns (no label)."""
        cutoff = max(self.feature_builder.elo._last_match.values(), default=datetime.min)
        if (fixtures["match_time"] <= cutoff).any():
            raise ValueError("Fixture times must follow the saved history cutoff")
        rows = []
        for _, fix in fixtures.iterrows():
            row = self._build_row(fix)
            if row is not None:
                rows.append(row)

        if not rows:
            raise ValueError("No feature rows could be built from fixtures")

        df = pd.DataFrame(rows)

        # Ensure discipline is categorical (same encoding as training)
        all_discs = DISCIPLINES + ["OTHER"]
        df["discipline"] = pd.Categorical(df["discipline"], categories=all_discs)

        return df[["match_id", "match_time"] + FEATURE_COLS]

    def _build_row(self, fix: pd.Series) -> dict[str, Any] | None:
        ids1 = _parse_player_ids(fix["team1_player_ids"])
        ids2 = _parse_player_ids(fix["team2_player_ids"])

        if not ids1 or not ids2:
            logger.warning("Skipping fixture %s: empty player IDs", fix.get("match_id"))
            return None

        disc = str(fix["discipline"])
        match_time: datetime = fix["match_time"].to_pydatetime()
        round_name = str(fix.get("round_name", "")) if pd.notna(fix.get("round_name", "")) else ""
        t_tier_raw = fix.get("tournament_tier", "5")
        try:
            t_tier = int(float(str(t_tier_raw))) if pd.notna(t_tier_raw) else 5
        except (ValueError, TypeError):
            t_tier = 5

        s1_raw = fix.get("team1_seed", "")
        s2_raw = fix.get("team2_seed", "")
        try:
            seed1 = int(float(str(s1_raw))) if pd.notna(s1_raw) and str(s1_raw).strip() else 0
        except (ValueError, TypeError):
            seed1 = 0
        try:
            seed2 = int(float(str(s2_raw))) if pd.notna(s2_raw) and str(s2_raw).strip() else 0
        except (ValueError, TypeError):
            seed2 = 0

        c1 = str(fix.get("team1_country", "")) if pd.notna(fix.get("team1_country", "")) else ""
        c2 = str(fix.get("team2_country", "")) if pd.notna(fix.get("team2_country", "")) else ""
        tournament_name = (
            str(fix.get("tournament_name", ""))
            if pd.notna(fix.get("tournament_name", ""))
            else ""
        )

        match = {
            "match_id": fix.get("match_id"),
            "match_time": match_time,
            "discipline": disc,
            "team1_ids": ids1,
            "team2_ids": ids2,
            "round_name": round_name,
            "tournament_name": tournament_name,
            "tournament_code": "",
            "tournament_tier": t_tier,
            "team1_seed": seed1,
            "team2_seed": seed2,
            "team1_country": c1,
            "team2_country": c2,
        }
        return self.feature_builder.preview(match)


def build_inference_features(
    fixtures_path: Path,
    state_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load fixtures and training state; return features and raw fixtures."""
    feature_builder = FeatureBuilder.load(state_path)
    fixtures = load_fixtures(fixtures_path)
    builder = InferenceBuilder(feature_builder)
    features = builder.build_features(fixtures)
    return features, fixtures
