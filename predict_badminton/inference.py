"""Inference module: generate training-compatible features for future fixtures.

Loads persisted Elo state and produces FEATURE_COLS-aligned rows (no labels)
that can be fed directly to the saved LightGBM model.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predict_badminton.config import (
    DISCIPLINES,
    FEATURE_COLS,
    ROUND_IMPORTANCE,
)
from predict_badminton.features import EloEngine, _k_factor, _elo_expected

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
    df["match_time"] = pd.to_datetime(df["match_time"], errors="coerce")
    bad_time = df["match_time"].isna()
    if bad_time.any():
        logger.warning("Dropping %d fixtures with unparseable match_time", bad_time.sum())
        df = df[~bad_time]

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
    """Generate prediction features from a loaded Elo state + fixtures CSV."""

    def __init__(self, elo: EloEngine) -> None:
        self.elo = elo

    def build_features(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with exactly FEATURE_COLS columns (no label)."""
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

        # Fill NaN rest_days with median of known values (safe default)
        for col in ("rest_days_t1", "rest_days_t2"):
            med = df[col].median()
            df[col] = df[col].fillna(0.0 if pd.isna(med) else med)

        # Ensure all feature cols exist (fill missing with 0)
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

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

        # Elo
        elo1_d, elo1_g = self.elo.team_elo(ids1, disc)
        elo2_d, elo2_g = self.elo.team_elo(ids2, disc)
        mc1 = self.elo.team_disc_count(ids1, disc)
        mc2 = self.elo.team_disc_count(ids2, disc)

        # Pair Elo
        pc1 = self.elo.pair_component(ids1, disc)
        pc2 = self.elo.pair_component(ids2, disc)
        pm1 = float(self.elo.pair_match_count(ids1, disc))
        pm2 = float(self.elo.pair_match_count(ids2, disc))

        # Seeds
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

        round_imp = ROUND_IMPORTANCE.get(round_name, 5)

        c1 = str(fix.get("team1_country", "")) if pd.notna(fix.get("team1_country", "")) else ""
        c2 = str(fix.get("team2_country", "")) if pd.notna(fix.get("team2_country", "")) else ""

        return {
            "match_id": fix.get("match_id"),
            "match_time": match_time,
            # Elo
            "elo_diff_disc": elo1_d - elo2_d,
            "elo_t1_disc": elo1_d,
            "elo_t2_disc": elo2_d,
            "elo_diff_global": elo1_g - elo2_g,
            "elo_t1_global": elo1_g,
            "elo_t2_global": elo2_g,
            "elo_matches_t1_disc": float(mc1),
            "elo_matches_t2_disc": float(mc2),
            "elo_reliability": float(min(mc1, mc2)),
            # Pair Elo
            "pair_component_t1": pc1,
            "pair_component_t2": pc2,
            "pair_component_diff": pc1 - pc2,
            "pair_matches_t1": pm1,
            "pair_matches_t2": pm2,
            "pair_reliability": min(pm1, pm2),
            # Form — unknown for future matches; fill with neutral defaults
            "form5_wr_t1": 0.5,
            "form10_wr_t1": 0.5,
            "form5_cnt_t1": 0.0,
            "form10_cnt_t1": 0.0,
            "rest_days_t1": float("nan"),
            "form5_wr_t2": 0.5,
            "form10_wr_t2": 0.5,
            "form5_cnt_t2": 0.0,
            "form10_cnt_t2": 0.0,
            "rest_days_t2": float("nan"),
            "form5_wr_diff": 0.0,
            "form10_wr_diff": 0.0,
            # H2H — not persisted in Elo state; default to neutral
            "h2h_wins_t1": 0.0,
            "h2h_wins_t2": 0.0,
            "h2h_total": 0.0,
            "h2h_wr_t1": 0.5,
            # Context
            "tournament_tier": float(t_tier),
            "round_importance": float(round_imp),
            "team1_seeded": float(seed1 > 0),
            "team2_seeded": float(seed2 > 0),
            "seed_diff": float(seed1 - seed2),
            "same_country": float(bool(c1) and c1 == c2),
            "discipline": disc,
        }


def build_inference_features(
    fixtures_path: Path,
    elo_state_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load fixtures and Elo state; return (features_df, raw_fixtures_df)."""
    elo = EloEngine.load(elo_state_path)
    fixtures = load_fixtures(fixtures_path)
    builder = InferenceBuilder(elo)
    features = builder.build_features(fixtures)
    return features, fixtures
