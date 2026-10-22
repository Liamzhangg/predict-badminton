"""Feature engineering: Elo ratings, rolling form, H2H, and match context.

Processing is strictly sequential (matches already sorted by time).
State is updated AFTER features are recorded → no look-ahead leakage.

Elo design:
  - Per-player, per-discipline Elo (disc-specific)
  - Per-player global Elo across all disciplines
  - For doubles (MD/WD/XD): hybrid team rating using pair Elo component
      team_rating = avg(player_elos) + w_pair(n_pair) * pair_component
      w_pair(n) = n / (n + PAIR_LAMBDA)   [Bayesian weight-in]
  - For singles: pair component = 0, standard player Elo only
  - K-factor scaled by round importance and tournament tier
  - Delta split: ALPHA_PLAYER_SHARE → players, (1 - ALPHA_PLAYER_SHARE) → pair

Form features (last N matches, before current match):
  - Win rate over last 5 / 10 matches
  - Match count in that window (fills to 0 if player is new)
  - Rest days since last match (NaN if no prior matches)
  - For doubles teams: mean across the two players

H2H:
  - Canonical key = (sorted_ids_a, sorted_ids_b) where a ≤ b lexicographically
  - Tracks [wins_first, wins_second, total] in canonical order
"""
from __future__ import annotations

import ast
import logging
import pickle
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predict_badminton.config import (
    ALPHA_PLAYER_SHARE,
    DEFAULT_ELO,
    DISCIPLINES,
    DOUBLES_DISCIPLINES,
    K_BASE,
    PAIR_LAMBDA,
    ROUND_IMPORTANCE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# K-factor helpers
# ---------------------------------------------------------------------------

_ROUND_K_MULT: dict[str, float] = {
    "Final": 1.5,
    "SF": 1.3,
    "QF": 1.2,
    "3/4": 1.2,
    "R16": 1.1,
    "R32": 1.0,
    "R64": 0.9,
    "R128": 0.8,
    "Qual. SF": 0.85,
    "Qual. QF": 0.80,
    "Qual. R16": 0.75,
    "Qual. R32": 0.70,
    "Qual. R64": 0.65,
}


def _k_factor(round_name: str, tournament_tier: int, k_base: float = K_BASE) -> float:
    round_mult = _ROUND_K_MULT.get(round_name, 1.0)
    tier_mult = 1.0 + max(0, 6 - tournament_tier) * 0.05
    return k_base * round_mult * tier_mult


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _pair_weight(n: int, lam: float = PAIR_LAMBDA) -> float:
    """Bayesian weight-in: w = n / (n + lambda). Approaches 1 as n grows."""
    return n / (n + lam)


def _pair_key(ids: list[str]) -> tuple[str, ...]:
    """Canonical sorted pair key for a doubles team."""
    return tuple(sorted(ids))


# ---------------------------------------------------------------------------
# EloEngine — player + pair hybrid
# ---------------------------------------------------------------------------


class EloEngine:
    """Maintains per-player discipline/global Elo and per-pair discipline components."""

    def __init__(
        self,
        default_elo: float = DEFAULT_ELO,
        k_base: float = K_BASE,
        pair_lambda: float = PAIR_LAMBDA,
        alpha_player: float = ALPHA_PLAYER_SHARE,
    ) -> None:
        self.default_elo = default_elo
        self.k_base = k_base
        self.pair_lambda = pair_lambda
        self.alpha_player = alpha_player

        # {player_id: {discipline: elo}}
        self._disc: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: default_elo)
        )
        # {player_id: global_elo}
        self._global: dict[str, float] = defaultdict(lambda: default_elo)
        # {player_id: {discipline: match_count}}
        self._disc_count: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # {pair_key: {discipline: pair_component}}  (pair_component starts at 0)
        self._pair: dict[tuple, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # {pair_key: {discipline: match_count}}
        self._pair_count: dict[tuple, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # {player_id: last match datetime}
        self._last_match: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # State export / import (for persistence)
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return a serialisable snapshot of all Elo state."""
        return {
            "disc": {pid: dict(d) for pid, d in self._disc.items()},
            "global": dict(self._global),
            "disc_count": {pid: dict(d) for pid, d in self._disc_count.items()},
            "pair": {str(k): dict(v) for k, v in self._pair.items()},
            "pair_count": {str(k): dict(v) for k, v in self._pair_count.items()},
            "last_match": {pid: str(dt) for pid, dt in self._last_match.items()},
            "default_elo": self.default_elo,
            "k_base": self.k_base,
            "pair_lambda": self.pair_lambda,
            "alpha_player": self.alpha_player,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "EloEngine":
        """Restore an engine from a state dictionary."""
        eng = cls(
            default_elo=state.get("default_elo", DEFAULT_ELO),
            k_base=state.get("k_base", K_BASE),
            pair_lambda=state.get("pair_lambda", PAIR_LAMBDA),
            alpha_player=state.get("alpha_player", ALPHA_PLAYER_SHARE),
        )
        for pid, values in state["disc"].items():
            eng._disc[pid].update(values)
        for pid, value in state["global"].items():
            eng._global[pid] = value
        for pid, values in state["disc_count"].items():
            eng._disc_count[pid].update(values)

        for raw_key, values in state["pair"].items():
            key = (
                raw_key
                if isinstance(raw_key, tuple)
                else tuple(ast.literal_eval(raw_key))
            )
            eng._pair[key].update(values)
        for raw_key, values in state["pair_count"].items():
            key = (
                raw_key
                if isinstance(raw_key, tuple)
                else tuple(ast.literal_eval(raw_key))
            )
            eng._pair_count[key].update(values)
        for pid, dt_str in state.get("last_match", {}).items():
            try:
                eng._last_match[pid] = datetime.fromisoformat(dt_str)
            except (TypeError, ValueError):
                continue
        return eng

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def pair_component(self, ids: list[str], disc: str) -> float:
        """Pair component value (0.0 for singles or unseen pairs)."""
        if disc not in DOUBLES_DISCIPLINES or len(ids) < 2:
            return 0.0
        return self._pair.get(_pair_key(ids), {}).get(disc, 0.0)

    def pair_match_count(self, ids: list[str], disc: str) -> int:
        if disc not in DOUBLES_DISCIPLINES or len(ids) < 2:
            return 0
        return self._pair_count.get(_pair_key(ids), {}).get(disc, 0)

    def team_elo(
        self, player_ids: list[str], discipline: str
    ) -> tuple[float, float]:
        """Returns (disc_elo, global_elo) for a team, incorporating pair component for doubles."""
        d_avg = float(np.mean([self._disc.get(pid, {}).get(discipline, self.default_elo) for pid in player_ids]))
        g_avg = float(np.mean([self._global.get(pid, self.default_elo) for pid in player_ids]))

        if discipline in DOUBLES_DISCIPLINES and len(player_ids) >= 2:
            pk = _pair_key(player_ids)
            n = self._pair_count.get(pk, {}).get(discipline, 0)
            w = _pair_weight(n, self.pair_lambda)
            comp = self._pair.get(pk, {}).get(discipline, 0.0)
            d_hybrid = d_avg + w * comp
            return d_hybrid, g_avg  # pair component only affects disc Elo
        return d_avg, g_avg

    def team_disc_count(self, player_ids: list[str], discipline: str) -> int:
        """Min disc match count across team members (conservative reliability)."""
        return min(self._disc_count.get(pid, {}).get(discipline, 0) for pid in player_ids)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        ids1: list[str],
        ids2: list[str],
        discipline: str,
        round_name: str,
        tournament_tier: int,
        winner: int,
        match_time: datetime | None = None,
    ) -> None:
        """Update Elo after a match. winner ∈ {1, 2}."""
        elo1_d, elo1_g = self.team_elo(ids1, discipline)
        elo2_d, elo2_g = self.team_elo(ids2, discipline)

        k = _k_factor(round_name, tournament_tier, self.k_base)

        E1_d = _elo_expected(elo1_d, elo2_d)
        E1_g = _elo_expected(elo1_g, elo2_g)

        actual1 = 1.0 if winner == 1 else 0.0
        actual2 = 1.0 - actual1

        n1, n2 = len(ids1), len(ids2)
        k1 = k / n1
        k2 = k / n2

        # Player updates (scaled by alpha_player share)
        alpha = self.alpha_player
        for pid in ids1:
            self._disc[pid][discipline] += alpha * k1 * (actual1 - E1_d)
            self._global[pid] += alpha * k1 * (actual1 - E1_g)
            self._disc_count[pid][discipline] += 1
            if match_time:
                self._last_match[pid] = match_time

        for pid in ids2:
            self._disc[pid][discipline] += alpha * k2 * (actual2 - (1.0 - E1_d))
            self._global[pid] += alpha * k2 * (actual2 - (1.0 - E1_g))
            self._disc_count[pid][discipline] += 1
            if match_time:
                self._last_match[pid] = match_time

        # Pair component update (doubles only, remaining share)
        if discipline in DOUBLES_DISCIPLINES:
            pair_share = 1.0 - alpha

            pk1 = _pair_key(ids1)
            pk2 = _pair_key(ids2)

            # Use full K (not per-player K) for pair component delta
            delta1 = pair_share * k * (actual1 - E1_d)
            delta2 = pair_share * k * (actual2 - (1.0 - E1_d))

            self._pair[pk1][discipline] += delta1
            self._pair_count[pk1][discipline] += 1

            self._pair[pk2][discipline] += delta2
            self._pair_count[pk2][discipline] += 1


# ---------------------------------------------------------------------------
# FeatureBuilder
# ---------------------------------------------------------------------------


class FeatureBuilder:
    """Builds leakage-safe feature rows; must process matches in time order."""

    _HISTORY_MAXLEN = 50

    def __init__(
        self,
        k_base: float = K_BASE,
        player_names: dict[str, str] | None = None,
    ) -> None:
        self.elo = EloEngine(k_base=k_base)
        self._player_names: dict[str, str] = player_names or {}
        # {player_id: deque of {"time": datetime, "won": bool}}
        self._player_hist: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._HISTORY_MAXLEN)
        )
        # {canonical_pair: [wins_first, wins_second, total]}
        self._h2h: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])

    def get_state(self) -> dict[str, Any]:
        """Return all historical state needed to reproduce training features."""
        return {
            "state_type": "feature_builder",
            "version": 1,
            "elo": self.elo.get_state(),
            "player_hist": {
                pid: list(history) for pid, history in self._player_hist.items()
            },
            "h2h": dict(self._h2h),
            "player_names": dict(self._player_names),
        }

    def save(self, path: Path) -> None:
        """Persist Elo, form, and H2H state for future inference."""
        with open(path, "wb") as fh:
            pickle.dump(self.get_state(), fh)
        logger.info("Feature state saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureBuilder":
        """Restore the complete, versioned inference state."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)

        required = {"elo", "player_hist", "h2h", "player_names"}
        if (not isinstance(state, dict) or state.get("state_type") != "feature_builder"
                or state.get("version") != 1 or not required.issubset(state)):
            raise ValueError("Incomplete or unsupported feature state; retrain and use feature_state.pkl")

        builder = cls(player_names=state.get("player_names"))
        builder.elo = EloEngine.from_state(state["elo"])
        for pid, history in state.get("player_hist", {}).items():
            builder._player_hist[pid].extend(history)
        for key, values in state.get("h2h", {}).items():
            builder._h2h[key] = list(values)
        logger.info("Feature state loaded from %s", path)
        return builder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_h2h(
        ids1: list[str], ids2: list[str]
    ) -> tuple[tuple, bool]:
        """Returns (canonical_key, team1_is_first_in_key)."""
        a = tuple(sorted(ids1))
        b = tuple(sorted(ids2))
        if a <= b:
            return (a, b), True
        return (b, a), False

    def _form(
        self, ids: list[str], match_time: datetime, n: int
    ) -> tuple[float, float, float]:
        """(win_rate, match_count, rest_days) for team over last n matches."""
        win_rates: list[float] = []
        counts: list[float] = []
        rest_list: list[float] = []

        for pid in ids:
            hist = list(self._player_hist.get(pid, ()))
            recent = hist[-n:]
            if recent:
                win_rates.append(sum(float(e["won"]) for e in recent) / len(recent))
                counts.append(float(len(recent)))
                days = (match_time - recent[-1]["time"]).days
                rest_list.append(float(days))
            else:
                win_rates.append(0.5)
                counts.append(0.0)

        wr = float(np.mean(win_rates)) if win_rates else 0.5
        cnt = float(np.mean(counts)) if counts else 0.0
        rd = float(np.mean(rest_list)) if rest_list else float("nan")
        return wr, cnt, rd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preview(self, match: dict[str, Any]) -> dict[str, Any]:
        """Return a feature row from current state without updating it."""
        ids1: list[str] = match["team1_ids"]
        ids2: list[str] = match["team2_ids"]
        disc: str = match["discipline"]
        round_name: str = match["round_name"]
        t_tier: int = match["tournament_tier"]
        match_time: datetime = match["match_time"]

        # ---- Elo (pre-match) -------------------------------------------
        elo1_d, elo1_g = self.elo.team_elo(ids1, disc)
        elo2_d, elo2_g = self.elo.team_elo(ids2, disc)
        mc1 = self.elo.team_disc_count(ids1, disc)
        mc2 = self.elo.team_disc_count(ids2, disc)

        # ---- Pair Elo (pre-match) --------------------------------------
        pc1 = self.elo.pair_component(ids1, disc)
        pc2 = self.elo.pair_component(ids2, disc)
        pm1 = float(self.elo.pair_match_count(ids1, disc))
        pm2 = float(self.elo.pair_match_count(ids2, disc))

        # ---- Form (pre-match) ------------------------------------------
        wr1_5, cnt1_5, rd1 = self._form(ids1, match_time, 5)
        wr1_10, cnt1_10, _ = self._form(ids1, match_time, 10)
        wr2_5, cnt2_5, rd2 = self._form(ids2, match_time, 5)
        wr2_10, cnt2_10, _ = self._form(ids2, match_time, 10)

        # ---- H2H (pre-match) -------------------------------------------
        h2h_key, t1_first = self._canonical_h2h(ids1, ids2)
        h2h_state = self._h2h.get(h2h_key, (0, 0, 0))
        if t1_first:
            h2h_w1, h2h_w2 = h2h_state[0], h2h_state[1]
        else:
            h2h_w1, h2h_w2 = h2h_state[1], h2h_state[0]
        h2h_total = h2h_state[2]
        h2h_wr1 = h2h_w1 / h2h_total if h2h_total > 0 else 0.5

        # ---- Seed context ----------------------------------------------
        s1 = match.get("team1_seed")
        s2 = match.get("team2_seed")
        seed1 = int(s1) if s1 is not None else 0
        seed2 = int(s2) if s2 is not None else 0

        round_imp = ROUND_IMPORTANCE.get(round_name, 5)

        row: dict[str, Any] = {
            # Metadata (not model features)
            "match_id": match.get("match_id"),
            "match_time": match_time,
            "tournament_code": match.get("tournament_code"),
            "tournament_name": match.get("tournament_name"),
            # -- Elo features --
            "elo_diff_disc": elo1_d - elo2_d,
            "elo_t1_disc": elo1_d,
            "elo_t2_disc": elo2_d,
            "elo_diff_global": elo1_g - elo2_g,
            "elo_t1_global": elo1_g,
            "elo_t2_global": elo2_g,
            "elo_matches_t1_disc": float(mc1),
            "elo_matches_t2_disc": float(mc2),
            "elo_reliability": float(min(mc1, mc2)),
            # -- Pair Elo features (0.0 for singles) --
            "pair_component_t1": pc1,
            "pair_component_t2": pc2,
            "pair_component_diff": pc1 - pc2,
            "pair_matches_t1": pm1,
            "pair_matches_t2": pm2,
            "pair_reliability": min(pm1, pm2),
            # -- Form features --
            "form5_wr_t1": wr1_5,
            "form10_wr_t1": wr1_10,
            "form5_cnt_t1": cnt1_5,
            "form10_cnt_t1": cnt1_10,
            "rest_days_t1": rd1,
            "form5_wr_t2": wr2_5,
            "form10_wr_t2": wr2_10,
            "form5_cnt_t2": cnt2_5,
            "form10_cnt_t2": cnt2_10,
            "rest_days_t2": rd2,
            "form5_wr_diff": wr1_5 - wr2_5,
            "form10_wr_diff": wr1_10 - wr2_10,
            # -- H2H features --
            "h2h_wins_t1": float(h2h_w1),
            "h2h_wins_t2": float(h2h_w2),
            "h2h_total": float(h2h_total),
            "h2h_wr_t1": h2h_wr1,
            # -- Context --
            "tournament_tier": float(t_tier),
            "round_importance": float(round_imp),
            "team1_seeded": float(seed1 > 0),
            "team2_seeded": float(seed2 > 0),
            "seed_diff": float(seed1 - seed2),
            "same_country": float(
                bool(match.get("team1_country"))
                and match.get("team1_country") == match.get("team2_country")
            ),
            "discipline": disc,  # categorical
        }

        if "label" in match:
            row["label"] = match["label"]
        return row

    def process(self, match: dict[str, Any]) -> dict[str, Any]:
        """Return feature dict (pre-match state), then update internal state."""
        row = self.preview(match)
        ids1: list[str] = match["team1_ids"]
        ids2: list[str] = match["team2_ids"]
        disc: str = match["discipline"]
        round_name: str = match["round_name"]
        t_tier: int = match["tournament_tier"]
        match_time: datetime = match["match_time"]
        winner: int = match["winner"]
        h2h_key, t1_first = self._canonical_h2h(ids1, ids2)
        h2h_state = self._h2h[h2h_key]

        # ---- Update state (AFTER features recorded) --------------------
        self.elo.update(ids1, ids2, disc, round_name, t_tier, winner, match_time)

        for pid in ids1:
            self._player_hist[pid].append({"time": match_time, "won": winner == 1})
        for pid in ids2:
            self._player_hist[pid].append({"time": match_time, "won": winner == 2})

        h2h_state[2] += 1
        if t1_first:
            h2h_state[0 if winner == 1 else 1] += 1
        else:
            h2h_state[1 if winner == 1 else 0] += 1

        return row

    def build_dataframe(
        self, matches: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Process all matches (must be sorted by time) → feature DataFrame."""
        rows = [self.process(m) for m in matches]
        df = pd.DataFrame(rows)

        all_discs = DISCIPLINES + ["OTHER"]
        df["discipline"] = pd.Categorical(df["discipline"], categories=all_discs)

        logger.info(
            "Feature matrix: %d rows × %d cols | label balance: %.3f",
            len(df),
            len(df.columns),
            df["label"].mean() if len(df) > 0 else float("nan"),
        )
        return df

    # ------------------------------------------------------------------
    # Leaderboard export
    # ------------------------------------------------------------------

    def export_leaderboards(self, artifacts_dir: Path) -> None:
        """Export per-discipline player Elo leaderboards and pair leaderboards."""
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # ---- Per-discipline player leaderboards ------------------------
        all_players = set(self.elo._disc.keys()) | set(self.elo._global.keys())

        for disc in DISCIPLINES:
            rows = []
            for pid in all_players:
                cnt = self.elo._disc_count[pid].get(disc, 0)
                if cnt == 0:
                    continue  # skip players who never played this discipline
                last_dt = self.elo._last_match.get(pid)
                rows.append({
                    "player_name": self._player_names.get(pid, pid),
                    "player_id": pid,
                    "elo": round(self.elo._disc[pid][disc], 2),
                    "matches_in_discipline": cnt,
                    "last_match_time": str(last_dt) if last_dt else "",
                })
            df_lb = (
                pd.DataFrame(rows, columns=["player_name", "player_id", "elo", "matches_in_discipline", "last_match_time"])
                .sort_values("elo", ascending=False)
                .reset_index(drop=True)
            )
            out = artifacts_dir / f"elo_leaderboard_{disc}.csv"
            df_lb.to_csv(out, index=False)
            logger.info("Leaderboard saved → %s (%d players)", out, len(df_lb))

        # ---- Global leaderboard ----------------------------------------
        rows_g = []
        for pid in all_players:
            total_matches = sum(self.elo._disc_count[pid].values())
            if total_matches == 0:
                continue
            last_dt = self.elo._last_match.get(pid)
            rows_g.append({
                "player_name": self._player_names.get(pid, pid),
                "player_id": pid,
                "elo": round(self.elo._global[pid], 2),
                "total_matches": total_matches,
                "last_match_time": str(last_dt) if last_dt else "",
            })
        df_global = (
            pd.DataFrame(rows_g, columns=["player_name", "player_id", "elo", "total_matches", "last_match_time"])
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
        df_global.to_csv(artifacts_dir / "elo_leaderboard_GLOBAL.csv", index=False)
        logger.info("Global leaderboard saved → %s", artifacts_dir / "elo_leaderboard_GLOBAL.csv")

        # ---- Doubles pair leaderboards ---------------------------------
        for disc in DOUBLES_DISCIPLINES:
            rows_p = []
            for pk, disc_map in self.elo._pair.items():
                if len(pk) != 2 or len(set(pk)) != 2:
                    continue
                cnt = self.elo._pair_count[pk].get(disc, 0)
                if cnt == 0:
                    continue
                comp = disc_map.get(disc, 0.0)
                rows_p.append({
                    "pair_name": " / ".join(self._player_names.get(pid) or pid for pid in pk),
                    "pair_key": "|".join(pk),
                    "elo": round(self.elo.team_elo(list(pk), disc)[0], 2),
                    "pair_elo_component": round(comp, 2),
                    "pair_matches": cnt,
                })
            df_pair = (
                pd.DataFrame(rows_p, columns=["pair_name", "pair_key", "elo", "pair_elo_component", "pair_matches"])
                .sort_values(["elo", "pair_key"], ascending=[False, True])
                .reset_index(drop=True)
            )
            out = artifacts_dir / f"elo_leaderboard_pairs_{disc}.csv"
            df_pair.to_csv(out, index=False)
            logger.info("Pair leaderboard saved → %s (%d pairs)", out, len(df_pair))


def build_feature_matrix(
    matches: list[dict[str, Any]],
    k_base: float = K_BASE,
    player_names: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, "FeatureBuilder"]:
    """Convenience wrapper: build feature DataFrame from sorted match list.

    Returns (df, builder) so callers can access Elo state for leaderboards/inference.
    """
    builder = FeatureBuilder(k_base=k_base, player_names=player_names)
    df = builder.build_dataframe(matches)
    return df, builder
