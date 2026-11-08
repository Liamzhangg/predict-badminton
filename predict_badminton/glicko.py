"""Causal weekly Glicko-2 ratings, using the tested glicko2-py math library."""
from __future__ import annotations

import math
from datetime import datetime

from glicko2.math import mu_to_rating, phi_to_rd, rd_to_phi, update_rating
from glicko2.math.algorithm import UpdatedRating

from predict_badminton.config import (
    GLICKO_FEATURE_COLS,
    GLICKO_INITIAL_RD,
    GLICKO_INITIAL_VOLATILITY,
    GLICKO_PERIOD_DAYS,
    GLICKO_TAU,
)

_EPOCH = datetime(2018, 1, 1)  # Monday, UTC (timestamps are UTC-naive internally).
TeamKey = tuple[str, tuple[str, ...]]


class GlickoEngine:
    """Freeze ratings within a period; only observed outcomes enter updates."""

    def __init__(
        self,
        period_days: int = GLICKO_PERIOD_DAYS,
        tau: float = GLICKO_TAU,
        initial_rd: float = GLICKO_INITIAL_RD,
        initial_volatility: float = GLICKO_INITIAL_VOLATILITY,
    ) -> None:
        if period_days < 1 or min(tau, initial_rd, initial_volatility) <= 0:
            raise ValueError("Glicko-2 parameters must be positive")
        self.parameters = dict(period_days=period_days, tau=tau, initial_rd=initial_rd,
                               initial_volatility=initial_volatility)
        self.period: int | None = None
        self.posteriors: dict[TeamKey, tuple[UpdatedRating, int]] = {}
        self.pre_period: dict[TeamKey, UpdatedRating] = {}
        self.pending: dict[TeamKey, list[tuple[float, float, float]]] = {}

    def get_state(self) -> dict:
        return {"parameters": dict(self.parameters), "period": self.period,
                "posteriors": dict(self.posteriors), "pre_period": dict(self.pre_period),
                "pending": {key: list(games) for key, games in self.pending.items()}}

    @classmethod
    def from_state(cls, state: dict) -> GlickoEngine:
        engine = cls(**state["parameters"])
        engine.period = state["period"]
        engine.posteriors = dict(state["posteriors"])
        engine.pre_period = dict(state["pre_period"])
        engine.pending = {key: list(games) for key, games in state["pending"].items()}
        return engine

    def _period(self, when: datetime) -> int:
        period = (when - _EPOCH).days // self.parameters["period_days"]
        if self.period is not None and period < self.period:
            raise ValueError("Glicko-2 matches must be chronological")
        return period

    @staticmethod
    def _keys(match: dict) -> tuple[TeamKey, TeamKey]:
        return tuple((match["discipline"], tuple(sorted(match[side])))
                     for side in ("team1_ids", "team2_ids"))

    def _updated(self, key: TeamKey) -> UpdatedRating:
        return update_rating(*self.pre_period[key], self.pending[key], tau=self.parameters["tau"])

    def _snapshot(self, key: TeamKey, period: int) -> UpdatedRating:
        if period == self.period and key in self.pre_period:
            return self.pre_period[key]
        # A future preview can use the observed week's posterior without committing it.
        if key in self.pending:
            previous, last_played = self._updated(key), self.period
        elif key in self.posteriors:
            previous, last_played = self.posteriors[key]
        else:
            return UpdatedRating(0.0, rd_to_phi(self.parameters["initial_rd"]),
                                 self.parameters["initial_volatility"])
        empty_periods = period - last_played - 1
        return UpdatedRating(previous.mu,
                             math.sqrt(previous.phi**2 + empty_periods * previous.sigma**2),
                             previous.sigma)

    def features(self, match: dict) -> dict[str, float]:
        """Read-only pre-period features, including for multiple future horizons."""
        period = self._period(match["match_time"])
        one, two = (self._snapshot(key, period) for key in self._keys(match))
        values = []
        for left, right in ((mu_to_rating(one.mu), mu_to_rating(two.mu)),
                            (phi_to_rd(one.phi), phi_to_rd(two.phi)),
                            (one.sigma, two.sigma)):
            values.extend((left, right, left - right))
        return dict(zip(GLICKO_FEATURE_COLS, values, strict=True))

    def observe(self, match: dict) -> None:
        """Record an actual result, finalizing the previous observed period first."""
        period = self._period(match["match_time"])
        if self.period is not None and period != self.period:
            for key in self.pending:
                self.posteriors[key] = (self._updated(key), self.period)
            self.pending.clear()
            self.pre_period.clear()
        self.period = period
        key1, key2 = self._keys(match)
        for key in (key1, key2):
            if key not in self.pre_period:
                self.pre_period[key] = self._snapshot(key, period)
        one, two = self.pre_period[key1], self.pre_period[key2]
        score = float(match["winner"] == 1)
        self.pending.setdefault(key1, []).append((two.mu, two.phi, score))
        self.pending.setdefault(key2, []).append((one.mu, one.phi, 1 - score))
