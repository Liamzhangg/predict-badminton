"""Causal player-based TrueSkill; calendar inactivity is not part of this policy."""
from __future__ import annotations

import math
from datetime import datetime

from trueskill import TrueSkill

from predict_badminton.config import (
    TRUESKILL_BETA,
    TRUESKILL_FEATURE_COLS,
    TRUESKILL_MU,
    TRUESKILL_SIGMA,
    TRUESKILL_TAU,
)


class TrueSkillEngine:
    def __init__(self, mu: float = TRUESKILL_MU, sigma: float = TRUESKILL_SIGMA,
                 beta: float = TRUESKILL_BETA, tau: float = TRUESKILL_TAU) -> None:
        if (not all(math.isfinite(v) for v in (mu, sigma, beta, tau))
                or min(sigma, beta) <= 0 or tau < 0):
            raise ValueError("Invalid TrueSkill parameters")
        self.parameters = dict(mu=mu, sigma=sigma, beta=beta, tau=tau)
        self.env = TrueSkill(**self.parameters, draw_probability=0.0, backend=None)
        self.ratings = {}
        self.last_time: datetime | None = None

    def get_state(self) -> dict:
        return {"parameters": dict(self.parameters), "last_time": self.last_time,
                "ratings": {key: (r.mu, r.sigma) for key, r in self.ratings.items()}}

    @classmethod
    def from_state(cls, state: dict) -> TrueSkillEngine:
        if (not isinstance(state, dict) or not {"parameters", "last_time", "ratings"}.issubset(state)
                or not isinstance(state["parameters"], dict)
                or set(state["parameters"]) != {"mu", "sigma", "beta", "tau"}
                or not isinstance(state["ratings"], dict)):
            raise ValueError("Incomplete TrueSkill history; use a matching bundle or retrain")
        engine = cls(**state["parameters"])
        engine.last_time = state["last_time"]
        engine.ratings = {key: engine.env.create_rating(*rating) for key, rating in state["ratings"].items()}
        return engine

    @staticmethod
    def _keys(match: dict) -> list[list[tuple[str, str]]]:
        return [[(match["discipline"], pid) for pid in sorted(match[side])]
                for side in ("team1_ids", "team2_ids")]

    def _groups(self, match: dict) -> list[tuple]:
        if self.last_time is not None and match["match_time"] < self.last_time:
            raise ValueError("TrueSkill matches must be chronological")
        return [tuple(self.ratings.get(key) or self.env.create_rating() for key in team)
                for team in self._keys(match)]

    def features(self, match: dict) -> dict[str, float]:
        """Read-only team skill and uncertainty before observing this result."""
        groups = self._groups(match)
        means = [sum(r.mu for r in team) for team in groups]
        sigmas = [math.sqrt(sum(r.sigma**2 for r in team)) for team in groups]
        values = [value for pair in (means, sigmas) for value in (*pair, pair[0]-pair[1])]
        return dict(zip(TRUESKILL_FEATURE_COLS, values, strict=True))

    def observe(self, match: dict) -> None:
        groups = self._groups(match)
        updated = self.env.rate(groups, ranks=[0, 1] if match["winner"] == 1 else [1, 0])
        for keys, ratings in zip(self._keys(match), updated, strict=True):
            self.ratings.update(zip(keys, ratings, strict=True))
        self.last_time = match["match_time"]
