"""Frozen country categories and availability-ordered country Elo context."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata

import pandas as pd


def normalized_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


class CountryContext:
    def __init__(self, *, countries, located_countries, host_codes, host_names, vocabulary):
        from predict_badminton.features import EloEngine

        self.registry = {name: sorted(set(values)) for name, values in (
            ("countries", countries), ("located_countries", located_countries),
            ("host_codes", host_codes), ("host_names", host_names), ("vocabulary", vocabulary))}
        if (any(not isinstance(value, str) or not value for values in self.registry.values() for value in values)
                or not set(self.registry["located_countries"]).issubset(self.registry["countries"])
                or not set(self.registry["vocabulary"]).issubset(self.registry["countries"])):
            raise ValueError("Invalid country registry or training vocabulary")
        self.countries = frozenset(self.registry["countries"])
        self.located = frozenset(self.registry["located_countries"])
        self.host_codes = frozenset(self.registry["host_codes"])
        self.host_names = frozenset(self.registry["host_names"])
        self.engine = EloEngine(k_base=20, alpha_player=1, score_margin_weight=0)

    @property
    def policy(self) -> dict:
        return {"version": 1, "entity": "country_per_discipline", "initial_rating": 1500,
                "k": 20, "scale": 400, "domestic_results": "excluded",
                "counts": "log1p_intercountry_matches", "unknown_identity": "missing",
                "registry_sha256": hashlib.sha256(json.dumps(self.registry, sort_keys=True).encode()).hexdigest()}

    def country(self, value) -> str | None:
        code = str(value).strip().upper() if pd.notna(value) else ""
        return code if code in self.countries else None

    def preview(self, match: dict) -> dict:
        if not {"team1_country", "team2_country"}.issubset(match):
            raise ValueError("Country-model fixtures require team1_country and team2_country; use empty values for unknown countries")
        name = normalized_name(str(match.get("tournament_name") or ""))
        edition = str(match.get("tournament_code") or name).casefold()
        row = {"host_known": float(edition in self.host_codes or name in self.host_names)}
        ratings, counts = [], []
        for side in (1, 2):
            country = self.country(match[f"team{side}_country"])
            row[f"country_id_t{side}"] = country
            row[f"country_known_t{side}"] = float(country in self.located)
            rating, count = float("nan"), float("nan")
            if country is not None:
                rating, _ = self.engine.team_elo([country], match["discipline"])
                count = math.log1p(self.engine.team_disc_count([country], match["discipline"]))
            ratings.append(rating)
            counts.append(count)
        for name, (first, second) in (("country_rating", ratings), ("country_log_n", counts)):
            row.update({f"{name}_t1": first, f"{name}_t2": second, f"{name}_diff": first - second})
        return row

    def observe(self, match: dict) -> None:
        first, second = [self.country(match.get(f"team{side}_country")) for side in (1, 2)]
        if first is not None and second is not None and first != second:
            # Neutral round/tier keeps K=20 in every discipline, including doubles.
            self.engine.update([first], [second], match["discipline"], "", 6, match["winner"])

    def encode_categories(self, frame: pd.DataFrame) -> None:
        vocabulary = self.registry["vocabulary"]
        dtype = pd.CategoricalDtype(categories=vocabulary)
        for name in ("country_id_t1", "country_id_t2"):
            frame[name] = pd.Categorical(frame[name].where(frame[name].isin(vocabulary)), dtype=dtype)

    def empty_copy(self) -> CountryContext:
        return CountryContext(**self.registry)

    def get_state(self) -> dict:
        return {"version": 1, "registry": self.registry, "elo": self.engine.get_state()}

    @classmethod
    def from_state(cls, state: dict) -> CountryContext:
        from predict_badminton.features import EloEngine

        if (not isinstance(state, dict) or state.get("version") != 1
                or not isinstance(state.get("registry"), dict) or not isinstance(state.get("elo"), dict)
                or set(state["registry"]) != {"countries", "located_countries", "host_codes", "host_names", "vocabulary"}):
            raise ValueError("Incomplete or unsupported country context state")
        context = cls(**state["registry"])
        parameters = ("default_elo", "k_base", "pair_lambda", "alpha_player",
                      "score_margin_weight", "score_margin_reference")
        if any(state["elo"].get(key) != getattr(context.engine, key) for key in parameters):
            raise ValueError("Country Elo parameters differ from the supported policy")
        context.engine = EloEngine.from_state(state["elo"])
        return context
