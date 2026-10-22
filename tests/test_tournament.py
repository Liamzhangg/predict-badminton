from __future__ import annotations

import copy
import json
import pickle
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from predict_badminton.features import FeatureBuilder
from predict_badminton.tournament import forecast, propagate, validate_draw


class FixedModel:
    def predict_proba(self, frame):
        return np.tile([0.5, 0.5], (len(frame), 1))


class FixedCalibrator:
    def predict(self, probabilities):
        return np.full(len(probabilities), 0.75)


def draw():
    return {"tournament_name": "Example Open", "tournament_code": "example", "tournament_tier": 3,
            "entries": [{"id": str(i), "name": f"Player {i}", "players": [{"id": str(i)}]} for i in range(4)],
            "rounds": [{"name": "SF", "matches": [{"time_utc": "2030-01-01 10:00:00"}] * 2},
                       {"name": "Final", "matches": [{"time_utc": "2030-01-02 10:00:00"}]}]}


class TournamentTests(unittest.TestCase):
    def test_balanced_draw_conserves_probability(self):
        advancement, matches = propagate(32, [0.5] * 496)
        np.testing.assert_allclose(advancement[:, -1], 1 / 32)
        np.testing.assert_allclose(advancement.sum(axis=0), [32, 16, 8, 4, 2, 1])
        self.assertTrue(np.all(np.diff(advancement, axis=1) <= 1e-12))
        for level in range(5):
            for slot in range(16 >> level):
                self.assertAlmostEqual(sum(m["meeting"] for m in matches if m["level"] == level and m["slot"] == slot), 1)

    def test_all_possible_routes_are_combined_not_just_favorites(self):
        advancement, matches = propagate(4, [0.7, 0.6, 0.8, 0.9, 0.4, 0.3])
        np.testing.assert_allclose(advancement[:, -1], [0.588, 0.108, 0.192, 0.112])
        self.assertAlmostEqual(matches[2]["meeting"], 0.42)
        self.assertEqual(matches[2]["left"], 0)
        self.assertEqual(matches[2]["right"], 2)

    def test_certain_outcomes_and_two_entry_draw(self):
        advancement, _ = propagate(4, [1.0] * 6)
        np.testing.assert_allclose(advancement[:, -1], [1, 0, 0, 0])
        advancement, _ = propagate(2, [0.3])
        np.testing.assert_allclose(advancement[:, -1], [0.3, 0.7])

    def test_invalid_probabilities_and_draw_sizes_are_rejected(self):
        for size, values in [(3, [0.5] * 3), (4, [0.5]), (2, [float("nan")]),
                             (2, [float("inf")]), (2, [-0.1]), (2, [1.1])]:
            with self.subTest(size=size, values=values), self.assertRaises(ValueError):
                propagate(size, values)

    def test_incomplete_or_repeated_player_ids_are_rejected(self):
        for players in ([], [{"id": ""}], [{"id": "1"}]):
            data = draw()
            data["entries"][0]["players"] = players
            with self.subTest(players=players), self.assertRaises(ValueError):
                validate_draw(data, "MS")
        with self.assertRaises(ValueError):
            validate_draw(draw(), "MD")

    def test_wrong_round_order_is_rejected(self):
        data = draw()
        data["rounds"].reverse()
        with self.assertRaises(ValueError):
            validate_draw(data, "MS")

    def test_forecast_never_changes_history_or_draw(self):
        data = draw()
        original = copy.deepcopy(data)
        history = FeatureBuilder()
        history.elo._last_match["known"] = datetime(2029, 1, 1)
        before = pickle.dumps(history.get_state())
        result = forecast(data, "MS", (FixedModel(), FixedCalibrator(), FixedCalibrator(), history))
        self.assertEqual(pickle.dumps(history.get_state()), before)
        self.assertEqual(data, original)
        self.assertEqual(result["table"]["Title"].tolist(), [25] * 4)
        self.assertEqual(result["table"]["History"].tolist(), [0] * 4)

    def test_probability_method_is_explicit_and_calibrators_are_used(self):
        bundle = (FixedModel(), FixedCalibrator(), FixedCalibrator(), FeatureBuilder())
        for method in ["Platt", "Isotonic"]:
            result = forecast(draw(), "MS", bundle, method)
            self.assertEqual(result["matchups"][0]["win"], 0.75)
        with self.assertRaises(ValueError):
            forecast(draw(), "MS", bundle, "unknown")

    def test_past_fixtures_cannot_use_future_history(self):
        history = FeatureBuilder()
        history.elo._last_match["known"] = datetime(2030, 1, 1, 10)
        with self.assertRaisesRegex(ValueError, "cutoff"):
            forecast(draw(), "MS", (FixedModel(), None, None, history))

    def test_french_open_snapshot_has_all_official_draw_slots(self):
        path = Path(__file__).resolve().parents[1] / "data/draws/french-open-2026.json"
        data = json.loads(path.read_text())
        self.assertEqual(set(data["draws"]), {"MS", "WS", "MD", "WD", "XD"})
        for discipline, event in data["draws"].items():
            validate_draw(event, discipline)
            self.assertEqual(len(event["entries"]), 32)
            self.assertEqual(sum(len(r["matches"]) for r in event["rounds"]), 31)
            self.assertTrue(event["source_sha256"])
        self.assertEqual(data["draws"]["MS"]["entries"][0]["players"][0]["id"], "64032")
        self.assertEqual(data["draws"]["MS"]["entries"][1]["players"][0]["id"], "61628")
        # French clocks change on finals day; 11:00 Paris is 10:00 UTC, not 09:00.
        self.assertEqual(data["draws"]["MS"]["rounds"][-1]["matches"][0]["time_utc"], "2026-10-25 10:00:00")

    def test_refreshed_snapshot_contains_only_confirmed_survivors(self):
        root = Path(__file__).resolve().parents[1] / "data/draws"
        original = json.loads((root / "french-open-2026.json").read_text())
        current = json.loads((root / "french-open-2026-after-r32.json").read_text())
        self.assertEqual(current["results_summary"]["completed"], 80)
        self.assertEqual(current["results_summary"]["played"], 78)
        self.assertEqual(current["results_summary"]["walkovers"], 2)
        for discipline, event in current["draws"].items():
            validate_draw(event, discipline)
            self.assertEqual(len(event["entries"]), 16)
            self.assertEqual([r["name"] for r in event["rounds"]], ["R16", "QF", "SF", "Final"])
            self.assertEqual(sum(len(r["matches"]) for r in event["rounds"]), 15)
            self.assertTrue({e["id"] for e in event["entries"]} < {e["id"] for e in original["draws"][discipline]["entries"]})
            self.assertTrue(all(not m["time_estimated"] and m["time_utc"].startswith("2026-10-22")
                                for m in event["rounds"][0]["matches"]))
        self.assertEqual(current["draws"]["MS"]["entries"][0]["players"][0]["id"], "61628")
        self.assertNotIn("64032", {p["id"] for e in current["draws"]["MS"]["entries"] for p in e["players"]})


if __name__ == "__main__":
    unittest.main()
