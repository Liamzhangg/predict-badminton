import copy
import unittest

from predict_badminton.slip import display_name, render_slip, start_label


class SlipRenderingTests(unittest.TestCase):
    def setUp(self):
        self.row = {"quote_id": "SYNTHETIC", "match_id": "synthetic-match", "selection_id": "synthetic:1",
                    "discipline": "MS", "round_name": "R32", "model_ev_percent": "0.000001",
                    "model_probability": "0.5", "american_odds": 100, "theoretical_positive_ev": False,
                    "source_selection_name": "Synthetic...", "selection_name": "Synthetic Player One",
                    "source_opponent_name": "Synthetic Player Two", "opponent_name": "Synthetic Player Two",
                    "selection_team_country_missing": True, "opponent_team_country_missing": False,
                    "low_player_history": True, "official_start_time_utc": "",
                    "forecast_reference_time_utc": "2030-01-02T01:00:00+00:00"}
        self.fixture = {"match_id": "synthetic-match", "time_estimated": True}
        self.forecast = {"prediction_generated_at_utc": "2030-01-01T00:00:00+00:00",
                         "selections": [self.row], "fixtures": [self.fixture]}
        self.coverage = {"verified_matches": 1, "verified_prices": 2, "missing_odds": []}

    def test_tiny_positive_edges_render_once_without_mutation(self):
        before = copy.deepcopy((self.forecast, self.coverage))
        report = render_slip(self.forecast, self.coverage)
        self.assertEqual(report.count("| SYNTHETIC |"), 1)
        self.assertIn("1 single paper plays", report)
        self.assertIn("Country input missing", report)
        self.assertIn("Low player history", report)
        self.assertEqual((self.forecast, self.coverage), before)

    def test_nonpositive_edges_are_excluded(self):
        for value in ("0", "-0.001"):
            self.row["model_ev_percent"] = value
            self.assertNotIn("| SYNTHETIC |", render_slip(self.forecast, self.coverage))

    def test_estimated_and_indicative_times_are_distinct(self):
        self.assertEqual(start_label(self.row, self.fixture), "Jan 1, 8:00 PM (est.)")
        self.row["official_start_time_utc"] = self.row["forecast_reference_time_utc"]
        self.assertEqual(start_label(self.row, self.fixture), "Jan 1, 8:00 PM (indic.)")
        self.fixture["time_estimated"] = False
        self.assertEqual(start_label(self.row, self.fixture), "Jan 1, 8:00 PM")

    def test_truncated_names_use_saved_full_name(self):
        self.assertEqual(display_name(self.row, "selection"), "Synthetic Player One")
        self.assertNotIn("Synthetic...", render_slip(self.forecast, self.coverage))


if __name__ == "__main__":
    unittest.main()
