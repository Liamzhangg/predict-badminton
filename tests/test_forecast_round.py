from contextlib import redirect_stderr, redirect_stdout
import copy
import csv
from datetime import UTC, datetime
from decimal import Decimal
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import forecast_round
from predict_badminton.data_sources.observations import record_results
from predict_badminton.features import FeatureBuilder
from predict_badminton.forecast_history import refresh_history
from predict_badminton.temporal import OBSERVED_AVAILABILITY_POLICY
from tests.test_observations import completed_match
from tests.test_settle_bets import ledger, result
from tests.test_tournament import FixedCalibrator, FixedModel


class SeedModel(FixedModel):
    def predict_proba(self, frame):
        chance = np.where(frame["team1_seeded"] == 1, 0.7123456789012345, 0.2876543210987655)
        return np.column_stack((1 - chance, chance))


class RoundWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch("forecast_round.REPO_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        clock_patch = patch("forecast_round.datetime", wraps=datetime)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.now = datetime(2026, 11, 14, 18, tzinfo=UTC)
        self.clock.now.return_value = self.now
        self.bundle = self.root / "outputs/runs/observed"
        self.bundle.mkdir(parents=True)
        history = FeatureBuilder(result_availability_policy=OBSERVED_AVAILABILITY_POLICY)
        history.save(self.bundle / "feature_state.pkl")
        self.model = SeedModel()
        self.model.feature_policy_ = history.model_policy
        calibrator = FixedCalibrator()
        calibrator.feature_policy_ = history.model_policy
        for name, value in (("lgbm_model.pkl", self.model), ("platt_calibrator.pkl", calibrator),
                            ("isotonic_calibrator.pkl", calibrator)):
            (self.bundle / name).write_bytes(pickle.dumps(value))
        (self.bundle / "feature_list.json").write_text(json.dumps(history.feature_columns))
        raw = self.root / "data/raw/bwf_matches"
        raw.mkdir(parents=True)
        match = {**completed_match(), "matchStatus": "F"}
        (raw / "example.json").write_text(json.dumps([[match]]))
        (raw / "second.json").write_text("[]")
        record_results([match], source_url="https://example.com/results", response_sha256="source",
                       observed_at=datetime(2026, 11, 11, 16, tzinfo=UTC),
                       path=self.root / "data/manifests/bwf_result_observations.sqlite3")
        self.fixtures = [{"match_id": "123", "match_time": "2030-01-01T10:00:00Z", "discipline": "MS",
                          "team1_player_ids": "1", "team2_player_ids": "2", "round_name": "SF",
                          "tournament_name": "Example Open", "tournament_id": "100", "tournament_code": "example",
                          "tournament_tier": "5", "team1_seed": "1", "team2_seed": "0",
                          "raw_file": "data/raw/bwf_matches/example.json"}]
        self.fixtures.append({**self.fixtures[0], "match_id": "456", "discipline": "MD",
                              "team1_player_ids": "3|4", "team2_player_ids": "5|6", "tournament_id": "200",
                              "tournament_name": "Second Open", "tournament_code": "second",
                              "team1_seed": "0", "team2_seed": "1", "raw_file": "data/raw/bwf_matches/second.json"})
        self.quotes = []
        for fix in self.fixtures:
            for side, odds in ((1, 150), (2, -110)):
                self.quotes.append({"match_id": fix["match_id"], "selection_id": f"{fix['match_id']}:{side}",
                                    "discipline": fix["discipline"], "round_name": fix["round_name"],
                                    "tournament_id": fix["tournament_id"], "selection_player_ids": fix[f"team{side}_player_ids"],
                                    "opponent_player_ids": fix[f"team{3-side}_player_ids"],
                                    "selection_name": f"Team {side}", "opponent_name": f"Team {3-side}",
                                    "american_odds": str(odds), "official_start_time_utc": fix["match_time"],
                                    "recorded_at_utc": "2026-11-14T17:00:00Z", "captured_at_utc": "",
                                    "sportsbook": "bet365", "jurisdiction": "Ontario", "source_images": "quote.png"})
        self.fixture_path, self.odds_path = self.root / "fixtures.csv", self.root / "odds.csv"
        self.write_inputs()
        self.saved = ledger()
        self.saved["batches"][0]["bets"][0].update(match_id="999", selection_id="999:1",
                                               selection_player_ids="91", opponent_player_ids="92")
        self.ledger_path = self.root / "outputs/profits.json"
        self.ledger_path.write_text(json.dumps(self.saved, indent=4) + "\n")
        (self.root / "outputs/profits.md").write_text("Existing report\n")
        (self.root / "outputs/saved_predictions.csv").write_text("Existing predictions\n")

    def write_csv(self, path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)

    def write_inputs(self):
        self.write_csv(self.fixture_path, self.fixtures)
        self.write_csv(self.odds_path, self.quotes)

    def run_round(self, name="run", **kwargs):
        return forecast_round.run_round(self.fixture_path, [self.odds_path], self.root / "outputs" / name,
                                        self.bundle, **kwargs)

    def snapshot(self):
        return {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def test_default_refreshes_once_and_preserves_every_existing_file(self):
        before = self.snapshot()
        with patch("predict_badminton.inference.refresh_history", wraps=refresh_history) as refresh:
            forecast = self.run_round()
        self.assertEqual(refresh.call_count, 1)
        self.assertEqual(refresh.call_args.kwargs["as_of"], self.now)
        self.assertEqual(refresh.call_args.kwargs["target_match_ids"], {"123", "456"})
        self.assertTrue(forecast["history_refreshed"])
        self.assertEqual(len(forecast["fixtures"]), 2)
        self.assertEqual(len(forecast["selections"]), 4)
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
        output = self.root / "outputs/run"
        self.assertEqual({p.name for p in output.iterdir()}, {"forecast.json", "comparison.csv", "pending_bets.json"})
        draft = json.loads((output / "pending_bets.json").read_text())
        self.assertEqual(len(draft["batches"]), 2)
        self.assertEqual([b["bets"][0]["stake"] for b in draft["batches"]], ["10.00", "10.00"])
        with (output / "comparison.csv").open() as handle:
            comparison = list(csv.DictReader(handle))
        for row, exported in zip(forecast["selections"], comparison, strict=True):
            self.assertEqual(exported["model_probability"], row["model_probability"])
            self.assertEqual(exported["american_odds"], str(row["american_odds"]))
            ev = Decimal(row["model_probability"]) * Decimal(row["decimal_odds"]) - 1
            self.assertEqual(Decimal(row["model_ev_percent"]), ev * 100)
            self.assertEqual(row["theoretical_positive_ev"], ev > 0)
        self.assertGreater(len(forecast["selections"][0]["model_probability"]), 10)

    def test_record_flag_appends_without_changing_original_bets_and_rejects_reruns(self):
        before = self.snapshot()
        self.run_round(record_bets=True)
        updated = json.loads(self.ledger_path.read_text())
        self.assertEqual(updated["batches"][:len(self.saved["batches"])], self.saved["batches"])
        self.assertEqual(len(updated["batches"]), 3)
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()
                            if p.name not in {"profits.json", "profits.md"}))
        recorded = self.snapshot()
        with self.assertRaisesRegex(ValueError, "already recorded"):
            self.run_round("duplicate", record_bets=True)
        self.assertFalse((self.root / "outputs/duplicate").exists())
        self.assertTrue(all(p.read_bytes() == content for p, content in recorded.items()))

    def test_new_pending_bets_work_with_existing_settlement(self):
        self.run_round(record_bets=True)
        saved = json.loads(self.ledger_path.read_text())
        frozen = copy.deepcopy(saved["batches"][1]["bets"][0])
        match = {**result(), "tournamentCode": "example", "roundName": "SF"}
        (self.root / "data/raw/bwf_matches/example.json").write_text(json.dumps([[match]]))
        self.assertEqual(forecast_round.settle_bets.settle_pending(saved, self.root, self.now), 1)
        settled = saved["batches"][1]["bets"][0]
        self.assertEqual((settled["status"], settled["profit"]), ("win", "15.00"))
        self.assertTrue(all(settled[key] == frozen[key] for key in (
            "american_odds", "model_probability", "model_ev_percent", "stake", "selection_player_ids")))

    def test_duplicate_detection_includes_alias_ids_and_reversed_sides(self):
        existing = copy.deepcopy(self.saved)
        batch = existing["batches"][0]
        repeated = copy.deepcopy(batch)
        bet = repeated["bets"][0]
        bet.update(match_id="alias", selection_id="alias:2", selection_player_ids="92", opponent_player_ids="91")
        with self.assertRaisesRegex(ValueError, "already recorded"):
            forecast_round.merge_pending(existing, {"currency": "CAD", "batches": [repeated]})
        self.assertEqual(existing, self.saved)

    def test_existing_or_external_output_directories_are_rejected(self):
        for output in (self.bundle, self.root / "data/new", self.root / "outside"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "new directory inside outputs"):
                forecast_round.run_round(self.fixture_path, [self.odds_path], output, self.bundle)

    def test_incomplete_and_inconsistent_quotes_fail_without_writes(self):
        original = copy.deepcopy(self.quotes)
        for changes in ({"selection_player_ids": "9"}, {"opponent_player_ids": "1"},
                        {"selection_player_ids": "1|1"}, {"selection_id": "123:2"}, {"match_id": "unknown"},
                        {"tournament_id": "999"}, {"round_name": "QF"}, {"discipline": "WS"},
                        {"official_start_time_utc": "2030-01-01T11:00:00Z"}, {"american_odds": "99"},
                        {"american_odds": "nan"}, {"recorded_at_utc": "2031-01-01T00:00:00Z"},
                        {"captured_at_utc": "2031-01-01T00:00:00Z"}, {"selection_name": ""},
                        {"sportsbook": "Other"}, {"jurisdiction": "Other"}):
            with self.subTest(changes=changes):
                self.quotes = copy.deepcopy(original)
                self.quotes[0].update(changes)
                self.write_inputs()
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    self.run_round()
                self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
                self.assertFalse((self.root / "outputs/run").exists())

    def test_invalid_fixtures_are_not_silently_dropped(self):
        original = copy.deepcopy(self.fixtures)
        for changes in ({"match_time": "2026-11-14T17:00:00Z"}, {"match_time": "2030-01-01 10:00:00"},
                        {"team1_player_ids": "1|1"}, {"team1_player_ids": "0"}, {"team2_player_ids": "1"},
                        {"discipline": "OTHER"}, {"round_name": "QF"}, {"raw_file": "../../outside.json"},
                        {"winner": "1"}, {"score_status": "1"}, {"status": "Walkover"}):
            with self.subTest(changes=changes):
                self.fixtures = copy.deepcopy(original)
                self.fixtures[0].update(changes)
                self.write_inputs()
                with self.assertRaises(ValueError):
                    self.run_round()
                self.assertFalse((self.root / "outputs/run").exists())

    def test_duplicate_quotes_and_fixtures_are_rejected(self):
        self.quotes.append(copy.deepcopy(self.quotes[0]))
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "duplicate quote"):
            self.run_round()
        self.quotes.pop()
        self.fixtures.append(copy.deepcopy(self.fixtures[0]))
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "duplicate fixture"):
            self.run_round()

    def test_partial_prices_and_reordered_doubles_ids_are_supported(self):
        self.quotes = [self.quotes[-1]]
        self.quotes[0].update(selection_player_ids="6|5", opponent_player_ids="4|3")
        self.write_inputs()
        forecast = self.run_round()
        self.assertEqual(len(forecast["fixtures"]), 2)
        self.assertEqual(len(forecast["selections"]), 1)
        draft = json.loads((self.root / "outputs/run/pending_bets.json").read_text())
        self.assertEqual(draft["batches"][0]["bets"][0]["selection_player_ids"], "6|5")

    def test_multiple_price_files_are_combined(self):
        other = self.root / "other.csv"
        self.write_csv(other, self.quotes[2:])
        self.write_csv(self.odds_path, self.quotes[:2])
        forecast = forecast_round.run_round(self.fixture_path, [self.odds_path, other], self.root / "outputs/run", self.bundle)
        self.assertEqual(len(forecast["selections"]), 4)
        self.assertEqual(len(forecast["input_sha256"]), 5)

    def test_missing_columns_and_empty_csvs_fail_before_writing(self):
        for content in ("match_id,selection_id\n123,123:1\n", "", "match_id,match_id\n123,123\n"):
            with self.subTest(content=content):
                self.odds_path.write_text(content)
                with self.assertRaises(ValueError):
                    self.run_round()
                self.assertFalse((self.root / "outputs/run").exists())

    def test_preview_does_not_need_an_existing_profit_ledger(self):
        self.ledger_path.unlink()
        self.run_round()
        self.assertFalse(self.ledger_path.exists())
        with self.assertRaises(FileNotFoundError):
            self.run_round("record", record_bets=True)
        self.assertFalse((self.root / "outputs/record").exists())

    def test_profit_ledger_change_is_not_overwritten(self):
        render = forecast_round.settle_bets.render_report

        def external_update(value):
            report = render(value)
            self.ledger_path.write_text(self.ledger_path.read_text() + "\n")
            return report

        original = self.ledger_path.read_text()
        with patch("forecast_round.settle_bets.render_report", side_effect=external_update):
            with self.assertRaisesRegex(ValueError, "ledger changed"):
                self.run_round(record_bets=True)
        self.assertEqual(self.ledger_path.read_text(), original + "\n")
        self.assertEqual((self.root / "outputs/profits.md").read_text(), "Existing report\n")
        self.assertFalse((self.root / "outputs/run").exists())

    def test_zero_ev_does_not_become_a_bet_or_rewrite_the_ledger(self):
        model = FixedModel()
        model.feature_policy_ = self.model.feature_policy_
        (self.bundle / "lgbm_model.pkl").write_bytes(pickle.dumps(model))
        for row in self.quotes:
            row["american_odds"] = "100" if row["selection_id"].endswith(":1") else "-100"
        self.write_inputs()
        before = self.snapshot()
        self.run_round(record_bets=True)
        draft = json.loads((self.root / "outputs/run/pending_bets.json").read_text())
        self.assertEqual(draft["batches"], [])
        self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))

    def test_fixture_starting_during_refresh_or_changed_input_prevents_saving(self):
        late = datetime(2031, 1, 1, tzinfo=UTC)
        self.clock.now.side_effect = [self.now, late, late]
        with self.assertRaisesRegex(ValueError, "started while"):
            self.run_round()
        self.assertFalse((self.root / "outputs/run").exists())
        self.clock.now.side_effect = None
        self.clock.now.return_value = self.now
        original = forecast_round.predict_probabilities

        def change_input(*args):
            result = original(*args)
            self.odds_path.write_text(self.odds_path.read_text() + "\n")
            return result

        with patch("forecast_round.predict_probabilities", side_effect=change_input):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.run_round()
        self.assertFalse((self.root / "outputs/run").exists())

    def test_cli_uses_selected_bundle_and_does_not_record_by_default(self):
        with patch("forecast_round.forecast_bundle_directory", return_value=self.bundle) as selected:
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(forecast_round.main(["--fixtures-csv", str(self.fixture_path),
                                                     "--odds-csv", str(self.odds_path)]), 0)
        selected.assert_called_once_with()
        self.assertIn("Draft only", output.getvalue())
        self.assertEqual(json.loads(self.ledger_path.read_text()), self.saved)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(forecast_round.main(["--fixtures-csv", "missing.csv", "--odds-csv", "missing.csv"]), 1)


if __name__ == "__main__":
    unittest.main()
