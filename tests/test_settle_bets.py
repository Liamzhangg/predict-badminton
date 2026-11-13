import copy
from datetime import UTC, datetime
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import settle_bets


def ledger():
    return {"currency": "CAD", "batches": [{"name": "Example Open", "date": "2026-11-14", "round": "SF",
        "tournament_code": "official-code", "raw_file": "data/raw/bwf_matches/example.json",
        "source": "outputs/frozen.json", "bets": [{"match_id": "123", "selection_id": "123:1",
            "discipline": "MS", "selection": "A", "opponent": "B", "selection_player_ids": "1",
            "opponent_player_ids": "2", "american_odds": 150, "model_probability": "0.6123456789012345",
            "model_ev_percent": "53.086419725308625", "stake": "10.00", "status": "pending",
            "profit": None, "returned": None, "score": ""}]}]}


def result():
    return {"id": 123, "tournamentCode": "official-code", "eventName": "MS", "roundName": "SF",
            "team1": {"players": [{"id": "1"}]}, "team2": {"players": [{"id": "2"}]},
            "matchStatus": "F", "scoreStatus": 0, "scoreStatusValue": "Normal", "winner": 1,
            "score": [{"home": 21, "away": 19}, {"home": 21, "away": 17}]}


class PaperSettlementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = ledger()
        self.bet = self.data["batches"][0]["bets"][0]
        self.now = datetime(2026, 11, 14, 18, tzinfo=UTC)

    def raw(self, *records):
        path = self.root / self.data["batches"][0]["raw_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([list(records)]))

    def settle(self):
        settle_bets.validate(self.data)
        return settle_bets.settle_pending(self.data, self.root, self.now)

    def test_positive_odds_and_frozen_values(self):
        frozen = {k: self.bet[k] for k in ("american_odds", "model_probability", "stake", "selection_player_ids")}
        self.raw(result())
        self.assertEqual(self.settle(), 1)
        self.assertEqual((self.bet["profit"], self.bet["returned"]), ("15.00", "25.00"))
        self.assertEqual(frozen, {k: self.bet[k] for k in frozen})

    def test_negative_odds_round_each_payout(self):
        self.bet["american_odds"] = -163
        self.raw(result())
        self.settle()
        self.assertEqual((self.bet["profit"], self.bet["returned"]), ("6.13", "16.13"))

    def test_loss_returns_zero(self):
        match = result()
        match["winner"] = 2
        match["score"] = [{"home": 19, "away": 21}, {"home": 17, "away": 21}]
        self.raw(match)
        self.settle()
        self.assertEqual((self.bet["status"], self.bet["profit"], self.bet["returned"]), ("loss", "-10.00", "0.00"))

    def test_swapped_doubles_and_player_order_use_ids_not_suffix(self):
        self.bet.update(discipline="MD", selection_player_ids="1|3", opponent_player_ids="2|4")
        match = result()
        match.update(eventName="MD", winner=2, team1={"players": [{"id": "4"}, {"id": "2"}]},
                     team2={"players": [{"id": "3"}, {"id": "1"}]},
                     score=[{"home": 19, "away": 21}, {"home": 17, "away": 21}])
        self.raw(match)
        self.settle()
        self.assertEqual(self.bet["status"], "win")
        self.assertEqual(self.bet["score"], "21-19, 21-17")

    def test_missing_and_live_results_remain_pending(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(self.settle(), 0)
        self.assertEqual(self.data, before)
        match = result()
        match.update(matchStatus="L", winner=0, score=[{"home": 12, "away": 9}])
        self.raw(match)
        self.assertEqual(self.settle(), 0)
        self.assertEqual(self.data, before)

    def test_exceptions_require_review_without_assuming_void_rules(self):
        for changes in ({"scoreStatusValue": "Retired"}, {"scoreStatus": 1, "scoreStatusValue": "Walkover"},
                        {"scoreStatusValue": "Cancelled"}, {"score": [{"home": 21, "away": 19}]},
                        {"tournamentCode": "wrong"}, {"team2": {"players": [{"id": "9"}]}},
                        {"eventName": "WS"}, {"roundName": "Final"}, {"winner": True}):
            with self.subTest(changes=changes):
                self.bet.update(status="pending", profit=None, returned=None)
                self.raw({**result(), **changes})
                self.assertEqual(self.settle(), 0)
                self.assertEqual(self.bet["status"], "review")
                self.assertIsNone(self.bet["profit"])
                self.assertIsNone(self.bet["returned"])

    def test_conflicting_duplicates_are_not_chosen_arbitrarily(self):
        self.raw(result(), {**result(), "matchStatus": "L", "winner": 0})
        self.settle()
        self.assertEqual(self.bet["status"], "review")
        self.assertIn("Conflicting", self.bet["note"])

    def test_identical_duplicates_do_not_double_count(self):
        self.raw(result(), result())
        self.assertEqual(self.settle(), 1)
        self.assertEqual(settle_bets.totals([self.bet])["profit"], Decimal(15))

    def test_repeated_run_leaves_settled_record_unchanged(self):
        self.raw(result())
        self.settle()
        before = copy.deepcopy(self.data)
        self.assertEqual(self.settle(), 0)
        self.assertEqual(before, self.data)

    def test_review_can_resolve_when_clean_result_arrives(self):
        self.raw({**result(), "scoreStatusValue": "Retired"})
        self.settle()
        self.raw(result())
        self.assertEqual(self.settle(), 1)
        self.assertEqual((self.bet["status"], self.bet["note"]), ("win", ""))

    def test_duplicate_bet_and_invalid_frozen_values_are_rejected(self):
        self.data["batches"][0]["bets"].append(copy.deepcopy(self.bet))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            settle_bets.validate(self.data)
        self.data["batches"][0]["bets"].pop()
        for field, value in (("model_probability", "NaN"), ("stake", "0"), ("american_odds", 99)):
            original = self.bet[field]
            self.bet[field] = value
            with self.assertRaises(ValueError):
                settle_bets.validate(self.data)
            self.bet[field] = original

    def test_pending_stakes_do_not_enter_settled_roi(self):
        pending = copy.deepcopy(self.bet)
        self.bet["american_odds"] = -163
        self.raw(result())
        self.settle()
        self.data["batches"][0]["bets"].append(pending)
        report = settle_bets.render_report(self.data)
        self.assertIn("Settled profit: +$6.13", report)
        self.assertIn("Settled ROI: 61.30%", report)
        self.assertIn("Unsettled stake: $10.00", report)
        self.assertIn("Total paper stakes: $20.00", report)

    def test_cli_dry_run_and_idempotent_writes(self):
        output = self.root / "outputs"
        output.mkdir()
        path, report = output / "profits.json", output / "profits.md"
        path.write_text(json.dumps(self.data))
        report.write_text("Original report\n")
        self.raw(result())
        original = (path.read_bytes(), report.read_bytes())
        with patch.object(settle_bets, "ROOT", self.root), redirect_stdout(io.StringIO()):
            self.assertEqual(settle_bets.main(["--dry-run"]), 0)
            self.assertEqual((path.read_bytes(), report.read_bytes()), original)
            self.assertEqual(settle_bets.main([]), 0)
            saved = (path.read_bytes(), report.read_bytes(), path.stat().st_mtime_ns, report.stat().st_mtime_ns)
            self.assertEqual(settle_bets.main([]), 0)
            self.assertEqual((path.read_bytes(), report.read_bytes(), path.stat().st_mtime_ns, report.stat().st_mtime_ns), saved)

    def test_corrupt_results_do_not_write_partial_updates(self):
        output = self.root / "outputs"
        output.mkdir()
        path, report = output / "profits.json", output / "profits.md"
        path.write_text(json.dumps(self.data))
        report.write_text("Original report\n")
        self.raw(result())
        (self.root / self.data["batches"][0]["raw_file"]).write_text("broken JSON")
        original = (path.read_bytes(), report.read_bytes())
        with patch.object(settle_bets, "ROOT", self.root), redirect_stderr(io.StringIO()):
            self.assertEqual(settle_bets.main([]), 1)
        self.assertEqual((path.read_bytes(), report.read_bytes()), original)
