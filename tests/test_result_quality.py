from datetime import UTC, datetime, timedelta
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fetch_bwf import main as fetch_main
from predict_badminton.data_parser import load_all_matches
from predict_badminton.data_sources.observations import (
    observations_before, record_results, result_version, verify_legacy_receipts,
)
from tests.test_observations import completed_match


class ResultQualityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db = self.root / "receipts.sqlite3"
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.when = datetime(2026, 11, 11, 10, tzinfo=UTC)

    def record(self, match, at=None, body=None):
        return record_results([match], source_url="https://example.test/results",
                              response_sha256=hashlib.sha256(body or b"source").hexdigest(),
                              observed_at=at or self.when, path=self.db)

    def legacy_receipt(self, match, at=None):
        body = json.dumps([[match]]).encode()
        self.record({**match, "matchStatus": "F"}, at, body)
        fingerprint, payload = result_version({**match, "matchStatus": "F"})
        snapshot = json.loads(payload)
        snapshot.pop("matchStatus")
        old_payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        old_hash = hashlib.sha256(old_payload.encode()).hexdigest()
        with sqlite3.connect(self.db) as connection:
            stamp = connection.execute("SELECT MAX(observed_at) FROM responses").fetchone()[0]
            response_id = connection.execute("SELECT MAX(response_id) FROM responses").fetchone()[0]
            connection.execute("INSERT OR IGNORE INTO result_versions VALUES (?,?,?,?)",
                               (str(match["id"]), old_hash, stamp, old_payload))
            connection.execute("UPDATE response_results SET result_sha256=? WHERE response_id=?",
                               (old_hash, response_id))
            connection.execute("DELETE FROM result_versions WHERE result_sha256=? AND NOT EXISTS (SELECT 1 FROM response_results WHERE result_sha256=?)",
                               (fingerprint, fingerprint))
        source = self.root / f"response-{response_id}.json"
        source.write_bytes(body)
        return source

    def receipt(self, at=None):
        return observations_before(at or self.when + timedelta(hours=1), self.db)["m1"]

    def test_live_missing_and_unknown_statuses_cannot_create_a_receipt(self):
        for status in ("L", "N", None, "", "unknown", {"value": "L"}):
            match = {**completed_match(), "matchStatus": status}
            with self.subTest(status=status):
                self.assertIsNone(result_version(match))
                self.assertEqual(self.record(match), (0, 0))
        self.assertFalse(self.db.exists())

    def test_live_to_final_cannot_reuse_a_provisional_timestamp(self):
        live = {**completed_match(), "matchStatus": "L"}
        self.record(live, self.when - timedelta(minutes=30))
        self.record(completed_match())
        self.assertIsNone(self.receipt(self.when - timedelta(minutes=10)))
        self.assertEqual(self.receipt()["available_at"], self.when.replace(tzinfo=None))
        payload = json.loads(result_version(completed_match())[1])
        self.assertEqual(payload["matchStatus"], "F")

    def test_explicit_live_reopening_invalidates_the_previous_completion_episode(self):
        self.record(completed_match())
        self.record({**completed_match(), "matchStatus": "L"}, self.when + timedelta(minutes=1))
        self.assertIsNone(self.receipt(self.when + timedelta(minutes=2)))
        self.record(completed_match(), self.when + timedelta(minutes=3))
        self.assertEqual(self.receipt()["available_at"], (self.when + timedelta(minutes=3)).replace(tzinfo=None))

    def test_legacy_receipts_fail_closed_until_exact_final_response_is_verified(self):
        source = self.legacy_receipt(completed_match())
        self.assertIsNone(self.receipt())
        with sqlite3.connect(self.db) as connection:
            before = connection.execute("SELECT * FROM result_versions").fetchall()
        report = verify_legacy_receipts([source], path=self.db)
        self.assertEqual(report["added_completion_receipts"], 1)
        self.assertEqual(self.receipt()["available_at"], self.when.replace(tzinfo=None))
        self.assertEqual(verify_legacy_receipts([source], path=self.db)["added_completion_receipts"], 0)
        with sqlite3.connect(self.db) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0], 1)
            self.assertTrue(set(before).issubset(set(connection.execute("SELECT * FROM result_versions"))))

    def test_live_legacy_payload_and_forged_final_body_cannot_prove_completion(self):
        source = self.legacy_receipt({**completed_match(), "matchStatus": "L"})
        self.assertEqual(verify_legacy_receipts([source], path=self.db)["added_completion_receipts"], 0)
        source.write_text(json.dumps([[completed_match()]]))
        self.assertEqual(verify_legacy_receipts([source], path=self.db)["added_completion_receipts"], 0)
        self.assertIsNone(self.receipt())

    def test_unverified_legacy_correction_breaks_the_old_availability_episode(self):
        source = self.legacy_receipt(completed_match())
        verify_legacy_receipts([source], path=self.db)
        self.legacy_receipt({**completed_match(), "winner": 2}, self.when + timedelta(minutes=1))
        self.assertIsNone(self.receipt(self.when + timedelta(minutes=2)))
        self.record(completed_match(), self.when + timedelta(minutes=3))
        self.assertEqual(self.receipt()["available_at"], (self.when + timedelta(minutes=3)).replace(tzinfo=None))

    def test_conflicting_live_and_final_copies_cannot_verify_a_legacy_response(self):
        source = self.legacy_receipt(completed_match())
        body = json.dumps([[completed_match(), {**completed_match(), "matchStatus": "L"}]]).encode()
        source.write_bytes(body)
        with sqlite3.connect(self.db) as connection:
            connection.execute("UPDATE responses SET response_sha256=?", (hashlib.sha256(body).hexdigest(),))
        self.assertEqual(verify_legacy_receipts([source], path=self.db)["added_completion_receipts"], 0)
        self.assertIsNone(self.receipt())

    def test_explicit_verification_cli_reports_the_updated_database_counts(self):
        self.legacy_receipt(completed_match())
        output = io.StringIO()
        with patch("sys.argv", ["fetch_bwf.py", "observations", "--db", str(self.db),
                                "--cache-dir", str(self.root), "--verify-cached-completions"]), redirect_stdout(output):
            self.assertEqual(fetch_main(), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["result_versions"], 2)
        self.assertEqual(report["completion_verification"]["added_completion_receipts"], 1)

    def test_valid_played_retirements_do_not_require_a_two_game_score(self):
        retired = {**completed_match(), "scoreStatusValue": "Retired", "scoreStatus": 2,
                   "score": [{"home": 7, "away": 5}]}
        self.assertEqual(self.record(retired), (1, 1))

    def test_quarantined_match_and_duplicate_alias_never_enter_training(self):
        original = {**completed_match(), "id": "1304511"}
        alias = {**original, "id": "different-id"}
        (self.raw / "matches.json").write_text(json.dumps([[original, alias]]))
        matches, drops, _ = load_all_matches(self.raw)
        self.assertEqual(matches, [])
        self.assertEqual(drops["quarantined_result"], 2)

    def test_fixture_signature_blocks_an_alias_without_the_original_id(self):
        path = self.raw / "matches.json"
        path.write_text(json.dumps([[completed_match()]]))
        matches, _, _ = load_all_matches(self.raw)
        key = "fixture:" + matches[0]["fixture_id"]
        path.write_text(json.dumps([[{**completed_match(), "id": "alias-only"}]]))
        with patch("predict_badminton.data_parser._quarantined_results", return_value=frozenset({key})):
            matches, drops, _ = load_all_matches(self.raw, deduplicate=False)
        self.assertEqual(matches, [])
        self.assertEqual(drops["quarantined_result"], 1)

    def test_untracked_live_winner_and_its_final_alias_are_excluded(self):
        original = {**completed_match(), "matchStatus": "L"}
        alias = {**original, "id": "alias", "matchStatus": "F"}
        (self.raw / "matches.json").write_text(json.dumps([[original, alias]]))
        matches, drops, _ = load_all_matches(self.raw)
        self.assertEqual(matches, [])
        self.assertEqual(drops["unfinished_result"], 2)


if __name__ == "__main__":
    unittest.main()
