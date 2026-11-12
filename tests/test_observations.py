from __future__ import annotations

from copy import deepcopy
from datetime import UTC, date, datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fetch_bwf import main
from predict_badminton.data_sources.bwf import fetch_url, normalize_source_file
from predict_badminton.data_sources.bwf_backfill import API_BASE, _api_json
from predict_badminton.data_sources.observations import observation_status, record_results


def completed_match() -> dict:
    return {
        "id": "m1", "tournamentCode": "100", "tournamentName": "Example Open",
        "eventName": "MS", "roundName": "R32", "winner": 1, "duration": 45,
        "matchTimeUtc": "2026-11-11 09:00:00", "scoreStatus": 0,
        "team1": {"countryCode": "CAN", "players": [{"id": "p1", "nameDisplay": "A"}]},
        "team2": {"countryCode": "USA", "players": [{"id": "p2", "nameDisplay": "B"}]},
        "score": [{"home": 21, "away": 15}, {"home": 21, "away": 19}],
    }


class ObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "observations.sqlite3"
        self.start = datetime(2026, 11, 11, 10, tzinfo=UTC)

    def record(self, matches: list[dict], at: datetime | None = None) -> tuple[int, int]:
        return record_results(matches, source_url="https://example.com/results", response_sha256="response-hash",
                              observed_at=at or self.start, path=self.db)

    def rows(self, query: str) -> list[tuple]:
        with sqlite3.connect(self.db) as connection:
            return connection.execute(query).fetchall()

    def test_no_database_for_absent_or_incomplete_results(self) -> None:
        self.assertEqual(observation_status(self.db)["matches"], 0)
        for winner in (None, 0, True, "1"):
            self.assertEqual(self.record([{**completed_match(), "winner": winner}]), (0, 0))
        self.assertFalse(self.db.exists())
        self.assertEqual(self.record([completed_match()]), (1, 1))
        self.assertEqual(self.rows("SELECT first_seen_at FROM result_versions"), [("2026-11-11T10:00:00.000000Z",)])

    def test_excludes_nonstandard_unplayed_and_unidentified_matches(self) -> None:
        changes = [
            {"id": None}, {"id": " "}, {"eventName": "OTHER"}, {"eventName": "MD"},
            {"tournamentName": "Para Badminton Open"}, {"tournamentName": "AirBadminton Open"},
            {"scoreStatus": 1}, {"scoreStatus": "1"}, {"scoreStatusValue": "Walkover"},
            {"team2": {"players": [{"id": "p1"}]}}, {"score": "21-15"},
        ]
        for change in changes:
            with self.subTest(change=change):
                self.assertEqual(self.record([{**completed_match(), **change}]), (0, 0))
        self.assertFalse(self.db.exists())

    def test_supports_all_five_disciplines_and_played_retirements(self) -> None:
        matches = []
        for discipline in ("MS", "WS", "MD", "WD", "XD"):
            match = {**completed_match(), "id": discipline, "eventName": discipline}
            if discipline in ("MD", "WD", "XD"):
                match = deepcopy(match)
                match["team1"]["players"].append({"id": "p3"})
                match["team2"]["players"].append({"id": "p4"})
            matches.append(match)
        matches.append({**completed_match(), "id": "retired", "scoreStatusValue": "Retired"})
        self.assertEqual(self.record(matches), (6, 6))

    def test_repeated_fetches_keep_first_seen_and_record_each_receipt(self) -> None:
        match = completed_match()
        self.assertEqual(self.record([match, deepcopy(match)]), (1, 1))
        self.assertEqual(self.record([match], self.start + timedelta(hours=6)), (1, 0))
        status = observation_status(self.db)
        self.assertEqual((status["responses"], status["matches"], status["result_versions"]), (2, 1, 1))
        self.assertEqual(status["last_observed_at"], "2026-11-11T16:00:00.000000Z")
        self.assertEqual(self.rows("SELECT first_seen_at FROM result_versions"), [("2026-11-11T10:00:00.000000Z",)])

    def test_cosmetic_fields_and_partner_order_do_not_change_result_version(self) -> None:
        original = completed_match()
        original["eventName"] = "MD"
        original["team1"]["players"].append({"id": "p3"})
        original["team2"]["players"].append({"id": "p4"})
        before = deepcopy(original)
        self.record([original])
        cosmetic = deepcopy(original)
        cosmetic["team1"]["players"][0].update(nameDisplay="New name", avatar="new.png")
        cosmetic["team1"]["players"].reverse()
        cosmetic["score"][0].update(serve=2, lastPointWinner=1)
        self.assertEqual(self.record([cosmetic]), (1, 0))
        self.assertEqual(original, before)

    def test_relevant_corrections_are_stored_separately(self) -> None:
        original = completed_match()
        self.record([original])
        corrections = [
            {"winner": 2}, {"duration": 50}, {"matchTimeUtc": "2026-11-11 08:00:00"},
            {"team1seed": 2}, {"score": [{"home": 21, "away": 16}]},
            {"team1": {"countryCode": "FRA", "players": [{"id": "p1"}]}},
            {"team1": {"countryCode": "CAN", "players": [{"id": "p3"}]}},
        ]
        for offset, correction in enumerate(corrections, 1):
            self.assertEqual(self.record([{**original, **correction}], self.start + timedelta(minutes=offset)), (1, 1))
        self.assertEqual(observation_status(self.db)["result_versions"], 8)

    def test_reversion_retains_the_full_receipt_history(self) -> None:
        original = completed_match()
        for hour, match in enumerate((original, {**original, "winner": 2}, original)):
            self.record([match], self.start + timedelta(hours=hour))
        rows = self.rows("""SELECT v.result_json,v.first_seen_at,r.observed_at FROM responses r
            JOIN response_results rr USING(response_id) JOIN result_versions v USING(match_id,result_sha256)
            ORDER BY r.observed_at""")
        self.assertEqual([json.loads(row[0])["winner"] for row in rows], [1, 2, 1])
        self.assertEqual(rows[0][1], rows[2][1])
        self.assertNotEqual(rows[0][2], rows[2][2])

    def test_conflicting_versions_in_one_response_are_not_silently_overwritten(self) -> None:
        self.assertEqual(self.record([completed_match(), {**completed_match(), "winner": 2}]), (2, 2))
        self.assertEqual(len(self.rows("SELECT * FROM response_results")), 2)

    def test_explicit_invalidations_only_track_previously_completed_ids(self) -> None:
        self.record([completed_match()])
        for offset, changes in enumerate(({"winner": 0}, {"winner": "0"}, {"scoreStatus": 1},
                                          {"scoreStatusValue": " Withdrawn "}, {"scoreStatusValue": "Cancelled"}), 1):
            with self.subTest(changes=changes):
                self.assertEqual(self.record([{**completed_match(), **changes}], self.start+timedelta(minutes=offset)), (0, 0))
        self.assertEqual(observation_status(self.db)["result_invalidations"], 5)
        self.assertEqual(observation_status(self.db)["responses"], 6)
        self.assertEqual(observation_status(self.db)["result_versions"], 1)
        self.assertEqual(self.record([{"id": "new-fixture", "winner": 0}]), (0, 0))
        self.assertEqual(observation_status(self.db)["responses"], 6)

    def test_missing_null_and_boolean_winners_do_not_invalidate_results(self) -> None:
        self.record([completed_match()])
        for response in ([], [{"id": "m1"}], [{"id": "m1", "winner": None}], [{"id": "m1", "winner": False}]):
            self.assertEqual(self.record(response), (0, 0))
        self.assertEqual(observation_status(self.db)["responses"], 1)
        self.assertEqual(observation_status(self.db)["result_invalidations"], 0)

    def test_invalidation_failure_rolls_back_its_response(self) -> None:
        self.record([completed_match()])
        self.record([{"id": "m1", "winner": 0}])
        with sqlite3.connect(self.db) as connection:
            connection.execute("""CREATE TRIGGER reject_invalidation BEFORE INSERT ON result_invalidations
                BEGIN SELECT RAISE(ABORT,'test write failure'); END""")
        with self.assertRaisesRegex(OSError, "Could not record result observations"):
            self.record([{"id": "m1", "winner": 0}])
        self.assertEqual(observation_status(self.db)["responses"], 2)
        self.assertEqual(observation_status(self.db)["result_invalidations"], 1)

    def test_timestamps_are_utc_and_keep_earliest_concurrent_receipt(self) -> None:
        match = completed_match()
        local = self.start.astimezone(timezone(timedelta(hours=-5)))
        self.record([match], self.start + timedelta(hours=1))
        self.record([match], local)
        self.assertEqual(self.rows("SELECT first_seen_at FROM result_versions"), [("2026-11-11T10:00:00.000000Z",)])
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            self.record([match], self.start.replace(tzinfo=None))

    def test_database_failure_is_reported_and_transaction_rolls_back(self) -> None:
        self.record([completed_match()])
        with sqlite3.connect(self.db) as connection:
            connection.execute("""CREATE TRIGGER reject_receipt BEFORE INSERT ON response_results
                BEGIN SELECT RAISE(ABORT,'test write failure'); END""")
        with self.assertRaisesRegex(OSError, "Could not record result observations"):
            self.record([{**completed_match(), "winner": 2}])
        self.assertEqual(observation_status(self.db)["responses"], 1)
        self.assertEqual(observation_status(self.db)["result_versions"], 1)

    def test_status_cli_is_read_only(self) -> None:
        for expected in (0, 1):
            with patch("sys.argv", ["fetch_bwf.py", "observations", "--db", str(self.db)]), patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main(), 0)
            self.assertEqual(json.loads(output.getvalue())["matches"], expected)
            if not expected:
                self.assertFalse(self.db.exists())
                self.record([completed_match()])

    def fetch(self, body: bytes, *, status: int = 200):
        with patch("urllib.request.urlopen") as open_url:
            response = open_url.return_value.__enter__.return_value
            response.status = status
            response.headers = {"Content-Type": "application/json"}
            response.read.return_value = body
            return fetch_url("https://example.com/results", self.root / "cache", observations_path=self.db)

    def test_fresh_standard_download_records_real_receipt_not_match_date(self) -> None:
        body = json.dumps({"matches": [completed_match()]}).encode()
        with patch("predict_badminton.data_sources.bwf.datetime") as clock:
            clock.now.return_value = self.start + timedelta(hours=6)
            result = self.fetch(body)
        self.assertEqual((result.completed_results_observed, result.result_versions_added), (1, 1))
        self.assertEqual(result.cache_path.read_bytes(), body)
        self.assertEqual(self.rows("SELECT observed_at,response_sha256 FROM responses"),
                         [("2026-11-11T16:00:00.000000Z", hashlib.sha256(body).hexdigest())])

    def test_fresh_download_records_a_reset_without_a_completed_result(self) -> None:
        self.fetch(json.dumps([completed_match()]).encode())
        self.fetch(json.dumps([{**completed_match(), "winner": 0}]).encode())
        self.assertEqual(observation_status(self.db)["result_invalidations"], 1)

    @unittest.skipUnless(importlib.util.find_spec("curl_cffi"), "Requires the fetch extra")
    def test_fresh_chrome_download_uses_the_same_logger(self) -> None:
        response = SimpleNamespace(status_code=200, headers={"Content-Type": "application/json"},
                                   content=json.dumps([completed_match()]).encode())
        with patch("predict_badminton.data_sources.bwf._chrome_session") as session:
            session.return_value.get.return_value = response
            result = fetch_url("https://example.com/results", self.root / "cache", transport="chrome", observations_path=self.db)
        self.assertEqual(result.result_versions_added, 1)

    def test_blocked_error_html_and_nonmatch_json_do_not_log(self) -> None:
        for status, body in ((403, b'{}'), (500, b'{}'), (200, b'<html>Results</html>'),
                             (200, b'{"matches": []}'), (200, b'{"matches": [{"team1": 1, "team2": 2}]}')):
            with self.subTest(status=status, body=body):
                self.assertEqual(self.fetch(body, status=status).completed_results_observed, 0)
        self.assertFalse(self.db.exists())

    def test_cached_replay_and_manual_file_reads_do_not_create_observations(self) -> None:
        body = json.dumps([completed_match()])
        url = API_BASE + "test?code=100"
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        cached = self.root / f"api-test-{digest}.json"
        cached.write_text(body)
        before = cached.read_bytes()
        with patch("predict_badminton.data_sources.bwf_backfill.fetch_url") as fetch, patch("predict_badminton.data_sources.bwf.record_results") as record:
            self.assertEqual(_api_json("test", {"code": 100}, self.root, "standard", 0, completed_on=date(2000, 1, 1)), json.loads(body))
            rounds, _ = normalize_source_file(cached)
            self.assertEqual(rounds[0][0]["id"], "m1")
            fetch.assert_not_called()
            record.assert_not_called()
        self.assertEqual(cached.read_bytes(), before)
        self.assertFalse(self.db.exists())

    def test_log_failure_preserves_previous_response_cache(self) -> None:
        body = json.dumps([completed_match()]).encode()
        cached = self.fetch(body).cache_path
        with patch("predict_badminton.data_sources.bwf.record_results", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.fetch(json.dumps([{**completed_match(), "winner": 2}]).encode())
        self.assertEqual(cached.read_bytes(), body)


if __name__ == "__main__":
    unittest.main()
