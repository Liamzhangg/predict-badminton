from __future__ import annotations

import copy
import json
import tempfile
import os
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

from predict_badminton.data_sources.bwf import FetchResult
from predict_badminton.data_sources.bwf_backfill import (
    SourceFetchError, _api_json, _draw_matches, _identity, _merge_match, backfill_matches, discover_calendar, flatten_matches,
)


def tournament(identifier: int = 1, name: str = "Example Open") -> dict:
    return {
        "id": identifier, "code": str(identifier), "name": name,
        "url": f"https://bwfbadminton.com/tournament/{identifier}/example-{identifier}",
        "start_date": "2025-12-07 00:00:00", "end_date": "2025-12-10 00:00:00",
        "status": {"status": "0", "code": "normal"},
    }


def match(identifier: int = 1, match_date: str = "2025-12-08", code: str = "1") -> dict:
    return {
        "id": identifier, "tournamentCode": code, "tournamentName": "Example Open",
        "winner": 1, "eventName": "MS", "roundName": "Final",
        "matchTimeUtc": match_date + " 10:00:00",
        "team1": {"players": [{"id": "p1"}]},
        "team2": {"players": [{"id": "p2"}]}, "matches": [],
    }


class BwfBackfillTests(unittest.TestCase):
    def test_team_rubbers_are_flattened_without_mutating_source(self) -> None:
        rubber = match()
        rubber.update(eventName="Men's Team", matchTypeValue="Men's Singles")
        parent = {**match(99), "isTeamMatch": True, "matches": [rubber]}
        original = copy.deepcopy(parent)
        flat = flatten_matches([parent])
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["id"], 1)
        self.assertEqual(flat[0]["eventName"], "MS")
        self.assertEqual(flat[0]["sourceEventName"], "Men's Team")
        self.assertEqual(flat[0]["teamTieId"], 99)
        self.assertTrue(flat[0]["isTeamMatch"])
        self.assertEqual(parent, original)

    def test_invalid_match_payload_is_rejected(self) -> None:
        for payload in ({"error": "unavailable"}, [None], [{"id": 1}]):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                flatten_matches(payload)

    def test_direct_team_draw_leaves_use_explicit_match_type(self) -> None:
        original = {**match(), "eventName": "Mixed Team", "matchTypeValue": "Women's Doubles"}
        result = flatten_matches([original])[0]
        self.assertEqual(result["eventName"], "WD")
        self.assertEqual(result["sourceEventName"], "Mixed Team")
        self.assertEqual(original["eventName"], "Mixed Team")

    def test_sparse_php_arrays_are_normalized(self) -> None:
        child = match(7)
        child["team1"]["players"] = {"2": {"id": "p2"}, "0": {"id": "p1"}}
        child["score"] = {"0": {"home": 21, "away": 10}}
        child["matches"] = {}
        parent = {**match(99), "matches": {"3": child}}
        payload = {"2": parent}
        original = copy.deepcopy(payload)
        result = flatten_matches(payload)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["team1"]["players"], [{"id": "p1"}, {"id": "p2"}])
        self.assertEqual(result[0]["score"], [{"home": 21, "away": 10}])
        self.assertEqual(result[0]["matches"], [])
        self.assertEqual(payload, original)

    def test_successful_source_is_reused_from_cache(self) -> None:
        payload = [match()]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            def fetch(url, directory, **kwargs):
                import hashlib
                from urllib.parse import urlparse
                from predict_badminton.data_sources.bwf import slugify

                digest = hashlib.sha256(url.encode()).hexdigest()[:16]
                path = directory / f"{slugify(urlparse(url).path)}-{digest}.json"
                path.write_text(json.dumps(payload))
                return FetchResult(url=url, status="cached", status_code=200, cache_path=path)
            with patch("predict_badminton.data_sources.bwf_backfill.fetch_url", side_effect=fetch) as get:
                first = _api_json("tournaments/day-matches", {"date": "2025-12-08"}, cache, "chrome", 0)
                second = _api_json("tournaments/day-matches", {"date": "2025-12-08"}, cache, "chrome", 0)
            self.assertEqual(first, payload)
            self.assertEqual(second, payload)
            self.assertEqual(get.call_count, 1)

    def test_calendar_paginates_and_excludes_para(self) -> None:
        first = tournament()
        second = tournament(2)
        para = tournament(3, "Para-Badminton International")
        responses = [
            {"results": {"data": [first, para], "current_page": 1, "last_page": 2}},
            {"results": {"data": [second], "current_page": 2, "last_page": 2}},
        ]
        with patch("predict_badminton.data_sources.bwf_backfill._api_json", side_effect=responses) as api:
            result = discover_calendar(date(2025, 12, 8), date(2025, 12, 10), Path("unused"), "chrome", 0)
        self.assertEqual([t["id"] for t in result], [1, 2])
        self.assertEqual(api.call_count, 2)
        self.assertEqual(api.call_args.args[1]["page"], 2)
        self.assertEqual(api.call_args.args[1]["discipline"], "Badminton")

    def test_calendar_retains_cancelled_entries_without_results_codes(self) -> None:
        entry = tournament()
        entry["code"] = None
        payload = {"results": {"data": [entry], "current_page": 1, "last_page": 1}}
        with patch("predict_badminton.data_sources.bwf_backfill._api_json", return_value=payload):
            found = discover_calendar(date(2025, 12, 8), date(2025, 12, 10), Path("unused"), "chrome", 0)
        self.assertEqual(found, [entry])

    def test_merge_keeps_completed_result_when_source_is_incomplete(self) -> None:
        old = {**match(), "score": [{"home": 21, "away": 10}]}
        self.assertEqual(_merge_match(old, {**match(), "winner": 0, "score": []}), old)
        updated = _merge_match({**match(), "winner": 0}, {**match(), "winner": 2, "matchTimeUtc": None})
        self.assertEqual(updated["winner"], 2)
        self.assertEqual(updated["matchTimeUtc"], match()["matchTimeUtc"])

    def test_idless_identity_is_stable_when_the_score_changes(self) -> None:
        record = {**match(), "id": None}
        self.assertEqual(_identity(record), _identity({**record, "winner": 2}))
        self.assertNotEqual(_identity(record), _identity({**record, "matchTimeUtc": "2026-01-01"}))

    def test_player_metadata_is_not_erased_by_blank_source_fields(self) -> None:
        old = {"players": [{"id": "p1", "countryName": "Canada"}]}
        self.assertEqual(_merge_match(old, {"players": [{"id": "p1", "countryName": ""}]}), old)

    def test_pre_completion_cache_is_refetched(self) -> None:
        import hashlib
        from predict_badminton.data_sources.bwf_backfill import API_BASE

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            digest = hashlib.sha256((API_BASE + "example?x=1").encode()).hexdigest()[:16]
            path = cache / f"api-example-{digest}.json"
            path.write_text("[]")
            stamp = datetime(2025, 12, 8, tzinfo=UTC).timestamp()
            os.utime(path, (stamp, stamp))
            def fetch(url, directory, **kwargs):
                path.write_text(json.dumps([match()]))
                return FetchResult(url, "cached", 200, path)
            with patch("predict_badminton.data_sources.bwf_backfill.fetch_url", side_effect=fetch) as get:
                result = _api_json("example", {"x": 1}, cache, "chrome", 0, completed_on=date(2025, 12, 10))
            self.assertEqual(len(result), 1)
            self.assertEqual(get.call_count, 1)

    def test_draw_fallback_uses_round_robin_results_and_full_knockout_matches(self) -> None:
        responses = [
            {"results": [{"value": "1", "type": 1}, {"value": "2", "type": 0},
                         {"value": "3", "type": 2}, {"value": "4", "type": 5}]},
            {"results": [match(1)]},
            {"results": {"ui-only": {"match": {"code": "1"}}}, "matches": [match(2)]},
            {"matches": [match(3)]},
            {"matches": [match(4)]},
        ]
        with patch("predict_badminton.data_sources.bwf_backfill._api_json", side_effect=responses) as api:
            results = _draw_matches(tournament(), Path("unused"), "chrome", 0)
        self.assertEqual([m["id"] for m in results], [1, 2, 3, 4])
        self.assertEqual(api.call_args_list[1].args[0], "vue-tournament-draw-data-round-robin")
        self.assertEqual(api.call_args_list[2].args[0], "vue-tournament-draw-data")
        self.assertEqual(api.call_args_list[3].args[0], "vue-tournament-draw-data")
        self.assertEqual(api.call_args_list[4].args[0], "vue-tournament-draw-data")
        self.assertEqual(api.call_args_list[0].args[1]["tmtType"], 1)

    def test_empty_team_view_falls_back_to_individual_view(self) -> None:
        responses = [{"results": None}, {"results": [{"value": "1", "type": 0}]}, {"matches": [match()]}]
        with patch("predict_badminton.data_sources.bwf_backfill._api_json", side_effect=responses) as api:
            self.assertEqual(_draw_matches(tournament(), Path("unused"), "chrome", 0), [match()])
        self.assertEqual([call.args[1]["tmtType"] for call in api.call_args_list], [1, 0, 0])

    def test_legacy_brackets_keep_draw_local_codes_and_missing_times(self) -> None:
        event = {**tournament(), "code": "official-guid"}
        legacy = {**match(), "code": "1", "tournamentCode": 1, "matchTimeUtc": None}
        del legacy["id"]
        responses = [
            {"results": None},
            {"results": [{"value": "MS", "type": 0}, {"value": "WS", "type": 0}]},
            {"results": {"0-0": {"match": legacy}}, "matches": []},
            {"results": {"0-0": {"match": legacy}}, "matches": []},
        ]
        original = copy.deepcopy(responses)
        with patch("predict_badminton.data_sources.bwf_backfill._api_json", side_effect=responses):
            found = _draw_matches(event, Path("unused"), "chrome", 0)
        self.assertEqual(len(found), 2)
        self.assertEqual(len({_identity(m) for m in found}), 2)
        self.assertTrue(all(m["tournamentCode"] == "official-guid" for m in found))
        self.assertTrue(all(m["sourceTournamentId"] == 1 for m in found))
        self.assertTrue(all(m["matchTimeUtc"] is None and "id" not in m for m in found))
        self.assertTrue(all("tmtType=0" in m["sourceUrl"] for m in found))
        self.assertEqual(responses, original)

    def test_legacy_brackets_reject_another_tournament(self) -> None:
        responses = [{"results": [{"value": "1", "type": 0}]},
                     {"results": {"0-0": {"match": {**match(), "tournamentCode": "wrong"}}}, "matches": []}]
        with patch("predict_badminton.data_sources.bwf_backfill._api_json", side_effect=responses), self.assertRaises(ValueError):
            _draw_matches(tournament(), Path("unused"), "chrome", 0)

    def test_known_no_draw_data_message_allows_other_official_view(self) -> None:
        responses = [{"results": [{"value": "1", "type": 0}]},
                     {"results": "No draw data - possible exception"},
                     {"results": [{"value": "1", "type": 0}]}, {"matches": [match()]}]
        with patch("predict_badminton.data_sources.bwf_backfill._api_json", side_effect=responses):
            self.assertEqual(_draw_matches(tournament(), Path("unused"), "chrome", 0), [match()])

    def test_server_error_tries_public_draws_and_reports_partial_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failure = SourceFetchError(FetchResult("https://example.com/day-matches", "failed", 500))
            with patch("predict_badminton.data_sources.bwf_backfill.discover_calendar", return_value=[tournament()]), patch(
                "predict_badminton.data_sources.bwf_backfill._api_json", side_effect=failure
            ), patch("predict_badminton.data_sources.bwf_backfill._draw_matches", return_value=[match()]):
                report = backfill_matches(date(2025, 12, 7), date(2025, 12, 10), today=date(2025, 12, 11),
                    delay_seconds=0, raw_dir=root / "raw", cache_root=root / "cache", manifest_path=root / "manifest.csv")
            result = report["tournaments"][0]
            self.assertEqual(result["source"], "draws_fallback")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["new_matches"], 1)
            self.assertIn("HTTP 500", result["daily_error"])
            self.assertTrue(Path(result["raw_path"]).exists())
            self.assertEqual(report["status"], "incomplete")

    def test_undated_draw_records_are_retained_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event = tournament()
            event["start_date"] = "2025-12-08 00:00:00"
            undated = match()
            undated["matchTimeUtc"] = None
            kwargs = dict(today=date(2025, 12, 11), delay_seconds=0, raw_dir=root / "raw",
                          cache_root=root / "cache", manifest_path=root / "manifest.csv")
            with patch("predict_badminton.data_sources.bwf_backfill.discover_calendar", return_value=[event]), patch(
                "predict_badminton.data_sources.bwf_backfill._api_json", return_value=[]
            ), patch("predict_badminton.data_sources.bwf_backfill._draw_matches", return_value=[undated]), patch(
                "predict_badminton.data_sources.bwf.ensure_data_dirs"
            ):
                report = backfill_matches(date(2025, 12, 8), date(2025, 12, 10), **kwargs)
                row = report["tournaments"][0]
                original = Path(row["raw_path"]).read_text()
                resumed = backfill_matches(date(2025, 12, 8), date(2025, 12, 10), **kwargs)
            self.assertEqual(row["source"], "draws_fallback")
            self.assertEqual(row["training_matches"], 0)
            self.assertEqual(row["parser_drops"], {"no_time": 1})
            self.assertIsNone(json.loads(original)[0][0]["matchTimeUtc"])
            self.assertEqual(resumed["new_matches"], 0)
            self.assertEqual(resumed["total_saved_matches"], 1)
            self.assertEqual(resumed["changes"]["new_matches"], 1)
            self.assertEqual(resumed["changes"]["new_files"], 1)
            self.assertTrue(Path(resumed["summary_path"]).exists())
            self.assertEqual(Path(row["raw_path"]).read_text(), original)

    def test_backfill_clips_dates_merges_and_backs_up_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            existing = raw / "example-1.json"
            original = json.dumps([[match(5)]])
            existing.write_text(original)
            def api(endpoint, params, *args, **kwargs):
                self.assertEqual(endpoint, "tournaments/day-matches")
                return [match(5), match(6), match(7, "2025-12-07"), match(8, "2025-12-11")]
            with patch("predict_badminton.data_sources.bwf_backfill.discover_calendar", return_value=[tournament()]), patch(
                "predict_badminton.data_sources.bwf_backfill._api_json", side_effect=api
            ) as get, patch("predict_badminton.data_sources.bwf.ensure_data_dirs"):
                report = backfill_matches(
                    date(2025, 12, 8), date(2026, 12, 7), today=date(2025, 12, 9),
                    delay_seconds=0, raw_dir=raw, cache_root=root / "cache", manifest_path=root / "manifest.csv",
                )
            backup = root / "cache" / "backfill-2025-12-08-2026-12-07" / "backups" / existing.name
            self.assertEqual(backup.read_text(), original)
            self.assertEqual(report["collected_through"], "2025-12-09")
            self.assertEqual(report["new_matches"], 1)
            row = report["tournaments"][0]
            self.assertEqual(row["status"], "partial")
            self.assertEqual(row["training_matches"], 2)
            self.assertEqual(row["parser_drops"], {})
            data = json.loads(Path(row["raw_path"]).read_text())
            self.assertEqual([m["id"] for r in data for m in r], [5, 6])
            self.assertEqual([c.args[1]["date"] for c in get.call_args_list], ["2025-12-07", "2025-12-08", "2025-12-09"])

    def test_existing_match_receives_finished_score_and_no_duplicate_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            path = raw / "example-1.json"
            path.write_text(json.dumps([[{**match(), "winner": 0}]]))
            with patch("predict_badminton.data_sources.bwf_backfill.discover_calendar", return_value=[tournament()]), patch(
                "predict_badminton.data_sources.bwf_backfill._api_json", return_value=[match()]
            ):
                report = backfill_matches(date(2025, 12, 7), date(2025, 12, 10), today=date(2025, 12, 11),
                    delay_seconds=0, raw_dir=raw, cache_root=root / "cache", manifest_path=root / "manifest.csv")
            self.assertEqual(report["new_matches"], 0)
            self.assertEqual(report["updated_matches"], 1)
            self.assertEqual(report["summary"], {"updated": 1})
            self.assertEqual(json.loads(path.read_text())[0][0]["winner"], 1)
            self.assertEqual(len(list(raw.glob("*.json"))), 1)

    def test_challenge_stops_batch_without_publishing_partial_tournament(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            block = SourceFetchError(FetchResult(url="https://example.com", status="blocked", status_code=403))
            with patch("predict_badminton.data_sources.bwf_backfill.discover_calendar", return_value=[tournament(), tournament(2)]), patch(
                "predict_badminton.data_sources.bwf_backfill._api_json", side_effect=[[match()], block]
            ), patch("predict_badminton.data_sources.bwf.ensure_data_dirs"):
                report = backfill_matches(
                    date(2025, 12, 8), date(2025, 12, 10), today=date(2025, 12, 11),
                    delay_seconds=0, raw_dir=root / "raw", cache_root=root / "cache", manifest_path=root / "manifest.csv",
                )
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["unattempted_tournaments"], 1)
            self.assertEqual(report["new_matches"], 0)
            self.assertFalse((root / "raw").exists())
            self.assertTrue(Path(report["report_path"]).exists())

    def test_network_failure_stops_without_marking_other_tournaments_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            error = SourceFetchError(FetchResult(url="https://example.com", status="failed", message="DNS lookup failed"))
            with patch("predict_badminton.data_sources.bwf_backfill.discover_calendar", return_value=[tournament(), tournament(2)]), patch(
                "predict_badminton.data_sources.bwf_backfill._api_json", side_effect=error
            ) as api:
                report = backfill_matches(date(2025, 12, 8), date(2025, 12, 10), today=date(2025, 12, 11),
                    delay_seconds=0, raw_dir=root / "raw", cache_root=root / "cache", manifest_path=root / "manifest.csv")
            self.assertEqual(report["status"], "network_error")
            self.assertEqual(report["summary"], {"failed": 1})
            self.assertEqual(report["unattempted_tournaments"], 1)
            self.assertEqual(api.call_count, 1)
            self.assertFalse((root / "raw").exists())

    def test_foreign_tournament_response_is_not_imported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("predict_badminton.data_sources.bwf_backfill.discover_calendar", return_value=[tournament()]), patch(
                "predict_badminton.data_sources.bwf_backfill._api_json", return_value=[match(code="wrong")]
            ), patch("predict_badminton.data_sources.bwf.ensure_data_dirs"):
                report = backfill_matches(
                    date(2025, 12, 8), date(2025, 12, 10), today=date(2025, 12, 11),
                    delay_seconds=0, raw_dir=root / "raw", cache_root=root / "cache", manifest_path=root / "manifest.csv",
                )
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["summary"], {"failed": 1})
            self.assertFalse((root / "raw").exists())


if __name__ == "__main__":
    unittest.main()
