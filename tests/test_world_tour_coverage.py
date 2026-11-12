import csv
from datetime import UTC, datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_data import parse_args, world_tour_audit, write_world_tour_report
from predict_badminton.tournament_catalog import TournamentCatalog, _FIELDS
from tests.test_observations import completed_match


class WorldTourCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw, self.cache = self.root / "raw", self.root / "cache"
        self.raw.mkdir()
        self.cache.mkdir()
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=_FIELDS)
        writer.writeheader()
        for tid, start, end, status in ((1, "2026-11-01", "2026-11-06", "normal"),
                                        (2, "2026-01-01", "2026-01-06", "cancelled"),
                                        (3, "2026-11-10", "2026-11-15", "normal")):
            writer.writerow({"tournament_id": tid, "tournament_code": f"code-{tid}", "name": f"Example {tid}",
                             "start_date": start, "end_date": end, "category": "HSBC BWF World Tour Super 500", "status": status})
        self.catalog = TournamentCatalog(buffer.getvalue())
        self.cutoff = datetime(2026, 11, 12, tzinfo=UTC)

    def finals(self):
        matches = []
        for disc in ("MS", "WS", "MD", "WD", "XD"):
            match = {**completed_match(), "id": disc, "eventName": disc, "roundName": "Final",
                     "tournamentCode": "code-1", "matchTimeUtc": "2026-11-06 10:00:00"}
            if disc in ("MD", "WD", "XD"):
                match = json.loads(json.dumps(match))
                match["team1"]["players"].append({"id": "p3"})
                match["team2"]["players"].append({"id": "p4"})
            matches.append(match)
        return matches

    def audit(self, raw, source):
        original = json.dumps([raw])
        path = self.raw / "matches.json"
        path.write_text(original)
        (self.cache / "api-tournaments-day-matches-example.json").write_text(json.dumps(source))
        result = world_tour_audit(self.raw, self.catalog, self.cache, self.cutoff, self.root / "missing.sqlite3")
        self.assertEqual(path.read_text(), original)
        return result

    def test_cached_agreement_is_not_reported_as_universal_completeness(self):
        report, rounds, gaps = self.audit(self.finals(), self.finals())
        self.assertEqual(report["coverage_statuses"], {"cancelled_or_postponed": 1, "cached_sources_match": 1, "no_usable_results": 1})
        self.assertEqual(report["raw_played_matches"], 5)
        self.assertEqual(report["eligible_matches"], 5)
        self.assertEqual(gaps, [])
        self.assertEqual(len(rounds), 5)
        self.assertTrue(any("not proof" in note for note in report["limits"]))

    def test_published_result_absent_from_raw_is_an_explicit_gap(self):
        source = self.finals()
        extra = {**completed_match(), "id": "missing", "roundName": "R16", "tournamentCode": "code-1",
                 "team1": {"players": [{"id": "new1"}]}, "team2": {"players": [{"id": "new2"}]}}
        report, _, gaps = self.audit(source, source+[extra, extra])
        self.assertEqual(report["missing_cached_records"], 1)
        self.assertEqual(gaps[0]["match_id"], "missing")
        self.assertEqual(report["tournaments"][1]["coverage_status"], "cached_result_gaps")

    def test_walkover_final_is_a_recorded_slot_not_a_missing_played_match(self):
        matches = self.finals()
        matches[1]["scoreStatus"] = 1
        report, _, _ = self.audit(matches, matches)
        edition = next(t for t in report["tournaments"] if t["tournament_id"] == "1")
        self.assertEqual(edition["coverage_status"], "cached_sources_match")
        self.assertEqual(report["raw_played_matches"], 4)
        self.assertEqual(len(edition["final_disciplines"].split("|")), 5)

    def test_in_progress_tournament_does_not_require_a_final(self):
        match = {**completed_match(), "tournamentCode": "code-3"}
        report, _, _ = self.audit([match], [match])
        edition = next(t for t in report["tournaments"] if t["tournament_id"] == "3")
        self.assertEqual(edition["coverage_status"], "in_progress")

    def test_explicit_no_match_final_is_accounted_for_without_inventing_a_winner(self):
        matches = self.finals()
        matches[0].update(winner=0, scoreStatus=4, scoreStatusValue="No Match", score=[])
        report, _, _ = self.audit(matches, matches)
        edition = next(t for t in report["tournaments"] if t["tournament_id"] == "1")
        self.assertEqual(edition["coverage_status"], "cached_sources_match")
        self.assertEqual(edition["unplayed_final_disciplines"], "MS")
        self.assertEqual(report["raw_played_matches"], 4)

    def test_missing_times_are_distinguished_from_raw_coverage(self):
        matches = self.finals()
        matches[0]["matchTimeUtc"] = None
        report, _, _ = self.audit(matches, matches)
        self.assertEqual(report["raw_played_matches"], 5)
        self.assertEqual(report["training_ready_matches"], 4)
        self.assertEqual(report["unusable_raw_matches"], 1)
        edition = next(t for t in report["tournaments"] if t["tournament_id"] == "1")
        self.assertEqual(edition["coverage_status"], "training_data_gaps")

    def test_report_writes_only_requested_outputs(self):
        report, rounds, gaps = self.audit(self.finals(), self.finals())
        output = self.root / "report/audit.json"
        write_world_tour_report(report, rounds, gaps, output)
        self.assertEqual(len(list(output.parent.iterdir())), 5)
        self.assertTrue(output.with_suffix(".md").exists())

    def test_unknown_round_is_metadata_uncertainty_not_an_unusable_match(self):
        matches = self.finals()
        matches[0]["roundName"] = None
        report, _, _ = self.audit(matches, matches)
        self.assertEqual(report["unusable_raw_matches"], 0)
        self.assertEqual(report["unknown_round_matches"], 1)
        self.assertEqual(report["training_ready_matches"], 5)
        edition = next(t for t in report["tournaments"] if t["tournament_id"] == "1")
        self.assertEqual(edition["coverage_status"], "round_metadata_needs_review")

    def test_world_tour_default_output_does_not_overwrite_general_audit(self):
        with patch("sys.argv", ["audit_data.py"]):
            general = parse_args().output
        with patch("sys.argv", ["audit_data.py", "--world-tour"]):
            world_tour = parse_args().output
        self.assertNotEqual(general, world_tour)


if __name__ == "__main__":
    unittest.main()
