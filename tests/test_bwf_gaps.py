from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from predict_badminton.data_sources.bwf_gaps import recover_gaps, write_gap_report


class BwfGapTests(unittest.TestCase):
    def test_only_empty_non_para_entries_are_retried_senior_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "backfill"
            source.mkdir()
            report = {"start_date": "2023-01-01", "collected_through": "2023-12-31", "tournaments": [
                {"tournament_id": 1, "status": "empty"}, {"tournament_id": 2, "status": "empty"},
                {"tournament_id": 3, "status": "verified"},
            ]}
            calendar = [{"id": identifier, "code": str(identifier), "name": name,
                         "start_date": "2023-04-26", "end_date": "2023-04-29"}
                        for identifier, name in ((1, "Junior U17 Open"), (2, "Pan Am Individual Championships"), (3, "Other Open"))]
            path = source / "report.json"
            path.write_text(json.dumps(report))
            (source / "tournaments.json").write_text(json.dumps(calendar))
            with patch("predict_badminton.data_sources.bwf_gaps.backfill_matches", return_value={}) as collect, patch(
                "predict_badminton.data_sources.bwf_gaps.write_gap_report"
            ):
                recover_gaps(path, today=date(2026, 10, 19))
            self.assertEqual([t["id"] for t in collect.call_args.kwargs["tournaments"]], [2, 1])
            self.assertEqual(collect.call_args.kwargs["run_dir"].name, "gap-recovery-2023-01-01-2023-12-31")
            self.assertEqual(json.loads(path.read_text()), report)

    def test_missing_calendar_entries_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps({"tournaments": [{"tournament_id": 1, "status": "empty"}]}))
            (path.parent / "tournaments.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "missing"):
                recover_gaps(path, today=date(2026, 10, 19))

    def test_para_entries_are_never_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps({"tournaments": [{"tournament_id": 1, "status": "empty"}]}))
            (path.parent / "tournaments.json").write_text(json.dumps([{"id": 1, "name": "Para Badminton Open", "start_date": "2023-01-01"}]))
            with self.assertRaisesRegex(ValueError, "non-para"):
                recover_gaps(path, today=date(2026, 10, 19))

    def test_recovery_report_cannot_overwrite_itself(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "gap-recovery-2023-01-01-2023-12-31"
            run.mkdir()
            source = run / "report.json"
            original = json.dumps({"start_date": "2023-01-01", "collected_through": "2023-12-31", "tournaments": []})
            source.write_text(original)
            (run / "tournaments.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "original backfill"):
                recover_gaps(source, today=date(2026, 10, 19))
            self.assertEqual(source.read_text(), original)

    def test_report_separates_scored_matches_draw_slots_and_cancellations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "matches.json"
            scored = {"winner": 1, "score": [{"home": 21, "away": 10}],
                      "team1": {"players": [{"id": "p1"}]}, "team2": {"players": [{"id": "p2"}]}}
            raw.write_text(json.dumps([[scored, {**scored, "score": []}]]))
            source = root / "previous.json"
            source.write_text(json.dumps({"tournaments": []}))
            (root / "evidence.json").write_text(json.dumps([
                {"tournament_id": 2, "assessment": "confirmed_cancelled", "urls": ["https://example.org/notice"]}]))
            report = {"report_path": str(root / "report.json"), "start_date": "2023-01-01",
                      "collected_through": "2023-12-31", "tournaments": [
                          {"tournament_id": 1, "name": "Example Open", "start_date": "2023-04-01", "source_url": "official", "raw_path": str(raw)},
                          {"tournament_id": 2, "name": "Cancelled Open", "start_date": "2023-05-01", "source_url": "official"},
                      ]}
            write_gap_report(report, source)
            self.assertEqual(report["gap_investigation"]["scored_player_matches"], 1)
            self.assertEqual(report["gap_investigation"]["training_matches"], 0)
            self.assertEqual(report["gap_investigation"]["assessments"], {
                "scored_results_missing_times": 1, "confirmed_cancelled": 1})
            self.assertTrue((root / "gap_investigation.csv").exists())
            self.assertIn("unplayed draw slots", (root / "gap_investigation.md").read_text())
            self.assertEqual(len(json.loads(raw.read_text())[0]), 2)
