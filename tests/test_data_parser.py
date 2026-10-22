from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from predict_badminton.data_parser import (
    build_data_report,
    is_para_match,
    load_all_matches,
    normalize_discipline,
)


def _raw_match(
    match_id: str | None,
    *,
    tournament: str = "Example Open",
    event: str = "MS",
    match_time: str = "2024-01-01 10:00:00",
    team1: str = "p1",
    team2: str = "p2",
) -> dict:
    return {
        "id": match_id,
        "winner": 1,
        "matchTimeUtc": match_time,
        "eventName": event,
        "roundName": "Final",
        "tournamentName": tournament,
        "team1": {"players": [{"id": team1}]},
        "team2": {"players": [{"id": team2}]},
    }


class DataParserTests(unittest.TestCase):
    def test_walkovers_do_not_update_training_history_but_retirements_remain(self) -> None:
        records = [{**_raw_match("code"), "scoreStatus": 1},
                   {**_raw_match("text"), "scoreStatusValue": "Walkover"},
                   {**_raw_match("retired"), "scoreStatus": 2, "scoreStatusValue": "Retired"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            original = json.dumps([records])
            path.write_text(original)
            matches, drops, _ = load_all_matches(Path(tmp))
            self.assertEqual(path.read_text(), original)
        self.assertEqual([m["match_id"] for m in matches], ["retired"])
        self.assertEqual(drops, {"walkover": 2})

    def test_dated_reports_do_not_invent_individual_match_times(self) -> None:
        individual = {
            **_raw_match(None, match_time=""),
            "matchTime": None,
            "matchDate": "2022-05-21",
            "timePrecision": "date",
            "sourceKind": "official_federation_report",
        }
        team_tie = {
            **individual,
            "eventName": "Mixed Team",
            "isTeamMatch": True,
            "teamTieTime": "2022-05-21 09:00:00",
            "teamScore": {"home": 3, "away": 1},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reports.json"
            original = json.dumps([[individual, team_tie]])
            path.write_text(original, encoding="utf-8")
            matches, drops, names = load_all_matches(Path(tmp))
            self.assertEqual(path.read_text(encoding="utf-8"), original)
        self.assertEqual(matches, [])
        self.assertEqual(drops, {"no_time": 2})
        self.assertEqual(names, {})

    def test_descriptive_bwf_discipline_names_are_recognized(self) -> None:
        labels = {"Men's Singles": "MS", "Women's Singles": "WS", "Men's Doubles": "MD",
                  "Women's Doubles": "WD", "Mixed Doubles": "XD", "Girl's Singles": "WS",
                  "Women\u2019s Singles": "WS", "ms": "MS", "MS U19": "MS", "Air Triples": "OTHER"}
        for label, expected in labels.items():
            with self.subTest(label=label):
                self.assertEqual(normalize_discipline(label), expected)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.json"
            path.write_text(json.dumps([[_raw_match("olympic", event="Men's Singles")]]))
            matches, drops, _ = load_all_matches(Path(tmp))
        self.assertEqual(matches[0]["discipline"], "MS")
        self.assertEqual(matches[0]["event_name"], "Men's Singles")
        self.assertEqual(drops, {})

    def test_junior_discipline_codes_preserve_the_standard_match_format(self) -> None:
        labels = {"BS U17": "MS", "GS U15": "WS", "BD U13": "MD", "GD U19": "WD",
                  "MSU19": "MS", "WSU19": "WS", "MDU19": "MD", "WDU19": "WD", "XDU19": "XD",
                  "Team Relay": "OTHER", "Men's Triples": "OTHER", "MSUNKNOWN": "OTHER"}
        for label, expected in labels.items():
            with self.subTest(label=label):
                self.assertEqual(normalize_discipline(label), expected)

    def test_para_detection_uses_tournament_and_classification_metadata(self) -> None:
        self.assertTrue(
            is_para_match(_raw_match("1", tournament="European Para-Badminton"))
        )
        self.assertTrue(is_para_match(_raw_match("2", event="MS - SL 4")))
        self.assertFalse(is_para_match(_raw_match("3")))

    def test_load_filters_para_and_deduplicates_without_touching_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            first = [
                _raw_match("regular"),
                _raw_match("para", tournament="BWF Para Badminton International"),
                _raw_match(
                    None,
                    match_time="2024-01-02 10:00:00",
                    team1="p3",
                    team2="p4",
                ),
            ]
            second = [
                _raw_match("regular"),
                _raw_match(
                    None,
                    match_time="2024-01-02 10:00:00",
                    team1="p4",
                    team2="p3",
                ),
            ]
            (raw_dir / "a.json").write_text(json.dumps([first]), encoding="utf-8")
            (raw_dir / "b.json").write_text(json.dumps([second]), encoding="utf-8")

            matches, drops, _ = load_all_matches(raw_dir)

            self.assertEqual(len(matches), 2)
            self.assertEqual(drops["para_match"], 1)
            self.assertEqual(drops["duplicate_match_id"], 1)
            self.assertEqual(drops["duplicate_signature"], 1)
            self.assertEqual(len(list(raw_dir.glob("*.json"))), 2)

            all_matches, _, _ = load_all_matches(raw_dir, include_para=True)
            self.assertEqual(len(all_matches), 3)

    def test_data_report_summarizes_included_and_excluded_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp)
            records = [
                _raw_match("one", match_time="2023-01-01 10:00:00"),
                _raw_match("two", match_time="2024-01-01 10:00:00"),
            ]
            (raw_dir / "matches.json").write_text(
                json.dumps([records]), encoding="utf-8"
            )
            matches, drops, _ = load_all_matches(raw_dir)
            report = build_data_report(
                matches,
                drops,
                files_scanned=1,
                include_para=False,
                deduplicate=True,
            )

            self.assertEqual(report["matches_included"], 2)
            self.assertEqual(report["matches_by_year"], {2023: 1, 2024: 1})
            self.assertFalse(report["policy"]["include_para"])

    def test_source_aliases_prefer_utc_without_removing_distinct_match_slots(self) -> None:
        legacy = {**_raw_match("legacy", match_time=""), "matchTime": "2024-01-01 10:00:00", "matchOrder": 1}
        current = {**_raw_match("current", match_time="2024-01-01 02:00:00"),
                   "matchTime": legacy["matchTime"], "matchOrder": 1}
        separate = {**current, "id": "separate", "matchOrder": 2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.json"
            for records in ([legacy, current, separate], [current, legacy, separate]):
                path.write_text(json.dumps([records]))
                matches, drops, _ = load_all_matches(Path(tmp))
                self.assertEqual({m["match_id"] for m in matches}, {"current", "separate"})
                self.assertEqual(drops["duplicate_source_alias"], 1)
                self.assertEqual(matches[0]["match_time"].hour, 2)
                unfiltered, _, _ = load_all_matches(Path(tmp), deduplicate=False)
                self.assertEqual(len(unfiltered), 3)
            path.write_text(json.dumps([[current, {**current, "id": "another-utc-record"}]]))
            matches, _, _ = load_all_matches(Path(tmp))
            self.assertEqual(len(matches), 2)


if __name__ == "__main__":
    unittest.main()
