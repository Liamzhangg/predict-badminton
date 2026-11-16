import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts/check_publication.py"
spec = importlib.util.spec_from_file_location("publication", PATH)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def test_private_paths_stay_blocked_even_with_force_add(self):
        for name in ("data/odds/new.csv", "outputs/profits.json", "outputs/rounds/new/report.md",
                     ".private/backup.bundle", "renamed/pending_bets.json", "outputs/experiments/new/forecast.json",
                     "outputs/experiments/new/paired-predictions.csv"):
            with self.subTest(name=name):
                self.assertTrue(publication.private_path(name))

    def test_source_and_aggregate_reports_remain_publishable(self):
        for name in ("predict_badminton/odds.py", "settle_bets.py", "data/catalogs/events.csv",
                     "outputs/experiments/study/report.json", "outputs/audits/review/report.md",
                     "outputs/forecast_bundle.json", "data/odds/.gitkeep"):
            with self.subTest(name=name):
                self.assertFalse(publication.private_path(name))
        self.assertFalse(publication.private_content("report.json", b'{"accuracy":0.77,"n":40000}'))

    def test_renamed_and_nested_betting_records_are_detected(self):
        self.assertTrue(publication.private_content("innocent.txt", json.dumps({"items": [{"american_odds": 125}]}).encode()))
        self.assertTrue(publication.private_content("renamed.csv", b"match_id,model_probability\nsynthetic,0.6\n"))
        self.assertTrue(publication.private_content("report.md", b"| Quote | Pick / Opponent | Odds | Model Win |\n"))
        self.assertFalse(publication.private_content("test_example.py", b'row = {"american_odds": 125}'))

    def test_history_check_rejects_records_deleted_in_a_later_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
            git("init", "-q")
            git("config", "user.name", "Synthetic Test")
            git("config", "user.email", "test@example.invalid")
            path = root / "renamed.json"
            path.write_text('{"stake":10}')
            git("add", ".")
            git("commit", "-qm", "Synthetic record")
            git("rm", "-q", "renamed.json")
            git("commit", "-qm", "Delete record")
            result = subprocess.run(["python3", str(PATH), "--history", "HEAD"], cwd=root, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn(b"renamed.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
