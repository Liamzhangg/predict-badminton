from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fetch_bwf import build_parser, cmd_fetch
from predict_badminton.data_sources.bwf import (
    FetchResult,
    fetch_manifest_rows,
    fetch_url,
    is_blocked_response,
    load_manifest,
    normalize_json_data,
    normalize_source_file,
)


class BwfFetchTests(unittest.TestCase):
    def test_import_collects_all_rounds_and_flattens_team_rubbers(self) -> None:
        from tests.test_bwf_backfill import match

        parent = {**match(99), "matches": {"2": match(1), "4": match(2)}}
        payload = {"results": [{"rounds": [[parent]]}, {"matches": [match(3)]}]}
        rounds, _ = normalize_json_data(payload, "example")
        self.assertEqual([m["id"] for rnd in rounds for m in rnd], [1, 2, 3])

    def test_import_rejects_json_fields_it_cannot_normalize(self) -> None:
        for payload in ([{"home": {}, "away": {}}], [{"team1_player_ids": "1", "team2_player_ids": "2"}]):
            with self.assertRaises(ValueError):
                normalize_json_data(payload, "example")

    def test_html_import_reads_json_scripts_not_javascript_assignments(self) -> None:
        from tests.test_bwf_backfill import match

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "saved.html"
            path.write_text('<script type="application/json">' + json.dumps([match()]) + '</script>')
            rounds, _ = normalize_source_file(path)
            self.assertEqual(rounds[0][0]["id"], 1)
            path.write_text('<script>window.data = ' + json.dumps([match()]) + ';</script>')
            with self.assertRaisesRegex(ValueError, "No embedded JSON"):
                normalize_source_file(path)

    def test_cloudflare_headers_do_not_block_successful_content(self) -> None:
        headers = {"server": "cloudflare", "cf-ray": "example-YYZ"}
        self.assertFalse(is_blocked_response(200, headers, b"<html>Results</html>"))
        self.assertFalse(is_blocked_response(
            200, {**headers, "content-type": "application/json"},
            b'{"matches": [], "captcha": false}',
        ))

    def test_passive_cloudflare_script_does_not_turn_server_errors_into_challenges(self) -> None:
        body = b'<title>Server Error</title><script src="/cdn-cgi/challenge-platform/scripts/jsd/main.js"></script>'
        self.assertFalse(is_blocked_response(500, {"server": "cloudflare"}, body))

    def test_challenges_and_access_restrictions_are_detected(self) -> None:
        cases = [
            (403, {}, b"Forbidden"),
            (429, {}, b"Too many requests"),
            (200, {"cf-mitigated": "challenge"}, b""),
            (200, {}, b"<html>Checking your browser</html>"),
            (503, {}, b"<html>Attention Required! | Cloudflare</html>"),
        ]
        for code, headers, body in cases:
            with self.subTest(code=code, headers=headers):
                self.assertTrue(is_blocked_response(code, headers, body))
        self.assertFalse(is_blocked_response(503, {}, b"Service unavailable"))

    def test_standard_transport_caches_cloudflare_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("urllib.request.urlopen") as open_url:
            response = open_url.return_value.__enter__.return_value
            response.status = 200
            response.headers = {"Content-Type": "application/json", "Server": "cloudflare", "CF-Ray": "example"}
            response.read.return_value = b'[[{"id": "match-1"}]]'
            result = fetch_url("https://example.com", Path(tmp))
            self.assertEqual(result.status, "cached")
            self.assertEqual(result.cache_path.read_bytes(), response.read.return_value)

    def test_missing_optional_dependency_has_setup_instruction(self) -> None:
        with patch.dict(sys.modules, {"curl_cffi": None}):
            with self.assertRaisesRegex(RuntimeError, "uv sync --extra fetch"):
                fetch_url("https://example.com", transport="chrome")

    @unittest.skipUnless(importlib.util.find_spec("curl_cffi"), "Requires the fetch extra")
    def test_chrome_client_is_reused(self) -> None:
        from predict_badminton.data_sources.bwf import _chrome_session

        _chrome_session.cache_clear()
        with patch("curl_cffi.requests.Session") as session:
            self.assertIs(_chrome_session(), _chrome_session())
            session.assert_called_once_with(impersonate="chrome")
        _chrome_session.cache_clear()

    @unittest.skipUnless(importlib.util.find_spec("curl_cffi"), "Requires the fetch extra")
    def test_chrome_transport_caches_success_in_requested_directory(self) -> None:
        response = SimpleNamespace(
            status_code=200,
            headers={"Content-Type": "application/json", "Server": "cloudflare", "CF-Ray": "example"},
            content=b'{"matches": []}',
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "curl_cffi.requests.Session.get", return_value=response
        ) as get:
            cache = Path(tmp) / "cache"
            result = fetch_url("https://example.com/results", cache, transport="chrome")
            self.assertEqual(result.status, "cached")
            self.assertEqual(result.cache_path.parent, cache)
            self.assertEqual(result.cache_path.read_bytes(), response.content)
            self.assertEqual(result.cache_path.suffix, ".json")
            get.assert_called_once_with(
                "https://example.com/results", timeout=30.0
            )

    @unittest.skipUnless(importlib.util.find_spec("curl_cffi"), "Requires the fetch extra")
    def test_chrome_transport_never_caches_challenges_or_http_errors(self) -> None:
        for code, body, expected in [
            (403, b"Forbidden", "blocked"),
            (200, b"<html><script>window._cf_chl_opt = {}</script></html>", "blocked"),
            (404, b"Not found", "failed"),
            (503, b"Service unavailable", "failed"),
        ]:
            with self.subTest(code=code, body=body), tempfile.TemporaryDirectory() as tmp, patch(
                "curl_cffi.requests.Session.get",
                return_value=SimpleNamespace(status_code=code, headers={}, content=body),
            ):
                cache = Path(tmp) / "cache"
                result = fetch_url("https://example.com", cache, transport="chrome")
                self.assertEqual(result.status, expected)
                self.assertEqual(result.status_code, code)
                self.assertIsNone(result.cache_path)
                self.assertFalse(cache.exists())

    @unittest.skipUnless(importlib.util.find_spec("curl_cffi"), "Requires the fetch extra")
    def test_network_errors_are_reported(self) -> None:
        from curl_cffi.requests.exceptions import Timeout

        with patch("curl_cffi.requests.Session.get", side_effect=Timeout("Timed out")):
            result = fetch_url("https://example.com", transport="chrome")
        self.assertEqual(result.status, "failed")
        self.assertIn("Timed out", result.message)

    def test_manifest_passes_transport_and_records_result(self) -> None:
        url = "https://example.com/results"
        with tempfile.TemporaryDirectory() as tmp, patch(
            "predict_badminton.data_sources.bwf.fetch_url",
            return_value=FetchResult(url=url, status="blocked", status_code=403),
        ) as fetch, patch("predict_badminton.data_sources.bwf.ensure_data_dirs"):
            manifest = Path(tmp) / "manifest.csv"
            results = fetch_manifest_rows([url], manifest, delay_seconds=0, transport="chrome")
            fetch.assert_called_once_with(url, transport="chrome")
            self.assertEqual(results[0].status, "blocked")
            self.assertEqual(load_manifest(manifest)[0]["status"], "blocked")

    def test_cli_returns_failure_for_blocked_fetch(self) -> None:
        args = build_parser().parse_args(["fetch", "--transport", "chrome", "--url", "https://example.com"])
        with patch("fetch_bwf.ensure_data_dirs"), patch("fetch_bwf.fetch_manifest_rows") as fetch, patch("builtins.print"):
            fetch.return_value = [FetchResult(url=args.url[0], status="blocked", status_code=403)]
            self.assertEqual(cmd_fetch(args), 1)
            self.assertEqual(fetch.call_args.kwargs["transport"], "chrome")
            fetch.return_value = [FetchResult(url=args.url[0], status="cached", status_code=200)]
            self.assertEqual(cmd_fetch(args), 0)


if __name__ == "__main__":
    unittest.main()
