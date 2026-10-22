#!/usr/bin/env python3
"""Fetch or import BWF match data into the local raw JSON directory.

Public HTTP requests support the standard client or a Chrome-compatible
transport. Sources that still return challenges are recorded in the manifest.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from predict_badminton.config import RAW_JSON_DIR
from predict_badminton.data_sources.bwf import (
    MANIFEST_PATH,
    SOURCE_CACHE_DIR,
    default_start_date,
    ensure_data_dirs,
    fetch_manifest_rows,
    import_source_file,
    latest_non_para_match_date,
    load_manifest,
    upsert_manifest_rows,
    validate_raw_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_bwf")


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire BWF match data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="discover tournaments using the public BWF calendar API")
    discover.add_argument("--start-date", type=parse_date, default=None, help="YYYY-MM-DD; defaults to the earliest local match year")
    discover.add_argument("--end-date", type=parse_date, default=date.today(), help="YYYY-MM-DD; defaults to today")
    discover.add_argument("--transport", choices=("standard", "chrome"), default="chrome", help="HTTP client; chrome requires uv sync --extra fetch")
    discover.add_argument("--delay-seconds", type=float, default=1.0)

    fetch = subparsers.add_parser("fetch", help="fetch discovered or supplied source URLs")
    fetch.add_argument("--url", action="append", default=[], help="URL to fetch; if omitted, uses discovered manifest rows")
    fetch.add_argument("--delay-seconds", type=float, default=1.0, help="polite delay between requests")
    fetch.add_argument("--force", action="store_true", help="retry rows even if already fetched or blocked")
    fetch.add_argument("--transport", choices=("standard", "chrome"), default="standard", help="HTTP client; chrome requires uv sync --extra fetch")

    import_file = subparsers.add_parser("import-file", help="normalize a saved official JSON/HTML/CSV file")
    import_file.add_argument("source_path", type=Path)
    import_file.add_argument("--slug", default=None, help="output filename slug without .json")
    import_file.add_argument("--output-dir", type=Path, default=RAW_JSON_DIR)
    import_file.add_argument("--force", action="store_true", help="overwrite existing raw JSON")

    validate = subparsers.add_parser("validate", help="validate raw BWF JSON files with the training parser")
    validate.add_argument("--raw-dir", type=Path, default=RAW_JSON_DIR)
    validate.add_argument("--max-files", type=int, default=None)

    subparsers.add_parser("list-blocked", help="show blocked fetch manifest rows")

    backfill = subparsers.add_parser("backfill", help="collect public calendar tournaments and daily matches")
    backfill.add_argument("--start-date", type=parse_date, default=None, help="defaults to the earliest local match year; does not skip historical gaps")
    backfill.add_argument("--end-date", type=parse_date, required=True, help="YYYY-MM-DD; future dates are capped at today")
    backfill.add_argument("--transport", choices=("standard", "chrome"), default="chrome")
    backfill.add_argument("--delay-seconds", type=float, default=1.0)
    backfill.add_argument("--refresh", action="store_true", help="refresh even cached completed results")

    gaps = subparsers.add_parser("recover-gaps", help="retry empty entries from a saved report, senior championships first")
    gaps.add_argument("--report", type=Path, required=True, help="previous backfill report.json")
    gaps.add_argument("--transport", choices=("standard", "chrome"), default="chrome")
    gaps.add_argument("--delay-seconds", type=float, default=1.0)

    return parser


def cmd_discover(args: argparse.Namespace) -> int:
    from predict_badminton.data_sources.bwf_backfill import discover_calendar

    ensure_data_dirs()
    start = args.start_date or default_start_date()
    if start > args.end_date or args.delay_seconds < 0:
        raise ValueError("Require start <= end and a nonnegative request delay")

    latest = latest_non_para_match_date()
    print(f"Latest local non-para match date: {latest or 'none'}")
    print(f"Discovery window: {start} to {args.end_date}")

    tournaments = discover_calendar(start, args.end_date, SOURCE_CACHE_DIR / "calendar", args.transport,
                                    args.delay_seconds, refresh=True)
    rows = [{"tournament_id": t["id"], "name": t["name"], "source_url": t["url"],
             "start_date": t["start_date"], "end_date": t["end_date"], "status": "discovered"}
            for t in tournaments]
    upsert_manifest_rows(rows)
    print(f"Discovered {len(rows)} candidate tournaments")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    ensure_data_dirs()
    results = fetch_manifest_rows(
        urls=list(args.url),
        delay_seconds=args.delay_seconds,
        force=args.force,
        transport=args.transport,
    )
    if not results:
        print("No URLs to fetch. Run discover first or pass --url.")
        return 0
    for result in results:
        cache_info = f" -> {result.cache_path}" if result.cache_path else ""
        status_info = f" (HTTP {result.status_code})" if result.status_code else ""
        message = f" | {result.message}" if result.message else ""
        print(f"{result.status}{status_info}: {result.url}{cache_info}{message}")
    print(f"Cache: {SOURCE_CACHE_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 1 if any(result.status != "cached" for result in results) else 0


def cmd_import_file(args: argparse.Namespace) -> int:
    result = import_source_file(
        source_path=args.source_path,
        output_dir=args.output_dir,
        slug=args.slug,
        force=args.force,
    )
    print(f"Imported {result.matches} matches")
    print(f"Tournament: {result.tournament_name}")
    print(f"Raw JSON: {result.raw_path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate_raw_dir(args.raw_dir, max_files=args.max_files)
    print(f"Raw dir: {report['raw_dir']}")
    print(f"Matches: {report['matches']}")
    print(f"Drops: {report['drops']}")
    print(f"Players with names: {report['players_with_names']}")
    return 0


def cmd_list_blocked(_: argparse.Namespace) -> int:
    rows = [row for row in load_manifest() if row.get("status") == "blocked"]
    if not rows:
        print("No blocked sources recorded.")
        return 0
    for row in rows:
        print(f"{row.get('source_url')} | {row.get('message')}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "discover":
            return cmd_discover(args)
        if args.command == "fetch":
            return cmd_fetch(args)
        if args.command == "import-file":
            return cmd_import_file(args)
        if args.command == "validate":
            return cmd_validate(args)
        if args.command == "list-blocked":
            return cmd_list_blocked(args)
        if args.command in {"backfill", "recover-gaps"}:
            from predict_badminton.data_sources.bwf_backfill import backfill_matches

            if args.command == "recover-gaps":
                from predict_badminton.data_sources.bwf_gaps import recover_gaps

                report = recover_gaps(args.report, today=date.today(), transport=args.transport, delay_seconds=args.delay_seconds)
            else:
                report = backfill_matches(
                    args.start_date or default_start_date(), args.end_date, today=date.today(),
                    transport=args.transport, delay_seconds=args.delay_seconds,
                    refresh=args.refresh,
                )
            print(f"Status: {report['status']}")
            print(f"New matches: {report.get('new_matches', 0)}")
            print(f"Updated matches: {report.get('updated_matches', 0)}")
            print(f"Total matches saved for this window: {report.get('total_saved_matches', 0)}")
            print(f"Tournament statuses: {report.get('summary', {})}")
            print(f"Report: {report['report_path']}")
            return 0 if report["status"] == "complete" else 1
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
