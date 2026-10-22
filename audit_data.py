#!/usr/bin/env python3
"""Audit normalized BWF data without modifying raw source files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from predict_badminton.data_parser import build_data_report, load_all_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bwf_matches"),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--include-para",
        action="store_true",
        help="Include para matches, which are excluded by default",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Report all parsed records without match deduplication",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/data_audit.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(args.data_dir.glob("*.json"))
    if args.max_files is not None:
        files = files[: args.max_files]

    matches, drop_counts, _ = load_all_matches(
        args.data_dir,
        max_files=args.max_files,
        include_para=args.include_para,
        deduplicate=not args.no_deduplicate,
    )
    report = build_data_report(
        matches,
        drop_counts,
        files_scanned=len(files),
        include_para=args.include_para,
        deduplicate=not args.no_deduplicate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nAudit written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
