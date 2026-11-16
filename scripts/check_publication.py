"""Reject private betting records in the index or any history being pushed."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import PurePosixPath
import re
import subprocess
import sys


PRIVATE_PREFIXES = (".private/", "data/odds/", "data/raw/", "data/source_cache/", "outputs/rounds/",
                    "outputs/predictions/", "outputs/experiments/japan-kaohsiung-finals-",
                    "outputs/experiments/japan-kaohsiung-settlement-", "outputs/experiments/kumamoto-r32-recovery-",
                    "outputs/experiments/kumamoto-settlement-", "outputs/experiments/sf-observed-")
PRIVATE_NAMES = {"profits.json", "profits.md", "pending_bets.json", "forecast.json", "comparison.csv",
                 "settlement.json", "recovery.json", "hong-kong-shadow-probabilities.csv", "shadow-manifest.json",
                 "paired-predictions.csv", "diagnostic-predictions.csv", "protected-before.json", "promotion-before.json",
                 "test_betting_slip.py"}
PRIVATE_FIELDS = {"american_odds", "decimal_odds", "model_probability", "model_ev_percent", "bets_detail",
                  "positive_ev_selections", "pending_bets", "stake_per_bet", "stake", "payout", "selections"}


def private_path(name: str) -> bool:
    path = PurePosixPath(name.lower())
    if name.endswith("/.gitkeep"):
        return False
    if name.lower().startswith(PRIVATE_PREFIXES) or path.name in PRIVATE_NAMES:
        return True
    if path.suffix in {".pkl", ".npz", ".sqlite3", ".bundle"}:
        return True
    if name.startswith("outputs/"):
        return not (name == "outputs/README.md" or name in {"outputs/forecast_bundle.json", "outputs/forecast_bundle.previous.json"}
                    or name.startswith(("outputs/audits/", "outputs/experiments/", "outputs/runs/"))
                    and path.suffix in {".py", ".md", ".json"})
    return False


def private_content(name: str, body: bytes) -> bool:
    if PurePosixPath(name).suffix in {".py", ".sh"} or name.startswith(".githooks/"):
        return False
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return False
    def has_record(value):
        if isinstance(value, dict):
            return bool(PRIVATE_FIELDS.intersection(value)) or any(has_record(v) for v in value.values())
        return isinstance(value, list) and any(has_record(v) for v in value)
    try:
        if has_record(json.loads(text)):
            return True
    except (ValueError, RecursionError):
        pass
    header = next(csv.reader(io.StringIO(text)), [])
    return bool(PRIVATE_FIELDS.intersection(header)) or bool(re.search(
        r"(?im)^\|[^\n]*(?:Pick / Opponent|American Odds|Model Win)[^\n]*\|", text))


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def inspect_index() -> set[str]:
    violations = set()
    for name in git("ls-files", "-z").decode().strip("\0").split("\0"):
        if name and (private_path(name) or private_content(name, git("show", ":" + name))):
            violations.add(name)
    return violations


def inspect_history(tips: list[str]) -> set[str]:
    violations = set()
    if not tips:
        return violations
    names = git("log", "--format=", "--name-only", "-z", *tips, "--").decode().split("\0")
    violations.update(name.strip("\n") for name in names if name.strip("\n") and private_path(name.strip("\n")))
    objects = []
    for line in git("rev-list", "--objects", *tips, "--").decode().splitlines():
        oid, _, name = line.partition(" ")
        if name and not private_path(name):
            objects.append((oid, name))
    request = "".join(oid + "\n" for oid, _ in objects).encode()
    response = subprocess.run(["git", "cat-file", "--batch"], input=request, stdout=subprocess.PIPE, check=True).stdout
    stream = io.BytesIO(response)
    for _, name in objects:
        _, kind, size = stream.readline().split()
        body = stream.read(int(size))
        assert stream.read(1) == b"\n"
        if kind == b"blob" and private_content(name, body):
            violations.add(name)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--history", nargs="+")
    mode.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if args.staged:
        violations = inspect_index()
    else:
        tips = args.history or []
        if args.push:
            for line in sys.stdin:
                local_ref, oid, remote_ref, _ = line.split()
                if not remote_ref.startswith(("refs/heads/", "refs/tags/")):
                    print("Refusing to publish non-branch/tag refs: " + remote_ref, file=sys.stderr)
                    return 1
                if set(oid) != {"0"}:
                    tips.append(oid)
        violations = inspect_history(tips)
    if violations:
        print("Publication blocked: private betting data or local-only artifacts found:", file=sys.stderr)
        print("\n".join(sorted(violations)), file=sys.stderr)
        return 1
    print("Publication check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
