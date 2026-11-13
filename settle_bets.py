#!/usr/bin/env python3
"""Settle frozen local paper bets after fetching results; never load a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from predict_badminton.config import DISCIPLINES
from predict_badminton.data_parser import normalize_discipline
from predict_badminton.match_rules import point_margin, unplayed_status

ROOT = Path(__file__).resolve().parent
SETTLED = {"win", "loss"}


def money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def validate(ledger: dict) -> None:
    seen = set()
    for batch in ledger["batches"]:
        for bet in batch["bets"]:
            key = (batch["tournament_code"].upper(), str(bet["match_id"]), bet["selection_id"])
            if key in seen:
                raise ValueError(f"Duplicate paper bet: {key}")
            seen.add(key)
            probability, stake = Decimal(bet["model_probability"]), Decimal(bet["stake"])
            odds = bet["american_odds"]
            if (not probability.is_finite() or not 0 <= probability <= 1
                    or not stake.is_finite() or stake <= 0 or Decimal(money(stake)) != stake
                    or isinstance(odds, bool) or not isinstance(odds, int) or abs(odds) < 100):
                raise ValueError(f"Invalid frozen probability, stake or odds: {key}")
            if bet["status"] not in SETTLED | {"pending", "review"}:
                raise ValueError(f"Unknown paper-bet status: {bet['status']}")
            if bet["discipline"] not in DISCIPLINES:
                raise ValueError(f"Unsupported discipline: {bet['discipline']}")
            players = [bet[field].split("|") for field in ("selection_player_ids", "opponent_player_ids")]
            ids = [set(team) for team in players]
            expected = 1 if bet["discipline"] in {"MS", "WS"} else 2
            if any(len(team) != expected for team in players) or any("" in team or len(team) != expected for team in ids) or ids[0] & ids[1]:
                raise ValueError(f"Invalid frozen participant IDs: {key}")
            if bet["status"] in SETTLED:
                returned, profit = Decimal(bet["returned"]), Decimal(bet["profit"])
                if not returned.is_finite() or not profit.is_finite() or returned != stake + profit:
                    raise ValueError(f"Stored payout does not reconcile: {key}")


def settle_pending(ledger: dict, root: Path, now: datetime) -> int:
    cache, settled = {}, 0
    for batch in ledger["batches"]:
        for bet in batch["bets"]:
            if bet["status"] in SETTLED:
                continue
            path = root / batch["raw_file"]
            if path not in cache:
                index = defaultdict(list)
                if path.exists():
                    for rnd in json.loads(path.read_text()):
                        for match in rnd:
                            index[str(match["id"])].append(match)
                cache[path] = index
            matches = cache[path].get(str(bet["match_id"]), [])
            if not matches:
                continue
            unique = {json.dumps(m, sort_keys=True): m for m in matches}
            if len(unique) != 1:
                bet.update(status="review", note="Conflicting result records")
                continue
            reason = ""
            match = matches[0]
            teams = {s: {str(p["id"]) for p in (match.get(f"team{s}") or {}).get("players", [])} for s in (1, 2)}
            sides = [s for s in teams if teams[s] == set(bet["selection_player_ids"].split("|"))]
            if (str(match.get("tournamentCode", "")).upper() != batch["tournament_code"].upper()
                    or normalize_discipline(match.get("eventName", "")) != bet["discipline"]
                    or match.get("roundName") != batch["round"] or len(sides) != 1
                    or teams[3 - sides[0]] != set(bet["opponent_player_ids"].split("|"))):
                reason = "Tournament, round or participant IDs do not match"
            elif unplayed_status(match) or str(match.get("scoreStatusValue") or "").casefold() not in {"", "normal"}:
                reason = "Non-normal result; sportsbook settlement needs review"
            elif match.get("matchStatus") != "F":
                continue
            elif isinstance(match.get("winner"), bool) or point_margin(match) is None:
                reason = "Finished score is incomplete or inconsistent"
            if reason:
                bet.update(status="review", note=reason)
                continue
            won = match["winner"] == sides[0]
            stake, odds = Decimal(bet["stake"]), bet["american_odds"]
            profit = stake * (Decimal(odds) / 100 if odds > 0 else Decimal(100) / -odds) if won else -stake
            profit = Decimal(money(profit))
            side = sides[0]
            scores = [g for g in match["score"] if g["home"] or g["away"]]
            bet.update(status="win" if won else "loss", profit=money(profit), returned=money(stake + profit),
                       score=", ".join(f"{g['home'] if side == 1 else g['away']}-{g['away'] if side == 1 else g['home']}" for g in scores),
                       settled_at_utc=now.isoformat(), note="",
                       result_sha256=hashlib.sha256(json.dumps(match, sort_keys=True).encode()).hexdigest())
            settled += 1
    return settled


def totals(bets: list[dict]) -> dict:
    stake = sum((Decimal(b["stake"]) for b in bets), Decimal(0))
    profit = sum((Decimal(b["profit"]) for b in bets), Decimal(0))
    return {"bets": len(bets), "wins": sum(b["status"] == "win" for b in bets),
            "losses": sum(b["status"] == "loss" for b in bets), "stake": stake,
            "profit": profit, "returned": stake + profit, "roi": profit / stake * 100 if stake else Decimal(0)}


def render_report(ledger: dict) -> str:
    all_bets = [b for batch in ledger["batches"] for b in batch["bets"]]
    closed = [b for b in all_bets if b["status"] in SETTLED]
    open_bets = [b for b in all_bets if b["status"] not in SETTLED]
    summary, statuses = totals(closed), Counter(b["status"] for b in all_bets)
    cash = lambda value: ("-" if value < 0 else "") + "$" + money(abs(value))
    signed = lambda value: ("+" if value > 0 else "") + cash(value)
    pending_stake = sum((Decimal(b["stake"]) for b in open_bets), Decimal(0))
    lines = ["# Paper Bet Profits", "",
             f"Hypothetical straight bets at saved bet365 Ontario prices. Currency: {ledger['currency']} (assumed). Not actual wagers or a guarantee of future profit.", "",
             f"**Settled profit: {signed(summary['profit'])} | Settled stake: {cash(summary['stake'])} | Returned: {cash(summary['returned'])} | Settled ROI: {money(summary['roi'])}%**", "",
             f"{len(all_bets)} paper bets tracked: **{len(closed)} settled** ({summary['wins']} wins, {summary['losses']} losses), **{statuses['pending']} pending**, **{statuses['review']} needing review**.", "",
             f"**Unsettled stake: {cash(pending_stake)}. Total paper stakes: {cash(summary['stake'] + pending_stake)}.** Unsettled returns and profit are unknown and excluded from settled profit and ROI.", "",
             "## Settled Summary", "",
             "| Date | Tournament | Round | Bets | Wins | Losses | Staked | Returned | Net Profit |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for batch in ledger["batches"]:
        bets = [b for b in batch["bets"] if b["status"] in SETTLED]
        if bets:
            t = totals(bets)
            lines.append(f"| {batch['date']} | {batch['name']} | {batch['round']} | {t['bets']} | {t['wins']} | {t['losses']} | {cash(t['stake'])} | {cash(t['returned'])} | {signed(t['profit'])} |")
    lines.extend([f"| **Total** | | | **{summary['bets']}** | **{summary['wins']}** | **{summary['losses']}** | **{cash(summary['stake'])}** | **{cash(summary['returned'])}** | **{signed(summary['profit'])}** |", ""])
    for settled, heading in ((False, "Pending Bets"), (True, "Settled Bet Results")):
        lines.extend([f"## {heading}", ""])
        for batch in ledger["batches"]:
            bets = [b for b in batch["bets"] if (b["status"] in SETTLED) == settled]
            if not bets:
                continue
            bets.sort(key=lambda bet: (DISCIPLINES.index(bet["discipline"]), bet["selection"].casefold()))
            lines.extend([f"### {batch['name']} {batch['round']} ({batch['date']})", "",
                          "| Event | Selection | Opponent | Odds | Frozen Win Chance | Stake | Score | Status | Net Profit |" if settled else
                          "| Event | Selection | Opponent | Odds | Frozen Win Chance | Model EV | Stake | Status |",
                          "| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |" if settled else
                          "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"])
            for bet in bets:
                status = bet["status"].title() + (f": {bet['note']}" if bet.get("note") else "")
                prefix = f"| {bet['discipline']} | {bet['selection']} | {bet['opponent']} | {bet['american_odds']:+d} | {Decimal(bet['model_probability']) * 100:.1f}% |"
                if settled:
                    lines.append(f"{prefix} {cash(Decimal(bet['stake']))} | {bet.get('score') or '-'} | {status} | {signed(Decimal(bet['profit']))} |")
                else:
                    ev = f"{Decimal(bet['model_ev_percent']):+.1f}%" if bet.get("model_ev_percent") is not None else "-"
                    lines.append(f"{prefix} {ev} | {cash(Decimal(bet['stake']))} | {status} |")
            if batch.get("note"):
                lines.extend(["", batch["note"]])
            lines.append("")
    lines.extend(["## Records", "", "[Structured paper-bet ledger](profits.json) stores verified match/player IDs, frozen odds and full-precision probabilities. Editing this generated Markdown does not change the ledger.", ""])
    for source in dict.fromkeys(batch["source"] for batch in ledger["batches"]):
        lines.append(f"- [Original betting record]({os.path.relpath(source, 'outputs')})")
    lines.extend(["", "After fetching results, run `uv run python settle_bets.py`. Use `--dry-run` to preview without writing.", "",
                  "Scores are shown from the selected side. Winning payouts are rounded per bet to the nearest cent. Missing or unfinished results stay pending. Retirements, walkovers and conflicting results need manual review; no sportsbook void rules are assumed. Previously settled bets are not automatically resettled.", "",
                  "These local outputs are Git-ignored. Screenshot prices and capture times are unverified. Model-positive EV does not establish a durable betting edge.", ""])
    return "\n".join(lines)


def write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="preview settlement counts without writing")
    args = parser.parse_args(argv)
    try:
        path = ROOT / "outputs/profits.json"
        ledger = json.loads(path.read_text())
        validate(ledger)
        count = settle_pending(ledger, ROOT, datetime.now(UTC))
        validate(ledger)
        report = render_report(ledger)
        if not args.dry_run:
            write_if_changed(path, json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
            write_if_changed(ROOT / "outputs/profits.md", report)
        statuses = Counter(b["status"] for batch in ledger["batches"] for b in batch["bets"])
        print(f"{'Preview' if args.dry_run else 'Updated'}: {count} newly settled; {statuses['pending']} pending; {statuses['review']} needing review.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation) as exc:
        print(f"Cannot settle paper bets: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
