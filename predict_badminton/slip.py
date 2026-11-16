"""Render saved paper-bet selections without rescoring or changing accounting."""
from collections import Counter
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo


EVENTS = {"MS": "Men's Singles", "WS": "Women's Singles", "MD": "Men's Doubles",
          "WD": "Women's Doubles", "XD": "Mixed Doubles"}


def display_name(row, field):
    name = row[f"source_{field}_name"]
    return row[f"{field}_name"] if "..." in name else name


def start_label(row, fixture):
    official = row["official_start_time_utc"]
    stamp = datetime.fromisoformat(official or row["forecast_reference_time_utc"]).astimezone(ZoneInfo("America/Toronto"))
    label = f"{stamp:%b} {stamp.day}, {stamp.hour % 12 or 12}:{stamp.minute:02d} {stamp:%p}"
    if not official:
        return label + " (est.)"
    return label + " (indic.)" if fixture["time_estimated"] in (True, "True") else label


def render_slip(forecast, coverage):
    fixtures = {str(row["match_id"]): row for row in forecast["fixtures"]}
    positives = sorted((row for row in forecast["selections"] if Decimal(row["model_ev_percent"]) > 0),
                       key=lambda row: (-Decimal(row["model_ev_percent"]), row["selection_id"]))
    stake = Decimal("10")
    expected = sum((Decimal(row["model_ev_percent"]) / 100 * stake for row in positives), Decimal(0))
    counts = Counter(row["discipline"] for row in positives)
    lines = ["# Theoretical +EV Betting Slip", "",
             f"**{len(positives)} single paper plays | $10 CAD each | ${len(positives) * stake:.2f} total stake | +${expected:.2f} theoretical expected net**", "",
             "Paper bets only. Positive model EV is not a guarantee of profit or current price availability.", "",
             f"Frozen forecast: **{forecast['prediction_generated_at_utc']}**. Saved prices and probabilities are unchanged.", "",
             "All times are Toronto / Eastern. `(est.)` is an unpublished start; `(indic.)` is a followed-by reference.", "",
             "| Event | Plays | Paper Stake | Theoretical Expected Net |",
             "| --- | ---: | ---: | ---: |"]
    for event, name in EVENTS.items():
        value = sum((Decimal(row["model_ev_percent"]) / 100 * stake for row in positives if row["discipline"] == event), Decimal(0))
        lines.append(f"| {name} ({event}) | {counts[event]} | ${counts[event] * stake:.2f} | +${value:.2f} |")
    for event, name in EVENTS.items():
        lines += ["", f"## {name} ({event})", "",
                  "| Quote | Round | Pick / Opponent | Odds | Model Win | EV | Net / $10 | Start (ET) |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for row in positives:
            if row["discipline"] != event:
                continue
            match = f"**{display_name(row, 'selection')}**<br>vs {display_name(row, 'opponent')}"
            if row["selection_team_country_missing"] or row["opponent_team_country_missing"]:
                match += "<br>Country input missing"
            if row["low_player_history"]:
                match += "<br>Low player history"
            ev = Decimal(row["model_ev_percent"])
            lines.append(f"| {row['quote_id']} | {row['round_name']} | {match} | {row['american_odds']:+d} | {Decimal(row['model_probability']) * 100:.2f}% | {ev:+.2f}% | +${ev / 100 * stake:.2f} | {start_label(row, fixtures[str(row['match_id'])])} |")
    lines += ["", "## Coverage And Exclusions", "",
              f"Verified matchups: {coverage['verified_matches']}. Verified prices: {coverage['verified_prices']}.", "",
              "| Event / Round | Match Without Usable Odds |", "| --- | --- |"]
    for missing in coverage["missing_odds"]:
        lines.append(f"| {missing['discipline']} / {missing['round']} | {missing['team1']} vs {missing['team2']} |")
    lines += ["", "EV = model win probability x decimal odds - 1. This rendering does not append or settle bets.", ""]
    return "\n".join(lines)
