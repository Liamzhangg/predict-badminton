"""Local R16 odds comparison. Run: uv run --extra app streamlit run odds_app.py"""
from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from predict_badminton.odds import american_to_decimal, expected_return, round_predictions
from predict_badminton.probabilities import PREDICTION_POLICY
from predict_badminton.tournament import load_bundle

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "data/draws/kumamoto-masters-japan-2026-r16.json"

st.set_page_config(page_title="Kumamoto R16 odds", layout="wide")
st.html("<style>[data-testid='stMainBlockContainer'] {padding-top: 1.5rem;}</style>")


@st.cache_resource
def bundle(directory: str, version: tuple, expected_hashes: tuple):
    model_dir = Path(directory)
    for filename, expected in zip(("lgbm_model.pkl", "feature_state.pkl"), expected_hashes, strict=True):
        with (model_dir / filename).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError("The model bundle differs from the pinned R16 snapshot")
    return load_bundle(model_dir)


@st.cache_data
def predictions(snapshot: str, directory: str, version: tuple):
    data = json.loads(snapshot)
    model = bundle(directory, version, (data["model_sha256"], data["state_sha256"]))
    table, cutoff = round_predictions(data, model, round_name=data["round_name"])
    table["model_sha256"] = data["model_sha256"]
    table["state_sha256"] = data["state_sha256"]
    table["generated_at_utc"] = datetime.now(UTC).isoformat()
    table["snapshot_retrieved_at_utc"] = data["retrieved_at_utc"]
    return table, cutoff


try:
    snapshot_text = SNAPSHOT.read_text(encoding="utf-8")
    data = json.loads(snapshot_text)
    model_dir = ROOT / data["model_run"]
    version = (PREDICTION_POLICY,) + tuple((model_dir / name).stat().st_mtime_ns for name in (
        "lgbm_model.pkl", "feature_state.pkl", "feature_list.json",
        "platt_calibrator.pkl", "isotonic_calibrator.pkl"))
    all_prices, cutoff = predictions(snapshot_text, str(model_dir), version)
except (ValueError, KeyError, OSError) as exc:
    st.error(str(exc))
    st.stop()

logo = ROOT / "data/source_cache/bwf/kumamoto-2026/tournament-logo.png"
if logo.exists():
    mark, heading = st.columns([1, 12], vertical_alignment="center")
    mark.image(str(logo), width=64)
    heading.header(data["name"])
else:
    st.header(data["name"])
st.caption("12 November 2026 (Japan) | Round of 16 | MS / WS / MD / WD / XD")
st.warning("Model-estimated value, not proven betting profit. Injuries, withdrawals and sportsbook settlement rules can change outcomes and payouts.")
st.caption(f"Model snapshot: {cutoff.replace('T', ' ')} UTC | Two-order averaged raw probabilities | Sportsbook prices: manual")
now = pd.Timestamp.now(tz="UTC")
prices = all_prices.loc[pd.to_datetime(all_prices["forecast_time_utc"], utc=True) > now].copy()
past_matches = all_prices["match_id"].nunique() - prices["match_id"].nunique()
if past_matches:
    st.warning(f"{past_matches} scheduled start times have passed. Those archived forecasts are excluded from current price comparisons.")

with st.expander("Sources and timing"):
    st.markdown(f"[Official BWF draw]({data['source_url']})")
    st.caption(f"Draw retrieved: {data['retrieved_at_utc']}")
    st.caption(data["assumptions"])
    st.caption(f"Model: {data['model_run']} | Availability: observed results + conservative older history")
    st.markdown("[American odds reference](https://help.smarkets.com/hc/en-gb/articles/214058369-How-to-calculate-implied-probability-in-betting) | [Expected value reference](https://help.smarkets.com/hc/en-gb/articles/214554985-How-to-calculate-expected-value-in-betting)")

event = st.segmented_control("Event", ["All", "MS", "WS", "MD", "WD", "XD"], default="All") or "All"
view = prices if event == "All" else prices.loc[prices["event"] == event]
display = view.set_index("selection_id").copy()
display["win_percent"] = display["probability"] * 100
display["fair_line"] = display["fair_american"].map(lambda value: f"{value:+.2f}")
display["positive_line"] = display["positive_ev_from"].map(lambda value: f"{value:+d}")
display["book_line"] = [st.session_state.get("offered_lines", {}).get(key) for key in display.index]
display["book_line"] = pd.to_numeric(display["book_line"], errors="coerce")
st.subheader(f"{view['match_id'].nunique()} matches")
edited = st.data_editor(
    display, key=f"prices_{event}", hide_index=True, width="stretch", height=560,
    column_order=(["event"] if event == "All" else []) + ["match", "selection", "opponent", "win_percent", "fair_line", "positive_line", "book_line"],
    disabled=[column for column in display.columns if column != "book_line"],
    column_config={
        "event": st.column_config.TextColumn("Event", width="small"),
        "match": st.column_config.NumberColumn("Match", width="small"),
        "selection": st.column_config.TextColumn("Player / pair", width="large"),
        "opponent": st.column_config.TextColumn("Opponent", width="large"),
        "win_percent": st.column_config.NumberColumn("Win %", width="small", format="%.1f%%", help="Model-estimated chance of winning."),
        "fair_line": st.column_config.TextColumn("Fair line", width="small", help="Zero expected profit under the model. Display rounded to two decimals; calculations use full precision. Not sportsbook odds."),
        "positive_line": st.column_config.TextColumn("+EV from", width="small", help="First whole-number American price strictly above model break-even. A higher-paying price also qualifies. This assumes the model probability is correct; it is not a sportsbook quote."),
        "book_line": st.column_config.NumberColumn("Book line", width="small", step=1, format="%+.2f", help="Observed moneyline. Blank means no price supplied. Valid lines are <= -100 or >= +100."),
    },
)
offered = st.session_state.setdefault("offered_lines", {})
for key, value in edited["book_line"].items():
    if pd.notna(value):
        offered[key] = float(value)
    else:
        offered.pop(key, None)

quoted = prices.loc[prices["selection_id"].isin(offered)].copy()
quoted["book_line"] = quoted["selection_id"].map(offered).astype(float)
valid_lines = quoted["book_line"].map(lambda value: math.isfinite(value) and abs(value) >= 100).astype(bool)
invalid = quoted.loc[~valid_lines]
if not invalid.empty:
    st.error("Invalid American odds: use a finite value at most -100 or at least +100. Invalid prices are excluded below.")
quoted = quoted.loc[valid_lines].copy()
if not quoted.empty:
    quoted["break_even_percent"] = quoted["book_line"].map(lambda value: 100 / american_to_decimal(value))
    quoted["model_win_percent"] = quoted["probability"] * 100
    quoted["edge_pp"] = quoted["model_win_percent"] - quoted["break_even_percent"]
    quoted["expected_return_percent"] = [100 * expected_return(p, line) for p, line in zip(quoted["probability"], quoted["book_line"])]
    quoted["model_value"] = quoted["expected_return_percent"].map(lambda value: "Positive" if value > 1e-10 else "Negative" if value < -1e-10 else "Break-even")
    st.subheader("Offered prices")
    st.dataframe(quoted.sort_values("expected_return_percent", ascending=False), hide_index=True, width="stretch",
                 column_order=["event", "selection", "opponent", "book_line", "model_win_percent", "break_even_percent", "edge_pp", "expected_return_percent", "model_value"],
                 column_config={"book_line": st.column_config.NumberColumn("Book line", format="%+.2f"),
                                "model_win_percent": st.column_config.NumberColumn("Model win %", format="%.1f%%"),
                                "break_even_percent": st.column_config.NumberColumn("Book break-even %", format="%.1f%%"),
                                "edge_pp": st.column_config.NumberColumn("Edge (pp)", format="%+.2f"),
                                "expected_return_percent": st.column_config.NumberColumn("Model EV / stake", format="%+.2f%%")})
    st.caption("EV assumes the model probability is correct, the quoted payout applies, and there are no fees. It is not a guarantee or a staking recommendation.")
    st.download_button("Download comparison", quoted.to_csv(index=False), "kumamoto-2026-odds-comparison.csv", "text/csv", icon=":material/download:")

excluded = [m for m in data["excluded_matches"] if event == "All" or m["discipline"] == event]
if excluded:
    st.subheader("Walkover · no betting forecast")
    st.dataframe(pd.DataFrame([{"Event": m["discipline"], "Player / pair": m["team1_name"],
                                "Opponent": m["team2_name"], "Status": m["status"]} for m in excluded]),
                 hide_index=True, width="stretch")

if (view["recorded_matches"] < 10).any():
    st.info("Some selections include a player with fewer than 10 recorded matches in this discipline.")
st.download_button("Download predictions", all_prices.to_csv(index=False), "kumamoto-2026-r16-predictions.csv", "text/csv", icon=":material/download:")
