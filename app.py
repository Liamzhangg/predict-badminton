"""Local French Open forecasts. Run: uv run --extra app streamlit run app.py"""
from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from predict_badminton.tournament import forecast, load_bundle

ROOT = Path(__file__).resolve().parent
DRAW = ROOT / "data/draws/french-open-2026-after-r32.json"
MODEL = ROOT / "outputs/runs/2026-10-21-french-open"

st.set_page_config(page_title="French Open 2026", layout="wide")


@st.cache_resource
def bundle(version: tuple):
    return load_bundle(MODEL)


@st.cache_data(show_spinner="Calculating the draw...")
def prediction(snapshot: str, event: str, method: str, version: tuple):
    data = json.loads(snapshot)
    return forecast(data["draws"][event], event, bundle(version), method)


def bracket(draw: dict, result: dict) -> str:
    entries, advance = draw["entries"], result["advance"]
    columns = []
    for level, rnd in enumerate(draw["rounds"]):
        nodes = []
        for slot in range(len(rnd["matches"])):
            width = 2 ** (level + 1)
            start = slot * width
            candidates = sorted(range(start, start + width), key=lambda i: advance[i, level + 1], reverse=True)
            shown = candidates[:2]
            lines = []
            for i in shown:
                name = html.escape(entries[i].get("short_name") or entries[i]["name"])
                title = html.escape(entries[i]["name"], quote=True)
                seed = f" [{entries[i]['seed']}]" if entries[i].get("seed") else ""
                lines.append(f'<div class="entrant"><span title="{title}">{name}{seed}</span>'
                             f'<b>{advance[i, level + 1]:.1%}</b></div>')
            remainder = 1 - sum(advance[i, level + 1] for i in shown)
            other = f'<small>Other players {remainder:.1%}</small>' if remainder > 0.0005 else ""
            status = " · Possible winners" if level else ""
            nodes.append(f'<div class="node" style="grid-row:span {2 ** level}">'
                         f'<article><small>{rnd["name"]} · {slot + 1}{status}</small>{"".join(lines)}{other}</article></div>')
        columns.append(f'<section><h4>{rnd["name"]}</h4><div class="round" '
                       f'style="grid-template-rows:repeat({len(entries) // 2},100px)">{"".join(nodes)}</div></section>')
    return """<style>
    .draw-scroll{overflow:auto;max-height:650px;border-top:1px solid #dce2e5;padding-top:12px}
    .draw-grid{display:flex;gap:20px;min-width:max-content;font-family:system-ui,sans-serif;color:#172026}
    .draw-grid section{width:220px;flex-shrink:0}.draw-grid h4{margin:0 0 8px;font-size:14px}
    .round{display:grid}.node{display:flex;align-items:center;position:relative}
    .node article{width:100%;padding:9px;border:1px solid #dce2e5;border-radius:4px;background:#fff;box-sizing:border-box}
    .node small{display:block;font-size:10px;color:#5d6a70;margin-bottom:3px}
    .entrant{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:12px;line-height:17px}
    .entrant span{overflow-wrap:anywhere;min-width:0}.entrant b{color:#176d64;white-space:nowrap;font-weight:600}
    </style><div class="draw-scroll"><div class="draw-grid">""" + "".join(columns) + "</div></div>"


if not DRAW.exists() or not (MODEL / "feature_state.pkl").exists():
    st.error("The local draw snapshot or model bundle is missing.")
    st.stop()

snapshot = DRAW.read_text(encoding="utf-8")
data = json.loads(snapshot)
metrics = json.loads((MODEL / "metrics_report.json").read_text(encoding="utf-8"))
version = tuple((MODEL / name).stat().st_mtime_ns for name in
                ("lgbm_model.pkl", "platt_calibrator.pkl", "isotonic_calibrator.pkl", "feature_state.pkl"))
st.title(data["name"])
st.caption(f"20–25 October 2026 · {data['location']} · {data['stage']}")
event = st.segmented_control("Event", list(data["draws"]), default="MS", key="event") or "MS"
with st.expander("Model and sources"):
    method = st.selectbox("Probabilities", ["Raw", "Isotonic", "Platt"])
    st.caption(f"Model: {data['model_date']} · Draw retrieved: {data['retrieved_at_utc']} · History frozen for future rounds")
    st.caption("2026 holdout log loss, before production refit: " + " · ".join(
        f"{name} {metrics[key]['log_loss']:.4f}" for name, key in
        (("Raw", "test_raw"), ("Isotonic", "test_isotonic"), ("Platt", "test_platt"))))
    st.markdown(f"[Official draw]({data['source_url']}) · [Official schedule]({data['schedule_source_url']})")
    st.caption(data["assumptions"])

try:
    result = prediction(snapshot, event, method, version)
except (ValueError, KeyError, OSError) as exc:
    st.error(str(exc))
    st.stop()
draw = data["draws"][event]
table = result["table"].sort_values("Title", ascending=False).reset_index(drop=True)
favorite = table.iloc[0]
a, b, c = st.columns([2, 1, 1])
a.caption("Title favorite")
a.markdown(f"**{favorite['Entry']}**")
b.metric("Title probability", f"{favorite['Title']:.1f}%")
c.metric("Remaining entries", len(draw["entries"]))
st.caption(f"History through {result['cutoff'][:10]} · {method} probabilities")
if (table["History"] < 10).any():
    st.warning(f"{int((table['History'] < 10).sum())} entries have fewer than 10 recorded matches for at least one player.")

draw_tab, odds_tab, match_tab = st.tabs(["Draw", "Advancement", "Matchup"])
with draw_tab:
    st.caption("R16: confirmed opponents · Later rounds: possible slot winners · All players included in calculations")
    st.html(bracket(draw, result))
with odds_tab:
    probability_columns = [r["name"] for r in draw["rounds"][1:]] + ["Title"]
    st.dataframe(table, hide_index=True, width="stretch", height=600,
                 column_config={name: st.column_config.ProgressColumn(name, min_value=0, max_value=100, format="%.1f%%")
                                for name in probability_columns})
    st.download_button("Download probabilities", table.to_csv(index=False), f"french-open-2026-{event}.csv", "text/csv", icon=":material/download:")
with match_tab:
    rnd = st.selectbox("Round", range(len(draw["rounds"])), format_func=lambda i: draw["rounds"][i]["name"])
    slot = st.selectbox("Match slot", range(len(draw["rounds"][rnd]["matches"])), format_func=lambda i: str(i + 1))
    choices = [m for m in result["matchups"] if m["level"] == rnd and m["slot"] == slot]
    labels = lambda i: draw["entries"][i]["name"]
    left = st.selectbox("Top half entrant", sorted({m["left"] for m in choices}), format_func=labels)
    right = st.selectbox("Bottom half entrant", sorted({m["right"] for m in choices}), format_func=labels)
    match = next(m for m in choices if m["left"] == left and m["right"] == right)
    a, b, c = st.columns(3)
    a.metric("Top entrant wins if they meet", f"{match['win']:.1%}")
    b.metric("Bottom entrant wins if they meet", f"{1 - match['win']:.1%}")
    c.metric("Chance this matchup occurs", f"{match['meeting']:.1%}")
    timing = draw["rounds"][rnd]["matches"][slot]
    st.caption(f"{'Estimated' if timing['time_estimated'] else 'Published schedule'}: {timing['time_utc']} UTC · Fixed history, no simulated Elo updates")
