"""PickleballVision dashboard. Run after app.py:  streamlit run dashboard.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pbv.draw import name

st.set_page_config(page_title="PickleballVision", page_icon="🏓", layout="wide")
runs = sorted(p.parent for p in Path("outputs").glob("*/stats.json"))
if not runs:
    st.info("No results yet. Run `python app.py <video>` first.")
    st.stop()
run = st.sidebar.selectbox("Match", runs, format_func=lambda p: p.name)
s = json.loads((run / "stats.json").read_text())
shots = s["shots"]

st.title(f"🏓 {run.name}")
if s.get("mode") == "pov":
    st.caption("POV mode: body-worn camera. Court positions, heatmaps, speeds and kitchen faults need fixed mode.")
c = st.columns(3)
c[0].metric("Duration", f"{s['frames'] / s['fps']:.0f} s")
c[1].metric("Rallies", len(s["rallies"]))
c[2].metric("Shots", len(shots))

LEAD_S = 1  # start a rally's clip this many seconds before the serve


def clock(sec):
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


video = run / "annotated.mp4"  # frame 0 of the annotated video is frame 0 of the analysis, as are rally frames
# rally table, start time as a link: a plain-text button styled as a hyperlink (a real <a href> would reload the
# page and the video), in a grid of columns since st.dataframe cells can't run code when clicked
st.markdown("""<style>
.st-key-rally-table button {color: #1a73e8; text-decoration: underline; padding: 0; min-height: 0;}
.st-key-rally-table button:hover {color: #1558b0;}
.st-key-rally-table [data-testid="stHorizontalBlock"] {border-bottom: 1px solid rgba(128, 128, 128, 0.2);}
.st-key-rally-table [data-testid="stMarkdownContainer"] p {margin: 0;}
</style>""", unsafe_allow_html=True)
jump = st.session_state.setdefault("jump", {})  # run name -> (rally start in s, clicks)


def go_to(start_s):
    _, clicks = jump.get(run.name, (0, 0))
    jump[run.name] = (start_s, clicks + 1)


WIDTHS = [1, 2, 1.3, 1.7]
col_video, col_rallies = st.columns([3, 1.2]) if s["rallies"] else (st.container(), None)
if col_rallies is not None:
    with col_rallies:
        st.markdown("**Rallies**")
        with st.container(height=440, key="rally-table"):
            for c, h in zip(st.columns(WIDTHS, vertical_alignment="center"), ("#", "start", "shots", "length (s)")):
                c.markdown(f"**{h}**")
            for i, r in enumerate(s["rallies"], 1):
                at = r["start"] / s["fps"]
                c = st.columns(WIDTHS, vertical_alignment="center")
                c[0].markdown(str(i))
                c[1].button(clock(at), key=f"rally-{run.name}-{i}", type="tertiary", on_click=go_to,
                            args=(max(0.0, at - LEAD_S),), help=f"Jump to rally {i}")
                c[2].markdown(str(r.get("shots", "")))
                c[3].markdown(str(r.get("duration_s", "")))
with col_video:
    if video.exists():
        start_s, clicks = jump.get(run.name, (0, 0))
        # the player only seeks when start_time changes (whole seconds), so a repeat click on the same rally
        # alternates between LEAD_S and LEAD_S + 1 s before the serve
        st.video(str(video), start_time=max(0, int(start_s) - (clicks + 1) % 2), autoplay=bool(clicks), muted=True)
    else:
        st.caption("No annotated video in this run.")

st.subheader("Players")
pids = list(s["players"]) + [p for p in s["shot_summary"] if p not in s["players"]]
st.dataframe(pd.DataFrame([{"player": name(pid), **s["players"].get(pid, {}), **s["shot_summary"].get(pid, {})}
                           for pid in pids]).fillna(0), hide_index=True)

if shots:
    st.subheader("Shot mix")
    df = pd.DataFrame(shots)
    st.bar_chart(df.assign(player=df["player"].map(name)).pivot_table(
        index="type", columns="player", values="frame", aggfunc="count", fill_value=0))

st.subheader("Court coverage")
maps = sorted(run.glob("heatmap_P*.png"))
if not maps:
    st.caption("Heatmaps need fixed mode.")
for col, m in zip(st.columns(max(len(maps), 1)), maps):
    col.image(str(m), caption=m.stem.removeprefix("heatmap_"))

st.subheader("Shot log")
if shots:
    log = pd.json_normalize(shots).drop(columns=["frame"], errors="ignore")
    if "faults" in log:  # POV mode
        log["faults"] = log["faults"].map(", ".join)
        if st.checkbox("Only shots with faults"):
            log = log[log["faults"] != ""]
    st.dataframe(log, hide_index=True)
