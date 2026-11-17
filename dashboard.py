"""PickleballVision dashboard. Run after app.py:  streamlit run dashboard.py"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pbv.constants import DASHBOARD_KEYFRAME_S
from pbv.draw import name as default_name

st.set_page_config(page_title="PickleballVision", page_icon="🏓", layout="wide")
runs = sorted(p.parent for p in Path("outputs").glob("*/stats.json"))
if not runs:
    st.info("No results yet. Run `python app.py <video>` first.")
    st.stop()
run = st.sidebar.selectbox("Match", runs, format_func=lambda p: p.name)
s = json.loads((run / "stats.json").read_text())
shots = s["shots"]

st.title(f"🏓 {run.name}")

# player names: edited here, kept in the run's names.json, and used everywhere below (the video's burned-in P1..P4
# labels can't change)
NAMES = run / "names.json"
try:
    names = json.loads(NAMES.read_text())
except (OSError, ValueError):
    names = {}
pids = list(s["players"]) + [p for p in s["shot_summary"] if p not in s["players"]]


def save_names():
    new = {pid: st.session_state[f"name-{run.name}-{pid}"].strip() for pid in pids}
    NAMES.write_text(json.dumps({pid: n for pid, n in new.items() if n}, indent=1))


with st.expander("Player names", expanded=not names):
    for col, pid in zip(st.columns(max(len(pids), 1)), pids):
        col.text_input(default_name(pid), value=names.get(pid, ""), placeholder=default_name(pid),
                       key=f"name-{run.name}-{pid}", on_change=save_names)
names = {pid: st.session_state.get(f"name-{run.name}-{pid}", names.get(pid, "")).strip() for pid in pids}


def name(pid):
    """The player's edited name, else P1..P4 / You."""
    return names.get(str(pid)) or default_name(pid)

if s.get("mode") == "pov":
    st.caption("POV mode: body-worn camera. Court positions, heatmaps, speeds and kitchen faults need fixed mode.")
c = st.columns(3)
c[0].metric("Duration", f"{s['frames'] / s['fps']:.0f} s")
c[1].metric("Rallies", len(s["rallies"]))
c[2].metric("Shots", len(shots))

LEAD_S = 1  # start a rally's clip this many seconds before the serve


def clock(sec):
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


# frame 0 of the video is frame 0 of the analysis, as are rally frames. dashboard.mp4: player boxes for the first
# 30 s, hit markers, no ball trail (app.py render()); runs analysed before it have only annotated.mp4
video = next((v for v in (run / "dashboard.mp4", run / "annotated.mp4") if v.exists()), run / "annotated.mp4")
# rally table, start time as a link: a plain-text button styled as a hyperlink (a real <a href> would reload the
# page and the video), in a grid of columns since st.dataframe cells can't run code when clicked
st.markdown("""<style>
.st-key-rally-table, .st-key-rally-table [data-testid="stVerticalBlock"] {gap: 0;}
.st-key-rally-table [data-testid="stHorizontalBlock"] {border-bottom: 1px solid rgba(128, 128, 128, 0.25);
    padding: 4px 0; align-items: center;}
.st-key-rally-table [data-testid="stElementContainer"], .st-key-rally-table [data-testid="stMarkdownContainer"],
.st-key-rally-table [data-testid="stMarkdownContainer"] p {margin: 0; padding: 0; line-height: 1.5;}
.st-key-rally-table button {color: #1a73e8; text-decoration: underline; padding: 0; min-height: 0; height: auto;
    line-height: 1.5; font-size: 1rem;}
.st-key-rally-table button p {font-size: 1rem; line-height: 1.5; margin: 0;}
.st-key-rally-table button:hover {color: #1558b0;}
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
# playback controls for the video, all in the browser (a Streamlit widget would rerun the page and reload the
# video from the start): 0.5x / 1x speed; d steps forward one frame (hold to keep going slowly), a jumps back to
# the previous keyframe (dashboard.mp4 has one every DASHBOARD_KEYFRAME_S s: a keyframe shows at once, while an
# earlier frame between them would have to be rebuilt from one, too slowly to step back a frame at a time).
# The component's iframe has the page's origin, so its script can reach st.video's <video>.
PLAYER_HTML = """<style>
body {margin: 0; font-family: "Source Sans Pro", sans-serif; font-size: 14px; color: #888;}
button {border: 1px solid rgba(128,128,128,.4); background: none; color: inherit; border-radius: 6px;
        padding: 2px 10px; margin-right: 4px; cursor: pointer; font-size: 14px;}
button.on {background: #1a73e8; border-color: #1a73e8; color: #fff;}
</style>
<button data-rate="0.5">0.5x</button><button data-rate="1">1x</button>
<span>&nbsp; d: forward one frame (hold to keep going) &nbsp;·&nbsp; a: back to the previous keyframe (every %(key)s s)</span>
<script>
const P = window.parent, D = P.document, FRAME = 1 / 30, HOLD_RATE = 0.25, KEY_S = %(key)s;
if (P.__pbvRate === undefined) P.__pbvRate = 1;
const video = () => D.querySelector('[data-testid="stVideo"]');
const show = () => document.querySelectorAll('button').forEach(b => b.classList.toggle('on', +b.dataset.rate === P.__pbvRate));
document.querySelectorAll('button').forEach(b => b.onclick = () => {
  P.__pbvRate = +b.dataset.rate; const v = video(); if (v) v.playbackRate = P.__pbvRate; show(); });
show();
setInterval(() => {  // st.video's element is replaced when a rally link is clicked: keep the speed on it
  const v = video();
  if (v && v.playbackRate !== P.__pbvRate) v.playbackRate = P.__pbvRate;
}, 250);
if (!P.__pbvKeys) {
  P.__pbvKeys = true;
  let timer = null, held = null;
  const step = (dir, dt) => {
    const v = video();
    if (!v) return;
    v.pause();
    v.currentTime = Math.max(0, Math.min(v.duration || 1e9, v.currentTime + dir * dt));
  };
  const keys = e => {
    const t = e.target.tagName;
    if (t === 'INPUT' || t === 'TEXTAREA' || e.ctrlKey || e.metaKey || e.altKey) return;
    const dir = e.key === 'a' ? -1 : e.key === 'd' ? 1 : 0;
    if (!dir) return;
    e.preventDefault();
    if (e.repeat || held === dir) return;  // holding: the timer moves it
    held = dir;
    if (dir < 0) {  // back to the keyframe before the current time (or the one before that, if on one)
      const v = video();
      if (v) { v.pause(); v.currentTime = Math.max(0, (Math.ceil(v.currentTime / KEY_S - 1e-3) - 1) * KEY_S); }
      return;
    }
    step(dir, FRAME);  // a tap: one frame
    clearInterval(timer);
    const t0 = Date.now();
    timer = setInterval(() => { if (Date.now() - t0 > 300) step(dir, HOLD_RATE * 0.05); }, 50);  // held: 0.25x
  };
  const up = e => { if (e.key === 'a' || e.key === 'd') { clearInterval(timer); timer = null; held = null; } };
  D.addEventListener('keydown', keys);
  D.addEventListener('keyup', up);
}
</script>"""

with col_video:
    if video.exists():
        components.html(PLAYER_HTML % {"key": DASHBOARD_KEYFRAME_S}, height=30)
        start_s, clicks = jump.get(run.name, (0, 0))
        # the player only seeks when start_time changes (whole seconds), so a repeat click on the same rally
        # alternates between LEAD_S and LEAD_S + 1 s before the serve
        st.video(str(video), start_time=max(0, int(start_s) - (clicks + 1) % 2), autoplay=bool(clicks), muted=True)
    else:
        st.caption("No annotated video in this run.")

st.subheader("Players")
SPEED = ("avg_speed_ms", "max_speed_ms", "avg_shot_kmh", "max_shot_kmh")  # not shown: not reliable yet
st.dataframe(pd.DataFrame([{"player": name(pid), **{k: v for k, v in {**s["players"].get(pid, {}),
                                                                      **s["shot_summary"].get(pid, {})}.items()
                                                     if k not in SPEED}}
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
    col.image(str(m), caption=name(m.stem.removeprefix("heatmap_P")))

st.subheader("Shot log")
if shots:
    log = pd.json_normalize(shots).drop(columns=["frame"], errors="ignore")
    if "player" in log:
        log["player"] = log["player"].map(name)
    if "faults" in log:  # POV mode
        log["faults"] = log["faults"].map(", ".join)
        if st.checkbox("Only shots with faults"):
            log = log[log["faults"] != ""]
    st.dataframe(log, hide_index=True)
