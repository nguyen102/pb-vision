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
.st-key-rally-table .pbv-current {background: rgba(26, 115, 232, 0.18); box-shadow: inset 3px 0 0 #1a73e8;}
.pbv-team {display: inline-block; padding: 0 6px; border-radius: 8px; font-size: 0.8rem; line-height: 1.4;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; vertical-align: middle;}
.pbv-team-0 {background: rgba(26, 115, 232, 0.2); color: #1a73e8;}
.pbv-team-1 {background: rgba(230, 120, 20, 0.22); color: #d36a00;}
.pbv-won {color: #1e8e3e; font-weight: 600;}
.pbv-lost {color: #d93025; font-weight: 600;}
.pbv-unknown {color: #888;}
</style>""", unsafe_allow_html=True)
jump = st.session_state.setdefault("jump", {})  # run name -> (rally start in s, clicks)


def go_to(start_s):
    _, clicks = jump.get(run.name, (0, 0))
    jump[run.name] = (start_s, clicks + 1)


# who served each rally, and the team they're on (teams: from app.py; older runs have none: just the server)
server = {sh["rally"]: str(sh["player"]) for sh in shots if sh.get("role", sh.get("type")) == "serve"}
team_of = {str(pid): k for k, t in enumerate(s.get("teams") or []) for pid in t["players"]}


def serving_tag(i):
    pid = server.get(i)
    if pid is None:
        return ""
    k = team_of.get(pid)
    if k is None:
        return name(pid)
    label = " & ".join(name(p) for p in s["teams"][k]["players"])
    return f'<span class="pbv-team pbv-team-{k % 2}" title="{label} serving ({name(pid)} served)">{label}</span>'


def won_tag(r):
    """Whether the serving team won the point (from who served next: app.py), else ? when it couldn't be told."""
    w, k = r.get("winner"), r.get("serving_team")
    if w is None or k is None:
        return '<span class="pbv-unknown" title="winner not known">?</span>' if "winner" in r else ""
    who = " & ".join(name(p) for p in s["teams"][w]["players"])
    return (f'<span class="pbv-won" title="serving team won: {who}">won</span>' if w == k
            else f'<span class="pbv-lost" title="serving team lost: point to {who}">lost</span>')


WIDTHS = [0.7, 1.6, 1.0, 1.2, 2.4, 1.0]
col_video, col_rallies = st.columns([3, 1.6]) if s["rallies"] else (st.container(), None)
if col_rallies is not None:
    with col_rallies:
        st.markdown("**Rallies**")
        with st.container(height=440, key="rally-table"):
            for c, h in zip(st.columns(WIDTHS, vertical_alignment="center"),
                            ("#", "start", "shots", "length (s)", "serving", "point")):
                c.markdown(f"**{h}**")
            for i, r in enumerate(s["rallies"], 1):
                at = r["start"] / s["fps"]
                c = st.columns(WIDTHS, vertical_alignment="center")
                c[0].markdown(str(i))
                c[1].button(clock(at), key=f"rally-{run.name}-{i}", type="tertiary", on_click=go_to,
                            args=(max(0.0, at - LEAD_S),), help=f"Jump to rally {i}")
                c[2].markdown(str(r.get("shots", "")))
                c[3].markdown(str(r.get("duration_s", "")))
                c[4].markdown(serving_tag(i), unsafe_allow_html=True)
                c[5].markdown(won_tag(r), unsafe_allow_html=True)
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
// the rally the video is on: the last one started (LEAD_S before its serve) by the current time, highlighted in
// the rally table and scrolled into view inside it. One timer for the page; each render gives it this match's rallies.
P.__pbvStarts = %(starts)s; P.__pbvPrefix = %(prefix)s; P.__pbvShown = null;
if (!P.__pbvRallyTimer) {
  P.__pbvRallyTimer = P.setInterval(() => {
    const v = video(), starts = P.__pbvStarts || [];
    let i = -1;
    if (v) for (let k = 0; k < starts.length && starts[k] <= v.currentTime + 1e-3; k++) i = k;
    const id = i < 0 ? null : P.__pbvPrefix + (i + 1);
    const btn = id && D.querySelector('.' + CSS.escape(id));
    const row = btn && btn.closest('[data-testid="stHorizontalBlock"]');
    if (row && row.classList.contains('pbv-current')) return;
    D.querySelectorAll('.pbv-current').forEach(e => e.classList.remove('pbv-current'));
    if (!row) return;
    row.classList.add('pbv-current');
    for (let box = row.parentElement; box && box !== D.body; box = box.parentElement) {  // the table's scroll box
      const st = P.getComputedStyle(box).overflowY;
      if ((st === 'auto' || st === 'scroll') && box.scrollHeight > box.clientHeight) {
        const r = row.getBoundingClientRect(), b = box.getBoundingClientRect();
        if (r.top < b.top || r.bottom > b.bottom) box.scrollTop += r.top - b.top - (b.height - r.height) / 2;
        break;
      }
    }
  }, 300);
}
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
        # rally starts (from LEAD_S before the serve, as the links jump) for highlighting the rally being played
        starts = [max(0.0, r["start"] / s["fps"] - LEAD_S) for r in s["rallies"]]
        components.html(PLAYER_HTML % {"key": DASHBOARD_KEYFRAME_S, "starts": json.dumps(starts),
                                       "prefix": json.dumps(f"st-key-rally-{run.name}-")}, height=30)
        start_s, clicks = jump.get(run.name, (0, 0))
        # the player only seeks when start_time changes (whole seconds), so a repeat click on the same rally
        # alternates between LEAD_S and LEAD_S + 1 s before the serve
        st.video(str(video), start_time=max(0, int(start_s) - (clicks + 1) % 2), autoplay=bool(clicks), muted=True)
    else:
        st.caption("No annotated video in this run.")

def with_kitchen(row, prefix=""):
    """Each kitchen percentage and the number of rallies it is of, as one readable column ("91% of 23")."""
    out = dict(row)
    for role in ("serving", "receiving"):
        pct, of = out.pop(f"{prefix}kitchen_{role}_pct", None), out.pop(f"{prefix}kitchen_{role}_of", None)
        if of is not None:
            label = f"{'Both at kitchen' if prefix else 'Kitchen'}, {role}"
            out[label] = f"{pct:.0f}% of {of}" if pct is not None else "–"
    return out


st.subheader("Players")
SPEED = ("avg_speed_ms", "max_speed_ms", "avg_shot_kmh", "max_shot_kmh")  # not shown: not reliable yet
DEPTH_KEYS = ("serve_", "return_")  # shown in their own table below
st.dataframe(pd.DataFrame([with_kitchen({"player": name(pid), **{k: v for k, v in {**s["players"].get(pid, {}),
                                                                                   **s["shot_summary"].get(pid, {})}.items()
                                                                  if k not in SPEED and not k.startswith(DEPTH_KEYS)}})
                           for pid in pids]).fillna(0), hide_index=True)
st.caption("Kitchen, serving / receiving: how often the player got up to the kitchen line when their team served "
           "or received, of the rallies long enough to get there (5+ shots serving, 3+ receiving).")

if s.get("teams"):
    st.subheader("Teams")
    st.caption("Both partners up at the kitchen line (or in the kitchen) together at some point in the rally, "
               "of the rallies long enough to get there: 5+ shots when serving, 3+ when receiving.")
    st.dataframe(pd.DataFrame([with_kitchen({"team": " & ".join(name(p) for p in t["players"]),
                                             **{k: v for k, v in t.items() if k != "players"}}, prefix="both_")
                               for t in s["teams"]]), hide_index=True)

DEPTH_ROLES = (("serve", "Serve"), ("return", "Return"))
if any(f"{role}_n" in s["players"].get(pid, {}) for pid in pids for role, _ in DEPTH_ROLES):
    st.subheader("Serve & return")
    st.caption("How deep each player's serves and returns went, and how often their team won the point after "
               "them. Each cell: how many, and the points won. Short: more than 3 m in from the baseline; "
               "medium: 1.5–3 m; deep: the back 1.5 m. Depth (m): the median, metres in from the baseline. "
               "The bounce is used when the ball tracker found it, else where the next player hit the ball. Who "
               "won is read from who serves next, so points before a gap or an undetected rally are left out.")

    def cell(n, pct):
        return "–" if not n else f"{n} · {pct:.0f}%" if pct is not None else f"{n} · ?"

    rows = []
    for pid in pids:
        p = s["players"].get(pid, {})
        row = {"player": name(pid)}
        for role, label in DEPTH_ROLES:
            for k in ("short", "medium", "deep"):
                row[f"{label} {k}"] = cell(p.get(f"{role}_{k}_n", 0), p.get(f"{role}_{k}_won_pct"))
            row[f"{label} all"] = cell(p.get(f"{role}_n", 0), p.get(f"{role}_won_pct"))
            row[f"{label} depth (m)"] = p.get(f"{role}_depth_m")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={f"{label} depth (m)": st.column_config.NumberColumn(format="%.1f")
                                for _, label in DEPTH_ROLES})

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
