# 🏓 Pickleball-Vision

**Computer-vision analytics for pickleball video, from a tripod camera or a camera you wear while playing.**

Pickleball-Vision turns raw match footage into player tracking, ball trajectories, hits and
bounces, shot types, rallies, rule faults, stroke form from body pose, court heatmaps and an
annotated replay. It also comes with an interactive dashboard.

It adapts the pipeline of [Badminton-VisionAI](https://github.com/amramer/Badminton-VisionAI)
(YOLO + ByteTrack tracking, homography court mapping, shot detection, dashboard) to pickleball's
court and rules. It adds the pose-based stroke analysis popularised by
[IsoCourt](https://www.isocourt.fit/). It also adds a **POV mode** for footage from a
chest/head-mounted action camera, which most recreational players actually have.

### 🌐 Try it in your browser: **https://arvanj.github.io/Pickleball-Vision/**

No install. **Live** uses your phone or laptop camera; **Analyze a clip** takes a recorded video. Detection
runs on your device with a small YOLO model, and the video is never uploaded.
[More below](#website-live-and-clip-analysis).

---

## Contents

- [Website: live and clip analysis](#website-live-and-clip-analysis)
- [Features](#features)
- [Two modes: fixed vs POV](#two-modes-fixed-vs-pov)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line reference](#command-line-reference)
- [Outputs](#outputs)
- [Dashboard](#dashboard)
- [Project structure](#project-structure)
- [Measured accuracy](#measured-accuracy)
- [Tuning](#tuning)
- [Recording tips](#recording-tips)
- [Testing](#testing)
- [Limitations and roadmap](#limitations-and-roadmap)
- [Credits and licences](#credits-and-licences)

---

## Website: live and clip analysis

**https://arvanj.github.io/Pickleball-Vision/** is a static site (in [`docs/`](docs)) that runs fixed-camera
analysis entirely in the browser:

- **Live:** start the camera, tap 4+ court landmarks on the video, and play. Players, the ball trail,
  projected court lines, the latest shot, rallies, faults, a top-down court and coach notes update every second.
  The screen stays awake during a session.
- **Analyze a clip:** choose or drop a video, pick a start and end time, tap the court landmarks, then press
  Analyze. Progress goes through reading frames, detecting players and ball, finding hits and bounces, and writing
  coach notes. Afterwards the video replays with the overlay, and clicking a shot jumps to it.
- **Ball detail:** *Standard* (640 px model, fast, the default for live) or *High* (1280 px, about 4× slower,
  the default for clips). High is what finds a small or distant ball.
- **Download results** as JSON (same fields as the Python `stats.json`). The calibration is remembered per
  camera resolution.

How it works:
- [`docs/engine.js`](docs/engine.js) is a line-for-line port of the Python court, event, shot and stats logic.
  [`tests/engine_parity.mjs`](tests/engine_parity.mjs) checks that it gives exactly the same events, shots and
  player stats as Python on a real output folder.
- [`docs/detector.js`](docs/detector.js) runs YOLOv8n (exported to ONNX) with onnxruntime-web: WebGPU when
  available, otherwise WebAssembly.
- Clips are sampled by playing the video slowly and catching frames as they're decoded, which is about 8× faster
  than seeking frame by frame.

Compared with the Python pipeline, the website uses the small YOLOv8n model and a simple IoU tracker, and has no
pose (stroke form, serve-height fault) or POV mode. On a Mac's GPU, Standard runs at about 6–8 frames per second
live, and High analyzes about 2 frames per second of clip.

## Features

### 🎥 Computer-vision pipeline
- **Player detection and tracking.** YOLOv8 finds people and ByteTrack keeps a persistent ID for each one across frames.
  In fixed mode, everyone near the court is recorded, then the players are picked out of them
  (`pbv/players.py`):
  - people more than 1 m past a sideline or 3 m behind a baseline are dropped (refs, benches, other courts)
  - IDs that stand still beside the court for their whole track are dropped (line judges, spectators)
  - at most 2 people are kept on each side of the net, preferring those on the court
  - each one gets one of 4 fixed IDs (`P1`–`P4`). While a player is in view the ID follows them by
    position, so a ByteTrack ID switch doesn't split one player into several.
  - **Clothing re-ID** (`pbv/appearance.py`): each ID keeps a colour profile of the player's head (cap or
    hair), shirt and shorts. A player who walks out of frame or is hidden gets their own ID back
    when they return, partners who were both lost don't come back swapped, and a stranger walking on
    court doesn't take a missing player's ID. Players keep their IDs after teams change ends.
    Players dressed alike (e.g. both in black) are told apart mostly by position, so a long
    disappearance by both of them can still swap them.
  In POV mode, your own arms, hands and legs are filtered out.
- **Ball tracking** (`pbv/balltrack.py`). YOLO reports up to 8 ball *candidates* per frame, not just the most
  confident box: a full-frame pass plus a full-resolution crop around where the ball is predicted to be (a far
  ball is a few pixels and vanishes when the frame is downscaled). Each candidate gets a motion score (frame
  difference: with a fixed camera, heads, lights and logos stand still). Candidates are linked into tracklets
  with a constant-velocity prediction, and the best-scoring tracklets win, so a blurred ball in flight beats a
  sharper head. In fixed mode a candidate must also fit the court at the depth its *size* implies (a 74 mm ball
  looks ~3× bigger at the near baseline than the far one), which drops balls on neighbouring courts that land
  on the same pixels. Short gaps are then filled by interpolation; long gaps are left empty (out of play).
  **The ball often leaves the picture** (a wide ball near the camera, a lob over the top). When the track runs
  into a frame edge and is heading out, the gap is marked *off-screen*: it is never interpolated (a straight
  line across the frame would invent a path), it counts as ball in play for serve detection, and its time
  doesn't count toward the pause that ends a rally.
- **Paddle detection.** In POV mode, paddles (COCO "tennis racket") are detected in the same pass as the ball,
  to recognise your own paddle in frame.
- **Court mapping (fixed mode).** You click 4 or more court landmarks once: corners, kitchen-line ends and centres,
  or baseline centres. Corners that are off-screen or hidden behind the net can be skipped. A homography then maps every
  pixel on the ground to real court metres (20 × 44 ft court, 7 ft non-volley zone).
- **Camera-motion compensation (POV mode).** Frame-to-frame camera pan is measured with phase correlation and
  removed from the ball track, so hits and bounces are found relative to the world, not to your moving head.
- **Pose estimation.** YOLOv8-pose keypoints at each contact frame give stroke form.

### 🏓 Shot intelligence
- **Hit and bounce detection.** These come from turning points in the ball's path plus how close it is to a player.
  A bounce always shows up as the ball's lowest point on screen. A hit is a turning point next to a player.
- **Your own swings (POV).** A sudden camera jolt while the ball is in play counts as your swing, but only
  if your own paddle is visible and an opponent hit the ball to you just before.
- **Shot types (fixed):** serve, return, drive, dink, drop (e.g. third-shot drop), lob, volley, groundstroke.
- **Shot types (POV):** serve, return, volley, groundstroke.
- **Volleys.** A shot is a volley when there was no bounce since the previous hit.
- **Shot speed (fixed).** Horizontal court-plane speed from the hitter to where the ball lands (or to the next hitter).
- **Landing spots (fixed).** Bounce positions in court metres.
- **Rallies.** Hits are grouped into rallies (a gap of more than 2.5 s starts a new rally), with shot counts and durations.

### ⚖️ Rule checks
| Fault | How it's detected | Mode |
|---|---|---|
| Kitchen volley | Volley hit while the hitter's feet are in the non-volley zone | fixed |
| Out | Bounce lands outside the court lines | fixed |
| Net | Ball bounces on the hitter's own side | fixed |
| Serve above waist | At serve contact the paddle wrist is clearly above the waist (pose) | fixed |

### 🧍 Stroke form (pose)
At every shot by a tracked player:
- **Paddle arm** (left or right)
- **Elbow angle** at contact
- **Contact height** (0 = waist, 1 = shoulders), plus **below waist** and **overhead** flags

### 📊 Player metrics (fixed)
- Distance covered (m)
- Average and top movement speed (m/s)
- Percentage of rally time up at the kitchen line (within 3 ft behind it, not inside it)
- **Getting to the kitchen**: how often a player gets up to the kitchen line (or into the kitchen) in a rally,
  when their team is serving and when it is receiving, and how often **both partners** are up there together.
  Only rallies long enough to get there count: 5+ shots when serving (the serving team moves in after its third
  shot), 3+ when receiving (in behind the return). "Got there" = 0.2 s there between the serve and the last hit.
- **Serve and return depth, and points won**: how far in from the baseline each serve and return went (deep: the
  back 1.5 m; medium: 1.5–3 m; short: nearer the net), and how often the player's team won the point after a
  short, medium or deep one. Depth is the ball's bounce when the tracker found it (about half the time), else
  where the next player hit it, plus the 1.8 m the ball typically carries on after bouncing. Who won is read
  from who serves next: only the serving team scores, so the same server again (from the other half) means
  the serving team won; the partner or the other team serving next means it lost. Points before a gap of more
  than a minute (between games) or a misread serve are left unknown. The server is right for about 95% of
  labelled serves, but an undetected rally makes the point before it wrong, so expect roughly 85% of winners right.
- Court side
- Positional **heatmap** per player

### 🖥️ Outputs and dashboard
- Annotated video with player boxes and IDs, ball trail, a live mini-court (fixed), and serve / return / 3rd-shot labels
- `stats.json` with every number above
- A Streamlit dashboard: replay, player table, shot mix chart, heatmaps, shot log with a fault filter, rally table
- **Caching.** Tracking (the slow part) is saved to `tracking.json` and reused when only analysis settings change.

---

## Two modes: fixed vs POV

| | `--mode fixed` (default) | `--mode pov` |
|---|---|---|
| Camera | Tripod or fence mount, whole court in view | Worn by a player (chest, head, paddle) |
| Setup | Click 4+ court landmarks once | None |
| Player tracking, ball trail, rallies | ✅ | ✅ (camera pan removed) |
| Other players' hits + stroke form | ✅ | ✅ |
| Your own swings | as a tracked player | ✅ camera jolt + your paddle in view |
| Shot types | serve, return, drive, dink, drop, lob, volley, groundstroke | serve, return, volley, groundstroke |
| Speed, landing spots, heatmaps, distance, kitchen % | ✅ | ❌ needs a fixed view of the court |
| Faults | kitchen volley, out, net, serve above waist | — (POV "serves" are too unreliable to fault) |

**Why POV can't measure court positions:** a homography maps the image to the court only while the
camera stays still. A worn camera moves dozens of pixels every frame and often sees only part of the
court through a fisheye lens. So any court-plane number would be wrong.

---

## How it works

```
                         ┌────────────────────────── fixed mode ──────────────────────────┐
video ─► frames ─┬─► court homography (4+ clicked landmarks) ─► feet → court metres
                 │
                 ├─► YOLOv8 + ByteTrack ─► people near the court ─► 4 players {P1–P4: bbox, court xy}
                 ├─► YOLOv8 (full frame + crop at predicted ball) ─► ball candidates + motion (+ paddles in POV)
                 └─► phase correlation ─► camera shift per frame            (POV only)
                                 │
                                 ▼
       ball path: candidates linked over time, best tracks inside the court (by size-implied depth) kept,
                  short gaps interpolated, camera pan removed (POV)
                                 │
                                 ▼
       events: turning points in ball y ─► hit (near a player) / bounce
               + your swings: camera jolt + own paddle + opponent hit just before   (POV)
               + de-duplication: one contact per 0.35 s, one hit per side per 1 s (fixed) / 0.5 s (POV)
                                 │
                                 ▼
       shots: type, volley, speed + landing (fixed), rally grouping, rule faults
                                 │
                                 ▼
       YOLOv8-pose at each contact ─► paddle arm, elbow angle, contact height
                                 │
                                 ▼
       stats.json · heatmaps · annotated.mp4 ─► Streamlit dashboard
```

**Hits vs bounces (fixed mode).** In the image, a bounce always appears as a local *maximum* of the
ball's y-coordinate (it falls, then rises). A far-side hit is a local *minimum* (the ball comes up the
screen toward the far player, then goes back down). A near-side hit is a local maximum right next to a
near-side player.

**Serves (fixed mode)** are found from the ball itself, not just from hits:
- the ball is out of play near a player within 3 m of their own baseline for 1 s: held or bounced
  at their body (not their head, since caps get detected as balls), or not seen at all while they
  stand still (a ball in a far server's hand is too small to detect)
- then it's hit below head height and flies on one continuous path to the other side within 0.6 s
- hits detected while the server bounces the ball beforehand are dropped, a serve always starts a new
  rally, and there's no second serve within 4 s (the ball is in play)

**Shot classification (fixed mode)**, in priority order:
1. A detected serve → **serve**; the next hit, from across the net → **return**
2. Speed ≥ `--drive-speed` (10 m/s) → **drive**
3. Lands in the kitchen → **dink** if hit from near the kitchen line, otherwise **drop**
4. Lands deep (within 2 m of the baseline) after more than 1 s in the air → **lob**
5. Otherwise **volley** (no bounce since the last hit) or **groundstroke**

---

## Installation

Requires Python 3.10+. Runs on Ubuntu (CUDA if available, otherwise CPU) and macOS (the Apple-silicon GPU is
used automatically).

Uses [uv](https://docs.astral.sh/uv/) for the virtual environment and packages. Install it once:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
git clone https://github.com/ArvanJ/Pickleball-Vision.git
cd Pickleball-Vision
uv venv .venv --python 3.12        # uv downloads that Python if it isn't installed
uv pip install -r requirements.txt
```

No need to activate the environment: prefix commands with `uv run` (as below), which uses `.venv/`.
Keep it at `.venv/`: `training/run_app.sh` runs `.venv/bin/python`.

YOLO weights (`yolov8m.pt`, `yolov8n-pose.pt`) download automatically on first run (~60 MB). The pickleball detector `models/ball-r1.pt` (22 MB) and the serve / return / third-shot detector `models/openings.pt` (1.5 MB) are in the repo.

**AV1 video** (most YouTube downloads, `.webm` / some `.mkv`): OpenCV often can't decode it. Convert to H.264 first:
```bash
ffmpeg -c:v libdav1d -i match.mkv -an -c:v libx264 -crf 20 match.mp4      # or -c:v h264_nvenc on an NVIDIA GPU
```

---

## Quick start

**Fixed camera:**
```bash
uv run app.py data/match.mp4
```
A window opens and asks for court landmarks one at a time, starting with the four corners. Click each
one, or press **s** to skip a landmark you can't see (for example a corner off the edge of the frame or
behind the net). Once at least 4 are placed (not all on one line), press **Enter**: the court lines those
points imply are drawn in red over the frame. If they sit on the painted lines, press **Enter** again to
save; any other key goes back to clicking. **u** undoes, **r** resets and **Esc** aborts. The calibration
is saved and reused.

**What fixed mode reports:** the **serve, return and third shot** of each rally (who hit each one, when),
found by the learned detector in `models/openings.pt` (rules are the fallback if the file is missing), plus
player movement stats and heatmaps. Players out of view are assumed to be where they were last seen. A long
video can run under a watchdog that restarts it if it crashes or stalls:
```bash
nohup training/run_app.sh outputs/match data/match.mp4 --no-pose >/dev/null 2>&1 &
training/run_app.sh --status outputs/match
```
To improve the detector on new footage, label rallies with `training/annotate_openings.py`, score with
`training/eval_openings.py` and retrain with `training/train_openings.py` (`--cv` for held-out scores).

Landmarks: `far-left`, `far-right`, `near-right`, `near-left`, `near-baseline-center`,
`far-baseline-center`, `near-kitchen-left|center|right`, `far-kitchen-left|center|right`.

**POV camera:**
```bash
uv run app.py data/my_gopro.mp4 --mode pov
```

**Analyse a slice of a long video** (recommended: POV clips are often 10–20 minutes):
```bash
uv run app.py data/my_gopro.mp4 --mode pov --start 120 --max-frames 1800
```

**Open the dashboard:**
```bash
uv run streamlit run dashboard.py
```

**Headless / scripted calibration** (all 4 corners visible):
```bash
uv run app.py data/match.mp4 --corners 412,318,1508,318,1840,1012,80,1012
```
Or with any 4+ named landmarks, e.g. a floor-level phone that sees only the near half:
```bash
uv run app.py IMG_1577.MOV --court-points "near-left=1537,1613;near-baseline-center=3330,1613;near-kitchen-left=1642,1398;near-kitchen-center=2354,1404"
```

---

## Command-line reference

### `app.py`
| Flag | Default | Description |
|---|---|---|
| `video` | — | Input video (any format OpenCV reads: MP4, MOV, AVI…) |
| `--mode {fixed,pov}` | `fixed` | Camera type |
| `--out DIR` | `outputs/<name>` or `outputs/<name>-pov` | Output folder |
| `--corners x1,y1,…,x4,y4` | — | Fixed mode: court corners in pixels (far-left, far-right, near-right, near-left); skips the click window |
| `--court-points "name=x,y;…"` | — | Fixed mode: 4+ named landmarks (see Quick start) when some corners aren't visible |
| `--start SECONDS` | `0` | Where to start in the video |
| `--max-frames N` | all | How many frames to analyse |
| `--player-model` | `yolov8m.pt` | YOLO model for people |
| `--ball-model` | fixed: `models/ball-r1.pt`; pov: `yolov8m.pt` | YOLO model for the ball. Fixed mode uses the pickleball detector trained with `training/` on fixed-camera video (round 1). POV keeps generic COCO "sports ball", which also detects the paddle used to confirm swings |
| `--ball-class` | from the model | Class id of the ball in `--ball-model`: found from the class names (`pickleball`, else COCO `sports ball` = 32); `-1` = any class |
| `--pose-model` | `yolov8n-pose.pt` | YOLO pose model |
| `--no-pose` | off | Skip stroke-form analysis |
| `--drive-speed` | `10.0` | Fixed mode: shot speed (m/s) above which a shot is a drive |
| `--force` | off | Re-run tracking even if a matching `tracking.json` exists |
| `--no-render` | off | Skip `annotated.mp4` and `dashboard.mp4` (about 1 GB per 40 minutes of play). Training needs only the data files |

### `calibrate.py`
```bash
uv run calibrate.py match.mp4 outputs/match/court.json --frame 300
```
Re-do the court calibration on any frame.

### `demo/build.py`
```bash
uv run demo/build.py demo/replay.html "outputs/IMG_1577:Fixed · floor phone" "outputs/clip-pov:POV · chest cam"
```
Builds a single self-contained HTML replay from one or more output folders. It shows synced camera and
court views, a shot timeline, a player table and a shot log. Only detections are embedded, never video
pixels, so the file is safe to share. Per-run captions come from `demo/meta.json`.

---

## Outputs

Each run writes to `outputs/<video name>[-pov]/`:

| File | Contents |
|---|---|
| `annotated.mp4` | Replay with boxes, IDs, ball trail, mini-court (fixed), shot + fault labels. H.264 so it plays in browsers |
| `stats.json` | Everything below |
| `tracking.json` | Raw per-frame detections (cache: everyone near the court as `people`, the selected `players`) plus the settings that produced them |
| `court.json` | Court calibration: clicked image points and their court coordinates (fixed) |
| `appearance.npy` | Clothing-colour descriptor per tracked person (cache for re-ID, fixed) |
| `heatmap_P<id>.png` | Court-coverage heatmap per player (fixed) |

### `stats.json` structure
```jsonc
{
  "video": "...", "mode": "fixed", "start_s": 0, "fps": 30, "frames": 900,
  "players": {                         // fixed: movement metrics; POV: seconds visible
    "3": {"seconds": 28.1, "side": "near", "distance_m": 41.2,
          "avg_speed_ms": 1.46, "max_speed_ms": 4.9, "kitchen_pct": 37.5,   // fixed: % of total rally time within 3 ft behind the kitchen line (not inside it)
          "kitchen_serving_pct": 60.0, "kitchen_serving_of": 25,       // fixed: got up to the kitchen in 60% of 25 rallies their team served (5+ shots)
          "kitchen_receiving_pct": 100.0, "kitchen_receiving_of": 25,  // ... and when receiving (3+ shots); pct null when _of is 0
          "serve_n": 17, "serve_depth_m": 1.6, "serve_won_pct": 38.9,   // fixed: serves with a depth, median metres from the baseline, % of points won
          "serve_deep_n": 9, "serve_deep_won_pct": 44.4,              // ... per depth (short / medium / deep); won_pct null when no winner is known
          "return_n": 15, "return_depth_m": 1.7, "return_won_pct": 71.4, "return_deep_n": 7, "return_deep_won_pct": 57.1}
  },
  "teams": [                           // fixed: both partners up at the kitchen together, same rallies as above
    {"players": ["1", "2"], "both_kitchen_serving_pct": 44.0, "both_kitchen_serving_of": 25,
     "both_kitchen_receiving_pct": 72.0, "both_kitchen_receiving_of": 25}
  ],
  "shot_summary": {                    // per player ("me" = you, in POV)
    "3": {"shots": 9, "n_serve": 3, "n_return": 3, "n_third": 3,   // fixed: serve / return / third shot only
          "avg_shot_kmh": 31.4, "max_shot_kmh": 52.0}
  },
  // fixed: every rally with a detected serve; "shots" counts all its shots (a run of hits by one side counts once), not just the three reported below
  // serving_team / winner: index into teams (fixed; winner null when it can't be told)
  "rallies": [{"start": 120, "end": 410, "shots": 7, "served": true, "duration_s": 9.7, "serving_team": 0, "winner": 1}],
  "shots": [{
    "frame": 120, "time_s": 4.0, "player": "3", "rally": 1, "n_in_rally": 1,
    "type": "serve", "speed_kmh": 38.2,
    "hitter": [3.1, 13.2], "landing": [1.9, 2.4],          // court metres (fixed)
    "depth_m": 2.4, "depth": "medium", "depth_from": "bounce", "won": false,   // serve / return only (fixed)
    "form": {"arm": "right", "elbow_angle": 142.0,
             "contact_below_waist": true, "overhead": false}
  }],
  "events": [{"frame": 150, "type": "bounce", "player": null, "px": [812.0, 604.5]}]
}
```
Court coordinates: `x` runs across the court (0–6.10 m) and `y` along it (0 = far baseline,
13.41 m = near baseline, 6.71 m = net).

---

## Dashboard

`uv run streamlit run dashboard.py` lists every analysed video in `outputs/` and shows:

- **Headline numbers:** duration, rallies, shots
- **Annotated replay** beside a rally table (#, start, shots, length, serving team, whether it won the point): click a start time to jump the video to it (1 s before the serve)
- **Player table:** movement metrics, shot counts by type, shot speeds
- **Serve & return:** per player, how many serves and returns were short / medium / deep, with the share of points their team won after each, and the median depth
- **Shot mix:** bar chart of shot types per player
- **Court coverage:** heatmaps (fixed mode)
- **Shot log:** every shot with timing, type, speed, landing spot, stroke form

---

## Project structure

```
Pickleball-Vision/
├── app.py              # Pipeline entry point (both modes)
├── calibrate.py        # Click-to-calibrate court corners
├── dashboard.py        # Streamlit dashboard
├── pbv/
│   ├── constants.py    # Official court geometry (metres)
│   ├── court.py        # Image ↔ court homography; kitchen / side / in-bounds checks
│   ├── tracking.py     # YOLO person tracker (ByteTrack), ball candidates + paddle detector, pose estimator
│   ├── balltrack.py    # Ball candidates -> track: motion score, tracklet linking, size-aware court region
│   ├── players.py      # Picks the 4 players from everyone tracked; stable per-player IDs
│   ├── appearance.py   # Clothing-colour descriptors for re-identifying players
│   ├── events.py       # Ball cleanup, hits/bounces, de-duplication, shot types, rallies, faults,
│   │                   #   POV camera compensation, swing detection
│   ├── analytics.py    # Player movement stats, shot summaries, stroke form from keypoints
│   └── draw.py         # Frame annotation, mini-court, heatmaps
├── docs/               # The website (GitHub Pages): index.html, app.js, engine.js, detector.js, models/
├── demo/
│   ├── build.py        # Builds the shareable HTML replay from output folders
│   ├── template.html   # Replay page (camera view, court view, timeline, tables)
│   └── meta.json       # Per-run captions for the replay
├── training/           # Ball-detector training from your own videos (see "Training a ball detector")
│   ├── scan.py         #   1. teacher model -> ball candidates per frame
│   ├── labels.py       #   2. ball-like tracks -> pseudo-labels -> full-resolution 640x640 crop dataset
│   ├── train.py        #   3. fine-tune YOLOv8s on the crops
│   ├── evaluate.py     #   4. detectors and trackers compared on a held-out video
│   └── annotate.py     #   hand-label a ground-truth test set
├── tests/
│   └── test_core.py    # Self-checks for all non-ML logic
└── requirements.txt
```

---

## Measured accuracy

> These are honest numbers from limited footage. Treat them as a baseline, not a benchmark.

**POV mode** was hand-checked on a 30-second, 4K/30 fps chest-mounted clip of recreational doubles
(3 live points, about 12 hits). Every detection was compared frame by frame with the video:

| Version | Detections | Precision | Recall |
|---|---|---|---|
| First version | 30 | 37% | 92% |
| + de-duplication, net-crossing rule, own-paddle check for your swings | 16 | **56%** | **75%** |

- Ball detected in **56%** of frames with the stock COCO model.
- Players detected reliably when in view. **IDs are not stable:** the same person gets a new ID after leaving and re-entering the frame.
- **Main remaining errors:**
  - players passing or bouncing the ball between points look like hits
  - your swings when your paddle is out of view are missed
  - players on neighbouring courts are occasionally picked up

**Fixed mode** was run on one minute of indoor doubles filmed by an iPhone lying on the floor
behind the near baseline (4K/60 fps). What it showed:
- Calibration works from 4 non-corner landmarks. The projected kitchen line, centre line and far
  sideline land on the painted lines.
- The ball was detected in only **23%** of frames. From floor level the far half of the court is about
  70 pixels tall, so the pickleball is a few pixels and far-side hits are missed.
- Near-side strokes by the player closest to the camera are found. False hits come from bouncing the
  ball before a serve and from carrying it between points. The net-crossing rule (one hit per side
  per second) and only computing speed toward a bounce or a different player came from this check.
- **A raised camera matters more than any setting.** See [Recording tips](#recording-tips).

**Speed:** about 3–6 frames per second on an Apple M-series GPU at 4K (two YOLO models per frame),
so a 10-minute POV clip takes roughly an hour. Use `--start` / `--max-frames`.

---

## Tuning

All thresholds are plain function arguments with docstrings:

| What | Where | Default |
|---|---|---|
| Rally gap | `classify_shots` / `pov_shots` `rally_gap_s` | 2.5 s |
| Duplicate-contact window | `filter_hits` `min_gap_s` | 0.35 s |
| One hit per side window | `filter_hits` `same_side_s` (fixed) / `pov_shots` `same_side_s` (POV) | 1.0 s / 0.5 s |
| Serve-above-waist fault | `add_stroke_form` in `app.py` | wrist > half-way from waist to shoulders |
| Drive speed | `--drive-speed` | 10 m/s |
| Dink / drop / lob zones | `shot_type` | kitchen ± 1 m, last 2 m before the baseline |
| Hit reach around a player | `find_events` `reach` | 0.6 × player height |
| Turning-point prominence | `find_events` `min_prominence` | 8 px |
| Swing jolt threshold (POV) | `find_swings` `rel_thresh` | 2% of frame width per frame (or 4× median motion) |
| Own-paddle size (POV) | `find_swings` `own_paddle` | 20% of frame height |
| Ball gap filling | `clean_ball` `max_gap` | 8 frames |
| Ball candidates per frame / min confidence | `BallDetector` `top_k` / `conf` | 8 / 0.05 |
| Ball track linking gate | `link_tracklets` `gate` | 2.5% of frame width (+ a share of the speed) |
| Ball off-screen detection | `offscreen_frames` `margin` / `max_frames` | within 5% of an edge and heading out / up to 3 s |
| Ball court region | `BallRegion` `side` / `end` / `up` / `loose` / `tight` | 1.5 m / 3 m / 8 m / box ≤2.6× the ball / never smaller |

Because tracking is cached, changing any of these and re-running `app.py` only re-does the analysis
(seconds to a few minutes), not the detection.

---

## Recording tips

**Fixed mode (best results):**
- Tripod or fence mount behind a baseline, as high as possible (2–3 m+). A phone on the floor
  squeezes the far half of the court into a sliver, and the ball becomes nearly invisible
- Ideally all four corners are in frame. If not, make sure at least 4 landmarks are visible (e.g. both
  near corners plus the two ends of the near kitchen line)
- 60 fps if your camera supports it. The ball moves fast and blurs at 30 fps
- Keep the camera still for the whole recording. If it moves, re-run `calibrate.py`

**POV mode:**
- Chest mount is steadier than head mount, which helps both camera compensation and swing detection
- Point it slightly up so the far court and net stay in frame during ready position
- Wide/fisheye is fine; the pipeline doesn't need straight lines in POV mode

---

## Testing

```bash
uv run tests/test_core.py        # or: uv run -m pytest tests
node tests/engine_parity.mjs outputs/<run>   # web engine vs Python on a real run
```

Covers:
- homography round-trips, kitchen checks, and calibration from non-corner landmarks
- ball outlier removal and gap filling; a ball leaving the frame is off-screen (not filled, rally not split)
- ball candidate selection: a moving ball beats a sharper still object; the size-aware court region keeps a
  big (near) ball beside the court and drops a small (far) one on the next court; pseudo-labels skip logos
- hit/bounce detection on a synthetic trajectory
- shot classification and faults (serve, return, drop, kitchen volley)
- movement metrics and stroke form from keypoints
- heatmap rendering
- camera-shift sign and compensation
- own-body filtering, swing detection with and without paddle data
- duplicate-hit removal (time and same-side) and POV rally/volley logic

---

## Limitations and roadmap

**Current limitations:**
- **Ball detection** uses COCO's generic "sports ball" class. Small, motion-blurred pickleballs are often missed.
- **Fixed mode calibration is manual**, and the whole court must be visible.
- **Fixed-mode speeds** are average horizontal speeds to the bounce point, not the speed off the paddle.
- **Shot types and faults are rule-based** heuristics.
- **POV mode** has no court geometry. Player IDs churn, and ball handling between points can look like play.

**Roadmap:**
- [x] Pickleball-specific ball detector trained on your own videos (`training/`), the default `--ball-model` (`models/ball-r1.pt`)
- [ ] TrackNet-style heatmap ball tracker for blur and occlusion (as Badminton-VisionAI uses)
- [ ] Automatic court-keypoint detection to replace manual calibration
- [ ] Learned re-ID embeddings (fixed mode uses colour histograms) for stable IDs, especially in POV
- [ ] Learned shot classifier from labelled rallies
- [ ] PDF coach report and Docker images
- [ ] A labelled evaluation set with a scoring script in the repo

---

## Credits and licences

- Pipeline design inspired by [Badminton-VisionAI](https://github.com/amramer/Badminton-VisionAI) by amramer
- Stroke-analysis idea inspired by [IsoCourt](https://www.isocourt.fit/)
- Detection, tracking and pose by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0) with ByteTrack
- Dashboard built with [Streamlit](https://streamlit.io)

Note that Ultralytics YOLO is AGPL-3.0 licensed. If you distribute or host this project as a service,
check that licence's terms.
