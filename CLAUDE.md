# CLAUDE.md

Pickleball video analytics: YOLO tracking of the ball and the four players, court homography, then rally and shot
statistics, a dashboard video and a Streamlit dashboard. README.md is the user guide (features, CLI, outputs); this
file is what to know when changing the code. The serve / return / third-shot model and the rally shot counter are
described in the "algorithms" doc (claude.ai), kept up to date with the code below.

## Commands

```bash
uv run app.py ~/yt-vids/<video>.mp4                     # full pipeline -> outputs/<video stem>/
nohup training/run_app.sh outputs/<run> <video> >/dev/null 2>&1 &   # same, under a watchdog (long videos)
training/run_app.sh --status outputs/<run>
uv run streamlit run dashboard.py
uv run tests/test_core.py                                # self-checks for the non-ML logic (no pytest installed)
```

`uv run` and `.venv/bin/python` are the same interpreter. GPU: one RTX 3060 under WSL2, which can hang mid-training;
long jobs run under the watchdog scripts in `training/` (`run_app.sh`, `run_train.sh`, `run_scan.sh`).

## Layout

- `app.py`: the pipeline. `analyse_fixed` is the fixed-camera analysis: ball cleanup and events
  (`pbv/events.py`), serves / returns / third shots from the openings model, rally shot counts from the hit model.
- `pbv/openings_model.py` (`models/openings.pt`): per-frame features (`frame_features`, 47 numbers: ball + 4 player
  slots) and `ContactNet` (temporal convs + bidirectional GRU), shared with the hit model.
- `pbv/hits_model.py` (`models/hits.pt`): rally shot counter. Inputs in this order: frame features, serve formation,
  pose features, audio (audio must stay last: training zeroes its columns). `rally_hits` turns hit peaks into a count.
- `pbv/audio.py`: paddle-pop features; `pbv/pose.py`: per-frame keypoints of every player (cached as
  `outputs/<run>/pose.npz`, a few minutes per video on first run).
- `training/`: labelling tools (`annotate_*.py`), trainers (`train_hits.py`, `train_openings.py`, `train.py` for the
  ball detector), evaluators. `training/TRAINING-NOTES.md` has the ball-detector history.

Learned models' settings travel in their checkpoint (`hidden`, `gru_layers`, `decode`, which inputs), so `app.py`
and the loaders must read them from there with defaults that match older checkpoints.

## Data

- Videos live in `~/yt-vids/` (copies of most on `/mnt/d/Videos/PB/BBB-Games/`; not all are byte-identical, and labels
  need the exact file they were made on). Never delete or overwrite hand labels.
- In git: hand labels (`training/data/{rallies,openings,gt,ids,hits}-*`) and, for each labelled game, the training
  inputs in `outputs/<run>/` (`tracking.json`, `court.json`, `stats.json`, `audio_feats.npy`, `pose.npz`), so
  training works from a clone without the videos. `outputs/` is otherwise ignored: when a new game is labelled,
  `git add -f` those five files. Built datasets (`ds-*`) and scan output (`train-*`, `test-*`) stay out.
- Labels removed from training go to `training/data/removed/` (not committed), outside the training globs.
- Each label file names its run in `meta.run` (an `outputs/<run>` folder, absolute path) and its source video in
  `meta.video` (absolute path in `~/yt-vids/`). Training skips labels whose run folder is gone.
  `removed/openings-bbb1..7` label `11-11-BBB.mp4`, whose video and runs no longer exist.
- `train_hits.py` caches per-run features next to the run (`openings_feats.npy`, `audio_feats.npy`,
  `pose_feats.npy`). Delete them after changing the feature code.

## Training and measuring the hit model

```bash
uv run training/train_hits.py training/data/rallies-* training/data/openings-* --cv 5 -v   # held-out score + save
```

- `--cv 5` trains fresh models per fold and counts each rally with models that never saw it, using the pipeline's
  own detected serves. Then it trains on everything and saves `--out` (default `models/hits.pt`). Write
  experiments to a scratch `--out`, not `models/hits.pt`.
- Results move by about ±3 rallies (of 154) between fold splits. Compare settings on two splits
  (`--split-seed 0` and `1`) before adopting anything; one split isn't enough.
- `--no-train RUN` keeps a game out of training but still tests on it (to judge a game's labels).
- A run takes 2–5 minutes; 256-wide networks are much slower (17 minutes) and were worse.

## Conventions

- Commit messages state the held-out result before and after (exact, within 1, mean abs error) for model changes,
  and list what was tried and not adopted.
- New behaviour goes behind a flag that defaults to the old behaviour until it has been measured.
- Comments and docstrings explain why, in plain sentences; keep the existing style.
