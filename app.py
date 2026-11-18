"""PickleballVision pipeline. Two modes:

fixed (default): tripod / fence camera that sees the whole court.
  video -> court calibration -> person tracking (YOLO + ByteTrack) + ball detection
        -> player selection (2 per side, bystanders dropped, stable ids, clothing re-ID)
        -> hits/bounces -> serve, return and third shot of each rally, faults -> stroke form (YOLO pose)
        -> movement stats, heatmaps, annotated video
pov: camera worn by a player (chest / head / paddle mount).
  video -> camera-motion estimate -> player + ball tracking (camera pan removed)
        -> other players' hits + your swings (camera jolts) -> rallies -> stroke form
  No court mapping, so no court positions, heatmaps, speeds, landing spots or kitchen faults.

Usage: python app.py data/match.mp4 [--mode pov] [--start 60 --max-frames 1800]
Then:  streamlit run dashboard.py
"""
import argparse
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pbv import analytics, audio, balltrack, draw, events
from pbv.constants import DASHBOARD_KEYFRAME_S
from pbv.court import Court
from pbv.players import fill_hidden as fill_hidden_players, select_players

PROGRESS_S = 5  # progress bars redraw at most this often, so a run's log stays small
OPENINGS_MODEL = Path(__file__).parent / "models" / "openings.pt"  # learned serve / return detector
HITS_MODEL = Path(__file__).parent / "models" / "hits.pt"  # learned hit detector: rally shot counts
BALL_MODEL = Path(__file__).resolve().parent / "models" / "ball-r1.pt"  # trained by training/ (round 1)


def camera_shift(prev, frame, width=480):
    """Gray thumbnail of frame and how far its content moved since prev, in full-res pixels."""
    scale = frame.shape[1] / width
    g = cv2.cvtColor(cv2.resize(frame, (width, int(frame.shape[0] / scale))), cv2.COLOR_BGR2GRAY).astype(np.float32)
    shift = [0.0, 0.0] if prev is None else [round(v * scale, 1) for v in cv2.phaseCorrelate(prev, g)[0]]
    return g, shift


def track(video, court, start, args):
    from pbv.tracking import BallDetector, PlayerTracker

    players_model = PlayerTracker(args.player_model)
    ball_model = BallDetector(args.ball_model, args.ball_class)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start
    if args.max_frames:
        n = min(n, args.max_frames)
    people, ball, paddle, cam, thumb = [], [], [], [], None
    for _ in tqdm(range(n), desc="tracking", mininterval=PROGRESS_S):
        ok, frame = cap.read()
        if not ok:
            # A file the reader can't keep up with (e.g. several runs reading one large video on /mnt/d) also ends
            # in a failed read: stop rather than save a short run as finished. The frame count in the header can
            # be off by a few frames at the true end.
            if n - len(people) > 30:
                raise RuntimeError(f"video read failed at frame {start + len(people)} of {start + n}")
            break
        people.append(players_model.track_pov(frame) if court is None else players_model.people_near_court(frame, court))
        c, p = ball_model.candidates(frame)
        ball.append(c)
        paddle.append(p)
        if court is None:
            thumb, shift = camera_shift(thumb, frame)
            cam.append(shift)
    # POV keeps the per-frame players directly; fixed mode keeps everyone near the court for select_players.
    return {"fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), ("players" if court is None else "people"): people,
            "ball_cands": ball, "paddle": paddle, "cam_shift": cam}


def add_stroke_form(video, shots, players, start, model, serve_fault=True):
    from pbv.tracking import PoseEstimator

    pose, cap = PoseEstimator(model), cv2.VideoCapture(str(video))
    for s in tqdm(shots, desc="pose", mininterval=PROGRESS_S):
        if s["player"] not in players[s["frame"]]:  # your own swings in POV: no box to pose
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + s["frame"])
        ok, frame = cap.read()
        box = players[s["frame"]].get(s["player"], {}).get("bbox")  # None: served from off screen
        kp = pose.keypoints_for(frame, box) if ok and box else None
        s["form"] = analytics.stroke_form(kp) if kp is not None else None
        # Rule 4.A: serve contact must be below the waist. The hit frame can land a few frames into
        # the follow-through, so only flag a wrist clearly above the waist (half-way to the shoulders).
        if serve_fault and s["type"] == "serve" and s["form"] and s["form"]["contact_height"] > 0.5:
            s["faults"].append("serve_above_waist")


def load_appearance(video, people, start, path):
    """Attach clothing descriptors to every tracked person, computing them (one video pass) if
    path doesn't hold a set that matches these people."""
    from pbv.appearance import add_appearance, attach_appearance

    n = sum(len(fp) for fp in people)
    if path.exists():
        looks = np.load(path)
        if len(looks) == n:
            attach_appearance(people, looks)
            return
    np.save(path, add_appearance(video, people, start))


def open_writer(path, fps, size):
    for codec in ("avc1", "mp4v"):  # avc1 if this OpenCV build has it; either way render() re-encodes with ffmpeg
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError(f"cannot open video writer for {path}")


OPENING_LABELS = {"serve": "serve", "return": "return", "third": "3rd shot"}


DASHBOARD_PLAYERS_S = 30  # dashboard.mp4: player boxes and IDs only this long, so you can see who is who
HIT_MARK_S = 0.4  # dashboard.mp4: how long a hit's ring stays up


def hit_spot(f, ball, shot, players, near=6):
    """Where to draw a hit at frame f: the ball (at f or the nearest frame within near where it's seen), else
    the top of the hitter's box (shot: the shot at f, if any), else None."""
    for d in sorted(range(-near, near + 1), key=abs):
        if 0 <= f + d < len(ball) and not np.isnan(ball[f + d][0]):
            return ball[f + d][:2]
    p = players[f].get(shot["player"]) if shot else None
    return ((p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][1]) if p else None


def render(video, dst, start, players, ball, evs, shots, court, fps, dash_dst=None, hit_frames=(), dash_sound=None):
    """annotated video -> dst (boxes and IDs, ball trail, shot labels, mini-court). dash_dst: also the dashboard's
    video, from the same frames: boxes and IDs for the first DASHBOARD_PLAYERS_S s only, no ball trail or court
    outline, and a growing ring where each hit in hit_frames happens. dash_sound: (video, seconds) for its
    soundtrack (to_browser_mp4), shifted so the pops play with the picture.

    Both are written to .part files and only replace dst / dash_dst once they're browser-playable H.264, so the
    dashboard keeps playing the previous videos while a run renders."""
    part = lambda p: Path(p).with_name(Path(p).stem + ".part.mp4")
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(part(dst), fps, size)
    dash = open_writer(part(dash_dst), fps, size) if dash_dst else None
    ball = np.asarray(ball, float)
    shot_at = {s["frame"]: s for s in shots}
    bounce_at = {e["frame"]: court.to_court([e["px"]])[0] for e in evs if e["type"] == "bounce"} if court else {}
    hit_len = max(1, int(HIT_MARK_S * fps))
    spots = {f: hit_spot(f, ball, shot_at.get(f), players) for f in sorted(set(hit_frames)) if f < len(players)}
    spots = {f: p for f, p in spots.items() if p is not None}
    label, label_until, bounces = "", -1, []
    for f in tqdm(range(len(players)), desc="rendering", mininterval=PROGRESS_S):
        ok, frame = cap.read()
        if not ok:
            break
        if f in shot_at and shot_at[f]["type"] in OPENING_LABELS:  # only serve / return / 3rd shot: no speed, fault or stroke type
            s = shot_at[f]
            label = f"{draw.name(s['player'])} {OPENING_LABELS[s['type']]}"
            label_until = f + int(fps * 1.5)
        if f in bounce_at:
            bounces = (bounces + [bounce_at[f]])[-3:]
        text = label if f <= label_until else ""
        if dash is not None:
            hits = [(spots[h], f - h, hit_len) for h in range(f - hit_len + 1, f + 1) if h in spots]
            dash.write(draw.annotate(frame.copy(), players[f], (), court, bounces, text,
                                     show_players=f < DASHBOARD_PLAYERS_S * fps, show_ball=False, hits=hits,
                                     show_court=False))
        trail = ball[max(0, f - 10):f + 1]
        writer.write(draw.annotate(frame, players[f], trail, court, bounces, text))
    writer.release()
    to_browser_mp4(part(dst))
    part(dst).replace(dst)
    if dash is not None:
        dash.release()
        to_browser_mp4(part(dash_dst), keyframe_s=DASHBOARD_KEYFRAME_S, sound=dash_sound)
        part(dash_dst).replace(dash_dst)


# H.264 encoders, fastest first: OpenCV's pip build can't write H.264, so the video is written as mp4v (which
# browsers won't play) and re-encoded with ffmpeg
H264 = (["-c:v", "h264_nvenc", "-cq", "26", "-preset", "p4"], ["-c:v", "libx264", "-crf", "23", "-preset", "veryfast"])


def to_browser_mp4(path, keyframe_s=None, sound=None):
    """Re-encode path in place to H.264 with faststart, so the dashboard can play it (and seek before it has
    all loaded). Leaves the file as it is, with a warning, if ffmpeg isn't installed or every encoder fails.
    sound: (video, seconds): add that video's soundtrack from that point on (negative: delayed by that much)."""
    path = Path(path)
    if not shutil.which("ffmpeg"):
        print(f"warning: ffmpeg not found; {path} stays mp4v, which browsers (the dashboard) can't play")
        return
    tmp = path.with_name(path.stem + ".h264.mp4")
    for enc in H264:
        # keyframe_s: a keyframe exactly every keyframe_s seconds (more only at scene cuts). -g sets the encoder's
        # keyframe interval (NVENC makes the forced frames plain I-frames, which players can't seek to, without it)
        fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS) or 30.0
        keys = ["-g", str(round(fps * keyframe_s)), "-force_key_frames", f"expr:gte(t,n_forced*{keyframe_s})"] \
            if keyframe_s else []
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), *enc, *keys, "-pix_fmt", "yuv420p", "-an",
               "-movflags", "+faststart", str(tmp)]
        if subprocess.run(cmd).returncode == 0:
            tmp.replace(path)
            if sound:
                mux_sound(path, *sound)
            return
    tmp.unlink(missing_ok=True)
    print(f"warning: ffmpeg couldn't re-encode {path} to H.264; it stays mp4v, which browsers can't play")


def mux_sound(path, src, at):
    """Put src's soundtrack into the video at path (the picture copied as it is): the sound from at seconds on,
    or delayed by -at if at is negative. It trims the decoded sound, as pbv.audio.load() does, rather than
    seeking in the file (which can land tens of ms off), so the pops play where the audio features put them."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".sound.mp4")
    shift = f"atrim=start={at:.4f},asetpts=PTS-STARTPTS" if at >= 0 else f"adelay={-at * 1000:.0f}:all=1"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-i", str(src), "-map", "0:v:0", "-map", "1:a:0?",
           "-c:v", "copy", "-af", shift, "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(tmp)]
    if subprocess.run(cmd).returncode == 0:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)
        print(f"warning: couldn't add the sound to {path}")


def pick_ball(data, court, min_conf=0.15):
    """Per-frame ball [x, y] or None from the detector's candidates.

    Fixed camera: link candidates over time and keep the best-scoring paths inside the court region
    (pbv.balltrack), so a blurred ball beats a sharper head or a ball on the next court.
    POV: the camera moves, so motion and court region mean little: most confident candidate per frame.
    """
    cands = data["ball_cands"]
    if court is None:
        return [c[0][:2] if c and c[0][4] >= min_conf else None for c in cands]
    return balltrack.select_path(cands, data["width"], balltrack.BallRegion(court))


def analyse_fixed(data, court, args, openings_only=True, fill_hidden=True):
    """openings_only=False: every shot and event (for evaluation), not just serve / return / third shot.
    fill_hidden: shots are attributed with players out of view kept where they were last / next seen."""
    fps, seen = data["fps"], data["players"]
    players = fill_hidden_players(seen) if fill_hidden else seen
    off = data["ball_offscreen"]
    ball = events.clean_ball(data["ball"], max_jump=0.08 * data["width"], offscreen=off)
    evs = events.filter_hits(events.find_events(ball, players), fps,
                             side=lambda e: court.side(players[e["frame"]][e["player"]]["court"]))
    ball_hits = [e["frame"] for e in evs if e["type"] == "hit"]  # from the ball track alone: syncs the sound
    detected = events.real_detections(data["ball"], ball)
    feats = None
    if OPENINGS_MODEL.exists() or HITS_MODEL.exists():
        import torch
        from pbv import openings_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        feats = openings_model.frame_features(ball, detected, players, court, data["width"], data["height"])
    if OPENINGS_MODEL.exists():
        # serves and returns from the detector trained on hand labels (training/train_openings.py)
        serves, model_hits = openings_model.detect(openings_model.load(OPENINGS_MODEL, device), ball, detected,
                                                   players, court, fps, data["width"], data["height"], device=device,
                                                   feats=feats)
        evs = events.merge_serves(evs, serves, players, court, fps)
        evs = sorted([e for e in evs if all(abs(e["frame"] - r["frame"]) > 3 or e.get("serve") for r in model_hits)]
                     + model_hits, key=lambda e: (e["frame"], e["type"] != "hit"))
    else:
        serves = events.find_serves(ball, players, court, fps, data["width"], detected=detected, offscreen=off)
        # serves find_serves misses (server off screen or without a box, so the ball is never seen in their hand):
        # everyone waiting still, then the ball comes into view and flies over the net
        entry = events.find_entry_serves(ball, players, court, fps, data["width"], data["height"],
                                         hits=[e["frame"] for e in evs if e["type"] == "hit"], detected=detected)
        serves = events.combine_serves(serves, entry, players, court, fps)
        evs = events.merge_serves(evs, serves, players, court, fps)
        evs = events.diagonal_servers(evs, players, court, fps)
        evs = events.infer_returns(evs, ball, players, court, fps)
    shots, rallies = events.classify_shots(evs, players, court, fps, drive_speed=args.drive_speed,
                                           serves_detected=True, offscreen=off)
    # every hit is found (the third shot's landing and speed depend on what follows). Shots: only the serve,
    # return and third shot of each rally are reported. Rallies: every rally with a detected serve, with its
    # full shot count and length.
    if openings_only:
        all_shots, all_rallies = shots, rallies
        shots, rallies, evs = events.opening_shots(shots, rallies, evs, players, court, fps)
        rallies = events.full_rallies(rallies, all_rallies, fps, shots=all_shots, court=court)
        if HITS_MODEL.exists():
            # shot counts from the hit detector trained on hand-marked rallies (training/train_hits.py)
            from pbv import hits_model
            models, dec, inputs = hits_model.load(HITS_MODEL, device)
            hit_feats = feats
            if inputs["formation"]:  # where players stand before a serve
                hit_feats = np.concatenate([hit_feats, hits_model.serve_formation(players, fps)], 1)
            if inputs["pose"]:  # every player's keypoints, every frame (pbv/pose.py; cached as pose.npz)
                hit_feats = np.concatenate([hit_feats, pose_features(args, data, seen, players, ball, court)], 1)
            af = None
            if inputs["audio"]:  # paddle pops (pbv/audio.py); zeros if the video has no sound or isn't given
                video = getattr(args, "video", None)
                af, sound_offset = audio.features(video, getattr(args, "start", 0.0), len(ball), fps, ball_hits) \
                    if video else (np.zeros((len(ball), audio.N_FEATS), np.float32), None)
                hit_feats = np.concatenate([hit_feats, af], 1)
                args.sound_offset = sound_offset  # render() syncs the dashboard video's sound with it
            prob = hits_model.detect(models, None, None, None, None, 0, 0, device, feats=hit_feats)
            peaks = hits_model.decode(prob, fps, thr=dec.get("thr", 0.5))
            starts = [r["start"] for r in rallies] + [len(ball)]
            for i, r in enumerate(rallies):
                hits = hits_model.rally_hits(r["start"], starts[i + 1], peaks, fps, max_gap_s=dec.get("max_gap_s", 3.0),
                                             play=prob[:, 3], play_thr=dec.get("play_thr", 0.3),
                                             play_at=dec.get("play_at", 0.0), alternate=dec.get("alternate", False))
                # the dashboard video's rings: on the paddle pop, when there's a clear one within 0.1 s
                rings = audio.snap_to_pops(hits, af) if af is not None and af[:, -1].any() else hits
                r.update(end=hits[-1], shots=len(hits), duration_s=round((hits[-1] - r["start"]) / fps, 2),
                         hit_frames=rings)  # for the dashboard video; not written to stats.json
    players = seen
    return ball, evs, shots, rallies, analytics.player_stats(players, fps, rallies=rallies, court=court)


def pose_features(args, data, seen, players, ball, court):
    """The hit model's pose inputs: keypoints from pose.npz in the output folder, extracted from the video (a few
    minutes) when missing; zeros, with a warning, when there's neither (training/train_hits.py's serve search)."""
    from pbv import hits_model, pose
    cache, video = getattr(args, "pose_cache", None), getattr(args, "video", None)
    if cache is None or (video is None and not Path(cache).exists()):
        print("warning: no video for the hit model's pose features: zeros", flush=True)
        return np.zeros((len(ball), 4 * hits_model.POSE_FEATS), np.float32)
    progress = lambda r: tqdm(r, desc="pose", mininterval=PROGRESS_S)
    kp, ids = pose.load_or_extract(cache, video, seen, data.get("params", {}).get("start", 0),
                                   getattr(args, "pose_model", "yolov8n-pose.pt"), progress=progress)
    return hits_model.pose_features(kp, ids, players, ball, court)


def analyse_pov(data):
    fps, players = data["fps"], data["players"]
    comp, offset = events.compensate(data["ball"], data["cam_shift"])
    ball = events.clean_ball(comp, max_jump=0.08 * data["width"])
    # Every visible player can hit; their boxes must share the ball's pan-compensated view.
    shifted = [{pid: {"bbox": [p["bbox"][0] - o[0], p["bbox"][1] - o[1], p["bbox"][2] - o[0], p["bbox"][3] - o[1]]}
                for pid, p in fp.items()} for fp, o in zip(players, offset)]
    evs = events.filter_hits(events.find_events(ball, shifted, near_side=lambda pl: True, far_side=lambda pl: True), fps)
    for e in evs:
        if e["type"] == "hit":  # your partner fills much of the frame; opponents across the net don't
            x1, y1, x2, y2 = players[e["frame"]][e["player"]]["bbox"]
            e["side"] = "near" if y2 - y1 > 0.35 * data["height"] else "far"
    swings = events.find_swings(data["cam_shift"], ball, fps, data["width"], data["paddle"], data["height"])
    shots, rallies = events.pov_shots(evs, swings, fps)
    for e in evs:  # back to on-screen pixels for the video overlay / stats
        e["px"] = [round(e["px"][0] + offset[e["frame"]][0], 1), round(e["px"][1] + offset[e["frame"]][1], 1)]
    seen = Counter(pid for fp in players for pid in fp)
    stats = {pid: {"seconds": round(n / fps, 1)} for pid, n in seen.items() if n >= fps}
    return ball + offset, evs, shots, rallies, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--mode", choices=["fixed", "pov"], default="fixed", help="fixed camera or body-worn POV camera")
    ap.add_argument("--out", type=Path, help="output dir (default outputs/<video name>[-pov])")
    ap.add_argument("--corners", help="fixed mode: 8 comma-separated pixel coords: far-left, far-right, near-right, near-left")
    ap.add_argument("--court-points", help='fixed mode: 4+ named landmarks when corners are hidden, e.g. '
                    '"near-left=1537,1613;near-baseline-center=3330,1613;near-kitchen-left=1642,1398;..." '
                    "(names: see LANDMARKS in pbv/court.py)")
    ap.add_argument("--player-model", default="yolov8m.pt")
    ap.add_argument("--ball-model", help="ball detector (default: fixed mode models/ball-r1.pt, the pickleball "
                    "model trained with training/ on fixed-camera video; pov mode yolov8m.pt, generic COCO, "
                    "which also finds the paddle used to confirm swings)")
    ap.add_argument("--ball-class", type=int, help="class id of the ball (default: from the model's class names; -1 = any)")
    ap.add_argument("--pose-model", default="yolov8n-pose.pt")
    ap.add_argument("--no-pose", action="store_true", help="skip stroke form at each shot (the hit model still uses "
                    "pose when it was trained with it: pose.npz in the output folder)")
    ap.add_argument("--drive-speed", type=float, default=10.0, help="fixed mode: m/s above which a shot is a drive")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the video to start at")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--force", action="store_true", help="re-run tracking even if tracking.json matches")
    ap.add_argument("--no-render", action="store_true", help="skip annotated.mp4 and dashboard.mp4 (training only "
                    "needs the data files; the videos take about 1 GB per 40 minutes of play)")
    args = ap.parse_args()
    if not args.video.exists():
        ap.error(f"{args.video} not found")
    pov = args.mode == "pov"
    args.ball_model = args.ball_model or ("yolov8m.pt" if pov else str(BALL_MODEL))

    out = args.out or Path("outputs") / (args.video.stem + ("-pov" if pov else ""))
    out.mkdir(parents=True, exist_ok=True)
    fps = cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FPS) or 30.0
    start = int(args.start * fps)
    court = None
    if not pov:
        court_path = out / "court.json"
        if args.corners:
            c = [float(v) for v in args.corners.split(",")]
            if len(c) != 8:
                ap.error("--corners needs exactly 8 numbers")
            Court(list(zip(c[::2], c[1::2]))).save(court_path)
        elif args.court_points:
            try:
                named = {n.strip(): tuple(float(v) for v in xy.split(","))
                         for n, xy in (part.split("=") for part in args.court_points.split(";") if part.strip())}
                Court.from_landmarks(named).save(court_path)
            except ValueError as e:
                ap.error(f"--court-points: {e}")
        elif not court_path.exists():
            from calibrate import calibrate
            calibrate(args.video, court_path, start)
        court = Court.load(court_path)

    params = {"version": 5, "mode": args.mode, "ball_model": Path(args.ball_model).name, "start": start, "max_frames": args.max_frames,
              "corners": None if pov else court.corners_px.tolist()}
    cache = out / "tracking.json"
    data = json.loads(cache.read_text()) if cache.exists() and not args.force else None
    if not data or data.get("params") != params:
        data = {**track(args.video, court, start, args), "params": params}
    if not pov:  # cheap, so always re-run: picks the 4 players and gives each a stable id
        load_appearance(args.video, data["people"], start, out / "appearance.npy")
        before = data.get("players")
        data["players"] = select_players(data["people"], data["fps"])
        if before:  # pose.npz holds keypoints per player id: follow the boxes to their new ids
            from pbv import pose
            if pose.remap(out / "pose.npz", before, data["players"]) > 0.01:
                (out / "pose.npz").unlink()  # many boxes never had keypoints: extract them again
        for fp in data["people"]:  # descriptors live in appearance.npy, not the JSON cache
            for p in fp:
                p.pop("look", None)
    data["height"] = int(cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FRAME_HEIGHT))
    data["ball"] = pick_ball(data, court)  # cheap, so always re-run (and kept in the cache for demo/build.py)
    # The ball leaving the picture (a lob over the top) is in play, not lost: no gap filling, no dead-ball serve.
    data["ball_offscreen"] = None if pov else balltrack.offscreen_frames(data["ball"], data["width"], data["height"])
    cache.write_text(json.dumps(data))
    players = data["players"]

    args.pose_cache = out / "pose.npz"  # the hit model's keypoints (analyse_fixed)
    ball, evs, shots, rallies, player_stats = analyse_pov(data) if pov else analyse_fixed(data, court, args)
    if not args.no_pose and shots:
        # POV "serves" are just the first hit after a pause (often ball handling), too unreliable to fault.
        add_stroke_form(args.video, shots, players, start, args.pose_model, serve_fault=not pov)

    summary = analytics.shot_stats(shots)
    if not pov:  # shot types beyond serve / return / third and fault detection aren't reliable yet: not reported
        for p in summary.values():
            p.pop("volleys"), p.pop("faults")
        for s in shots:
            s.pop("volley"), s.pop("faults")
    # every counted hit, for the dashboard video's hit markers: the hit detector's, else every detected shot
    hit_frames = [f for r in rallies for f in r.pop("hit_frames", [])] or [s["frame"] for s in shots]
    stats = {
        "video": str(args.video), "mode": args.mode, "start_s": args.start, "fps": data["fps"], "frames": len(players),
        "players": player_stats, "shot_summary": summary,
        "rallies": rallies, "shots": shots, "events": evs,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    if not pov:
        for pid in player_stats:
            cv2.imwrite(str(out / f"heatmap_P{pid}.png"),
                        draw.heatmap_image([fp[pid]["court"] for fp in players if pid in fp]))
    if not args.no_render:
        render(args.video, out / "annotated.mp4", start, players, ball, evs, shots, court, data["fps"],
               dash_dst=out / "dashboard.mp4", hit_frames=hit_frames,
               # the soundtrack runs sound_offset s late (or early): start it that much later, so pops play with hits
               dash_sound=(args.video, args.start + args.sound_offset) if getattr(args, "sound_offset", None) is not None
               else None)
    print(f"done ({args.mode}): {len(player_stats)} players, {len(rallies)} rallies, {len(shots)} shots -> {out}")


if __name__ == "__main__":
    main()
