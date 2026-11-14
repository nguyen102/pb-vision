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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pbv import analytics, balltrack, draw, events
from pbv.court import Court
from pbv.players import select_players

PROGRESS_S = 5  # progress bars redraw at most this often, so a run's log stays small
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
    for codec in ("avc1", "mp4v"):  # avc1 plays in browsers (dashboard); mp4v is the fallback
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError(f"cannot open video writer for {path}")


def render(video, dst, start, players, ball, evs, shots, court, fps):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(dst, fps, size)
    shot_at = {s["frame"]: s for s in shots}
    bounce_at = {e["frame"]: court.to_court([e["px"]])[0] for e in evs if e["type"] == "bounce"} if court else {}
    label, label_until, bounces = "", -1, []
    for f in tqdm(range(len(players)), desc="rendering", mininterval=PROGRESS_S):
        ok, frame = cap.read()
        if not ok:
            break
        if f in shot_at:
            s = shot_at[f]
            kind = "3rd shot" if s["type"] == "third" else s["type"]
            label = f"{draw.name(s['player'])} {kind}" + (f" {s['speed_kmh']:.0f} km/h" if s["speed_kmh"] else "")
            if s["faults"]:
                label += "  FAULT: " + ", ".join(s["faults"])
            label_until = f + int(fps * 1.5)
        if f in bounce_at:
            bounces = (bounces + [bounce_at[f]])[-3:]
        trail = ball[max(0, f - 10):f + 1]
        writer.write(draw.annotate(frame, players[f], trail, court, bounces, label if f <= label_until else ""))
    writer.release()


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


def analyse_fixed(data, court, args, openings_only=True):
    """openings_only=False: every shot and event (for evaluation), not just serve / return / third shot."""
    fps, players = data["fps"], data["players"]
    off = data["ball_offscreen"]
    ball = events.clean_ball(data["ball"], max_jump=0.08 * data["width"], offscreen=off)
    evs = events.filter_hits(events.find_events(ball, players), fps,
                             side=lambda e: court.side(players[e["frame"]][e["player"]]["court"]))
    detected = events.real_detections(data["ball"], ball)
    serves = events.find_serves(ball, players, court, fps, data["width"], detected=detected, offscreen=off)
    # serves find_serves misses (ball never seen in the server's hand, server off screen): everyone still, then
    # the ball flies over the net
    still = events.find_still_serves(ball, players, court, fps, data["width"], detected=detected)
    serves = events.combine_serves(serves, still, fps, evs, players, court)
    evs = events.merge_serves(evs, serves, players, court, fps)
    evs = events.diagonal_servers(evs, players, court, fps)
    evs = events.infer_returns(evs, ball, players, court, fps)
    shots, rallies = events.classify_shots(evs, players, court, fps, drive_speed=args.drive_speed,
                                           serves_detected=True, offscreen=off)
    # every hit is found (the third shot's landing and speed depend on what follows), but only the
    # serve, return and third shot of each rally are reported
    if openings_only:
        shots, rallies, evs = events.opening_shots(shots, rallies, evs, players, court, fps)
    return ball, evs, shots, rallies, analytics.player_stats(players, fps)


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
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--drive-speed", type=float, default=10.0, help="fixed mode: m/s above which a shot is a drive")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the video to start at")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--force", action="store_true", help="re-run tracking even if tracking.json matches")
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
        data["players"] = select_players(data["people"], data["fps"])
        for fp in data["people"]:  # descriptors live in appearance.npy, not the JSON cache
            for p in fp:
                p.pop("look", None)
    data["height"] = int(cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FRAME_HEIGHT))
    data["ball"] = pick_ball(data, court)  # cheap, so always re-run (and kept in the cache for demo/build.py)
    # The ball leaving the picture (a lob over the top) is in play, not lost: no gap filling, no dead-ball serve.
    data["ball_offscreen"] = None if pov else balltrack.offscreen_frames(data["ball"], data["width"], data["height"])
    cache.write_text(json.dumps(data))
    players = data["players"]

    ball, evs, shots, rallies, player_stats = analyse_pov(data) if pov else analyse_fixed(data, court, args)
    if not args.no_pose and shots:
        # POV "serves" are just the first hit after a pause (often ball handling), too unreliable to fault.
        add_stroke_form(args.video, shots, players, start, args.pose_model, serve_fault=not pov)

    stats = {
        "video": str(args.video), "mode": args.mode, "start_s": args.start, "fps": data["fps"], "frames": len(players),
        "players": player_stats, "shot_summary": analytics.shot_stats(shots),
        "rallies": rallies, "shots": shots, "events": evs,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    if not pov:
        for pid in player_stats:
            cv2.imwrite(str(out / f"heatmap_P{pid}.png"),
                        draw.heatmap_image([fp[pid]["court"] for fp in players if pid in fp]))
    render(args.video, out / "annotated.mp4", start, players, ball, evs, shots, court, data["fps"])
    print(f"done ({args.mode}): {len(player_stats)} players, {len(rallies)} rallies, {len(shots)} shots -> {out}")


if __name__ == "__main__":
    main()
