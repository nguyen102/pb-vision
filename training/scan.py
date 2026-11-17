"""Step 1 of ball-detector training: scan a video with a "teacher" model and keep every ball candidate.

For every scanned frame: the teacher's ball candidates [x, y, w, h, conf, motion] (pbv.tracking.BallDetector,
so a full-frame pass plus a full-res crop around the predicted ball). Frames that aren't scanned, or
(with --ref) aren't the main camera of a broadcast, are null. Writes training/data/<name>/candidates.json.

Long videos: --window 180 --every 600 scans the first 3 minutes of every 10 (decoding the rest is cheap).
Self-training: pass a trained model as --teacher (and a new --name) to relabel with the better detector.

Usage: python training/scan.py /mnt/d/Videos/PB/BBB-Games/match.mp4 [--teacher yolov8m.pt --imgsz 1920 --name match-r0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbv.tracking import BallDetector  # noqa: E402


def thumb(frame):
    """Small blurred greyscale of the picture below the scoreboard, for comparing camera views."""
    h = frame.shape[0]
    g = cv2.cvtColor(cv2.resize(frame[h // 4:], (80, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (3, 3), 0).astype("float32")


def is_main_camera(frame, ref, max_diff=18.0):
    """Main camera = a fixed view, so it matches a reference main-camera frame (players aside)."""
    return float(abs(thumb(frame) - ref).mean()) < max_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--name", help="output folder name (default: video stem)")
    ap.add_argument("--teacher", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--crop", type=int, default=640, help="full-res crop around the predicted ball (0 = off)")
    ap.add_argument("--ref", type=float, help="broadcast video: time (s) of a main-camera frame; other views are skipped")
    ap.add_argument("--window", type=float, help="seconds scanned per --every seconds (default: everything)")
    ap.add_argument("--every", type=float, default=600.0)
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()

    out = Path("training/data") / (args.name or args.video.stem)
    out.mkdir(parents=True, exist_ok=True)
    det = BallDetector(args.teacher, conf=args.conf, imgsz=args.imgsz, crop=args.crop)
    cap = cv2.VideoCapture(str(args.video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        n = min(n, args.max_frames)
    ref = None
    if args.ref is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.ref * fps))
        ref = thumb(cap.read()[1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    meta = {"video": str(args.video.resolve()), "fps": fps, "width": int(cap.get(3)), "height": int(cap.get(4)),
            "teacher": args.teacher, "imgsz": args.imgsz, "conf": args.conf, "crop": args.crop,
            "window": args.window, "every": args.every, "ref": args.ref}
    # Resumable: a long scan checkpoints every 2500 scanned frames (a GPU driver reset can wedge the process;
    # training/run_scan.sh restarts it and it carries on from the checkpoint).
    part = out / "candidates.partial.json"
    frames = []
    if part.exists():
        saved = json.loads(part.read_text())
        if saved["meta"] == meta:
            frames = saved["frames"]
            print(f"resuming at frame {len(frames)}", flush=True)
    for _ in range(len(frames)):
        cap.grab()
    t0, scanned = time.time(), sum(f is not None for f in frames)
    for i in range(len(frames), n):
        in_window = args.window is None or (i / fps) % args.every < args.window
        if not in_window:
            if not cap.grab():
                break
            frames.append(None)
            det.reset()
            continue
        ok, f = cap.read()
        if not ok:
            break
        if ref is not None and not is_main_camera(f, ref):
            frames.append(None)
            det.reset()
            continue
        frames.append(det.candidates(f)[0])
        scanned += 1
        if scanned % 2500 == 0:
            part.write_text(json.dumps({"meta": meta, "frames": frames}))
            print(f"{i}/{n} frames, {scanned} scanned, {time.time() - t0:.0f}s", flush=True)
    (out / "candidates.json").write_text(json.dumps({"meta": meta, "frames": frames}))
    part.unlink(missing_ok=True)
    print(f"done: {scanned} frames scanned, {sum(bool(f) for f in frames)} with candidates -> {out / 'candidates.json'}")


if __name__ == "__main__":
    main()
