"""Hand-label every shot of whole rallies: ground truth for a learned hit detector and for
training/eval_rallies.py (rally shot counts).

Like training/annotate_openings.py, but you mark every contact until the point is over, not just the serve,
return and third shot. For each serve the pipeline detects (random order) it loops the clip from --before s
before it up to the next detected serve (at most --after s, default 30, after it; e extends by 5 s), with every
player's box and id drawn and the ball circled. Nothing the pipeline detected after the serve is shown, so
missed hits get labelled too.

Step to the frame where the paddle meets the ball and press the hitter's id, for every shot in order:
  1-4    that box hit the ball on this frame         m  someone hit it here but has no box
  n      the point is over: save this rally and go to the next
  x      no serve here (ball handling between points, a warm-up hit, a ball from another court)
  u      undo the last mark (or, with none, go back to the previous rally)
  e      show 5 s more (the rally runs past the end of the clip)
  s      skip (unsure)       q  save and quit
  space  pause / play        a / d  one frame back / forward      j / l  10 frames back / forward
Count only shots in the point: not ball handling or knocking the ball back to the server after it.
If the real serve is before the window (the detected one is late), mark it anyway as long as it's on screen.

Labels are saved after every rally to <out>/rallies.json, each shot with its contact frame, hitter's box and
court position (so they still score after a re-track renumbers ids). The first three shots have the roles
serve / return / third shot, as in openings.json, and the rest "shot".

Usage: python training/annotate_rallies.py outputs/10-11-BBB-mb-tim-cut /mnt/d/Videos/PB/BBB-Games/10-11-BBB-mb-tim-cut.mkv \
           --out training/data/rallies-1011 [--n 60]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_hits import KEYS, VIEW_W, load_analysis  # noqa: E402
from annotate_openings import _draw_scaled  # noqa: E402

ROLES = ("serve", "return", "third shot")
role = lambda n: ROLES[n] if n < len(ROLES) else "shot"
MORE_S = 5.0


class Clip:
    """Frames lo..hi of the video, resized to VIEW_W and kept JPEG-compressed (a 15 s clip raw is ~1 GB)."""

    def __init__(self, cap, start, lo, hi, s):
        self.cap, self.start, self.s, self.lo, self.frames = cap, start, s, lo, []
        self.extend(hi)

    def extend(self, hi):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start + self.lo + len(self.frames))
        while self.lo + len(self.frames) <= hi:
            ok, img = self.cap.read()
            if not ok:
                self.frames.append(None)
                continue
            img = cv2.resize(img, (VIEW_W, int(img.shape[0] * self.s)))
            self.frames.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1])
        self.hi = hi

    def __getitem__(self, f):
        b = self.frames[f - self.lo]
        return None if b is None else cv2.imdecode(b, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="analysed clip dir with tracking.json and court.json")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60, help="rallies to label")
    ap.add_argument("--before", type=float, default=2.0, help="seconds shown before the detected serve")
    ap.add_argument("--after", type=float, default=30.0, help="at most this many seconds shown after it")
    ap.add_argument("--video-start", type=int, default=None,
                    help="video frame of tracking frame 0 (default: the run's --start; 0 for its annotated.mp4)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rallies.json"
    data, court, evs = load_analysis(args.run)
    players, ball, fps = data["players"], data["ball"], data["fps"]
    meta = {"run": str(args.run.resolve()), "video": str(args.video.resolve()), "fps": fps}
    labels = {int(k): v for k, v in json.loads(path.read_text())["labels"].items()} if path.exists() else {}
    serves = sorted((e for e in evs if e.get("serve")), key=lambda e: e["frame"])
    next_serve = {a["frame"]: b["frame"] for a, b in zip(serves, serves[1:])}
    random.Random(0).shuffle(serves)
    todo = [e for e in serves if e["frame"] not in labels][: max(args.n - len(labels), 0)]

    cap = cv2.VideoCapture(str(args.video))
    start = args.video_start  # tracking frame 0 is this video frame (app.py --start)
    if start is None:
        start = data.get("params", {}).get("start", 0)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start, len(players))
    s = VIEW_W / int(cap.get(3))
    save = lambda: path.write_text(json.dumps({"meta": meta, "labels": {str(f): v for f, v in sorted(labels.items())}},
                                              indent=1))
    cv2.namedWindow("rallies")
    placed = False  # WSLg can open a window off-screen: move it once it's shown
    done, k = [], 0
    while k < len(todo):
        e = todo[k]
        lo = max(e["frame"] - int(args.before * fps), 0)
        hi = min(next_serve.get(e["frame"], total) - 1, e["frame"] + int(args.after * fps), total - 1)
        clip = Clip(cap, start, lo, hi, s)
        f, paused, marks, result = lo, False, [], None
        while result is None:
            img = clip[f]
            if img is None:
                img = np.zeros((int(VIEW_W * 9 / 16), VIEW_W, 3), np.uint8)
            view = _draw_scaled(img, players, ball, f, s)
            for m in marks:  # marked contacts, as a reminder
                if m["contact"] == f:
                    cv2.putText(view, f"shot {marks.index(m) + 1} ({m['role']}) here", (10, 95), 0, 0.9, (0, 255, 255), 2)
            last = marks[-1] if marks else None
            info = (f"{len(labels)}/{args.n}  shots marked: {len(marks)}"
                    + (f" (last P{last.get('hitter', '?')} {(last['contact'] - e['frame']) / fps:+.1f}s)" if last else "")
                    + f"  next: {role(len(marks)).upper()}   {(f - e['frame']) / fps:+.1f}s from detected serve")
            cv2.putText(view, info, (10, 30), 0, 0.65, (255, 255, 255), 2)
            cv2.putText(view, "1-4 hitter  m no box  n point over  x no serve  u undo  e +5s  s skip  q quit  "
                        "space pause  a/d 1 frame  j/l 10", (10, view.shape[0] - 15), 0, 0.5, (255, 255, 255), 2)
            cv2.imshow("rallies", view)
            if not placed:
                cv2.waitKey(50)  # let it map first: a move before that is ignored
                cv2.moveWindow("rallies", 0, 0)
                placed = True
            key = cv2.waitKey(0 if paused else int(1000 / fps)) & 0xFF
            if not paused and key == 255:
                f = lo if f >= clip.hi else f + 1
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("a"), ord("d"), ord("j"), ord("l")):
                step = {"a": -1, "d": 1, "j": -10, "l": 10}[chr(key)]
                paused, f = True, min(clip.hi, max(lo, f + step))
            elif key == ord("e"):
                clip.extend(min(clip.hi + int(MORE_S * fps), total - 1))
            elif ((key in KEYS and KEYS[key] in players[f]) or key == ord("m")) and last and f <= last["contact"]:
                paused = True  # the clip looped back: a shot can't come before the last one marked
            elif (key in KEYS and KEYS[key] in players[f]) or key == ord("m"):
                if key == ord("m"):
                    marks.append({"role": role(len(marks)), "contact": f, "untracked": True})
                else:
                    p = players[f][KEYS[key]]
                    marks.append({"role": role(len(marks)), "contact": f, "hitter": KEYS[key], "bbox": p["bbox"],
                                  "court": p["court"]})
                paused = False
            elif key == ord("n") and marks:
                result = {"kind": "rally", "shots": marks}
            elif key == ord("x"):
                result = {"kind": "none"}
            elif key == ord("u"):
                if marks:
                    marks.pop()
                elif done:
                    labels.pop(done.pop(), None)
                    save()
                    k -= 2
                    result = "undo"
            elif key == ord("s"):
                result = "skip"
            elif key == ord("q"):
                k = len(todo)
                result = "quit"
        if isinstance(result, dict):
            labels[e["frame"]] = {**result, "detected_player": e["player"]}
            done.append(e["frame"])
            save()
        k += 1
    cv2.destroyAllWindows()
    rallies = [v for v in labels.values() if v["kind"] == "rally"]
    n = [len(v["shots"]) for v in rallies]
    print(f"{len(labels)} labelled: {len(rallies)} rallies, {sum(n)} shots, "
          f"{len(labels) - len(rallies)} not a serve -> {path}")


if __name__ == "__main__":
    main()
