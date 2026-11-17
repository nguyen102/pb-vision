"""Hand-label the serve, return and third shot of rallies: ground truth for training/eval_openings.py.

For each serve the pipeline detects (random order) it loops the clip from 2 s before to 7 s after, with every
player's box and id drawn and the ball circled. Nothing the pipeline detected after the serve is shown, so
missed returns and third shots get labelled too.

Step to the frame where the paddle meets the ball and press the hitter's id. Do it for the serve, then the
return, then the third shot (the counter at the top says which one is next):
  1-4    that box hit the ball on this frame         m  someone hit it here but has no box
  n      save this rally and go to the next (press it early if the rally ended before the third shot,
         e.g. a serve fault or a missed return; after the third shot it saves by itself)
  x      no serve here (ball handling between points, a warm-up hit, a ball from another court)
  u      undo the last mark (or, with none, go back to the previous rally)
  s      skip (unsure)       q  save and quit
  space  pause / play        a / d  one frame back / forward      j / l  10 frames back / forward
If the real serve is before the window (the detected one is late), mark it anyway as long as it's on screen.
Labels are saved after every rally to <out>/openings.json, with each hitter's box and court position, so
they still score after a re-track renumbers ids.

Usage: python training/annotate_openings.py outputs/11-11-jon-bel-1 /mnt/d/Videos/PB/BBB-Games/11-11-jon-bel-1.mp4 \
           --out training/data/openings-jb [--n 60]
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

ROLES = ("serve", "return", "third shot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="analysed clip dir with tracking.json and court.json")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60, help="rallies to label")
    ap.add_argument("--before", type=float, default=2.0, help="seconds shown before the detected serve")
    ap.add_argument("--after", type=float, default=7.0, help="seconds shown after it")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "openings.json"
    data, court, evs = load_analysis(args.run)
    players, ball, fps = data["players"], data["ball"], data["fps"]
    meta = {"run": str(args.run.resolve()), "video": str(args.video.resolve()), "fps": fps}
    labels = {int(k): v for k, v in json.loads(path.read_text())["labels"].items()} if path.exists() else {}
    serves = [e for e in evs if e.get("serve")]
    random.Random(0).shuffle(serves)
    todo = [e for e in serves if e["frame"] not in labels][: max(args.n - len(labels), 0)]

    cap = cv2.VideoCapture(str(args.video))
    start = data.get("params", {}).get("start", 0)  # tracking frame 0 is this video frame (app.py --start)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start, len(players))
    s = VIEW_W / int(cap.get(3))
    save = lambda: path.write_text(json.dumps({"meta": meta, "labels": {str(f): v for f, v in sorted(labels.items())}},
                                              indent=1))
    cv2.namedWindow("openings")
    placed = False  # WSLg can open a window off-screen: move it once it's shown
    done, k = [], 0
    while k < len(todo):
        e = todo[k]
        lo, hi = max(e["frame"] - int(args.before * fps), 0), min(e["frame"] + int(args.after * fps), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + lo)
        clip = []
        for _ in range(lo, hi + 1):
            ok, img = cap.read()
            clip.append(cv2.resize(img, (VIEW_W, int(img.shape[0] * s))) if ok else None)
        f, paused, marks, result = lo, False, [], None
        while result is None:
            img = clip[f - lo]
            if img is None:
                img = np.zeros((int(VIEW_W * 9 / 16), VIEW_W, 3), np.uint8)
            view = _draw_scaled(img, players, ball, f, s)
            for m in marks:  # marked contacts, as a reminder
                if m["contact"] == f:
                    cv2.putText(view, f"{m['role']} here", (10, 95), 0, 0.9, (0, 255, 255), 2)
            nxt = ROLES[len(marks)] if len(marks) < 3 else "-"
            info = (f"{len(labels)}/{args.n}  next: {nxt.upper()}  marked: "
                    + (", ".join(f"{m['role']} P{m.get('hitter', '?')} {m['contact'] - e['frame']:+d}" for m in marks) or "none")
                    + f"   f{f} ({f - e['frame']:+d} from detected serve)")
            cv2.putText(view, info, (10, 30), 0, 0.65, (255, 255, 255), 2)
            cv2.putText(view, "1-4 hitter  m no box  n next rally  x no serve  u undo  s skip  q quit  "
                        "space pause  a/d 1 frame  j/l 10", (10, view.shape[0] - 15), 0, 0.55, (255, 255, 255), 2)
            cv2.imshow("openings", view)
            if not placed:
                cv2.waitKey(50)  # let it map first: a move before that is ignored
                cv2.moveWindow("openings", 0, 0)
                placed = True
            key = cv2.waitKey(0 if paused else int(1000 / fps)) & 0xFF
            if not paused and key == 255:
                f = lo if f >= hi else f + 1
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("a"), ord("d"), ord("j"), ord("l")):
                step = {"a": -1, "d": 1, "j": -10, "l": 10}[chr(key)]
                paused, f = True, min(hi, max(lo, f + step))
            elif (key in KEYS and KEYS[key] in players[f]) or key == ord("m"):
                if key == ord("m"):
                    marks.append({"role": ROLES[len(marks)], "contact": f, "untracked": True})
                else:
                    p = players[f][KEYS[key]]
                    marks.append({"role": ROLES[len(marks)], "contact": f, "hitter": KEYS[key], "bbox": p["bbox"],
                                  "court": p["court"]})
                paused = False
                if len(marks) == 3:
                    result = {"kind": "rally", "shots": marks}
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
    print(f"{len(labels)} labelled: {len(rallies)} rallies ({n.count(3)} to the third shot), "
          f"{len(labels) - len(rallies)} not a serve -> {path}")


def _draw_scaled(img, players, ball, f, s):
    """draw() for a frame already resized to VIEW_W: boxes and ball scaled by s, no hitter highlighted."""
    view = img.copy()
    for pid, p in players[f].items():
        x1, y1, x2, y2 = (int(v * s) for v in p["bbox"])
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(view, pid, (x1, max(y1 - 6, 20)), 0, 0.9, (0, 255, 0), 2)
    b = ball[f] if f < len(ball) else None
    if b is not None:
        cv2.circle(view, (int(b[0] * s), int(b[1] * s)), 12, (0, 255, 255), 2)
    return view


if __name__ == "__main__":
    main()
