"""Hand-label who hit the ball, for a ground-truth test of hitter attribution (training/eval_hits.py scores it).

Takes the hits the pipeline detects in an analysed clip (outputs/<name>/tracking.json + court.json), in random
order, and loops a short clip around each one with every player's box and id drawn and the ball circled.
Find the frame where the paddle meets the ball, then press the hitter's id on that frame:
  1-4    the hitter is that box; the frame on screen is saved as the contact frame
  m      a real hit, but the hitter has no box (the tracker lost them)
  x      not a hit (ball handling between points, a bounce, the top of a lob, a ball on another court)
  s      skip (unsure)            u  undo the last label          q  save and quit
  space  pause / play             a / d  step one frame back / forward while paused
The hitter's box and court position are saved too, so labels still score after a re-track renumbers ids.
Labels are saved after every hit to <out>/hits.json.

Usage: python training/annotate_hits.py outputs/10-11-BBB-mb-tim-cut ~/yt-vids/10-11-BBB-mb-tim-cut.mkv \
           --out training/data/hits-1011 [--n 150]
"""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402
from pbv.court import Court  # noqa: E402

VIEW_W = 1280
KEYS = {ord(str(k)): str(k) for k in range(1, 5)}


def load_analysis(run):
    """(tracking data, court, detected events) for an analysed clip, with the current pipeline code."""
    data = json.loads((run / "tracking.json").read_text())
    data.setdefault("ball_offscreen", [False] * len(data["ball"]))  # tracking.json from before off-screen detection
    court = Court.load(run / "court.json")
    _, evs, *_ = app.analyse_fixed(data, court, SimpleNamespace(drive_speed=10.0), openings_only=False)
    return data, court, evs


def draw(img, players, ball, f, pred, s):
    view = cv2.resize(img, (VIEW_W, int(img.shape[0] * s)))
    for pid, p in players[f].items():
        x1, y1, x2, y2 = (int(v * s) for v in p["bbox"])
        c = (0, 0, 255) if pid == pred else (0, 255, 0)
        cv2.rectangle(view, (x1, y1), (x2, y2), c, 2)
        cv2.putText(view, pid, (x1, max(y1 - 6, 20)), 0, 0.9, c, 2)
    b = ball[f] if f < len(ball) else None
    if b is not None:
        cv2.circle(view, (int(b[0] * s), int(b[1] * s)), 12, (0, 255, 255), 2)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="analysed clip dir with tracking.json and court.json")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--span", type=int, default=20, help="frames shown before and after each detected hit")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "hits.json"
    data, court, evs = load_analysis(args.run)
    players, ball, fps = data["players"], data["ball"], data["fps"]
    meta = {"run": str(args.run.resolve()), "video": str(args.video.resolve()), "fps": fps}
    labels = {int(k): v for k, v in json.loads(path.read_text())["labels"].items()} if path.exists() else {}
    hits = [e for e in evs if e["type"] == "hit"]
    random.Random(0).shuffle(hits)
    todo = [e for e in hits if e["frame"] not in labels][: max(args.n - len(labels), 0)]

    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = VIEW_W / int(cap.get(3))
    cv2.namedWindow("hits")
    done, k = [], 0
    while k < len(todo):
        e = todo[k]
        lo, hi = max(e["frame"] - args.span, 0), min(e["frame"] + args.span, total - 1, len(players) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        clip = [cap.read()[1] for _ in range(lo, hi + 1)]
        f, paused, result = e["frame"], False, None
        while result is None:
            img = clip[f - lo]
            if img is None:
                img = np.zeros((int(VIEW_W / s * 9 / 16), int(VIEW_W / s), 3), np.uint8)
            view = draw(img, players, ball, f, e["player"], s)
            info = f"{len(labels)}/{args.n}  detected hit f{e['frame']} -> P{e['player']} (red)  showing f{f} ({f - e['frame']:+d})"
            cv2.putText(view, info, (10, 30), 0, 0.7, (255, 255, 255), 2)
            cv2.putText(view, "1-4 hitter  m untracked  x no hit  s skip  u undo  q quit  space pause  a/d step",
                        (10, view.shape[0] - 15), 0, 0.6, (255, 255, 255), 2)
            cv2.imshow("hits", view)
            key = cv2.waitKey(0 if paused else int(1000 / fps)) & 0xFF
            if not paused and key == 255:
                f = lo if f >= hi else f + 1
            elif key == ord(" "):
                paused = not paused
            elif key == ord("a"):
                paused, f = True, max(lo, f - 1)
            elif key == ord("d"):
                paused, f = True, min(hi, f + 1)
            elif key in KEYS and KEYS[key] in players[f]:
                p = players[f][KEYS[key]]
                result = {"kind": "hit", "contact": f, "hitter": KEYS[key], "bbox": p["bbox"], "court": p["court"]}
            elif key == ord("m"):
                result = {"kind": "untracked", "contact": f}
            elif key == ord("x"):
                result = {"kind": "none"}
            elif key == ord("s"):
                result = "skip"
            elif key == ord("u") and done:
                labels.pop(done.pop(), None)
                k -= 2
                result = "undo"
            elif key == ord("q"):
                k = len(todo)
                result = "quit"
        if isinstance(result, dict):
            labels[e["frame"]] = {**result, "detected_player": e["player"]}
            done.append(e["frame"])
            path.write_text(json.dumps({"meta": meta, "labels": {str(f): v for f, v in sorted(labels.items())}}, indent=1))
        k += 1
    cv2.destroyAllWindows()
    kinds = [v["kind"] for v in labels.values()]
    print(f"{len(labels)} hits labelled ({kinds.count('hit')} hit, {kinds.count('untracked')} untracked, "
          f"{kinds.count('none')} not a hit) -> {path}")


if __name__ == "__main__":
    main()
