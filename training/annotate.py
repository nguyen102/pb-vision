"""Hand-label ball positions for a ground-truth test set (pseudo-labels can't score what the teacher misses).

Shows random frames from a video (at least --gap frames apart). Click the centre of the ball in play on
YOUR court (ignore balls on other courts and balls in a hand or on the floor), or press a key:
  n  no ball visible (out of play, hidden, off-screen)     s  skip this frame (unsure)
  u  undo the last label                                     q  save and quit
Left half of the window is the frame; the right half is a 3x zoom around the mouse.
Labels are saved after every frame to <out>/labels.json in the format training/evaluate.py --gt reads
(frame -> [[x, y, w, h, False]]; a frame with no ball has an empty list).

Usage: python training/annotate.py ~/yt-vids/10-11-BBB-mb-tim-cut.mkv --out training/data/gt-1011 [--n 300]
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

VIEW_W = 1280  # displayed frame width; the zoom pane is beside it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gap", type=int, default=45, help="min frames between labelled frames")
    ap.add_argument("--frames", type=int, help="only sample from the first N frames")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "labels.json"
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4))
    meta = {"video": str(args.video.resolve()), "fps": cap.get(cv2.CAP_PROP_FPS), "width": W, "height": H, "hand": True}
    labels = {int(k): v for k, v in json.loads(path.read_text())["labels"].items()} if path.exists() else {}
    pool = list(range(0, min(total, args.frames or total), args.gap))
    random.Random(0).shuffle(pool)
    todo = [i for i in pool if i not in labels][: max(args.n - len(labels), 0)]
    s = VIEW_W / W
    state = {"mouse": (W // 2, H // 2), "click": None}

    def on_mouse(ev, x, y, *_):
        if x < VIEW_W:
            state["mouse"] = (int(x / s), int(y / s))
            if ev == cv2.EVENT_LBUTTONDOWN:
                state["click"] = state["mouse"]

    cv2.namedWindow("annotate")
    placed = False  # WSLg can open a window off-screen: move it once it's shown
    cv2.setMouseCallback("annotate", on_mouse)
    done, k = [], 0
    while k < len(todo):
        i = todo[k]
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            k += 1
            continue
        state["click"], result = None, None
        while result is None:
            mx, my = state["mouse"]
            view = cv2.resize(img, (VIEW_W, int(H * s)))
            x0, y0 = int(np.clip(mx - 80, 0, W - 160)), int(np.clip(my - 80, 0, H - 160))
            zoom = cv2.resize(img[y0:y0 + 160, x0:x0 + 160], (480, 480), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(zoom, ((mx - x0) * 3, (my - y0) * 3), (0, 255, 255), cv2.MARKER_CROSS, 30, 1)
            pane = np.zeros((view.shape[0], 480, 3), np.uint8)
            pane[:480] = zoom[: view.shape[0]]
            cv2.putText(pane, f"{len(labels)}/{args.n}  frame {i}", (8, view.shape[0] - 40), 0, 0.6, (255, 255, 255), 1)
            cv2.putText(pane, "click=ball n=none s=skip u=undo q=quit", (8, view.shape[0] - 15), 0, 0.5, (200, 200, 200), 1)
            cv2.imshow("annotate", np.hstack([view, pane]))
            if not placed:
                cv2.waitKey(50)  # let it map first: a move before that is ignored
                cv2.moveWindow("annotate", 0, 0)
                placed = True
            key = cv2.waitKey(20) & 0xFF
            if state["click"]:
                x, y = state["click"]
                result = [[x, y, 10, 10, False]]
            elif key == ord("n"):
                result = []
            elif key == ord("s"):
                result = "skip"
            elif key == ord("u") and done:
                labels.pop(done.pop(), None)
                k -= 2
                result = "undo"
            elif key == ord("q"):
                k = len(todo)
                result = "quit"
        if isinstance(result, list):
            labels[i] = result
            done.append(i)
            path.write_text(json.dumps({"meta": meta, "labels": {str(f): v for f, v in sorted(labels.items())}}))
        k += 1
    cv2.destroyAllWindows()
    print(f"{len(labels)} frames labelled ({sum(bool(v) for v in labels.values())} with a ball) -> {path}")


if __name__ == "__main__":
    main()
