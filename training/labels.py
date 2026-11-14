"""Step 2: turn teacher candidates into ball pseudo-labels and a YOLO dataset of full-resolution crops.

Candidates are linked into tracklets (pbv.balltrack), minus static hotspots (a rack of spare balls);
only tracklets that persist, travel and visibly move are kept as balls (logos, lights, heads and shoes don't), and short gaps inside a track are filled,
so the dataset also has balls the teacher missed. Every ball-like track is kept, including balls on
other courts: they're real pickleballs, which is what the detector should learn.

The dataset is 640x640 crops at native resolution, not whole frames: a far ball is ~5 px in a 1080p
frame, and training on whole frames at imgsz 640 would shrink it to ~2 px. The detector is then run at
native resolution (BallDetector uses imgsz 1920 for a trained model). Crops are:
  - positives: a labelled ball at a random spot in the crop (any other labelled ball inside is labelled too)
  - hard negatives: centred on a confident teacher candidate that isn't on any ball track (a head, a light)
  - random negatives: anywhere with no candidate nearby
Train/val are split by time (the last --val-frac of each video) to avoid near-duplicate leakage.

Usage: python training/labels.py training/data/train-1104-r0 [more dirs] --out training/data/ds-r0
       python training/labels.py training/data/test-1011-r0 --labels-only    (labels.json for evaluation)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbv.balltrack import drop_static, is_ball_track, link_tracklets, static_hotspots  # noqa: E402

CROP = 640


def ball_labels(frames, width, hot=None):
    """frame -> [[x, y, w, h, interpolated], ...] from the ball-like tracklets; gaps filled linearly.
    Candidates at static hotspots (a rack of spare balls) are left out: they'd link into fake tracks."""
    hot = static_hotspots(frames) if hot is None else hot
    tracks = [t for t in link_tracklets(drop_static(frames, hot), width) if is_ball_track(t, width)]
    out = {}
    for t in tracks:
        a = np.asarray(t, float)
        w, h = (float(np.clip(np.median(a[:, k]), 5, 30)) for k in (3, 4))
        for p, q in zip(a, a[1:]):
            for i in range(int(p[0]), int(q[0])):
                u = (i - p[0]) / (q[0] - p[0])
                out.setdefault(i, []).append([p[1] + u * (q[1] - p[1]), p[2] + u * (q[2] - p[2]), w, h, bool(i != p[0])])
        out.setdefault(int(a[-1, 0]), []).append([float(a[-1, 1]), float(a[-1, 2]), w, h, False])
    return out, tracks


def hard_negatives(frames, labels, tracks, hot, min_conf=0.25, away=40.0):
    """(frame, x, y) of confident candidates on no ball track and nowhere near a labelled ball.
    Hotspots are excluded: they're often real (spare) balls, which mustn't be taught as background."""
    on_track = {(int(p[0]), round(p[1], 1), round(p[2], 1)) for t in tracks for p in t}
    out = []
    for i, fr in enumerate(drop_static(frames, hot)):
        for c in fr or []:
            if c[4] < min_conf or (i, round(c[0], 1), round(c[1], 1)) in on_track:
                continue
            if all(np.hypot(c[0] - b[0], c[1] - b[1]) > away for b in labels.get(i, [])):
                out.append((i, c[0], c[1]))
    return out


def crop_origin(x, y, W, H, rnd, margin=24):
    """Top-left of a CROP x CROP window containing (x, y) at a random position, at least margin px from its edge."""
    lo_x, hi_x = max(0, int(x) - CROP + margin), min(W - CROP, int(x) - margin)
    lo_y, hi_y = max(0, int(y) - CROP + margin), min(H - CROP, int(y) - margin)
    return rnd.randint(lo_x, max(lo_x, hi_x)), rnd.randint(lo_y, max(lo_y, hi_y))


def yolo_rows(balls, x0, y0):
    rows = []
    for x, y, w, h, *_ in balls:
        cx, cy = x - x0, y - y0
        if 2 <= cx <= CROP - 2 and 2 <= cy <= CROP - 2:
            rows.append(f"0 {cx / CROP:.6f} {cy / CROP:.6f} {w / CROP:.6f} {h / CROP:.6f}\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, nargs="+", help="scan folders (each with candidates.json)")
    ap.add_argument("--out", type=Path, help="dataset folder (default: <first data>/yolo)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth labelled frame (neighbours are near-duplicates)")
    ap.add_argument("--neg-frac", type=float, default=0.25, help="negatives per positive (half hard, half random)")
    ap.add_argument("--labels-only", action="store_true")
    args = ap.parse_args()

    root = args.out or args.data[0] / "yolo"
    rnd = random.Random(0)
    counts = {"train": 0, "val": 0}
    for data in args.data:
        cand = json.loads((data / "candidates.json").read_text())
        meta, frames = cand["meta"], cand["frames"]
        W, H = meta["width"], meta["height"]
        hot = static_hotspots(frames)
        labels, tracks = ball_labels(frames, W, hot)
        n_scanned = sum(f is not None for f in frames)
        print(f"{data.name}: {n_scanned} scanned frames, {len(tracks)} ball tracks -> {len(labels)} frames with a ball "
              f"({sum(all(b[4] for b in v) for v in labels.values())} only by gap filling); "
              f"{len(hot)} static hotspot cells ignored")
        (data / "labels.json").write_text(json.dumps({"meta": meta, "labels": {str(k): v for k, v in labels.items()}}))
        if args.labels_only:
            continue

        pos = sorted(labels)[::args.stride]
        hard = hard_negatives(frames, labels, tracks, hot)
        rnd.shuffle(hard)
        hard = hard[: int(len(pos) * args.neg_frac / 2)]
        scanned = [i for i, f in enumerate(frames) if f is not None]
        rand = rnd.sample(scanned, min(len(scanned), int(len(pos) * args.neg_frac / 2)))
        jobs = {}  # frame -> [(kind, x, y)]
        for i in pos:
            jobs.setdefault(i, []).append(("pos",) + tuple(rnd.choice(labels[i])[:2]))
        for i, x, y in hard:
            jobs.setdefault(i, []).append(("hard", x, y))
        for i in rand:
            jobs.setdefault(i, []).append(("rand", None, None))
        split_at = scanned[int(len(scanned) * (1 - args.val_frac))]
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            (root / sub).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(meta["video"])
        last = max(jobs)
        for i in range(last + 1):  # one sequential pass: far faster than seeking per frame
            if i not in jobs:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            split = "val" if i >= split_at else "train"
            balls = labels.get(i, [])
            for c in frames[i]:  # blur out hotspot objects (spare balls in a rack): unlabelled, but not background
                if (int(c[0] // 8), int(c[1] // 8)) in hot:
                    r = int(max(c[2], c[3]) / 2) + 4
                    y0, y1, x0, x1 = max(int(c[1]) - r, 0), int(c[1]) + r, max(int(c[0]) - r, 0), int(c[0]) + r
                    img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), 6)
            for k, (kind, x, y) in enumerate(jobs[i]):
                if kind == "rand":
                    for _ in range(20):  # somewhere with no candidate at all
                        x0, y0 = rnd.randint(0, W - CROP), rnd.randint(0, H - CROP)
                        if not any(x0 - 8 <= c[0] <= x0 + CROP + 8 and y0 - 8 <= c[1] <= y0 + CROP + 8 for c in frames[i]):
                            break
                    else:
                        continue
                else:
                    x0, y0 = crop_origin(x, y, W, H, rnd)
                rows = yolo_rows(balls, x0, y0)
                if kind == "hard" and rows:
                    continue  # a labelled ball is in this crop after all; positives cover it
                name = f"{data.name}_{i:06d}_{k}"
                cv2.imwrite(str(root / f"images/{split}/{name}.jpg"), img[y0:y0 + CROP, x0:x0 + CROP],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                (root / f"labels/{split}/{name}.txt").write_text("".join(rows))
                counts[split] += 1
        print(f"  {len(pos)} positive, {len(hard)} hard-negative, {len(rand)} random crops queued")
    if not args.labels_only:
        (root / "data.yaml").write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: pickleball\n")
        print(f"dataset: {counts['train']} train / {counts['val']} val crops -> {root / 'data.yaml'}")


if __name__ == "__main__":
    main()
