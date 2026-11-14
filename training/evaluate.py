"""Step 4: compare ball detectors and trackers on a held-out video (a different day from training).

Each detector is run over the same test frames with training/scan.py (one candidates.json per detector).
Reference labels come from the union of all their candidates: merged per frame, linked into tracklets and
kept only if they persist, travel and move like a ball. Using every detector keeps the reference from
favouring whichever one built it. It is still a pseudo-label, so --sheet writes crops of disagreements
to check by eye, and --gt takes a hand-labelled labels.json (training/annotate.py) instead.

Reported per detector, on frames where the reference has a ball on our court:
  top-1 recall   the most confident candidate is the ball (the old pipeline's rule)
  any recall     some candidate is the ball (what tracking has to work with)
and for the ball track the pipeline would produce:
  old  top-1 candidate with conf >= 0.15, then events.clean_ball (outliers out, gaps filled)
  new  balltrack.select_path over all candidates, limited to the court region, then clean_ball
  recall = reference ball frames tracked within --radius px; false = tracked frames with no reference
  ball within --radius px (wrong object, or a ball the reference missed: see the sheet).

Usage: python training/evaluate.py training/data/test-coco1280 training/data/test-r1 ... \
           --court outputs/10-11-BBB-mb-tim-cut/court.json [--frames 9000 --sheet out.jpg]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbv import events  # noqa: E402
from pbv.balltrack import BallRegion, merge_candidates, select_path  # noqa: E402
from pbv.court import Court  # noqa: E402
from training.labels import ball_labels  # noqa: E402


def near(p, balls, r):
    """p within r px of a ball, or within the ball's own size for a big (near-camera) one."""
    return p is not None and any(np.hypot(p[0] - b[0], p[1] - b[1]) <= max(r, 0.6 * min(b[2], b[3])) for b in balls)


def old_track(frames, width, conf=0.15):
    top = [fr[0][:2] if fr and fr[0][4] >= conf else None for fr in frames]
    return events.clean_ball(top, max_jump=0.08 * width)


def new_track(frames, width, region):
    return events.clean_ball(select_path(frames, width, region), max_jump=0.08 * width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans", type=Path, nargs="+")
    ap.add_argument("--court", type=Path, required=True)
    ap.add_argument("--gt", type=Path, help="labels.json to use as the reference instead of the union")
    ap.add_argument("--frames", type=int, help="only the first N frames")
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--sheet", type=Path, help="jpg of disagreements between the reference and the last scan's new track")
    args = ap.parse_args()

    scans = {}
    for d in args.scans:
        c = json.loads((d / "candidates.json").read_text())
        scans[d.name] = (c["meta"], c["frames"][: args.frames])
    meta = next(iter(scans.values()))[0]
    W, H, n = meta["width"], meta["height"], min(len(f) for _, f in scans.values())
    region = BallRegion(Court.load(args.court))

    if args.gt:
        gt = {int(k): v for k, v in json.loads(args.gt.read_text())["labels"].items()}
    else:
        union = [None if all(f[i] is None for _, f in scans.values()) else
                 merge_candidates([c for _, f in scans.values() for c in (f[i] or [])], k=20) for i in range(n)]
        gt, _ = ball_labels(union, W)
    scanned = [i for i in range(n) if all(f[i] is not None for _, f in scans.values())]
    if args.gt:  # hand labels: only the labelled frames count, and the labeller marked our court's ball only
        scanned = [i for i in scanned if i in gt]
    else:  # pseudo-labels include balls on other courts
        gt = {i: [b for b in v if region.contains(b)] for i, v in gt.items() if i < n}
    gt = {i: v for i, v in gt.items() if v}
    pos = [i for i in scanned if i in gt]
    print(f"{len(scanned)} frames, reference ball on our court in {len(pos)} ({100 * len(pos) / len(scanned):.0f}%)\n")
    print(f"{'detector':24s} {'top-1 rec':>9s} {'any rec':>8s} | {'old track rec':>13s} {'false':>6s} "
          f"| {'new track rec':>13s} {'false':>6s}")
    tracks = {}
    for name, (_, frames) in scans.items():
        frames = frames[:n]
        top1 = sum(near(frames[i][0][:2] if frames[i] else None, gt[i], args.radius) for i in pos)
        anyr = sum(any(near(c[:2], gt[i], args.radius) for c in frames[i]) for i in pos)
        row = f"{name:24s} {100 * top1 / len(pos):8.1f}% {100 * anyr / len(pos):7.1f}% |"
        for kind, tr in (("old", old_track(frames, W)), ("new", new_track(frames, W, region))):
            tracks[name, kind] = tr
            have = [i for i in scanned if not np.isnan(tr[i, 0])]
            rec = sum(near(tr[i], gt[i], args.radius) for i in pos if not np.isnan(tr[i, 0]))
            false = sum(not near(tr[i], gt.get(i, []), args.radius) for i in have)
            row += f" {100 * rec / len(pos):12.1f}% {100 * false / max(len(have), 1):5.1f}% |"
        print(row)
    print("\nfalse = share of tracked frames with no reference ball within the radius")

    if args.sheet:  # green circle = reference, yellow cross = new track of the last scan
        last = list(scans)[-1]
        tr = tracks[last, "new"]
        miss = [i for i in pos if not near(tr[i], gt[i], args.radius)]
        extra = [i for i in scanned if not np.isnan(tr[i, 0]) and not near(tr[i], gt.get(i, []), args.radius)]
        rnd = random.Random(0)
        pick = sorted(rnd.sample(miss, min(12, len(miss))) + rnd.sample(extra, min(12, len(extra))))
        cap, tiles = cv2.VideoCapture(meta["video"]), []
        for i in range(max(pick) + 1):
            if i not in pick:
                cap.grab()
                continue
            img = cap.read()[1]
            for b in gt.get(i, []):
                cv2.circle(img, (int(b[0]), int(b[1])), 12, (0, 255, 0), 1)
            if not np.isnan(tr[i, 0]):
                cv2.drawMarker(img, (int(tr[i, 0]), int(tr[i, 1])), (0, 255, 255), cv2.MARKER_CROSS, 18, 1)
            cx, cy = (tr[i] if i in extra else gt[i][0][:2])
            x0, y0 = int(np.clip(cx - 120, 0, W - 240)), int(np.clip(cy - 68, 0, H - 135))
            tile = cv2.resize(img[y0:y0 + 135, x0:x0 + 240], (480, 270), interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile, f"{i} {'MISS' if i in miss else 'EXTRA'}", (6, 20), 0, 0.6, (255, 255, 255), 2)
            tiles.append(tile)
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
        cv2.imwrite(str(args.sheet), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))
        print(f"sheet: {args.sheet} (top rows/ordered by frame; label says which kind)")


if __name__ == "__main__":
    main()
