"""Where does the tracker lose the ball? For each hand-labelled frame, follow the true ball's candidate
through the tracker's stages and report the first one that loses it (or what was tracked instead).

Ball frames (the labelled ball is in play):
  detector   no candidate near the ball: the detector missed it (not a tracker problem)
  static     dropped as a static hotspot
  region     dropped by the court check (BallRegion)
  single     its tracklet was a single, not-confident-enough detection
  overlap    its tracklet lost the frame to a better-scoring track (the path has something else there)
  dropped    its tracklet was skipped because most of its frames were already taken
  clean      the path had the ball but clean_ball removed or moved it
  ok         tracked correctly
No-ball frames: 'quiet' (nothing tracked) or 'false' (something tracked).

Usage: python training/diagnose.py training/data/test-1011-r1 --court outputs/10-11-BBB-mb-tim-cut/court.json \
           --gt training/data/gt-1011/labels.json [--show]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbv import events  # noqa: E402
from pbv.balltrack import BallRegion, drop_static, link_tracklets, select_path, static_hotspots, tracklet_score  # noqa: E402
from pbv.court import Court  # noqa: E402
from evaluate import size_hand_labels  # noqa: E402


def near(p, b, r):
    return p is not None and np.hypot(p[0] - b[0], p[1] - b[1]) <= max(r, 0.6 * min(b[2], b[3]))


def diagnose(frames, width, region, gt, r=10.0, select=select_path, single_conf=0.4, min_frac=0.5):
    """frame -> (cause, detail) for every labelled frame."""
    path = select(frames, width, region)
    clean = events.clean_ball(path, max_jump=0.08 * width)
    hot = static_hotspots(frames)
    no_static = drop_static(frames, hot)
    in_region = [None if fr is None else [c for c in fr if region.contains(c)] for fr in no_static]
    tracks = link_tracklets(in_region, width)
    owner = {}  # (frame, x, y) -> tracklet
    for tr in tracks:
        for p in tr:
            owner[(p[0], round(p[1], 1), round(p[2], 1))] = tr
    out = {}
    for i, labels in gt.items():
        if i >= len(frames) or frames[i] is None:
            continue
        cp = clean[i] if not np.isnan(clean[i][0]) else None
        if not labels:
            out[i] = ("false", cp) if cp is not None else ("quiet", None)
            continue
        b = labels[0]
        if near(cp, b, r):
            out[i] = ("ok", None)
            continue
        if near(path[i], b, r):
            out[i] = ("clean", cp)
            continue

        def hit(fr):
            m = [c for c in fr or [] if near(c, b, r)]
            return max(m, key=lambda c: c[4]) if m else None
        c = hit(frames[i])
        if c is None:
            out[i] = ("detector", path[i])
        elif hit(no_static[i]) is None:
            out[i] = ("static", c)
        elif (c := hit(in_region[i])) is None:
            out[i] = ("region", hit(no_static[i]))
        else:
            tr = owner.get((i, round(c[0], 1), round(c[1], 1)))
            if tr is not None and len(tr) == 1 and tr[0][5] < single_conf:
                out[i] = ("single", c)
            elif path[i] is not None:
                out[i] = ("overlap", {"ours": round(tracklet_score(tr, width), 2) if tr else None, "len": len(tr) if tr else 0,
                                      "path": path[i]})
            else:
                out[i] = ("dropped", {"len": len(tr) if tr else 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", type=Path)
    ap.add_argument("--court", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--show", action="store_true", help="list each non-ok frame")
    args = ap.parse_args()
    cand = json.loads((args.scan / "candidates.json").read_text())
    gt = {int(k): v for k, v in json.loads(Path(args.gt).read_text())["labels"].items()}
    gt = size_hand_labels(gt, [cand["frames"]])
    res = diagnose(cand["frames"], cand["meta"]["width"], BallRegion(Court.load(args.court)), gt, args.radius)
    balls = Counter(c for i, (c, _) in res.items() if gt[i])
    empty = Counter(c for i, (c, _) in res.items() if not gt[i])
    nb = sum(balls.values())
    print(f"{args.scan.name}: {nb} ball frames")
    for k, v in balls.most_common():
        print(f"  {k:9s} {v:4d}  {100 * v / nb:5.1f}%")
    print(f"{sum(empty.values())} no-ball frames: " + ", ".join(f"{k} {v}" for k, v in empty.items()))
    if args.show:
        for i, (c, d) in sorted(res.items()):
            if c not in ("ok", "quiet"):
                print(f"  {i:6d} {c:9s} gt={[round(v) for v in gt[i][0][:2]] if gt[i] else None} {d}")


if __name__ == "__main__":
    main()
