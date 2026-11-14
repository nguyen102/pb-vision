"""Ball candidates -> ball track.

The detector reports several ball candidates per frame (low confidence threshold). Picking the
single most confident box per frame lets a head, a light or a ball on the next court win whenever
the real ball is blurred. Instead candidates are:
  1. scored for motion (frame difference at the box; a fixed camera makes still things still), and
     dropped at static hotspots (spots with a candidate in many frames: a rack of spare balls, a light),
  2. linked into tracklets with a constant-velocity prediction,
  3. filtered to where our ball can be (BallRegion: court position at the depth its size implies),
  4. combined into one path: best-scoring tracklets first, each claiming the frames it covers.
offscreen_frames then tells "left the picture" (a lob over the top) apart from "not detected".

A candidate is [x, y, w, h, conf, motion] in image pixels (centre, size).
"""
import cv2
import numpy as np

from .constants import COURT_L, COURT_W


def motion_score(prev_gray, gray, box, pad=2, full=48.0):
    """0..1: how much the picture changed since the previous frame inside box (x, y, w, h)."""
    if prev_gray is None:
        return 0.5  # unknown, neutral
    x, y, w, h = box
    r = max(w, h) / 2 + pad
    H, W = gray.shape
    x0, x1 = int(max(x - r, 0)), int(min(x + r + 1, W))
    y0, y1 = int(max(y - r, 0)), int(min(y + r + 1, H))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    d = cv2.absdiff(gray[y0:y1, x0:x1], prev_gray[y0:y1, x0:x1])
    return float(min(np.percentile(d, 90) / full, 1.0))


def merge_candidates(cands, min_dist=6.0, k=10):
    """Drop near-duplicate candidates (full-frame and crop passes find the same ball), keep the top k."""
    out = []
    for c in sorted(cands, key=lambda c: -c[4]):
        if all(np.hypot(c[0] - o[0], c[1] - o[1]) > min_dist for o in out):
            out.append(c)
    return out[:k]


def static_hotspots(cands, cell=8, frac=0.02, min_frames=90, min_conf=0.1):
    """Grid cells (x // cell, y // cell) where a candidate shows up in many frames: fixed objects
    (a rack of spare balls behind the court, a light, a logo). Real play never parks the ball in one
    spot for long. A cell is hot when it or its 8 neighbours has a candidate in > frac of the
    scanned frames (and at least min_frames, so short clips aren't over-filtered)."""
    counts, n = {}, 0
    for fr in cands:
        if fr is None:
            continue
        n += 1
        for key in {(int(c[0] // cell), int(c[1] // cell)) for c in fr if c[4] >= min_conf}:
            counts[key] = counts.get(key, 0) + 1
    need = max(frac * n, min_frames)
    hot = set()
    for (i, j) in counts:
        if sum(counts.get((i + di, j + dj), 0) for di in (-1, 0, 1) for dj in (-1, 0, 1)) >= need:
            hot.add((i, j))
    return hot


def drop_static(cands, hot, cell=8):
    """Candidates minus those in hot cells (or next to one)."""
    def is_hot(c):
        i, j = int(c[0] // cell), int(c[1] // cell)
        return any((i + di, j + dj) in hot for di in (-1, 0, 1) for dj in (-1, 0, 1))
    return [None if fr is None else [c for c in fr if not is_hot(c)] for fr in cands]


def link_tracklets(cands, width, max_gap=4, gate=0.025, first_step=0.07):
    """Per-frame candidate lists -> tracklets [(frame, x, y, w, h, conf, motion), ...].

    A tracklet predicts its next position from its last velocity; a candidate joins the tracklet
    whose prediction is nearest within the gate (grown with the frames since it was last seen).
    Hits and bounces change the velocity abruptly, so a tracklet usually ends there and a new one starts.
    """
    tracks, live = [], []
    g0, step0 = gate * width, first_step * width
    for t, fr in enumerate(cands):
        if fr is None:  # camera cut / frame not scanned: nothing continues across it
            live = []
            continue
        live = [tr for tr in live if t - tr[-1][0] <= max_gap + 1]
        pairs = []
        for k, tr in enumerate(live):
            j, px, py = tr[-1][:3]
            dt = t - j
            if len(tr) >= 2:
                j2, qx, qy = tr[-2][:3]
                vx, vy = (px - qx) / (j - j2), (py - qy) / (j - j2)
                pred, r = (px + vx * dt, py + vy * dt), g0 * (1 + 0.5 * (dt - 1)) + 0.25 * np.hypot(vx, vy)
            else:
                pred, r = (px, py), step0 * dt
            for m, c in enumerate(fr):
                d = np.hypot(c[0] - pred[0], c[1] - pred[1])
                if d <= r:
                    pairs.append((d, k, m))
        used_t, used_c = set(), set()
        for d, k, m in sorted(pairs):
            if k in used_t or m in used_c:
                continue
            live[k].append((t, *fr[m][:6]))
            used_t.add(k)
            used_c.add(m)
        for m, c in enumerate(fr):
            if m not in used_c:
                tr = [(t, *c[:6])]
                live.append(tr)
                tracks.append(tr)
    return tracks


def tracklet_score(tr, width, min_move=0.008):
    """Evidence that a tracklet is a ball in play: confident, moving detections over several frames.
    A still tracklet (a head, a ball on the floor or in a hand) keeps a fraction of its score."""
    a = np.asarray(tr, float)
    s = float(np.sum(a[:, 5] * (0.3 + a[:, 6])))
    moving = len(a) > 1 and np.ptp(a[:, 1:3], axis=0).max() >= min_move * width
    return s if moving else 0.3 * s


def is_ball_track(tr, width, min_len=5, min_move=0.008, min_conf=0.1, min_motion=0.15):
    """Strict test for pseudo-labels: long, moving, confident on average, and visibly in motion."""
    a = np.asarray(tr, float)
    if len(a) < min_len or np.ptp(a[:, 1:3], axis=0).max() < min_move * width:
        return False
    return a[:, 5].mean() >= min_conf and np.median(a[:, 6]) >= min_motion


BALL_M = 0.074  # pickleball diameter


class BallRegion:
    """Where our ball can be, using its apparent size as a depth cue.

    In a behind-the-baseline view, image position alone can't tell a ball 2 m above our near sideline
    from a ball on the next court: they land on the same pixels. But a 74 mm ball looks ~3x bigger near
    the camera than at the far baseline. So for a candidate of size s: the depths along the court where
    the ball would look that big (within a box-noise factor), and at any of them the candidate must
    be between our sidelines (plus margin), no lower than the floor, and no higher than `up` metres.
    Vertical metres are approximated by horizontal pixels-per-metre at that depth.
    """

    def __init__(self, court, side=1.5, end=3.0, up=8.0, loose=2.6, tight=1.0, step=0.25):
        d = np.arange(-end, COURT_L + end + step, step)
        mid = np.c_[np.full_like(d, COURT_W / 2), d]
        left, right = court.to_image(np.c_[np.full_like(d, -side), d]), court.to_image(np.c_[np.full_like(d, COURT_W + side), d])
        self.ppm = np.linalg.norm(court.to_image(mid + [0.5, 0]) - court.to_image(mid - [0.5, 0]), axis=1)
        self.xl, self.xr = left[:, 0], right[:, 0]
        self.floor = court.to_image(mid)[:, 1]
        self.up, self.loose, self.tight = up, loose, tight

    def contains(self, c):
        x, y, w, h = c[:4]
        # Blur stretches one side, so use the short one. Boxes on a few-pixel ball are often looser
        # than the ball but rarely tighter: the box may be up to `loose` x the ball, down to 1/`tight`.
        ppm = min(w, h) / BALL_M
        ok = (self.ppm >= ppm / self.loose) & (self.ppm <= ppm * self.tight)
        ok &= (x >= np.minimum(self.xl, self.xr)) & (x <= np.maximum(self.xl, self.xr))
        ok &= (y <= self.floor + 0.3 * self.ppm) & (y >= self.floor - self.up * self.ppm)
        return bool(ok.any())


def select_path(cands, width, region=None, min_frac=0.5, single_conf=0.4):
    """Per-frame candidates -> per-frame ball [x, y] or None.

    Tracklets are taken best score first; each claims the frames it covers that are still free,
    and is dropped if most of them are already taken (it overlaps a better track). One-frame
    tracklets survive only when very confident.
    """
    cands = drop_static(cands, static_hotspots(cands))
    if region is not None:
        cands = [None if fr is None else [c for c in fr if region.contains(c)] for fr in cands]
    tracks = link_tracklets(cands, width)
    scored = sorted(((tracklet_score(tr, width), tr) for tr in tracks), key=lambda s: -s[0])
    out = [None] * len(cands)
    for s, tr in scored:
        if len(tr) == 1 and tr[0][5] < single_conf:
            continue
        free = [p for p in tr if out[p[0]] is None]
        if len(free) < min_frac * len(tr):
            continue
        for p in free:
            out[p[0]] = [round(p[1], 1), round(p[2], 1)]
    return out


def offscreen_frames(path, width, height, margin=0.05, lookahead=4, max_frames=90):
    """Per-frame True where the ball has left the picture (a lob over the top edge, a ball out the side or
    past the camera) rather than gone undetected or out of play.

    A gap counts as off-screen when the last tracked point before it is within `margin` of a frame edge and
    its velocity carries it past that edge within `lookahead` frames. The gap is marked until the ball
    reappears, for at most max_frames (3 s at 30 fps): longer than that, the point is probably over.
    """
    n = len(path)
    out = [False] * n
    mx, my = margin * width, margin * height
    t = 1
    while t < n:
        if path[t] is not None or path[t - 1] is None:
            t += 1
            continue
        (x, y), prev = path[t - 1], (path[t - 2] if t >= 2 else None)
        vx, vy = (x - prev[0], y - prev[1]) if prev is not None else (0.0, 0.0)
        near_edge = x < mx or x > width - mx or y < my or y > height - my
        fx, fy = x + vx * lookahead, y + vy * lookahead
        leaving = fx < 0 or fx >= width or fy < 0 or fy >= height
        end = t
        while end < n and path[end] is None:
            end += 1
        if near_edge and leaving:
            for k in range(t, min(end, t + max_frames)):
                out[k] = True
        t = end
    return out
