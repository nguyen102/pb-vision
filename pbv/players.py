"""Pick the 4 players out of everyone the person tracker sees, and give each one a stable id.

The tracker records every person near the court with ByteTrack ids, which churn (a player hidden behind
another player, or behind the net, comes back with a new id) and include refs and spectators.
select_players works on the whole clip at once:
1. tracklets: each ByteTrack id's detections inside the play area, cut where the position jumps further
   than a person can move (the id slid onto someone else);
2. each tracklet's clothes: the median colour descriptor over its frames where nobody overlaps it;
3. four players seeded from the moment four long tracklets are on court together, so everyone's clothes
   are known from the start;
4. tracklets go to players longest first. A tracklet costs clothing difference plus motion error: how far
   it starts from where that player was heading when last seen (last position + velocity, allowed error
   growing with the time hidden), and, if the player is seen again later, how far it ends from where they
   reappear. A player can't be in two tracklets at once, a tracklet that is too far to reach or dressed
   like nobody stays unassigned, and the players' clothes are re-estimated and the assignment repeated.
"""
import numpy as np

from .appearance import distance
from .constants import COURT_W
from .court import Court

PLAYER_IDS = ("1", "2", "3", "4")


def static_bystanders(frames, fps, min_move=1.0, long_s=10.0):
    """ByteTrack ids that stand still off the court for their whole life: refs, line judges, benches.

    An id is dropped when 90% of its positions stay within min_move metres of its median and that
    median is beside the court (past a sideline), or behind a baseline for at least long_s seconds.
    A server waiting behind the baseline is still for a few seconds only, so short tracks there stay.
    """
    tracks = {}
    for fp in frames:
        for p in fp:
            tracks.setdefault(p["id"], []).append(p["court"])
    static = set()
    for pid, pts in tracks.items():
        a = np.asarray(pts, float)
        med = np.median(a, axis=0)
        if np.percentile(np.linalg.norm(a - med, axis=1), 90) >= min_move:
            continue
        beside = med[0] < 0 or med[0] > COURT_W
        if beside or (not Court.inside(med) and len(a) >= long_s * fps):
            static.add(pid)
    return static


def overlaps_someone(fp, min_frac=0.25):
    """Per person in one frame: does another box cover >= min_frac of theirs? Then their colours are mixed."""
    out = []
    for i, p in enumerate(fp):
        x1, y1, x2, y2 = p["bbox"]
        area = max((x2 - x1) * (y2 - y1), 1e-9)
        cover = 0.0
        for j, q in enumerate(fp):
            if i != j:
                ix = max(0.0, min(x2, q["bbox"][2]) - max(x1, q["bbox"][0]))
                iy = max(0.0, min(y2, q["bbox"][3]) - max(y1, q["bbox"][1]))
                cover = max(cover, ix * iy / area)
        out.append(cover >= min_frac)
    return out


def _velocity(pts, fps):
    """m/s from a few court positions (least-squares slope), 0 for a single point."""
    if len(pts) < 2:
        return np.zeros(2)
    t = np.arange(len(pts)) / fps
    return np.polyfit(t, np.asarray(pts, float), 1)[0]


PERSON_M = 1.7  # a standing player's height, to turn metres into box heights


def _jump(a, b):
    """How far the top-centre of box a moved to b, in box heights (of the taller box). The top (head),
    not the feet: a far player's feet go behind the net, which moves the box bottom, and their court
    position, by a metre or more between frames."""
    h = max(a[3] - a[1], b[3] - b[1], 1e-9)
    return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, a[1] - b[1])) / h


def build_tracklets(frames, fps, skip=(), max_speed=7.0, max_gap_s=1.0, motion_s=0.5):
    """Per-frame raw people -> tracklets: one ByteTrack id's run in the play area, split where its box jumps
    further than a person moving at max_speed could (plus a quarter box height of jitter), or where it's
    unseen for more than max_gap_s.

    Each tracklet: frames, people, start/end frame, start/end court position (median over motion_s, since a
    far player's position jumps when their feet go behind the net) and velocity (fitted over motion_s), and
    look (median descriptor over frames where nobody overlaps them, or None).
    """
    runs, live = [], {}
    gap = int(max_gap_s * fps)
    for f, fp in enumerate(frames):
        mixed = dict(zip(map(id, fp), overlaps_someone(fp)))
        for p in fp:
            if p["id"] in skip or not Court.in_play_area(p["court"]):
                continue
            t = live.get(p["id"])
            if t is not None:
                dt = f - t["frames"][-1]
                if dt > gap or _jump(t["people"][-1]["bbox"], p["bbox"]) > 0.25 + max_speed * dt / fps / PERSON_M:
                    t = None
            if t is None:
                t = live[p["id"]] = {"frames": [], "people": [], "looks": []}
                runs.append(t)
            if p.get("look") is not None and not mixed[id(p)]:
                t["looks"].append(np.asarray(p["look"], float))
            t["frames"].append(f)
            t["people"].append(p)
    k = max(2, int(motion_s * fps))
    for t in runs:
        pts = np.array([p["court"] for p in t["people"]], float)
        t["start"], t["end"] = t["frames"][0], t["frames"][-1]
        t["p0"], t["p1"] = np.median(pts[:k], axis=0), np.median(pts[-k:], axis=0)
        t["v0"], t["v1"] = _velocity(pts[:k], fps), _velocity(pts[-k:], fps)
        looks = t.pop("looks")
        t["look"] = np.median(looks, axis=0) if looks else None
    return runs


def _anchor(tracks):
    """Up to 4 tracklets that are on court together for longest: the seed for the four players."""
    best, best_len = [], -1
    starts = sorted({t["start"] for t in tracks})
    for f in starts + [0]:
        here = sorted((t for t in tracks if t["start"] <= f <= t["end"]), key=lambda t: -len(t["frames"]))[:4]
        total = sum(len(t["frames"]) for t in here) + 10 ** 6 * len(here)  # most people first, then longest
        if total > best_len:
            best, best_len = here, total
    return best


class Assigner:
    """Costs of tracklets for players (see the module docstring); None = not allowed.

    look_w: weight of clothing difference (0..1) against motion error (1 = one allowed-error radius).
    max_look: a tracklet whose clothes differ more than this from a player's is not them (a stranger).
    Motion: a player last seen at p moving at v is expected at p + v * min(dt, horizon_s) after dt s,
    give or take 0.5 + walk * dt metres; further than 0.5 + max_speed * dt is out of reach.
    unassigned: cost of leaving a tracklet out (a stranger, or a duplicate box).
    """

    def __init__(self, fps, look_w=8.0, max_look=0.6, walk=2.0, max_speed=7.0, horizon_s=1.0, overlap=2,
                 unassigned=6.0):
        self.fps, self.look_w, self.max_look, self.walk = fps, look_w, max_look, walk
        self.max_speed, self.horizon_s, self.overlap, self.unassigned = max_speed, horizon_s, overlap, unassigned

    def _motion(self, here, there, dt):
        """Normalised error of seeing someone at `here` when `there` was predicted dt s away; None if unreachable."""
        err = float(np.hypot(*(here - there)))
        if err > 0.5 + self.max_speed * dt:
            return None
        return err / (0.5 + self.walk * dt)

    def cost(self, t, look, owned):
        """Cost of tracklet t for a player with clothes `look` who already has tracklets `owned`."""
        lo, hi = t["start"] + self.overlap, t["end"] - self.overlap
        if any(o["start"] <= hi and lo <= o["end"] for o in owned):
            return None  # they're already somewhere else then
        c = 0.0
        if t["look"] is not None and look is not None:
            d = distance(t["look"], look)
            if d > self.max_look:
                return None
            c += self.look_w * d
        before = max((o for o in owned if o["end"] <= lo), key=lambda o: o["end"], default=None)
        after = min((o for o in owned if o["start"] >= hi), key=lambda o: o["start"], default=None)
        if before is not None:
            dt = max(t["start"] - before["end"], 1) / self.fps
            m = self._motion(t["p0"], before["p1"] + before["v1"] * min(dt, self.horizon_s), dt)
            if m is None:
                return None
            c += m
        if after is not None:
            dt = max(after["start"] - t["end"], 1) / self.fps
            m = self._motion(t["p1"], after["p0"] - after["v0"] * min(dt, self.horizon_s), dt)
            if m is None:
                return None
            c += m
        return c


def _profiles(owned, n):
    """Each player's clothes: frame-weighted mean of their tracklets' descriptors (None if none known)."""
    out = []
    for k in range(n):
        ls = [(t["look"], len(t["frames"])) for t in owned[k] if t["look"] is not None]
        out.append(np.average([l for l, _ in ls], axis=0, weights=[w for _, w in ls]) if ls else None)
    return out


def select_players(frames, fps, rounds=3, **assigner_args):
    """Per-frame raw people [{"id", "bbox", "court", "look"?}] -> per-frame {player_id: {"bbox", "court"}}.

    Drops people outside the play area and ids that stand still beside the court, splits the rest into
    tracklets and gives each tracklet to one of the 4 players or to nobody (module docstring).
    """
    tracks = build_tracklets(frames, fps, skip=static_bystanders(frames, fps))
    out = [{} for _ in frames]
    if not tracks:
        return out
    seeds = _anchor(tracks)
    # ids in a fixed order at the seed moment: near side left to right, then far side left to right
    seeds.sort(key=lambda t: (Court.side(t["p0"]) != "near", t["p0"][0]))
    n = len(seeds)
    a = Assigner(fps, **assigner_args)
    looks = [t["look"] for t in seeds]
    order = sorted((t for t in tracks if not any(t is s for s in seeds)), key=lambda t: -len(t["frames"]))
    for _ in range(rounds):
        owned = [[s] for s in seeds]
        for t in order:
            costs = [a.cost(t, looks[k], owned[k]) for k in range(n)]
            ok = [(c, k) for k, c in enumerate(costs) if c is not None and c < a.unassigned]
            if ok:
                owned[min(ok)[1]].append(t)
        new = _profiles(owned, n)
        looks = [nl if nl is not None else ol for nl, ol in zip(new, looks)]
    for k in range(n):
        for t in sorted(owned[k], key=lambda t: t["start"]):
            for f, p in zip(t["frames"], t["people"]):
                out[f][PLAYER_IDS[k]] = {"bbox": p["bbox"], "court": p["court"]}
    return out
