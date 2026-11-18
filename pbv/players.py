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
5. teams keep their end: players never cross the net during a game, only when the teams change ends between
   games. When that happens is read from the clothes seen at each end over time (change_ends), and a player
   can't take a tracklet on the other team's side. Without this, a player hidden for a few seconds could be
   "found" on the far side, and two similar-looking players on opposite teams traded ids all game.
6. partners: tracklets of the two players of a team that are on court at the same time are different people,
   so stretches of overlapping tracklets are swapped as a whole when the pooled clothes and apparent heights
   fit the other partner better (split_partners). Partners in the same dark kit at the far end were decided
   wrongly tracklet by tracklet.

Clothes and apparent height are kept per player and per end: the same person looks different near the camera.
Scored with training/eval_ids.py on hand labels of three games (training/data/ids-*): 84% of boxes with the
right id before steps 5-6, height and ByteTrack links; 95% after (wrong ids 2 of 373; the rest have no id).
"""
import cv2
import numpy as np

from .appearance import distance
from .constants import COURT_W, NET_Y
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


def apparent_heights(frames):
    """{id(person): box height in metres}, measured with the court's scale at their feet: pixels per metre
    across the court there, from the camera's ground homography (refitted from the people's feet, whose court
    positions came from it). Not a true height (the camera looks down at an angle), but partners on one side
    are measured alike, and it tells a tall player from a short one when their clothes don't."""
    people = [p for fp in frames for p in fp]
    if len(people) < 8:
        return {}
    px = np.array([((p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][3]) for p in people], np.float64)
    m = np.array([p["court"] for p in people], np.float64)
    H, _ = cv2.findHomography(m, px)
    if H is None:
        return {}
    a = cv2.perspectiveTransform((m - [0.5, 0])[None], H)[0]
    b = cv2.perspectiveTransform((m + [0.5, 0])[None], H)[0]
    ppm = np.maximum(np.linalg.norm(b - a, axis=1), 1e-6)
    h = np.array([p["bbox"][3] - p["bbox"][1] for p in people])
    return {id(p): float(v) for p, v in zip(people, h / ppm)}


def build_tracklets(frames, fps, skip=(), max_speed=7.0, max_gap_s=1.0, motion_s=0.5, cut_overlaps=True,
                    heights=None, min_height_n=5):
    """Per-frame raw people -> tracklets: one ByteTrack id's run in the play area, split where its box jumps
    further than a person moving at max_speed could (plus a quarter box height of jitter), where it's
    unseen for more than max_gap_s, and (cut_overlaps) where another box starts or stops covering it.

    Each tracklet: frames, people, start/end frame, start/end court position (median over motion_s, since a
    far player's position jumps when their feet go behind the net) and velocity (fitted over motion_s), and
    look (median descriptor over frames where nobody overlaps them, or None), height (median apparent height
    over those frames, from heights = apparent_heights(frames); None under min_height_n of them), and per frame
    whether they are on the near side of the net.
    """
    runs, live = [], {}
    gap = int(max_gap_s * fps)
    for f, fp in enumerate(frames):
        mixed = dict(zip(map(id, fp), overlaps_someone(fp)))
        for p in fp:
            if p["id"] in skip or not Court.in_play_area(p["court"]):
                continue
            t, prev = live.get(p["id"]), None
            if t is not None:
                dt = f - t["frames"][-1]
                if dt > gap or _jump(t["people"][-1]["bbox"], p["bbox"]) > 0.25 + max_speed * dt / fps / PERSON_M:
                    t = None
                elif cut_overlaps and t["mixed"] != mixed[id(p)]:
                    # ByteTrack swaps people when their boxes overlap, without any jump: a tracklet that ran on
                    # would carry the other player's frames under one colour. Usually it doesn't swap, so the
                    # pieces stay linked (same_as) and the matcher prefers to give them the same player.
                    prev, t = t, None
            if t is None:
                t = live[p["id"]] = {"frames": [], "people": [], "looks": [], "heights": [],
                                     "mixed": mixed[id(p)], "same_as": prev}
                runs.append(t)
            if p.get("look") is not None and not mixed[id(p)]:
                t["looks"].append(np.asarray(p["look"], float))
            if heights and id(p) in heights and not mixed[id(p)]:
                t["heights"].append(heights[id(p)])
            t["frames"].append(f)
            t["people"].append(p)
    k = max(2, int(motion_s * fps))
    for t in runs:
        pts = np.array([p["court"] for p in t["people"]], float)
        t["start"], t["end"] = t["frames"][0], t["frames"][-1]
        t["near"] = pts[:, 1] >= NET_Y
        t["p0"], t["p1"] = np.median(pts[:k], axis=0), np.median(pts[-k:], axis=0)
        t["v0"], t["v1"] = _velocity(pts[:k], fps), _velocity(pts[-k:], fps)
        t.pop("mixed")
        t.setdefault("same_as", None)
        looks = t.pop("looks")
        t["look"] = np.median(looks, axis=0) if looks else None
        hs = t.pop("heights")
        t["height"] = float(np.median(hs)) if len(hs) >= min_height_n else None
    return runs


def _anchor(tracks):
    """Up to 4 tracklets that are on court together for longest: the seed for the four players."""
    best, best_len = [], -1
    starts = sorted({t["start"] for t in tracks})
    for f in starts + [0]:
        here = sorted((t for t in tracks if t["start"] <= f <= t["end"]), key=lambda t: -len(t["frames"]))[:4]
        near = sum(Court.side(t["p0"]) == "near" for t in here)
        # most people first, then two on each side (a stranger in the seed would cost a player their id), then longest
        total = sum(len(t["frames"]) for t in here) + 10 ** 8 * len(here) + 10 ** 7 * (near == 2)
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
    height_w: weight of the apparent-height difference, per height_sd metres (partners in the same dark kit
    are often only told apart by height).
    ends: per frame, the side each player should be on (True = near), from change_ends; None = anywhere.
    A tracklet must spend most of its settled frames (not within changeover_s of the teams changing ends,
    when players walk round the net) on that side.
    """

    def __init__(self, fps, look_w=8.0, max_look=0.6, walk=2.0, max_speed=7.0, horizon_s=1.0, overlap=2,
                 unassigned=6.0, changeover_s=30.0, height_w=1.0, height_sd=0.1):
        self.fps, self.look_w, self.max_look, self.walk = fps, look_w, max_look, walk
        self.max_speed, self.horizon_s, self.overlap, self.unassigned = max_speed, horizon_s, overlap, unassigned
        self.changeover_s, self.height_w, self.height_sd = changeover_s, height_w, height_sd
        self.ends = self.settled = None

    def set_ends(self, ends):
        """ends: (n_frames, n_players) bool, where each player should be (True = near side)."""
        self.ends = ends
        flips = np.flatnonzero(np.any(ends[1:] != ends[:-1], axis=1)) + 1
        self.settled = np.ones(len(ends), bool)
        m = int(self.changeover_s * self.fps)
        for f in flips:
            self.settled[max(f - m, 0):f + m] = False

    def wrong_side(self, t, k):
        """Is tracklet t mostly on the other team's side for player k?"""
        if self.ends is None:
            return False
        fr = np.asarray(t["frames"])
        ok = self.settled[fr]
        if not ok.any():
            return False
        return np.mean(t["near"][ok] == self.ends[fr[ok], k]) < 0.5

    def _motion(self, here, there, dt):
        """Normalised error of seeing someone at `here` when `there` was predicted dt s away; None if unreachable."""
        err = float(np.hypot(*(here - there)))
        if err > 0.5 + self.max_speed * dt:
            return None
        return err / (0.5 + self.walk * dt)

    def cost(self, t, look, owned, k=None, height=None):
        """Cost of tracklet t for player k, with clothes `look` and apparent height `height`, who already has
        tracklets `owned`."""
        lo, hi = t["start"] + self.overlap, t["end"] - self.overlap
        if any(o["start"] <= hi and lo <= o["end"] for o in owned):
            return None  # they're already somewhere else then
        if k is not None and self.wrong_side(t, k):
            return None  # the other team's end of the court
        c = 0.0
        if t["look"] is not None and look is not None:
            d = distance(t["look"], look)
            if d > self.max_look:
                return None
            c += self.look_w * d
        if t.get("height") is not None and height is not None:
            c += self.height_w * abs(t["height"] - height) / self.height_sd
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


def _side_looks(tracks, n_win, win, fps):
    """Per side (0 = near, 1 = far) and window: the summed clothes of everyone seen there, weighted by seconds."""
    dim = next(len(t["look"]) for t in tracks if t["look"] is not None)
    looks, secs = np.zeros((2, n_win, dim)), np.zeros((2, n_win))
    for t in tracks:
        if t["look"] is None:
            continue
        fr = np.asarray(t["frames"])
        for si, side in enumerate((True, False)):
            c = np.bincount(fr[t["near"] == side] // win, minlength=n_win)[:n_win] / fps
            looks[si] += c[:, None] * np.asarray(t["look"], float)[None]
            secs[si] += c
    return looks, secs


def change_ends(tracks, n_frames, fps, seed_frame, window_s=20.0, switch=1.0, min_s=3.0, rounds=4):
    """When are the teams at the other end from where they were at seed_frame? -> per frame bool.

    The clothes seen at the near end in a window_s window are those of whichever team is there, and the same
    for the far end, so the windows fall into two groups: teams at home and teams swapped. Each end is only
    compared with itself (a player looks different near the camera than far from it). Starting from the
    windows around the seed and the window least like them, every window goes to the group whose clothes it
    fits (mean distance over the ends seen for min_s seconds), with `switch` (a whole window that fits neither
    group) for each change, and the groups' clothes are re-estimated. A clip without a change of ends leaves
    the second group empty: splitting one team's windows in two never pays for a change.
    """
    win = int(window_s * fps)
    n_win = max(1, -(-n_frames // win))
    if not any(t["look"] is not None for t in tracks):
        return np.zeros(n_frames, bool)
    looks, secs = _side_looks(tracks, n_win, win, fps)
    seen = [w for w in range(n_win) if (secs[:, w] >= min_s).all()]
    if not seen:
        return np.zeros(n_frames, bool)
    sw = min(seed_frame // win, n_win - 1)

    def fit(w, c):
        ds = [distance(looks[si, w], c[si]) for si in range(2) if secs[si, w] >= min_s and c[si].any()]
        return float(np.mean(ds)) if ds else 0.0

    centre = lambda ws: looks[:, ws].sum(1)
    groups = [centre([w for w in seen if abs(w - sw) <= 3] or seen)]
    groups.append(centre([max(seen, key=lambda w: fit(w, groups[0]))]))
    state = np.zeros(n_win, int)
    for _ in range(rounds):
        cost = np.array([[fit(w, groups[0]), fit(w, groups[1])] for w in range(n_win)])
        cost[sw, 1] = np.inf  # the seed defines home
        acc, back = cost[0].copy(), np.zeros((n_win, 2), int)
        for w in range(1, n_win):  # Viterbi over the two groups
            stay, flip = acc, acc[::-1] + switch
            back[w] = np.where(stay <= flip, [0, 1], [1, 0])
            acc = np.minimum(stay, flip) + cost[w]
        state[-1] = int(np.argmin(acc))
        for w in range(n_win - 1, 0, -1):
            state[w - 1] = back[w, state[w]]
        for g in (0, 1):
            ws = [w for w in seen if state[w] == g]
            if ws:
                groups[g] = centre(ws)
    return np.repeat(state, win)[:n_frames].astype(bool)


def _near(t):
    return bool(np.mean(t["near"]) >= 0.5)


def _profiles(owned, n):
    """Each player's clothes at each end ({True: near, False: far}): frame-weighted mean of their tracklets'
    descriptors there (None if never seen there), and their apparent height there (key ("height", side)). Far from the camera the same kit looks different (smaller,
    blurrier), so partners are compared at the same scale."""
    out = []
    for k in range(n):
        per = {}
        for side in (True, False):
            ls = [(t["look"], len(t["frames"])) for t in owned[k] if t["look"] is not None and _near(t) == side]
            per[side] = np.average([l for l, _ in ls], axis=0, weights=[w for _, w in ls]) if ls else None
            hs = [(t["height"], len(t["frames"])) for t in owned[k] if t.get("height") is not None and _near(t) == side]
            per[("height", side)] = float(np.average([h for h, _ in hs], weights=[w for _, w in hs])) if hs else None
        out.append(per)
    return out


def _look_at(profile, side, key=None):
    """A player's clothes at one end, or at the other end while they've not been seen at this one yet. With
    key="height", their apparent height at this end only: how tall people come out depends on the end with
    some camera angles."""
    if key is not None:
        return profile.get((key, side))
    return profile[side] if profile[side] is not None else profile[not side]


def split_partners(owned, team, looks, a, rounds=3, singles=True, motion_s=1.0):
    """Second pass over each team's two players: tracklets of the two that are on court at the same time are
    two different people, so chains of such overlaps (stretches with both partners in view) fix who is who up to
    one swap per stretch. Each stretch is swapped if the clothes and apparent heights pooled over all its
    tracklets fit the other player better; the players' profiles are re-estimated and this is repeated.
    Partners in the same kit, or out of view one at a time, are often decided wrongly tracklet by tracklet; a
    whole stretch has much more to go on. singles: also reconsider tracklets that overlap neither partner.
    owned: per player their tracklets (changed in place); team: pairs of player indices; looks: profiles."""
    def fit(t, k, outside):
        """Clothes and height misfit of t for player k, per frame, plus (once, weighted as motion_s seconds) how
        well t follows on from and leads into k's tracklets outside the stretch."""
        side = _near(t)
        c, look, h = 0.0, _look_at(looks[k], side), _look_at(looks[k], side, "height")
        if t["look"] is not None and look is not None:
            c += a.look_w * distance(t["look"], look)
        if t.get("height") is not None and h is not None:
            c += a.height_w * abs(t["height"] - h) / a.height_sd
        m = a.cost(t, None, outside, k)
        return c * len(t["frames"]) + (2 * a.unassigned if m is None else m) * motion_s * a.fps

    for _ in range(rounds):
        for k1, k2 in team:
            items = [(t, 0) for t in owned[k1]] + [(t, 1) for t in owned[k2]]
            items.sort(key=lambda it: it[0]["start"])
            parent = list(range(len(items)))

            def root(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i
            # sweep: join every tracklet with the other player's tracklets that overlap it in time
            for i, (t, g) in enumerate(items):
                for j in range(i + 1, len(items)):
                    u, h = items[j]
                    if u["start"] > t["end"]:
                        break
                    if h != g:
                        parent[root(j)] = root(i)
            groups = {}
            for i in range(len(items)):
                groups.setdefault(root(i), []).append(items[i])
            new1, new2 = [], []
            for members in groups.values():
                if len(members) == 1 and not singles:
                    swap = False
                else:
                    inside = {id(t) for t, _ in members}
                    outside = {k: [u for u in owned[k] if id(u) not in inside] for k in (k1, k2)}
                    keep = sum(fit(t, (k1, k2)[g], outside[(k1, k2)[g]]) for t, g in members)
                    flip = sum(fit(t, (k2, k1)[g], outside[(k2, k1)[g]]) for t, g in members)
                    swap = flip < keep
                for t, g in members:
                    (new1 if (g == 0) != swap else new2).append(t)
            owned[k1], owned[k2] = new1, new2
        new = _profiles(owned, len(owned))
        looks[:] = [{key: nl[key] if nl[key] is not None else ol[key] for key in nl} for nl, ol in zip(new, looks)]


def select_players(frames, fps, rounds=3, fixed_ends=True, cut_overlaps=True, use_height=True, link_w=2.0,
                   split=True, split_singles=True, split_motion_s=0.03,
                   **assigner_args):
    """Per-frame raw people [{"id", "bbox", "court", "look"?}] -> per-frame {player_id: {"bbox", "court"}}.

    Drops people outside the play area and ids that stand still beside the court, splits the rest into
    tracklets and gives each tracklet to one of the 4 players or to nobody (module docstring).
    fixed_ends: teams stay on their side of the net until they change ends (step 5); False = the older
    matcher, where a player's id could cross the net. cut_overlaps: see build_tracklets. use_height: match on
    apparent height as well as clothes (Assigner height_w). link_w: cost taken off for giving a piece of a
    ByteTrack id the same player as the piece before or after it (and added for giving it another).
    """
    tracks = build_tracklets(frames, fps, skip=static_bystanders(frames, fps), cut_overlaps=cut_overlaps,
                             heights=apparent_heights(frames) if use_height else None)
    out = [{} for _ in frames]
    if not tracks:
        return out
    seeds = _anchor(tracks)
    # ids in a fixed order at the seed moment: near side left to right, then far side left to right
    seeds.sort(key=lambda t: (Court.side(t["p0"]) != "near", t["p0"][0]))
    n = len(seeds)
    a = Assigner(fps, **assigner_args)
    looks = [{**{side: t["look"] if _near(t) == side else None for side in (True, False)},
              **{("height", side): t["height"] if _near(t) == side else None for side in (True, False)}} for t in seeds]
    order = sorted((t for t in tracks if not any(t is s for s in seeds)), key=lambda t: -len(t["frames"]))
    after = {id(t["same_as"]): t for t in tracks if t["same_as"] is not None}
    # a seed set without two players on each side can't say who belongs to which team
    teams = fixed_ends and n == 4 and sum(Court.side(t["p0"]) == "near" for t in seeds) == 2
    if teams:
        seed_frame = max(t["start"] for t in seeds)
        home = np.array([bool(t["near"][np.searchsorted(t["frames"], seed_frame)]) for t in seeds])
        swapped = change_ends(tracks, len(frames), fps, seed_frame)
        a.set_ends(home[None, :] != swapped[:, None])
    for _ in range(rounds):
        owned = [[s] for s in seeds]
        owner = {id(s): k for k, s in enumerate(seeds)}
        for t in order:
            costs = [a.cost(t, _look_at(looks[k], _near(t)), owned[k], k, _look_at(looks[k], _near(t), "height"))
                     for k in range(n)]
            # the pieces of one ByteTrack id cut at an overlap: same player unless the evidence says otherwise
            linked = [owner.get(id(u)) for u in (t["same_as"], after.get(id(t))) if u is not None]
            costs = [c if c is None else c - link_w * sum(1 if o == k else -1 for o in linked if o is not None)
                     for k, c in enumerate(costs)]
            ok = [(c, k) for k, c in enumerate(costs) if c is not None and c < a.unassigned]
            if ok:
                owned[min(ok)[1]].append(t)
                owner[id(t)] = min(ok)[1]
        new = _profiles(owned, n)
        looks = [{key: nl[key] if nl[key] is not None else ol[key] for key in nl} for nl, ol in zip(new, looks)]
    if split and teams:
        split_partners(owned, [tuple(np.flatnonzero(home == side)) for side in (True, False)], looks, a,
                       singles=split_singles, motion_s=split_motion_s)
    for k in range(n):
        for t in sorted(owned[k], key=lambda t: t["start"]):
            for f, p in zip(t["frames"], t["people"]):
                out[f][PLAYER_IDS[k]] = {"bbox": p["bbox"], "court": p["court"]}
    return out


def fill_hidden(players, ids=PLAYER_IDS, max_gap=None):
    """Per-frame players with everyone who is out of view put back where they were last (or next) seen.

    A player missing from a frame has walked off the edge of the picture (a near player behind the
    baseline or wide of the sideline) or is hidden behind someone (a far player behind a near one), and is
    still close to where they were seen: for each gap, the nearer in time of their last and next sighting
    (a player who walks into view just after the ball came from that side hit it). Filled entries are
    copies with "hidden": True; gaps longer than max_gap frames (None = any) are left empty.
    For hit attribution only: stats and drawing use the real boxes.
    """
    n = len(players)
    out = [dict(fp) for fp in players]
    for pid in ids:
        seen = [f for f in range(n) if pid in players[f]]
        if not seen:
            continue
        bounds = [(-1, seen[0])] + [(a, b) for a, b in zip(seen, seen[1:]) if b - a > 1] + [(seen[-1], n)]
        for a, b in bounds:
            if max_gap is not None and b - a - 1 > max_gap:
                continue
            for f in range(a + 1, b):
                src = a if a >= 0 and (b >= n or f - a <= b - f) else b
                out[f][pid] = {**players[src][pid], "hidden": True}
    return out
