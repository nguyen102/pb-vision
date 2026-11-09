"""Pick the 4 players out of everyone the person tracker sees, and give each one a stable id.

The tracker records every person near the court with ByteTrack ids, which churn (a player who is
hidden or walks out of frame comes back with a new id) and include refs, line judges and spectators.
select_players turns that into at most 2 players per side under 4 fixed ids "1".."4":
- while a player is in view, their id follows them by position (and ByteTrack id);
- when they come back after being lost, their clothing colours pick which id they get back,
  so partners who both left the frame don't come back swapped, and a stranger walking on court
  doesn't inherit a missing player's id.
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


class SlotMatcher:
    """Costs of giving a person to a player id (slot). Lower is better; None = not allowed.

    Clothing limits come from gold-match-short (appearance.describe, 48-bin HSV): a player's own
    frames sit at a median 0.38 (p90 0.46, p99 ~0.6) from their running profile, other players at
    a median 0.48. So max_look=0.5 lets returning players back while turning away clear strangers,
    and the joint assignment does the finer work of telling two lost players apart.

    - active slot (seen < reacquire_s ago): only people its player could have reached and whose
      clothes aren't clearly different (<= max_look_active), ranked by distance, clothing
      difference (look_w per unit of distance) and a same-ByteTrack-id bonus
    - lost slot: a returning player, if their clothes match (distance <= max_look; after relax_s
      lost, any clothes, so one bad early match can't lock a player out for good)
    - empty slot (never used): anyone, but dearer than a good clothes match to a lost slot
    """

    def __init__(self, fps, max_speed=7.0, reacquire_s=1.0, look_w=1.0, max_look=0.5, max_look_active=0.7,
                 return_w=4.0, relax_s=8.0, unassigned=10.0):
        self.fps, self.max_speed, self.reacquire_s = fps, max_speed, reacquire_s
        self.max_look_active = max_look_active
        self.look_w, self.max_look, self.return_w, self.relax_s = look_w, max_look, return_w, relax_s
        self.unassigned = unassigned
        self.empty_cost = 1.0 + return_w * max_look  # a lost player matching below max_look wins over a new id

    def cost(self, slot, person, frame):
        """Cost of person taking slot at frame, or None."""
        outside = Court.outside_distance(person["court"])
        if slot is None:
            return self.empty_cost + outside
        look = distance(slot["look"], person.get("look"))
        lost_s = (frame - slot["frame"]) / self.fps
        if lost_s <= self.reacquire_s:
            d = float(np.hypot(*np.subtract(person["court"], slot["court"])))
            if d > 0.5 + self.max_speed * lost_s or look > self.max_look_active:
                return None
            return d + self.look_w * look - (0.3 if person["id"] == slot["track"] else 0.0)
        if look > self.max_look and lost_s < self.relax_s:
            return None
        return 1.0 + outside + self.return_w * look

    def assign(self, slots, people, frame, per_side=2):
        """Best joint assignment of slots to people, at most per_side people per side of the net.
        -> [person index or None] per slot."""
        costs = [[self.cost(s, p, frame) for p in people] for s in slots]
        sides = [Court.side(p["court"]) for p in people]
        best = {"cost": np.inf, "pick": [None] * len(slots)}

        def search(i, used, count, total, pick):
            if total >= best["cost"]:
                return
            if i == len(slots):
                best["cost"], best["pick"] = total, pick
                return
            for c in range(len(people)):
                if c not in used and costs[i][c] is not None and count[sides[c]] < per_side:
                    search(i + 1, used | {c}, {**count, sides[c]: count[sides[c]] + 1},
                           total + costs[i][c], pick + [c])
            search(i + 1, used, count, total + self.unassigned, pick + [None])

        search(0, frozenset(), {"near": 0, "far": 0}, 0.0, [])
        return best["pick"]


def select_players(frames, fps, max_candidates=6, look_rate=0.1, **matcher_args):
    """Per-frame raw people [{"id", "bbox", "court", "look"?}] -> per-frame {player_id: {"bbox", "court"}}.

    1. drop people outside the play area (tight sideline margin, loose baseline margin)
    2. drop ids that stand still beside the court for their whole track
    3. keep at most 2 people per side of the net
    4. map them to 4 fixed ids by position while in view and by clothing colours when they come back
    Each id's clothing profile is a running average (look_rate) over frames where nobody overlaps them.
    """
    static = static_bystanders(frames, fps)
    matcher = SlotMatcher(fps, **matcher_args)
    slots = [None] * len(PLAYER_IDS)
    out = []
    for f, fp in enumerate(frames):
        mixed = dict(zip(map(id, fp), overlaps_someone(fp)))
        cands = [p for p in fp if p["id"] not in static and Court.in_play_area(p["court"])]
        cands = sorted(cands, key=lambda p: Court.outside_distance(p["court"]))[:max_candidates]
        frame_out = {}
        for k, c in enumerate(matcher.assign(slots, cands, f)):
            if c is None:
                continue
            p = cands[c]
            look = slots[k]["look"] if slots[k] else None
            new = p.get("look")
            if new is not None and not mixed[id(p)]:
                new = np.asarray(new, float)
                look = new if look is None else (1 - look_rate) * look + look_rate * new
            slots[k] = {"court": p["court"], "track": p["id"], "frame": f, "look": look}
            frame_out[PLAYER_IDS[k]] = {"bbox": p["bbox"], "court": p["court"]}
        out.append(frame_out)
    return out
