"""Per-player movement, shot and stroke-form metrics."""
from collections import Counter

import numpy as np

from .constants import COURT_L, COURT_W, NET_Y
from .court import Court, kitchen_line_zone

# COCO keypoint indices
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 5, 6, 7, 8, 9, 10, 11, 12


def player_stats(players, fps, sample_s=0.2, max_speed=8.0, rallies=None, court=None):
    """Distance, speed, kitchen time and side per player id. Tracks shorter than 1 s are noise.

    kitchen_pct: time up at the kitchen line (within 3 ft behind it, not inside the kitchen),
    measured from court's painted kitchen lines (Court.at_kitchen_line), or the nominal 7 ft lines without a court.
    With rallies, that time during rallies divided by the total time of all rallies (frames start..end), so it
    ignores time between points; a frame where the player isn't tracked doesn't count. Without rallies, the
    share of the player's tracked frames.
    """
    in_rally = None
    if rallies is not None:
        in_rally = np.zeros(len(players), bool)
        for r in rallies:
            in_rally[r["start"]:r["end"] + 1] = True
    tracks = {}
    for f, fp in enumerate(players):
        for pid, p in fp.items():
            tracks.setdefault(pid, []).append((f, *p["court"]))
    step, stats = max(1, round(fps * sample_s)), {}
    for pid, tr in tracks.items():
        if len(tr) < fps:
            continue
        a = np.array(tr)
        s = a[::step]  # subsample so box jitter doesn't inflate distance
        dt = np.diff(s[:, 0]) / fps
        d = np.linalg.norm(np.diff(s[:, 1:], axis=0), axis=1)
        v = d / np.maximum(dt, 1e-9)
        ok = v <= max_speed  # faster than a sprinting human = ID swap / bad box
        dist = float(d[ok].sum())
        sides = Counter(Court.side(p) for p in a[:, 1:])
        stats[pid] = {
            "seconds": round(len(tr) / fps, 1),
            "side": sides.most_common(1)[0][0],
            "distance_m": round(dist, 1),
            "avg_speed_ms": round(dist / max(float(dt[ok].sum()), 1e-9), 2),
            "max_speed_ms": round(float(v[ok].max()), 2) if ok.any() else 0.0,
            "kitchen_pct": round(100 * kitchen_share(a, in_rally, court), 1),
        }
    return stats


def kitchen_share(track, in_rally=None, court=None):
    """track: rows (frame, x, y). Fraction of rally frames spent at the kitchen line (in_rally: per-frame bool
    mask), or of the track's own frames when there is no mask."""
    at_line = court.at_kitchen_line if court is not None else kitchen_line_zone
    kit = np.array([at_line(p) for p in track[:, 1:]], bool)
    if in_rally is None:
        return float(kit.mean())
    total = int(in_rally.sum())
    return float(in_rally[track[kit, 0].astype(int)].sum() / total) if total else 0.0


def kitchen_arrival(players, rallies, shots, fps, court=None, min_shots_receiving=3, min_shots_serving=5,
                    min_s=0.2):
    """How often players, and both members of a team together, get up to the kitchen in a rally.

    A player has got there when they spend min_s seconds (not necessarily in one go) at the kitchen line or in
    the kitchen between the serve and the rally's last hit; a team, when both of them are there at the same
    time for min_s seconds. Split by whether their team is serving or receiving. A rally only counts when it
    lasts long enough to get up: the receiving team runs in behind the return, so the third shot must be hit
    (min_shots_receiving); the serving team waits for its third shot to land and moves in after it, so the
    rally needs min_shots_serving shots. players: per-frame players as seen (not filled in while hidden: a
    player out of the picture isn't at the kitchen, which is always in view).

    -> ({pid: {"kitchen_serving_pct", "kitchen_serving_of", "kitchen_receiving_pct", "kitchen_receiving_of"}},
        [{"players": [pid, pid], "both_kitchen_serving_pct", "both_kitchen_serving_of", ...receiving...}]).
    _of: how many rallies the percentage is of; the pct is None when that's 0.
    """
    at_line = (lambda p: court.at_kitchen_line(p, inside=True)) if court is not None else \
        (lambda p: kitchen_line_zone(p, inside=True))
    serves = {s["rally"]: s for s in shots if s.get("role") == "serve" or s.get("type") == "serve"}
    need = int(round(min_s * fps))
    per, teams = {}, {}
    for i, r in enumerate(rallies, 1):
        serve = serves.get(i)
        if serve is None or serve.get("hitter") is None:
            continue
        lo, hi = r["start"], r["end"] + 1
        serving_side = Court.side(serve["hitter"])
        # each player's side this rally: where they were seen most
        sides = {}
        for fp in players[lo:hi]:
            for pid, p in fp.items():
                sides.setdefault(pid, Counter())[Court.side(p["court"])] += 1
        sides = {pid: c.most_common(1)[0][0] for pid, c in sides.items()}
        up = {pid: np.array([pid in fp and at_line(fp[pid]["court"]) for fp in players[lo:hi]], bool)
              for pid in sides}
        for side in ("near", "far"):
            role = "serving" if side == serving_side else "receiving"
            if r["shots"] < (min_shots_serving if role == "serving" else min_shots_receiving):
                continue
            team = sorted(pid for pid, sd in sides.items() if sd == side)
            for pid in team:
                c = per.setdefault(pid, Counter())
                c[role] += 1
                c[role + "_up"] += int(up[pid].sum() >= need)
            if len(team) == 2:
                c = teams.setdefault(tuple(team), Counter())
                c[role] += 1
                c[role + "_up"] += int((up[team[0]] & up[team[1]]).sum() >= need)

    def pct(c, role):
        return round(100 * c[role + "_up"] / c[role], 1) if c[role] else None

    players_out = {pid: {f"kitchen_{role}_pct": pct(c, role) for role in ("serving", "receiving")}
                   | {f"kitchen_{role}_of": c[role] for role in ("serving", "receiving")} for pid, c in per.items()}
    teams_out = [{"players": list(t)} | {f"both_kitchen_{role}_pct": pct(c, role) for role in ("serving", "receiving")}
                 | {f"both_kitchen_{role}_of": c[role] for role in ("serving", "receiving")}
                 for t, c in sorted(teams.items())]
    return players_out, teams_out


DEPTHS = ("short", "medium", "deep")


def depth_class(d, deep_m=1.52, short_m=3.05):
    """Thirds of the service area (baseline to kitchen line, 4.57 m): deep is the back 5 ft, short the front."""
    return "deep" if d <= deep_m else "medium" if d <= short_m else "short"


def point_winners(rallies, shots, teams, fps, max_gap_s=60.0):
    """Which team won each rally, from who serves the next one: only the serving team scores, so the same server
    again (from the other half: servers switch after a point) means the serving team won; their partner or the
    other team serving next means they lost. Unknown (None) for the last rally, when the next serve comes more
    than max_gap_s later (between games, or rallies missed in between), and when the next serve doesn't fit
    (from the same half, or by someone with no team): a misread server or a missed rally.
    teams: [{"players": [pid, pid]}, ...] -> per rally (1-based numbering, as in shots): team index or None."""
    team_of = {str(p): k for k, t in enumerate(teams) for p in t["players"]}
    serves = sorted((s for s in shots if s.get("role", s.get("type")) == "serve" and s.get("hitter") is not None),
                    key=lambda s: s["frame"])
    half = lambda s: s["hitter"][0] < COURT_W / 2  # noqa: E731
    out = {r: None for r in range(1, len(rallies) + 1)}
    for a, b in zip(serves, serves[1:]):
        ta, tb = team_of.get(str(a["player"])), team_of.get(str(b["player"]))
        if ta is None or tb is None or (b["frame"] - a["frame"]) / fps > max_gap_s:
            continue
        if ta != tb:
            out[a["rally"]] = tb  # side out: the receiving team won
        elif half(a) != half(b):  # same team serves again, from the other half
            out[a["rally"]] = ta if str(a["player"]) == str(b["player"]) else 1 - ta
    return out


def serve_return_depth(rallies, shots, teams, fps, receiver_offset_m=1.8, max_gap_s=60.0):
    """How deep each serve and return went, and whether the hitter's team won the point.

    Both must bounce, so the depth (metres in from the baseline it was hit towards) comes from the ball's
    bounce when one was found in that half of the court, else from where the next shot was hit: players take
    the ball after it has bounced and carried on towards them, standing a median 1.8 m deeper than the bounce
    (receiver_offset_m, measured on four games where both were found). Bounces are found for about half of
    serves and returns. Adds to each serve / return shot "depth_m", "depth" (short / medium / deep:
    depth_class), "depth_from" (bounce / receiver) and "won" (None: unknown); to each rally "serving_team"
    and "winner" (team index, None: unknown, see point_winners).
    -> {pid: {"serve_n", "serve_depth_m" (median), "serve_<depth>_n", "serve_<depth>_won_pct", "serve_won_pct",
              same for return_}}: _won_pct only over points whose winner is known, None when there are none."""
    team_of = {str(p): k for k, t in enumerate(teams) for p in t["players"]}
    winners = point_winners(rallies, shots, teams, fps, max_gap_s)
    by_rally = {}
    for s in shots:
        by_rally.setdefault(s.get("rally"), []).append(s)
    for i, r in enumerate(rallies, 1):
        r["winner"] = winners.get(i)
        sh = sorted(by_rally.get(i, []), key=lambda s: s["frame"])
        r["serving_team"] = team_of.get(str(sh[0]["player"])) if sh and sh[0].get("role") == "serve" else None
        for s, nxt in zip(sh[:2], sh[1:3] + [None]):
            if s.get("role") not in ("serve", "return") or s.get("hitter") is None:
                continue
            near = Court.side(s["hitter"]) == "near"  # hit from the near half: towards the far baseline (y = 0)
            depth = (lambda y: y) if near else (lambda y: COURT_L - y)
            d, src = None, None
            land = s.get("landing")
            if land is not None and 0 <= land[0] <= COURT_W and -0.5 <= depth(land[1]) <= NET_Y:
                d, src = depth(land[1]), "bounce"
            elif nxt is not None and nxt.get("hitter") is not None and Court.side(nxt["hitter"]) != Court.side(s["hitter"]):
                d, src = depth(nxt["hitter"][1]) + receiver_offset_m, "receiver"
            if d is None:
                continue
            d = float(np.clip(d, 0.0, NET_Y))
            k = team_of.get(str(s["player"]))
            s.update(depth_m=round(d, 2), depth=depth_class(d), depth_from=src,
                     won=None if r["winner"] is None or k is None else r["winner"] == k)

    per = {}
    for s in shots:
        if "depth_m" not in s:
            continue
        role = s["role"]
        c = per.setdefault(str(s["player"]), {}).setdefault(role, [])
        c.append(s)

    def won_pct(xs):
        known = [x["won"] for x in xs if x["won"] is not None]
        return round(100 * sum(known) / len(known), 1) if known else None

    out = {}
    for pid, roles in per.items():
        o = out.setdefault(pid, {})
        for role, xs in roles.items():
            o[f"{role}_n"] = len(xs)
            o[f"{role}_depth_m"] = round(float(np.median([x["depth_m"] for x in xs])), 2)
            o[f"{role}_won_pct"] = won_pct(xs)
            for k in DEPTHS:
                ys = [x for x in xs if x["depth"] == k]
                o[f"{role}_{k}_n"] = len(ys)
                o[f"{role}_{k}_won_pct"] = won_pct(ys)
    return out


def shot_stats(shots):
    """Shot counts by type, volleys, faults, and shot speed per player."""
    per = {}
    for s in shots:
        p = per.setdefault(s["player"], {"shots": 0, "volleys": 0, "faults": 0, "_speeds": []})
        p["shots"] += 1
        p["volleys"] += s["volley"]
        p["faults"] += len(s["faults"])
        p[f"n_{s['type']}"] = p.get(f"n_{s['type']}", 0) + 1
        if s["speed_kmh"] is not None:
            p["_speeds"].append(s["speed_kmh"])
    for p in per.values():
        sp = p.pop("_speeds")
        p["avg_shot_kmh"] = round(float(np.mean(sp)), 1) if sp else None
        p["max_shot_kmh"] = max(sp) if sp else None
    return per


def _angle(a, b, c):
    """Angle at b in degrees."""
    u, w = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9), -1, 1))))


def stroke_form(kp):
    """Contact-frame form from 17x2 COCO keypoints, or None if the arm/hips weren't seen."""
    kp = np.asarray(kp, float)
    hip = kp[[L_HIP, R_HIP]].mean(0)
    # The paddle arm is the one whose wrist is furthest from the body at contact.
    wrist = max((L_WRIST, R_WRIST), key=lambda w: np.linalg.norm(kp[w] - hip))
    shoulder, elbow = (L_SHOULDER, L_ELBOW) if wrist == L_WRIST else (R_SHOULDER, R_ELBOW)
    if not kp[[shoulder, elbow, wrist, L_HIP, R_HIP]].all(axis=1).all():  # (0,0) = keypoint missing
        return None
    shoulder_y = kp[[L_SHOULDER, R_SHOULDER]][:, 1].min()
    torso = max(hip[1] - kp[[L_SHOULDER, R_SHOULDER]][:, 1].mean(), 1e-6)
    return {
        "arm": "left" if wrist == L_WRIST else "right",
        "elbow_angle": round(_angle(kp[shoulder], kp[elbow], kp[wrist]), 1),
        "contact_below_waist": bool(kp[wrist, 1] > hip[1]),  # image y grows downward
        "overhead": bool(kp[wrist, 1] < shoulder_y),
        "contact_height": round(float((hip[1] - kp[wrist, 1]) / torso), 2),  # 0 = waist, 1 = shoulders
    }
