"""Ball trajectory cleanup, hit/bounce detection, shot classification and rallies."""
import numpy as np

from .constants import COURT_L, KITCHEN, NET_Y


def clean_ball(track, max_jump=150, max_gap=8, offscreen=None):
    """Per-frame [x, y] or None -> Nx2 array (NaN = unknown), outliers removed, short gaps filled.
    offscreen[i] = True: the ball has left the picture there (balltrack.offscreen_frames), so the gap is
    never filled: a straight line across the frame would invent a path the ball didn't take."""
    a = np.array([p if p is not None else (np.nan, np.nan) for p in track], float).reshape(-1, 2)
    orig, known = a.copy(), np.flatnonzero(~np.isnan(a[:, 0]))
    # A detection far from both neighbouring detections is a false positive (head, shoe, sign).
    for k in range(1, len(known) - 1):
        p, i, n = known[k - 1:k + 2]
        if min(np.linalg.norm(orig[i] - orig[p]), np.linalg.norm(orig[i] - orig[n])) > max_jump:
            a[i] = np.nan

    idx, ok = np.arange(len(a)), ~np.isnan(a[:, 0])
    if ok.sum() < 2:
        return a
    out = np.column_stack([np.interp(idx, idx[ok], a[ok, d]) for d in range(2)])
    known = idx[ok]
    out[:known[0]] = np.nan
    out[known[-1] + 1:] = np.nan
    for s, e in zip(known[:-1], known[1:]):
        if e - s - 1 > max_gap:  # long gap = ball out of play, don't invent a path
            out[s + 1:e] = np.nan
    if offscreen is not None:
        out[np.asarray(offscreen, bool)[:len(out)]] = np.nan
    return out


def nearest_player(p, frame_players, reach=0.6, allow=None, centre_tiebreak=False):
    """Player whose bbox (grown by reach * bbox height) contains p, closest first.

    centre_tiebreak: when p is inside several boxes (distance 0 for all), the box whose centre line is
    closest wins, in units of box width, so a small (far) box the ball is inside beats a big (near) box
    that just covers it; otherwise the first player listed wins. allow(pl) -> False leaves a player out.
    """
    best, best_key = None, (np.inf, np.inf)
    for pid, pl in frame_players.items():
        if allow is not None and not allow(pl):
            continue
        x1, y1, x2, y2 = pl["bbox"]
        d = np.hypot(max(x1 - p[0], 0, p[0] - x2), max(y1 - p[1], 0, p[1] - y2))
        key = (d, abs(p[0] - (x1 + x2) / 2) / max(x2 - x1, 1e-9) if centre_tiebreak else 0.0)
        if d <= (y2 - y1) * reach and key < best_key:
            best, best_key = pid, key
    return best


def _far_hitter(ball, players, t, reach, far_side, first):
    """(far player nearest the ball, frame) at t, or failing that at the latest frame back to `first`
    where one is in reach: a far lob keeps rising after contact, so its local min comes after the ball
    has left the paddle, and the contact frame is where the hit belongs. (None, t) if nobody is."""
    for f in range(t, max(first, 0) - 1, -1):
        if np.isnan(ball[f, 0]):
            break
        pid = nearest_player(ball[f], players[f], reach, allow=far_side, centre_tiebreak=True)
        if pid is not None:
            return pid, f
    return None, t


def find_events(ball, players, window=5, min_prominence=8, min_gap=6, reach=0.6,
                near_side=lambda pl: pl["court"][1] > NET_Y, far_side=lambda pl: pl["court"][1] <= NET_Y,
                lookback=8):
    """Hits and bounces are local extrema of the ball's image y.

    A bounce (ball falls then rises) is always a local max of image y. A near-side hit is a
    local max next to a near-side player (ball comes down the screen, then goes back up).
    A far-side hit is a local min (ball goes up the screen, then comes back down), so only a
    far_side player can make one: a near player's box often covers the far court on screen,
    and a local min there is a far hit or the top of a lob, never a near player's shot. The top of a
    far player's lob comes a few frames after contact, so a local min looks back up to lookback frames.
    In POV mode every visible player counts as near and as far.
    """
    y, events = ball[:, 1], []
    for t in range(window, len(y) - window):
        w = y[t - window:t + window + 1]
        if np.isnan(w).any() or (events and t - events[-1]["frame"] < min_gap):
            continue
        left, right = w[:window], w[window + 1:]
        at = t
        if y[t] == w.max() and min(y[t] - left.min(), y[t] - right.min()) >= min_prominence:
            pid = nearest_player(ball[t], players[t], reach)
            etype = "hit" if pid is not None and near_side(players[t][pid]) else "bounce"
        elif y[t] == w.min() and min(left.max() - y[t], right.max() - y[t]) >= min_prominence:
            first = max(t - lookback, events[-1]["frame"] + 1 if events else 0)
            pid, at = _far_hitter(ball, players, t, reach, far_side, first)
            etype = "hit" if pid is not None else None  # else: apex of a lob, not an event
        else:
            continue
        if etype:
            events.append({"frame": at, "type": etype, "player": pid if etype == "hit" else None,
                           "px": [round(float(v), 1) for v in ball[at]]})
    return events


def filter_hits(events, fps, min_gap_s=0.35, side=None, same_side_s=1.0):
    """Drop hits that can't be separate shots.

    - re-detections of one contact: a hit < min_gap_s after the previous kept hit (time only,
      because player IDs churn and one person can show up under two IDs)
    - with side(event) -> "near"/"far": a hit by the same side < same_side_s after the previous
      kept hit, since the ball has to cross the net and come back first
    """
    out, last = [], None
    for e in events:
        if e["type"] == "hit":
            gap = e["frame"] - last["frame"] if last else np.inf
            if gap < min_gap_s * fps or (side and gap < same_side_s * fps and side(e) == side(last)):
                continue
            last = e
        out.append(e)
    return out


def _box_distance(p, bbox):
    """Pixels from point p to bbox (0 inside)."""
    x1, y1, x2, y2 = bbox
    return float(np.hypot(max(x1 - p[0], 0, p[0] - x2), max(y1 - p[1], 0, p[1] - y2)))


def _flight(ball, detected, t, go, max_step, max_gap=4):
    """Detected ball positions after t that form one continuous path from ball[t] (each step < max_step
    px per frame, no more than max_gap frames unseen), stopping at the first jump or long gap: a
    detection on another court is not this ball. Interpolated positions are skipped: a straight line
    drawn across a gap would look like a flight."""
    path, last, last_f = [], ball[t], t
    for k in range(t + 1, min(t + go + 1, len(ball))):
        if not detected[k]:
            continue
        if k - last_f > max_gap or np.linalg.norm(ball[k] - last) > max_step * (k - last_f):
            break
        path.append(ball[k])
        last, last_f = ball[k], k
    return np.array(path).reshape(-1, 2)


def _ball_out_of_play(ball, known, players, pid, t, hold, near_h, away_h, min_near, max_away, head, still_m,
                      offscreen=None):
    """find_serves' hold check: over frames [t - hold, t) was the ball held by pid, or dead while pid stood still?
    A ball that was off-screen in that window (a lob over the top) was in play, not dead."""
    if offscreen is not None and any(offscreen[t - hold:t]):
        return False
    box = lambda f: players[f][pid]["bbox"] if pid in players[f] else players[t][pid]["bbox"]
    seen = [f for f in range(t - hold, t) if known[f]]
    if not seen:
        here = [players[f][pid]["court"] for f in range(t - hold, t + 1) if pid in players[f]]
        return len(here) >= hold // 2 and np.ptp(np.asarray(here), axis=0).max() < still_m
    at_body = away = 0
    for f in seen:
        x1, y1, x2, y2 = box(f)
        d = _box_distance(ball[f], (x1, y1, x2, y2)) / (y2 - y1)
        at_body += d <= near_h and (ball[f, 1] - y1) / (y2 - y1) >= head
        away += d > away_h
    return at_body >= min_near and away <= max_away * len(seen)


def find_serves(ball, players, court, fps, width, hold_s=1.0, go_s=0.6, near_h=0.5, away_h=1.5,
                min_near=3, max_away=0.2, baseline_m=3.0, min_contact_y=0.15, still_m=0.5, step_frac=0.1,
                same_contact_s=0.5, min_gap_s=4.0, detected=None, offscreen=None):
    """Fixed mode: serves straight from the ball track -> [{"frame", "player", "px"}].

    A serve is the ball held (or bounced) by a player near their own baseline, then hit so it
    crosses the net quickly. Contact is a frame t where:
    - the ball is within near_h box-heights of a player standing within baseline_m of their baseline,
      and not in the top min_contact_y of their box (serves are underhand: a ball at head height is
      a volley, an overhead, or the player's cap detected as the ball)
    - over the hold_s before t the ball was out of play, one of:
      held: seen at the server's body (not their head: caps get detected as balls) >= min_near times,
      and away from them (> away_h box-heights) in at most max_away of its sightings. Unseen frames
      are neutral: a held ball is often hidden in the hand
      dead: not seen at all while the server stood still (moved < still_m). On the far side the ball
      in the server's hand is too small to detect, so its first sighting is the serve itself
    - the ball is detected at t, and within go_s after t (detected[k]: a real detection, not filled in
      or replaced by clean_ball) flies on
      one continuous path (steps < step_frac * width px per frame) that moves >= 3 m toward the net
      and reaches the other side
    Frames within same_contact_s of each other are one contact (the one where the ball speeds up
    most is kept), and a serve puts the ball in play, so nothing within min_gap_s after it is a serve.
    """
    from .constants import COURT_L

    hold, go, max_step = max(2, int(hold_s * fps)), max(2, int(go_s * fps)), step_frac * width
    known = ~np.isnan(ball[:, 0])
    detected = known if detected is None else np.asarray(detected) & known
    serves = []
    for t in range(hold, len(ball) - 1):
        if not detected[t]:
            continue
        pid = nearest_player(ball[t], players[t], near_h)
        if pid is None:
            continue
        pl = players[t][pid]
        x1, y1, x2, y2 = pl["bbox"]
        box_h = y2 - y1
        side = court.side(pl["court"])
        to_baseline = COURT_L - pl["court"][1] if side == "near" else pl["court"][1]
        if to_baseline > baseline_m or (ball[t, 1] - y1) / box_h < min_contact_y:
            continue
        if not _ball_out_of_play(ball, known, players, pid, t, hold, near_h, away_h, min_near, max_away,
                                 min_contact_y, still_m, offscreen):
            continue
        path = _flight(ball, detected, t, go, max_step)
        if len(path) < 2:
            continue
        on_court = court.to_court(path)
        start = court.to_court([ball[t]])[0]
        toward = (start[1] - on_court[:, 1]) if side == "near" else (on_court[:, 1] - start[1])
        if toward.max() >= 3.0 and any(court.side(p) != side for p in on_court):
            serves.append({"frame": t, "player": pid, "px": [round(float(v), 1) for v in ball[t]]})
    # several frames around one contact qualify; contact is where the ball speeds up most
    groups = []
    for s in serves:
        if groups and s["frame"] - groups[-1][-1]["frame"] <= same_contact_s * fps:
            groups[-1].append(s)
        else:
            groups.append([s])

    def speed_up(f, k=3):
        a, b = ball[max(f - k, 0)], ball[min(f + k, len(ball) - 1)]
        after = np.linalg.norm(b - ball[f]) if not np.isnan(b[0]) else 0.0
        before = np.linalg.norm(ball[f] - a) if not np.isnan(a[0]) else 0.0
        return after - before

    out = []
    for s in (max(g, key=lambda s: speed_up(s["frame"])) for g in groups):
        if not out or s["frame"] - out[-1]["frame"] >= min_gap_s * fps:
            out.append(s)
    return out


def real_detections(raw, ball):
    """Per frame: is the cleaned ball position an actual detection (not gap-filled, not a removed outlier)?"""
    return np.array([b is not None and not np.isnan(c[0]) and np.allclose(c, b) for b, c in zip(raw, ball)])


def merge_serves(events, serves, players, court, fps, handling_s=3.0, after_s=0.3, hold_s=0.7):
    """Put detected serves into the event list as hits flagged "serve".

    Hits by the server's team in the handling_s before a serve are ball handling (bouncing it,
    passing it to the server), and any hit within after_s after it is the same contact detected late:
    the served ball can't have reached the other side yet, so a "hit" there (the ball passing over a
    receiver on screen) is the serve too. Any other event in the hold_s before contact (a bounce of the
    held ball) goes too.
    """
    side = lambda f, pid: court.side(players[f][pid]["court"])
    drop = set()
    for s in serves:
        f, srv_side = s["frame"], side(s["frame"], s["player"])
        for i, e in enumerate(events):
            if e["type"] == "hit" and f < e["frame"] <= f + after_s * fps:
                drop.add(i)
            elif f - handling_s * fps <= e["frame"] <= f and e["type"] == "hit" \
                    and side(e["frame"], e["player"]) == srv_side:
                drop.add(i)
            elif f - hold_s * fps <= e["frame"] < f:
                drop.add(i)
    kept = [e for i, e in enumerate(events) if i not in drop]
    added = [{"frame": s["frame"], "type": "hit", "player": s["player"], "px": s["px"], "serve": True}
             for s in serves]
    return sorted(kept + added, key=lambda e: (e["frame"], e["type"] != "hit"))


def shot_type(n_in_rally, hitter, landing, speed, airtime, volley, drive_speed=10.0):
    """Rule-based pickleball shot label. hitter/landing are court metres, speed m/s."""
    if n_in_rally == 1:
        return "serve"
    if n_in_rally == 2:
        return "return"
    if speed is not None and speed >= drive_speed:
        return "drive"
    if landing is not None and abs(landing[1] - NET_Y) <= KITCHEN:
        return "dink" if abs(hitter[1] - NET_Y) <= KITCHEN + 1.0 else "drop"
    if landing is not None and abs(landing[1] - NET_Y) > COURT_L / 2 - 2.0 and airtime > 1.0:
        return "lob"
    return "volley" if volley else "groundstroke"


def _type_position(n, e, rally, serves_detected, side):
    """Rally position shot_type should see: with detected serves, 1 = serve, and 2 = return only for
    the next hit after a detected serve, from across the net; otherwise 3+ (rule-based type)."""
    if not serves_detected:
        return n
    if e.get("serve"):
        return 1
    return 2 if n == 2 and rally["served"] and side != rally["server_side"] else max(n, 3)


def classify_shots(events, players, court, fps, rally_gap_s=2.5, drive_speed=10.0, serves_detected=False,
                   offscreen=None):
    """-> (shots, rallies). Every hit becomes a shot with type, speed, landing and faults.

    offscreen[i] = True: the ball was out of the picture (a high lob); that time doesn't count toward
    the rally_gap_s pause that ends a rally.

    serves_detected=False: the first hit of a rally (after a rally_gap_s pause) is the serve.
    serves_detected=True (events went through merge_serves): only hits flagged "serve" are serves,
    each one starts a new rally, and "return" is the hit right after a serve. A rally that starts
    without a detected serve (clip starts mid-point, serve missed) has no serve or return.
    """
    hits = [i for i, e in enumerate(events) if e["type"] == "hit"]
    off = np.zeros(0, bool) if offscreen is None else np.asarray(offscreen, bool)
    pause = lambda a, b: (b - a - int(off[a:b].sum())) / fps  # seconds between frames a and b, off-screen excluded
    shots, rallies, prev = [], [], None
    for k, i in enumerate(hits):
        e, f = events[i], events[i]["frame"]
        new_rally = prev is None or pause(events[prev]["frame"], f) > rally_gap_s or bool(e.get("serve"))
        if new_rally:
            rallies.append({"start": f, "end": f, "shots": 0, "served": bool(e.get("serve")),
                            "server_side": court.side(players[f][e["player"]]["court"])})
        rallies[-1]["shots"] += 1
        rallies[-1]["end"] = f
        volley = not new_rally and not any(b["type"] == "bounce" for b in events[prev + 1:i])

        nxt = hits[k + 1] if k + 1 < len(hits) else len(events)
        landing_ev = next((b for b in events[i + 1:nxt] if b["type"] == "bounce"), None)
        # destination: the bounce, else the next hitter in the same rally (a re-hit by the same
        # player, or a hit in the next point, says nothing about this shot's speed)
        nxt_ev = events[nxt] if nxt < len(events) else None
        same_rally = nxt_ev and nxt_ev["player"] != e["player"] and pause(f, nxt_ev["frame"]) <= rally_gap_s \
            and not nxt_ev.get("serve")
        target_ev = landing_ev or (nxt_ev if same_rally else None)
        hitter = np.array(players[f][e["player"]]["court"])
        landing = speed = None
        airtime = 0.0
        if target_ev:
            if landing_ev:
                landing = court.to_court([landing_ev["px"]])[0]
                tgt = landing
            else:  # volleyed: use the next hitter's position as the ball's destination
                tgt = np.array(players[target_ev["frame"]][target_ev["player"]]["court"])
            airtime = (target_ev["frame"] - f) / fps
            speed = float(np.linalg.norm(tgt - hitter) / airtime)

        n = rallies[-1]["shots"]
        faults = []
        if volley and court.in_kitchen(hitter):
            faults.append("kitchen_volley")
        if landing is not None and not court.inside(landing):
            faults.append("out")
        if landing is not None and court.side(landing) == court.side(hitter):
            faults.append("net")  # bounced on the hitter's own side
        shots.append({
            "frame": f, "time_s": round(f / fps, 2), "player": e["player"], "rally": len(rallies),
            "n_in_rally": n, "type": shot_type(_type_position(n, e, rallies[-1], serves_detected, court.side(hitter)),
                                               hitter, landing, speed, airtime, volley, drive_speed),
            "volley": volley, "speed_kmh": None if speed is None else round(speed * 3.6, 1),
            "hitter": hitter.round(2).tolist(),
            "landing": None if landing is None else landing.round(2).tolist(), "faults": faults,
        })
        prev = i
    for r in rallies:
        r["duration_s"] = round((r["end"] - r["start"]) / fps, 2)
        r.pop("server_side")
    return shots, rallies


def compensate(track, cam_shift):
    """POV: undo the cumulative camera pan so ball positions share one reference view.

    Only valid locally (pan, not rotation/fisheye), which is all hit/bounce detection needs.
    """
    off = np.cumsum(np.asarray(cam_shift, float).reshape(-1, 2), axis=0)
    return [None if p is None else [p[0] - o[0], p[1] - o[1]] for p, o in zip(track, off)], off


def find_swings(cam_shift, ball, fps, width, paddles=None, height=None, rel_thresh=0.02,
                window_s=0.3, rally_s=1.0, own_paddle=0.2):
    """POV: your own swing jolts the camera -> spikes in camera motion while the ball is in play.

    Head turns jolt the camera too, so when paddle boxes are available a jolt also needs your
    own paddle in view (a paddle box taller than own_paddle * frame height) within window_s.
    """
    # shortcut: thresholds come from one hand-checked 30 s clip (chest mount); re-check per mount.
    m = np.linalg.norm(np.asarray(cam_shift, float).reshape(-1, 2), axis=1)
    thr = max(4 * np.median(m), rel_thresh * width)
    w, r = max(1, int(window_s * fps)), int(rally_s * fps)
    seen = ~np.isnan(ball[:, 0])
    own = None
    if paddles and any(paddles):
        own = np.array([p is not None and p[3] - p[1] > own_paddle * height for p in paddles])
    swings = []
    for t in range(len(m)):
        if m[t] < thr or m[t] < m[max(0, t - w):t + w + 1].max():
            continue
        if not seen[max(0, t - r):t + r + 1].any() or (swings and t - swings[-1] < w):
            continue
        if own is not None and not own[max(0, t - w):t + w + 1].any():
            continue
        swings.append(t)
    return swings


def pov_shots(events, swings, fps, rally_gap_s=2.5, same_side_s=0.5):
    """POV: no court, so a shot is who hit when (you = "me"), grouped into rallies.

    Hit events carry a "side" ("near" = you/partner, "far" = across the net). The ball must
    cross the net between hits, so:
    - a hit by the same side as the previous one within same_side_s is dropped;
    - a camera jolt is your swing only if the latest hit before it came from the far side
      0.15-2 s earlier (head turns and walking jolt the camera too; your serves are missed).
    Not done: dropping far-side-only "rallies" (ball passed between points) also dropped real
    points whose near-side hits were missed - worse recall for little precision on test footage.
    """
    hits = sorted((e["frame"], e["player"], e.get("side", "far")) for e in events if e["type"] == "hit")

    def is_swing(f):
        prior = [h for h in hits if h[0] < f]
        return bool(prior) and prior[-1][2] == "far" and 0.15 * fps < f - prior[-1][0] <= 2 * fps

    kept = []
    for h in sorted(hits + [(f, "me", "near") for f in swings if is_swing(f)]):
        if kept and h[2] == kept[-1][2] and h[0] - kept[-1][0] < same_side_s * fps:
            continue
        kept.append(h)
    groups = []
    for h in kept:
        if not groups or (h[0] - groups[-1][-1][0]) / fps > rally_gap_s:
            groups.append([])
        groups[-1].append(h)

    bounces = [e["frame"] for e in events if e["type"] == "bounce"]
    shots, rallies = [], []
    for g in groups:
        rallies.append({"start": g[0][0], "end": g[-1][0], "shots": len(g),
                        "duration_s": round((g[-1][0] - g[0][0]) / fps, 2)})
        for n, (f, pid, side) in enumerate(g, 1):
            volley = n > 1 and not any(g[n - 2][0] < b < f for b in bounces)
            shots.append({"frame": f, "time_s": round(f / fps, 2), "player": pid, "side": side,
                          "rally": len(rallies), "n_in_rally": n,
                          "type": shot_type(n, None, None, None, 0.0, volley),
                          "volley": volley, "speed_kmh": None, "faults": []})
    return shots, rallies
