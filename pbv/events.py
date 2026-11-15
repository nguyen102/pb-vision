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


def _far_hitter(ball, players, t, reach, far_side, first, hand_h=0.5, dist_w=0.5):
    """(hitter, frame) for a local min of image y at t: (None, t) if no far player is in reach back to `first`.

    A far shot keeps going up the screen after contact, so the local min comes up to ~half a second later,
    when the ball may have passed the hitter's partner. The hitter is the far player the ball came closest
    to (in box heights) over [first, t]. The contact frame is the one, among the frames they were in
    reach, where the ball is nearest hand height (hand_h of the way down their box) and to the box
    (dist_w weights the box distance): a far ball rising to the apex stays in reach well after contact,
    and a far bounce is in reach too but at the feet.
    """
    near, best = {}, (np.inf, None)
    for f in range(t, max(first, 0) - 1, -1):
        if np.isnan(ball[f, 0]):
            break
        for pid, pl in players[f].items():
            if not far_side(pl):
                continue
            x1, y1, x2, y2 = pl["bbox"]
            h = max(y2 - y1, 1e-9)
            d = _box_distance(ball[f], pl["bbox"]) / h
            if d <= reach:
                near.setdefault(pid, []).append((abs((ball[f, 1] - y1) / h - hand_h) + dist_w * d, -f))
                if d < best[0]:
                    best = (d, pid)
    if best[1] is None:
        return None, t
    return best[1], -min(near[best[1]])[1]


def _turn(ball, t, k=3):
    """How sharply the ball changes velocity at frame t (px/frame, over k frames each side); 0 if unseen."""
    if t - k < 0 or t + k >= len(ball):
        return 0.0
    a, b = ball[t - k:t + 1], ball[t:t + k + 1]
    if np.isnan(a).any() or np.isnan(b).any():
        return 0.0
    return float(np.linalg.norm((b[-1] - b[0]) - (a[-1] - a[0]))) / k


def find_events(ball, players, window=5, min_prominence=8, min_gap=6, reach=0.6,
                near_side=lambda pl: pl["court"][1] > NET_Y, far_side=lambda pl: pl["court"][1] <= NET_Y,
                lookback=15, swing=(4, 15)):
    """Hits and bounces are local extrema of the ball's image y.

    A bounce (ball falls then rises) is always a local max of image y. A near-side hit is a
    local max next to a near-side player (ball comes down the screen, then goes back up).
    A far-side hit is a local min (ball goes up the screen, then comes back down), so only a
    far_side player can make one: a near player's box often covers the far court on screen,
    and a local min there is a far hit or the top of a lob, never a near player's shot. The top of a
    far player's lob comes a few frames after contact, so a local min looks back up to lookback frames.
    A near local max next to a player can be the bounce in front of them (a return or a third shot is
    always played off a bounce), with their contact swing[0]..swing[1] frames later, where the ball
    rises off the court and is struck: no extremum there, but a sharper turn. When the ball, still in
    their reach, turns more sharply then than at the local max, the local max is a bounce and the hit
    is at that turn.
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
            if etype == "hit":
                later = [f for f in range(t + swing[0], min(t + swing[1] + 1, len(y)))
                         if pid in players[f] and not np.isnan(ball[f, 0])
                         and _box_distance(ball[f], players[f][pid]["bbox"])
                         <= reach * (players[f][pid]["bbox"][3] - players[f][pid]["bbox"][1])]
                turn = max(later, key=lambda f: _turn(ball, f), default=None)
                if turn is not None and _turn(ball, turn) >= _turn(ball, t):
                    events.append({"frame": t, "type": "bounce", "player": None,
                                   "px": [round(float(v), 1) for v in ball[t]]})
                    at = turn
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


def filter_hits(events, fps, min_gap_s=0.35, side=None, same_side_s=1.0, bounce_hit_s=(0.2, 0.6)):
    """Drop hits that can't be separate shots.

    - re-detections of one contact: a hit < min_gap_s after the previous kept hit (time only,
      because player IDs churn and one person can show up under two IDs)
    - with side(event) -> "near"/"far": a hit by the same side < same_side_s after the previous
      kept hit, since the ball has to cross the net and come back first
    Except: two near-side hits bounce_hit_s[0]..[1] s apart are a bounce in front of the player (the
    ball's lowest point on screen, in reach) and then their contact, so the later one is the hit and
    the earlier one becomes a bounce (once: the hit that replaced it isn't replaced in turn). Every
    return and third shot is played off a bounce; closer pairs are one contact detected twice.
    """
    out, last, replaced = [], None, False
    for e in events:
        e = dict(e)
        if e["type"] == "hit":
            gap = e["frame"] - last["frame"] if last else np.inf
            if side and not replaced and bounce_hit_s[0] * fps <= gap < bounce_hit_s[1] * fps \
                    and side(e) == side(last) == "near":
                last.update(type="bounce", player=None)
                last, replaced = e, True
            elif gap < min_gap_s * fps or (side and gap < same_side_s * fps and side(e) == side(last)):
                continue
            else:
                last, replaced = e, False
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


def find_serves(ball, players, court, fps, width, hold_s=1.0, go_s=0.6, go_far_s=1.0, near_h=0.5, away_h=1.5,
                min_near=3, max_away=0.2, baseline_m=1.0, receiver_m=1.5, min_contact_y=0.15, still_m=0.5, step_frac=0.1,
                same_contact_s=0.5, min_gap_s=2.0, keep_later=True, across_s=1.5, max_across=11, max_across_far=0, far_delay_s=0.27,
                detected=None, offscreen=None):
    """Fixed mode: serves straight from the ball track -> [{"frame", "player", "px"}].

    A serve is the ball held (or bounced) by a player near their own baseline, then hit so it
    crosses the net quickly. Contact is a frame t where:
    - the ball is within near_h box-heights of a player standing within baseline_m of their baseline
      (in front of it; behind it is fine), and, when both players across the net are in view, one of them
      stands within receiver_m of their own baseline to receive (balls fed across between points are
      hit from mid-court, or to players who aren't back to receive), and the ball is not in the top min_contact_y of their box (serves are underhand: a ball at head height is
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
    - the ball didn't come from across the net: in the across_s before t it was detected beyond the net
      (on screen: above the net line for a near server, below it for a far one) at most max_across
      times (max_across_far for a far server: before a real far serve the ball is never seen on the near
      half, while a far return has the near serve's flight behind it). Otherwise t is a return, or the end of a ball fed across between points
    Frames within same_contact_s of each other are one contact (the one where the ball speeds up
    most is kept), and a serve puts the ball in play, so nothing within min_gap_s after it is a serve,
    except from the same side (keep_later): then the first one was ball handling before the serve (a
    bounce, a toss to the server) and the later one is kept.
    """
    from .constants import COURT_L, COURT_W, NET_Y

    net_px = court.to_image([[COURT_W / 2, NET_Y]])[0][1]
    margin = 0.02 * width
    hold, go, max_step = max(2, int(hold_s * fps)), max(2, int(go_s * fps)), step_frac * width
    known = ~np.isnan(ball[:, 0])
    detected = known if detected is None else np.asarray(detected) & known
    serves = []
    for t in range(hold, len(ball) - 1):
        if not detected[t]:
            continue
        # only players at their baseline can serve (a near player at the kitchen has a box tall enough to
        # cover a far server on screen); a server out of view (no box, or behind someone: players.fill_hidden)
        # only when nobody seen is near
        back = {k: p for k, p in players[t].items()
                if (COURT_L - p["court"][1] if court.side(p["court"]) == "near" else p["court"][1]) <= baseline_m}
        visible = {k: p for k, p in back.items() if not p.get("hidden")}
        pid = nearest_player(ball[t], visible, near_h) or nearest_player(ball[t], back, near_h)
        if pid is None:
            continue
        pl = players[t][pid]
        x1, y1, x2, y2 = pl["bbox"]
        box_h = y2 - y1
        side = court.side(pl["court"])
        if (ball[t, 1] - y1) / box_h < min_contact_y:
            continue
        visible = {k: p for k, p in players[t].items() if not p.get("hidden")}
        across = [p["court"] for p in visible.values() if court.side(p["court"]) != side]
        depth = lambda c: c[1] if side == "near" else COURT_L - c[1]  # receiver's distance in from their baseline
        if len(across) >= 2 and min(depth(c) for c in across) > receiver_m:
            continue
        w = slice(max(0, t - int(across_s * fps)), t)
        ys = ball[w, 1][detected[w]]
        if (ys < net_px - margin if side == "near" else ys > net_px + margin).sum() > \
                (max_across if side == "near" else max_across_far):
            continue
        if not _ball_out_of_play(ball, known, players, pid, t, hold, near_h, away_h, min_near, max_away,
                                 min_contact_y, still_m, offscreen):
            continue
        path = _flight(ball, detected, t, go if side == "near" else max(2, int(go_far_s * fps)), max_step)
        if len(path) < 2:
            continue
        on_court = court.to_court(path)
        start = court.to_court([ball[t]])[0]
        toward = (start[1] - on_court[:, 1]) if side == "near" else (on_court[:, 1] - start[1])
        if toward.max() >= 3.0 and any(court.side(p) != side for p in on_court):
            serves.append({"frame": t, "player": pid, "px": [round(float(v), 1) for v in ball[t]]})
    # several frames around one contact qualify; contact is where the ball speeds up most, except on the
    # far side: the ball isn't seen in the server's hand, the speed-up frames come well after contact and
    # the first qualifying frame a little before it, so far_delay_s after the first one
    for s in serves:
        s["side"] = court.side(players[s["frame"]][s["player"]]["court"])
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
    def pick(g):
        if g[0]["side"] == "far":  # the frame nearest far_delay_s after the first one (tuned on hand labels)
            return min(g, key=lambda s: abs(s["frame"] - g[0]["frame"] - far_delay_s * fps))
        return max(g, key=lambda s: speed_up(s["frame"]))

    for s in (pick(g) for g in groups):
        if not out or s["frame"] - out[-1]["frame"] >= min_gap_s * fps:
            out.append(s)
        elif keep_later and s["side"] == out[-1]["side"]:
            out[-1] = s  # same side again within min_gap_s: the first was ball handling (a bounce, a toss)
    for s in out:
        del s["side"]
    return out


def real_detections(raw, ball):
    """Per frame: is the cleaned ball position an actual detection (not gap-filled, not a removed outlier)?"""
    return np.array([b is not None and not np.isnan(c[0]) and np.allclose(c, b) for b, c in zip(raw, ball)])


def hitter_court(players, e):
    """Court position of an event's player at its frame; for a player off screen (infer_returns), where
    they were last seen."""
    fp = players[e["frame"]]
    return fp[e["player"]]["court"] if e["player"] in fp else e["court"]


def find_entry_serves(ball, players, court, fps, width, height, still_s=1.0, swing_s=0.3, still_h=0.3,
                      min_people=2, unseen=5, edge=0.1, held_s=0.33, held_frac=0.02, quiet_s=2.0, hits=(),
                      go_s=1.0, across_s=1.5,
                      max_across=11, max_across_far=0, baseline_m=1.5, step_frac=0.1, same_s=1.0,
                      detected=None):
    """Serves inferred from the wait before one: everyone in view stands still, nobody has played the ball,
    then the ball comes into view and flies over the net. For serves find_serves can't see: the server is
    off the edge of the picture, or has no box (a far server the person detector misses), so the ball is
    never seen in their hand.

    players: per-frame players with hidden ones filled in (players.fill_hidden). Contact is the first frame
    t where:
    - the ball comes into play: detected at t, and either not detected in the `unseen` frames before, at
      the edge of the picture (within `edge` of the frame width / height, sides and bottom), or held: its
      detections over the held_s before stay within held_frac of the frame width with no player box in
      reach (in the hand of a server the person detector missed)
    - at least min_people players are really in view and none of them moved over the still_s before t
      (ending swing_s before t): the top-centre of their box stays within still_h box heights
    - no hit (from `hits`, frames) in the quiet_s before t: the ball was dead
    - the ball didn't come from across the net (as find_serves, max_across / max_across_far sightings)
    - it flies on (one continuous path, as find_serves) to past the net within go_s
    - the server: of the players (seen or hidden) on the ball's side of the net (above the net line on screen:
      far) within baseline_m of their baseline, the one nearest the ball
    -> [{"frame", "player", "px", "entry": True}], at most one per same_s.
    """
    from .constants import COURT_W

    known = ~np.isnan(ball[:, 0])
    detected = known if detected is None else np.asarray(detected) & known
    net_px = court.to_image([[COURT_W / 2, NET_Y]])[0][1]
    margin = 0.02 * width
    still, swing, go = int(still_s * fps), int(swing_s * fps), int(go_s * fps)
    hit_frames = np.asarray(sorted(hits), int)
    out = []
    for t in range(still + swing, len(ball) - 1):
        if not detected[t] or (out and t - out[-1]["frame"] < same_s * fps):
            continue
        x, y = ball[t]
        at_edge = x < edge * width or x > (1 - edge) * width or y > (1 - edge) * height
        before = ball[max(0, t - int(held_s * fps)):t][detected[max(0, t - int(held_s * fps)):t]]
        ball_side = "far" if y < net_px else "near"
        held = len(before) >= 3 and np.ptp(before, axis=0).max() < held_frac * width \
            and nearest_player(ball[t], {k: p for k, p in players[t].items() if not p.get("hidden")
                                         and court.side(p["court"]) == ball_side}, 0.5) is None
        if detected[max(0, t - unseen):t].any() and not at_edge and not held:
            continue
        i = np.searchsorted(hit_frames, t)
        if i and t - hit_frames[i - 1] <= quiet_s * fps:
            continue
        seen = {k: p for k, p in players[t].items() if not p.get("hidden")}
        if len(seen) < min_people:
            continue
        moved = False
        for pid in seen:
            boxes = [players[f][pid]["bbox"] for f in range(t - swing - still, t - swing)
                     if pid in players[f] and not players[f][pid].get("hidden")]
            if len(boxes) < still // 2:
                moved = True
                break
            b = np.array([((x1 + x2) / 2, y1, y2 - y1) for x1, y1, x2, y2 in boxes])
            if np.ptp(b[:, :2], axis=0).max() / np.median(b[:, 2]) > still_h:
                moved = True
                break
        if moved:
            continue
        side = "far" if y < net_px else "near"
        w = slice(max(0, t - int(across_s * fps)), t)
        ys = ball[w, 1][detected[w]]
        if (ys < net_px - margin if side == "near" else ys > net_px + margin).sum() > \
                (max_across if side == "near" else max_across_far):
            continue
        path = np.asarray(_flight(ball, detected, t, go, step_frac * width))
        if len(path) < 2 or not (path[:, 1] < net_px - margin if side == "near" else path[:, 1] > net_px + margin).any():
            continue
        back = {k: p for k, p in players[t].items() if court.side(p["court"]) == side
                and (COURT_L - p["court"][1] if side == "near" else p["court"][1]) <= baseline_m}
        if not back:
            continue
        pid = min(back, key=lambda k: _box_distance(ball[t], back[k]["bbox"]))
        out.append({"frame": t, "player": pid, "px": [round(float(v), 1) for v in ball[t]], "entry": True})
    return out


def combine_serves(serves, entry_serves, players, court, fps, same_s=1.0, return_s=2.5):
    """find_serves' serves plus find_entry_serves'.

    An entry serve and a find_serves serve on the same side within same_s are one serve: find_serves fires
    later when it only saw the ball once it was flying (next to the server's partner), so the earlier one
    is kept. An entry serve also turns a find_serves serve from across the net within return_s after it
    into a plain hit ("return": True; merge_serves adds it as a hit): that was the return.
    """
    side = lambda s: court.side(players[s["frame"]][s["player"]]["court"])
    out = list(serves)
    for e in entry_serves:
        f = e["frame"]
        same = [o for o in out if not o.get("return") and abs(o["frame"] - f) <= same_s * fps and side(o) == side(e)]
        if same and min(o["frame"] for o in same) <= f:
            continue  # find_serves already has it, as early
        out = [o for o in out if o not in same]
        out = [{**o, "return": True} if 0 < o["frame"] - f <= return_s * fps and side(o) != side(e)
               and not o.get("return") else o for o in out]
        if all(abs(o["frame"] - f) > same_s * fps for o in out if not o.get("return")):
            out.append(e)
    return sorted(out, key=lambda s: s["frame"])


def merge_serves(events, serves, players, court, fps, handling_s=3.0, after_s=0.3, hold_s=0.7):
    """Put detected serves into the event list as hits flagged "serve".

    Hits by the server's team in the handling_s before a serve are ball handling (bouncing it,
    passing it to the server), and any hit within after_s after it is the same contact detected late:
    the served ball can't have reached the other side yet, so a "hit" there (the ball passing over a
    receiver on screen) is the serve too. Any other event in the hold_s before contact, or at it (a bounce
    of the held ball, or the contact itself seen as a bounce), goes too.
    """
    side = lambda f, pid: court.side(players[f][pid]["court"]) if pid in players[f] else "near"  # off screen: near
    returns = [{"frame": s["frame"], "type": "hit", "player": s["player"], "px": s["px"]} for s in serves if s.get("return")]
    serves = [s for s in serves if not s.get("return")]
    events = sorted([e for e in events if all(abs(e["frame"] - r["frame"]) > 2 for r in returns)] + returns,
                    key=lambda e: e["frame"])
    drop = set()
    for s in serves:
        f, srv_side = s["frame"], side(s["frame"], s["player"])
        for i, e in enumerate(events):
            if e["type"] == "hit" and f < e["frame"] <= f + after_s * fps:
                drop.add(i)
            elif f - handling_s * fps <= e["frame"] <= f and e["type"] == "hit" \
                    and side(e["frame"], e["player"]) == srv_side:
                drop.add(i)
            elif f - hold_s * fps <= e["frame"] < f or (e["frame"] == f and e["type"] != "hit"):
                drop.add(i)
    kept = [e for i, e in enumerate(events) if i not in drop]
    added = [{"frame": s["frame"], "type": "hit", "player": s["player"], "px": s["px"], "serve": True,
              **{k: s[k] for k in ("offscreen", "court") if k in s}} for s in serves]
    return sorted(kept + added, key=lambda e: (e["frame"], e["type"] != "hit"))


def diagonal_servers(events, players, court, fps, within_s=2.5, centre_m=0.3):
    """Serves go cross-court: give each serve to the server's partner when it landed on the server's own half.

    A far server's ball is too small to see in their hand, so a far serve is first seen in flight, often
    next to the server's partner, and goes to whoever it passes. Where it lands is on the ground, so its
    court position is exact: the first bounce across the net within within_s, else where the next hit from
    across the net (the return) is played. If that is on the same half (left / right of the centre line)
    as the chosen server, and their partner stands on the other half, the partner served. Landings within
    centre_m of the centre line, or servers without a partner in view, are left alone.
    """
    from .constants import COURT_W

    half = lambda x: x < COURT_W / 2
    out = [dict(e) for e in events]
    for i, e in enumerate(out):
        if not e.get("serve") or e.get("offscreen"):  # off screen: the one player not in view served
            continue
        f, srv = e["frame"], hitter_court(players, e)
        side = court.side(srv)
        ref = None
        for nxt in out[i + 1:]:
            if nxt["frame"] > f + within_s * fps or nxt.get("serve"):
                break
            if nxt["type"] == "bounce":
                at = court.to_court([nxt["px"]])[0]
                if court.side(at) != side:
                    ref = at[0]
                    break
            elif nxt["type"] == "hit" and nxt["player"] in players[nxt["frame"]]:
                at = players[nxt["frame"]][nxt["player"]]["court"]
                if court.side(at) != side:
                    ref = at[0]
                    break
        if ref is None or abs(ref - COURT_W / 2) < centre_m or half(srv[0]) != half(ref):
            continue
        partner = [pid for pid, p in players[f].items()
                   if pid != e["player"] and court.side(p["court"]) == side and half(p["court"][0]) != half(ref)]
        if partner:
            e["player"] = partner[0]
    return out


def infer_returns(events, ball, players, court, fps, ids=("1", "2", "3", "4"), gap_s=(0.5, 2.5), after_bounce_s=1.0,
                  search_s=(0.8, 1.8), reach_h=0.6):
    """Add the return to serves that have none: a far return is often too flat to be a local extremum of the
    ball's image y, and a near receiver at the edge of the picture has no box, so their contact is taken
    for a bounce.

    For a serve with no hit from across the net gap_s after it, the return is the frame where the ball turns
    most sharply (_turn): in the after_bounce_s after the serve's bounce on the receiver's side, or, with
    no such bounce detected, search_s after the serve while the ball is on the receiver's side of the net.
    The returner is the receiver-side player the ball is within reach_h box heights of, else the one
    diagonally across from the server, else the one player not in view (off screen: "court" = last seen).
    Inferred returns are hits with "inferred": True.
    """
    from .constants import COURT_W

    ball = np.asarray(ball, float)
    net_px = court.to_image([[COURT_W / 2, NET_Y]])[0][1]
    half = lambda x: x < COURT_W / 2
    out = [dict(e) for e in events]
    added = []
    for e in out:
        if not e.get("serve"):
            continue
        f, srv = e["frame"], hitter_court(players, e)
        side = court.side(srv)
        across = lambda h: court.side(hitter_court(players, h)) != side
        if any(h["type"] == "hit" and across(h) and gap_s[0] * fps <= h["frame"] - f <= gap_s[1] * fps for h in out):
            continue
        bounce = next((b for b in out if b["type"] == "bounce" and gap_s[0] * fps <= b["frame"] - f <= gap_s[1] * fps
                       and court.side(court.to_court([b["px"]])[0]) != side), None)
        if bounce is not None:
            win = range(bounce["frame"] + 4, bounce["frame"] + int(after_bounce_s * fps))
        else:
            win = [k for k in range(f + int(search_s[0] * fps), f + int(search_s[1] * fps))
                   if k < len(ball) and not np.isnan(ball[k, 1])
                   and (ball[k, 1] < net_px if side == "near" else ball[k, 1] > net_px)]
        win = [k for k in win if 3 <= k < len(ball) - 3]
        if not win:
            continue
        k = max(win, key=lambda k: _turn(ball, k))
        if _turn(ball, k) == 0:
            continue
        fp = players[k]
        mine = {pid: p for pid, p in fp.items() if court.side(p["court"]) != side}
        pid = nearest_player(ball[k], mine, reach_h) if not np.isnan(ball[k, 0]) else None
        hit = {"frame": k, "type": "hit", "player": pid, "px": [round(float(v), 1) for v in ball[k]],
               "inferred": True}
        if pid is None:
            diag = [p for p, v in mine.items() if half(v["court"][0]) != half(srv[0])]
            if diag:
                hit["player"] = diag[0]
            else:
                missing = [i for i in ids if i not in fp and i != e["player"]]
                last = next((players[g][missing[0]]["court"] for g in range(k, max(-1, k - int(10 * fps)), -1)
                             if missing and missing[0] in players[g]), None)
                if last is None:
                    continue
                hit.update(player=missing[0], offscreen=True, court=[float(v) for v in last])
        added.append(hit)
    return sorted(out + added, key=lambda e: (e["frame"], e["type"] != "hit"))


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
                            "server_side": court.side(hitter_court(players, e))})
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
        hitter = np.array(hitter_court(players, e))
        landing = speed = None
        airtime = 0.0
        if target_ev and target_ev["frame"] > f:
            if landing_ev:
                landing = court.to_court([landing_ev["px"]])[0]
                tgt = landing
            else:  # volleyed: use the next hitter's position as the ball's destination
                tgt = np.array(hitter_court(players, target_ev))
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


ROLES = ("serve", "return", "third")


def opening_shots(shots, rallies, events, players, court, fps, min_gap_s=0.5, max_gap_s=2.5,
                  unseen_return_s=(1.8, 3.5)):
    """Keep only the serve, return and third shot of each served rally -> (shots, rallies, events).

    Roles go by side and timing, not by position in the rally, so a spurious hit (a bounce taken for a
    hit, a contact detected twice) doesn't push the real return or third shot out:
    - serve: a shot flagged as a serve (merge_serves)
    - return: the first shot from across the net min_gap_s..max_gap_s after the serve
    - third: the first shot by the serving team min_gap_s..max_gap_s after the return, or, if no return
      was detected, unseen_return_s after the serve (the return happened in between, unseen)
    all before the next serve. A rally keeps what was found (a serve alone if no return is). Every shot
    gets "role" and "type" (both serve / return / third: no drive / drop / lob classification) and its
    rally is renumbered; rallies keep start (the serve), end and duration_s (to the last kept shot).
    Events are cut to those that belong to the kept shots: from the serve up to, not including, the next hit after the last kept
    shot (so the third shot's bounce stays, later play goes).
    """
    side = lambda s: court.side(s["hitter"])
    lo, hi = min_gap_s * fps, max_gap_s * fps
    serves = [s for s in shots if s["type"] == "serve"]
    out_shots, out_rallies, windows = [], [], []
    for k, srv in enumerate(serves):
        until = serves[k + 1]["frame"] if k + 1 < len(serves) else float("inf")

        def first(after, same_side_as_server, a=lo, b=hi):
            return next((s for s in shots if after["frame"] + a <= s["frame"] <= min(after["frame"] + b, until - 1)
                         and s["type"] != "serve" and (side(s) == side(srv)) == same_side_as_server), None)

        kept = [(srv, "serve")]
        ret = first(srv, False)
        if ret is not None:
            kept.append((ret, "return"))
        third = first(ret, True) if ret is not None else \
            first(srv, True, unseen_return_s[0] * fps, unseen_return_s[1] * fps)
        if third is not None:
            kept.append((third, "third"))
        n = len(out_rallies) + 1
        rally = next(r for r in rallies if r["start"] <= srv["frame"] <= r["end"])
        roles = [r for _, r in kept]
        kept = [s for s, _ in kept]
        for role, s in zip(roles, kept):
            out_shots.append({**s, "rally": n, "n_in_rally": ROLES.index(role) + 1, "role": role, "type": role})
        out_rallies.append({**rally, "start": srv["frame"], "end": kept[-1]["frame"], "shots": len(kept),
                            "duration_s": round((kept[-1]["frame"] - srv["frame"]) / fps, 2)})
        last = kept[-1]["frame"]
        stop = next((e["frame"] for e in events if e["type"] == "hit" and e["frame"] > last), None)
        windows.append((srv["frame"], stop))
    out_events = [e for e in events
                  if any(a <= e["frame"] and (b is None or e["frame"] < b) for a, b in windows)]
    return out_shots, out_rallies, out_events


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
