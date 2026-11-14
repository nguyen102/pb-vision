"""Self-checks for the non-ML logic. Run: python -m pytest tests  (or python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbv import analytics, draw, events  # noqa: E402
from pbv.constants import COURT_L, COURT_W, KITCHEN, NET_Y  # noqa: E402
from pbv.court import LANDMARKS, Court  # noqa: E402

COURT = Court([(300, 100), (700, 100), (900, 600), (100, 600)])


def test_homography_roundtrip():
    assert np.allclose(COURT.to_court([(300, 100), (900, 600)]), [[0, 0], [COURT_W, COURT_L]], atol=1e-3)
    p = np.array([[2.0, 5.0], [4.0, 11.0]])
    assert np.allclose(COURT.to_court(COURT.to_image(p)), p, atol=1e-3)
    assert Court.in_kitchen((3, NET_Y + 1)) and not Court.in_kitchen((3, NET_Y + KITCHEN + 0.5))


def test_court_from_landmarks_when_corners_hidden():
    # Only the near half is visible: rebuild the same homography from 4 non-corner landmarks.
    named = {n: COURT.to_image([LANDMARKS[n]])[0].tolist()
             for n in ("near-left", "near-baseline-center", "near-kitchen-left", "near-kitchen-center")}
    c = Court.from_landmarks(named)
    assert np.allclose(c.corners_px, COURT.corners_px, atol=0.5)
    assert np.allclose(c.to_court([(500, 300)]), COURT.to_court([(500, 300)]), atol=1e-3)
    for bad in ({"near-left": (0, 0)}, {"middle": (1, 1), **named}):
        try:
            Court.from_landmarks(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_clean_ball_fills_short_gaps_only():
    track = [[10.0 * i, 100.0] for i in range(30)]
    track[5] = track[6] = None                  # short gap -> filled
    for i in range(15, 27):                     # long gap -> left empty
        track[i] = None
    track[10] = [900.0, 900.0]                  # outlier -> dropped then interpolated
    b = events.clean_ball(track, max_gap=8)
    assert np.allclose(b[5], [50, 100]) and np.allclose(b[10], [100, 100])
    assert np.isnan(b[20]).all()


def _player(cx, cy_court, h=150):
    """Player standing at court (cx, cy_court) with an h px tall box."""
    fx, fy = COURT.to_image([[cx, cy_court]])[0]
    return {"bbox": [fx - h / 5, fy - h, fx + h / 5, fy], "court": [cx, cy_court]}


def test_find_events_hit_bounce_hit():
    near, far = _player(3, 12.5), _player(3, 1.0, h=60)  # far player is smaller on screen
    nb, fb = near["bbox"], far["bbox"]
    bounce_px = COURT.to_image([[4.5, 4.0]])[0]
    # Piecewise-linear image-y: near hit (max) -> apex (min, no player) -> bounce (max) -> far hit (min) -> down.
    keys = [(0, nb[1] + 20), (10, nb[1] + 60), (22, 60), (32, bounce_px[1]), (40, fb[1] + 40), (55, 400)]
    xs = [(0, nb[0] + 30), (10, nb[0] + 30), (22, 750), (32, bounce_px[0]), (40, fb[0] + 12), (55, 450)]
    f = np.arange(56)
    ball = np.c_[np.interp(f, *zip(*xs)), np.interp(f, *zip(*keys))]
    players = [{"1": near, "2": far} for _ in f]
    evs = events.find_events(ball, players)
    assert [(e["frame"], e["type"], e["player"]) for e in evs] == [(10, "hit", "1"), (32, "bounce", None), (40, "hit", "2")]


def _vee(f, top, bottom, x, n=31):
    """n frames of ball at image x, going up the screen to `top` at frame f, then back down."""
    t = np.arange(n)
    return np.c_[np.full(n, x, float), bottom - (bottom - top) * (1 - np.abs(t - f) / f)]


def test_find_events_local_min_is_never_a_near_players_hit():
    near, far = _player(3, 12.5), _player(0.5, 1.0, h=60)
    nb = near["bbox"]
    # Top of a lob over the near player's box on screen, nowhere near the far player: no event,
    # where the nearest box (the near player's) used to get a hit.
    ball = _vee(15, nb[1] + 20, nb[1] + 120, (nb[0] + nb[2]) / 2)
    assert events.find_events(ball, [{"1": near, "2": far}] * len(ball)) == []
    # Same, with the far player right there on screen too: it's theirs.
    far2 = {"bbox": [nb[0], nb[1] - 10, nb[0] + 25, nb[1] + 50], "court": [3, 1.0]}
    evs = events.find_events(ball, [{"1": near, "2": far2}] * len(ball))
    assert [(e["type"], e["player"]) for e in evs] == [("hit", "2")]


def test_find_events_far_lob_goes_to_the_player_it_left():
    far = _player(3, 1.0, h=60)
    fb = far["bbox"]
    # The ball rises past the far player (reach 0.6 * 60 = 36 px) and keeps rising after contact:
    # its top, at frame 15, is 60 px above their box. The hit is theirs, at the last frame in reach.
    ball = _vee(15, fb[1] - 60, fb[1] + 60, fb[2] + 5)
    evs = events.find_events(ball, [{"2": far}] * len(ball))
    assert [(e["type"], e["player"]) for e in evs] == [("hit", "2")]
    assert 8 <= evs[0]["frame"] < 15


def test_find_events_far_lob_passing_the_partner_is_the_hitters():
    hitter, partner = _player(1.5, 1.0, h=60), _player(4.5, 1.0, h=60)
    hb, pb = hitter["bbox"], partner["bbox"]
    # Contact inside the hitter's box at frame 3; the ball rises across to the partner (just in reach
    # at the top of the arc, frame 15), then comes down. The partner is nearer at the local min.
    f = np.arange(31)
    xs = np.interp(f, [0, 3, 15, 30], [hb[0] + 5, hb[2] - 5, pb[0] - 10, pb[0] - 10])
    ys = np.interp(f, [0, 3, 15, 30], [hb[1] + 40, hb[1] + 20, pb[1] - 15, pb[1] + 60])
    evs = events.find_events(np.c_[xs, ys], [{"1": hitter, "2": partner}] * len(f))
    assert [(e["type"], e["player"]) for e in evs] == [("hit", "1")]


def test_classify_shots_types_and_faults():
    fps = 30
    pos = {"A": (3, 12.8), "B": (3, 0.5), "A2": (3, 12.8), "B2": (3, NET_Y - 1)}  # B2 at the kitchen
    frames = [{"1": _player(*pos["A"]), "2": _player(*pos["B"])} for _ in range(200)]
    for f in range(100, 200):
        frames[f]["2"] = _player(*pos["B2"])
    bpx = lambda x, y: COURT.to_image([[x, y]])[0].tolist()
    evs = [
        {"frame": 0, "type": "hit", "player": "1", "px": [0, 0]},
        {"frame": 30, "type": "bounce", "player": None, "px": bpx(3, 2.0)},      # serve lands deep
        {"frame": 40, "type": "hit", "player": "2", "px": [0, 0]},
        {"frame": 70, "type": "bounce", "player": None, "px": bpx(3, 12.0)},
        {"frame": 80, "type": "hit", "player": "1", "px": [0, 0]},               # third shot from baseline
        {"frame": 110, "type": "bounce", "player": None, "px": bpx(3, NET_Y - 1)},  # ... into the kitchen = drop
        {"frame": 120, "type": "hit", "player": "2", "px": [0, 0]},
        {"frame": 150, "type": "hit", "player": "1", "px": [0, 0]},              # volley, from baseline
        {"frame": 165, "type": "hit", "player": "2", "px": [0, 0]},              # volley from the kitchen = fault
    ]
    shots, rallies = events.classify_shots(evs, frames, COURT, fps)
    assert [s["type"] for s in shots][:3] == ["serve", "return", "drop"]
    assert shots[4]["volley"] and shots[5]["volley"]
    assert shots[5]["faults"] == ["kitchen_volley"] and shots[4]["faults"] == []
    assert len(rallies) == 1 and rallies[0]["shots"] == 6
    assert abs(shots[0]["speed_kmh"] - 10.8 / 1.0 * 3.6) < 0.5  # 10.8 m in 1 s
    summary = analytics.shot_stats(shots)
    assert summary["2"]["faults"] == 1 and summary["1"]["n_serve"] == 1


def test_player_stats_distance():
    fps = 30
    frames = [{"7": {"bbox": [0, 0, 1, 1], "court": [1.0 + f / fps, 1.0]}} for f in range(10 * fps)]  # 1 m/s
    st = analytics.player_stats(frames, fps)["7"]
    assert abs(st["distance_m"] - 10) < 0.2 and abs(st["avg_speed_ms"] - 1) < 0.05
    assert st["side"] == "far" and st["kitchen_pct"] == 0


def test_stroke_form():
    kp = np.zeros((17, 2))
    kp[5], kp[6] = (90, 100), (110, 100)       # shoulders
    kp[11], kp[12] = (95, 200), (105, 200)     # hips
    kp[7], kp[9] = (85, 150), (80, 160)        # left arm resting
    kp[8], kp[10] = (140, 150), (180, 230)     # right arm swinging low
    form = analytics.stroke_form(kp)
    assert form["arm"] == "right" and form["contact_below_waist"] and not form["overhead"]
    assert form["contact_height"] < 0                          # wrist 30 px below the hips
    kp[12] = (0, 0)
    assert analytics.stroke_form(kp) is None


def test_heatmap_renders():
    img = draw.heatmap_image([(3, 3)] * 20 + [(1, 10)] * 5)
    assert img.ndim == 3 and img.sum() > 0


# ---- POV mode ----

def test_camera_shift_and_compensation():
    import cv2
    from app import camera_shift
    rng = np.random.default_rng(0)
    a = (cv2.GaussianBlur(rng.random((540, 960)), (0, 0), 3) * 255).astype(np.uint8)
    f0 = cv2.merge([a] * 3)
    f1 = np.roll(f0, 20, axis=1)                      # scene content moves 20 px right
    g, _ = camera_shift(None, f0)
    _, shift = camera_shift(g, f1)
    assert abs(shift[0] - 20) < 1.5 and abs(shift[1]) < 1.5
    # A ball standing still in the world drifts with the pan on screen; compensation removes it.
    comp, _ = events.compensate([[100, 50], [120, 50], None], [[0, 0], [20, 0], [0, 0]])
    assert comp[0] == comp[1] == [100, 50] and comp[2] is None


def test_own_body_filter():
    from pbv.tracking import is_own_body
    assert is_own_body([900, 0, 1280, 180], 1280, 720)        # arm in the top-right corner
    assert is_own_body([300, 500, 900, 720], 1280, 720) is False  # touches only the bottom edge
    assert not is_own_body([830, 330, 880, 450], 1280, 720)   # opponent mid-frame
    assert is_own_body([650, 610, 740, 720], 1280, 720)       # own hand on the paddle, bottom edge
    assert not is_own_body([980, 290, 1075, 720], 1280, 720)  # partner standing close, feet cut off


def test_find_swings_needs_ball_and_own_paddle():
    fps, n = 30, 120
    cam = [[2.0, 0.0]] * n
    cam[30] = [150.0, 0.0]                            # swing jolt, ball visible
    cam[60] = [150.0, 0.0]                            # head turn, ball visible, no own paddle
    cam[100] = [150.0, 0.0]                           # walking jolt, no ball nearby
    ball = np.full((n, 2), np.nan)
    ball[20:70] = 100.0
    assert events.find_swings(cam, ball, fps, width=1920) == [30, 60]       # no paddle data: jolts only
    paddles = [None] * n
    paddles[28] = [1500, 100, 1800, 700]              # own paddle: 600 px of a 1080 px frame
    paddles[58] = [900, 500, 930, 560]                # someone else's paddle, far away
    assert events.find_swings(cam, ball, fps, 1920, paddles, 1080) == [30]


def test_filter_hits_drops_double_detections():
    evs = [{"frame": 0, "type": "hit", "player": "1"}, {"frame": 5, "type": "hit", "player": "7"},
           {"frame": 8, "type": "bounce", "player": None}, {"frame": 20, "type": "hit", "player": "1"},
           {"frame": 35, "type": "hit", "player": "2"}]
    assert [e["frame"] for e in events.filter_hits(evs, fps=30)] == [0, 8, 20, 35]
    side = {"1": "near", "7": "near", "2": "far"}       # frame 20: near side again 0.67 s later
    assert [e["frame"] for e in events.filter_hits(evs, 30, side=lambda e: side[e["player"]])] == [0, 8, 35]


def test_pov_shots_rallies_and_net_crossing():
    fps = 30
    evs = [{"frame": 0, "type": "hit", "player": "4", "side": "far"},
           {"frame": 20, "type": "bounce", "player": None},
           {"frame": 50, "type": "hit", "player": "4", "side": "far"},       # volley: no bounce after your swing
           {"frame": 60, "type": "hit", "player": "6", "side": "far"},       # same side 0.33 s later: impossible
           {"frame": 400, "type": "hit", "player": "5", "side": "near"},     # long gap -> new rally (partner serves)
           ]
    swings = [30, 410]                                # 410 follows your partner's hit, not the opponent's
    shots, rallies = events.pov_shots(evs, swings, fps)
    assert [(s["player"], s["type"]) for s in shots] == [("4", "serve"), ("me", "return"), ("4", "volley"), ("5", "serve")]
    assert [r["shots"] for r in rallies] == [3, 1]

def _ball_frames(n=20, missed=(8, 9)):
    """Candidates [x, y, w, h, conf, motion]: a sharp static logo (conf 0.6) and a blurred moving ball (0.3)."""
    frames = []
    for i in range(n):
        c = [[300.0, 50.0, 6, 6, 0.6, 0.0]]
        if i not in missed:
            c.append([100.0 + 10 * i, 200.0 + 0.5 * i * i, 6, 6, 0.3, 0.8])
        frames.append(c)
    return frames


def test_training_labels_keep_moving_ball_and_fill_gaps():
    from training.labels import ball_labels
    frames = _ball_frames() + [None] + _ball_frames(6, missed=())  # None = camera cut / unscanned
    labels, tracks = ball_labels(frames, width=1920)
    assert all(t[0][1] == 100.0 for t in tracks)                 # the logo never becomes a label
    assert sorted(labels) == list(range(20)) + list(range(21, 27))
    x, y, w, h, filled = labels[8][0]
    assert abs(x - 180) < 1e-6 and filled and not labels[7][0][4]


def test_select_path_prefers_moving_ball_over_sharper_static_object():
    from pbv.balltrack import select_path
    path = select_path(_ball_frames(), width=1920)
    assert path[5] == [150.0, 212.5] and path[8] is None          # gaps are clean_ball's job
    assert all(p is None or p[0] != 300.0 for p in path[:8] + path[10:])


def test_ball_leaving_the_frame_is_offscreen_not_lost():
    from pbv.balltrack import offscreen_frames
    W, H = 1920, 1080
    # Lob: rises 30 px/frame, crosses the top edge, comes back 6 frames later. Then a mid-court miss.
    path = [[900.0, 200.0 - 30 * i] for i in range(7)] + [None] * 6 + [[960.0, 30.0], [965.0, 60.0]]
    path += [[970.0, 400.0], None, None, [990.0, 420.0]]
    off = offscreen_frames(path, W, H)
    assert off[7:13] == [True] * 6 and not any(off[:7]) and not any(off[13:])
    # Short gap: clean_ball would join (900, 20) and (960, 30) with a line across the top; off-screen stays empty.
    ball = events.clean_ball(path, offscreen=off)
    assert np.isnan(ball[7:13]).all() and not np.isnan(ball[17:19]).any()   # the mid-court miss is still filled
    assert offscreen_frames([[900.0, 500.0], [900.0, 480.0], None, None], W, H) == [False] * 4
    assert offscreen_frames([[900.0, 30.0], [900.0, 31.0], None], W, H) == [False] * 3  # at the edge, not leaving


def test_offscreen_lob_does_not_split_the_rally():
    fps = 30
    players = [{"1": {"bbox": [0, 0, 1, 1], "court": [3.0, 12.0]}, "2": {"bbox": [0, 0, 1, 1], "court": [3.0, 1.0]}}
               for _ in range(200)]
    evs = [{"frame": 10, "type": "hit", "player": "1", "px": [0, 0]},
           {"frame": 100, "type": "hit", "player": "2", "px": [0, 0]}]  # 3 s later, 2 s of it a lob above the frame
    off = [10 <= f < 70 for f in range(200)]
    _, rallies = events.classify_shots(evs, players, COURT, fps)
    assert len(rallies) == 2
    _, rallies = events.classify_shots(evs, players, COURT, fps, offscreen=off)
    assert len(rallies) == 1 and rallies[0]["shots"] == 2


def test_ball_region_uses_size_to_drop_the_next_court():
    from pbv.balltrack import BallRegion
    court = Court([(642, 473), (1205, 468), (1915, 1005), (2, 1017)])  # 1080p behind-baseline view
    r = BallRegion(court)
    assert r.contains([920, 700, 14, 14]) and r.contains([920, 300, 8, 8])     # mid-court; lob over the far half
    assert r.contains([1700, 470, 22, 22])       # big = near the camera: 1.7 m above our near sideline
    assert not r.contains([1700, 470, 8, 8])     # small = far away: that's the next court's ball
    assert not r.contains([920, 1060, 8, 8])     # small and below the near baseline: under the floor




def _person(pid, cx, cy_court, h=150):
    """Raw tracker detection: ByteTrack id + bbox + feet in court metres."""
    return {"id": pid, **_player(cx, cy_court, h)}


def test_play_area_margins_are_tight_on_sidelines_loose_on_baselines():
    assert Court.in_play_area((COURT_W + 0.5, 5)) and not Court.in_play_area((COURT_W + 1.7, 5))
    assert not Court.in_play_area((-1.7, 10))
    assert Court.in_play_area((3, COURT_L + 2.5)) and Court.in_play_area((3, -2.5))
    assert not Court.in_play_area((3, COURT_L + 4))


def test_select_players_keeps_two_per_side_and_drops_bystanders():
    from pbv.players import select_players
    fps = 30
    frames = []
    for f in range(3 * fps):
        t = f / fps
        frames.append([
            _person("1", 1.5 + 0.3 * t, 12.5, h=200), _person("2", 4.5, 12.0 - 0.3 * t, h=200),  # near players
            _person("3", 1.5, 1.0 + 0.3 * t, h=60), _person("4", 4.5 - 0.3 * t, 1.5, h=60),        # far players
            _person("8", COURT_W + 1.7, 10.0, h=170),   # line judge past the sideline: outside the margin
            _person("9", COURT_W + 0.5, 11.0, h=180),   # still bystander inside the margin, near side
            _person("10", -0.6, 3.0, h=70),              # still bystander inside the margin, far side
        ])
    out = select_players(frames, fps)
    assert all(len(fp) == 4 for fp in out)
    sides = {pid: Court.side(p["court"]) for pid, p in out[0].items()}
    assert sorted(sides.values()) == ["far", "far", "near", "near"]
    kept_x = sorted(round(p["court"][0], 1) for p in out[0].values())
    assert kept_x == [1.5, 1.5, 4.5, 4.5]  # the four players, none of the bystanders


def test_select_players_keeps_one_id_through_bytetrack_switches():
    from pbv.players import select_players
    fps = 30
    frames = []
    for f in range(4 * fps):
        x = 1.0 + 0.5 * f / fps  # near player walks right at 0.5 m/s ...
        near_id = "5" if f < 2 * fps else "17"  # ... and ByteTrack gives them a new id halfway
        frames.append([_person(near_id, x, 12.0), _person("6", 5.0, 11.0),
                       _person("7", 2.0, 1.0, h=60), _person("8", 4.0, 1.0, h=60)])
    frames[70] = [p for p in frames[70] if p["id"] != "6"]  # partner missed for one frame
    out = select_players(frames, fps)
    walker = {pid for fp in out for pid, p in fp.items() if abs(p["court"][1] - 12.0) < 1e-6}
    partner = {pid for fp in out for pid, p in fp.items() if abs(p["court"][1] - 11.0) < 1e-6}
    assert len(walker) == 1 and len(partner) == 1 and walker != partner
    assert len({pid for fp in out for pid in fp}) == 4
    assert len(out[70]) == 3  # a missed detection leaves the slot empty, no stand-in


def _look(top, bottom, head=(120, 160, 210)):
    """Appearance descriptor of a person wearing top/bottom BGR colours (head: skin tone by default)."""
    from pbv.appearance import describe
    img = np.zeros((300, 200, 3), np.uint8)
    img[:60] = head
    img[60:150], img[150:] = top, bottom
    return describe(img, [0, 0, 200, 300])


def test_appearance_distance_tells_clothes_apart():
    from pbv.appearance import distance
    red_blue, blue_red = _look((0, 0, 220), (220, 0, 0)), _look((220, 0, 0), (0, 0, 220))
    assert distance(red_blue, _look((0, 0, 220), (220, 0, 0))) < 0.05
    assert distance(red_blue, blue_red) > 0.5
    assert distance(None, red_blue) == 0.0  # unknown look is neutral


def _walkers(fps, seconds, plan):
    """plan(f) -> [(track_id, x, y, look), ...] per frame -> raw people frames."""
    frames = []
    for f in range(int(seconds * fps)):
        frames.append([{**_person(pid, x, y, h=200 if y > NET_Y else 60), "look": look} for pid, x, y, look in plan(f)])
    return frames


def test_reid_gives_returning_players_their_own_ids():
    from pbv.players import select_players
    fps = 30
    a, b = _look((0, 0, 220), (40, 40, 40)), _look((220, 220, 220), (220, 0, 0))
    far = [("7", 2.0, 1.0, None), ("8", 4.0, 1.0, None)]

    def plan(f):
        if f < 60:                      # A on the left, B on the right
            return [("1", 1.5, 12.0, a), ("2", 4.5, 12.0, b)] + far
        if f < 150:                     # both walk out of frame
            return far
        return [("21", 1.5, 12.0, b), ("22", 4.5, 12.0, a)] + far  # back in, sides swapped, new ByteTrack ids

    out = select_players(_walkers(fps, 6, plan), fps)
    id_a_before = next(pid for pid, p in out[0].items() if p["court"] == [1.5, 12.0])
    id_a_after = next(pid for pid, p in out[170].items() if p["court"] == [4.5, 12.0])
    assert id_a_before == id_a_after
    assert len({pid for fp in out for pid in fp}) == 4


def test_reid_keeps_strangers_out_of_a_missing_players_id():
    from pbv.players import select_players
    fps = 30
    a, b, stranger = _look((0, 0, 220), (40, 40, 40)), _look((220, 220, 220), (220, 0, 0)), \
        _look((0, 200, 0), (0, 200, 200), head=(30, 30, 30))  # green kit, dark hair
    far = [("7", 2.0, 1.0, None), ("8", 4.0, 1.0, None)]

    def plan(f):
        if f < 60:
            return [("1", 1.5, 12.0, a), ("2", 4.5, 12.0, b)] + far
        return [("2", 4.5, 12.0, b), ("30", 1.5, 12.5, stranger)] + far  # A gone, a stranger walks on

    out = select_players(_walkers(fps, 4, plan), fps)
    assert all(len(fp) == 3 for fp in out[61:])  # A's id stays empty; the stranger gets none
    assert len({pid for fp in out for pid in fp}) == 4


def _serve_scene(fps, server_y=13.8, go=True, incoming=False):
    """Server on the near side holds the ball still for 1.5 s, then (go=True) serves it over the net.
    incoming=True: instead of being held, the ball flies in from the far side and is hit straight back."""
    server, partner = _player(2.0, server_y, h=100), _player(4.5, 12.5, h=100)  # real scale: ~1/8 of court width
    far1, far2 = _player(2.0, 1.0, h=60), _player(4.5, 1.0, h=60)
    x1, y1, x2, y2 = server["bbox"]
    hand = np.array([x2, (y1 + y2) / 2])
    n_hold, n_fly = int(1.5 * fps), int(0.8 * fps)
    target = COURT.to_image([[4.5, 2.0]])[0]
    if incoming:
        ball = [target + (hand - target) * (k + 1) / n_hold for k in range(n_hold)]
    else:
        ball = [hand + [0, 3 * np.sin(f)] for f in range(n_hold)]  # jiggling in the hand
    for k in range(1, n_fly + 1):
        ball.append(hand + (target - hand) * k / n_fly if go else hand + [0, 3 * np.sin(k)])
    players = [{"1": server, "2": partner, "3": far1, "4": far2} for _ in ball]
    return np.array(ball, float), players, n_hold - 1


def test_find_serves_needs_still_ball_at_baseline_then_fast_crossing():
    fps = 30
    ball, players, contact = _serve_scene(fps)
    serves = events.find_serves(ball, players, COURT, fps, 1000)
    assert [(s["frame"], s["player"]) for s in serves] == [(contact, "1")]
    assert events.find_serves(*_serve_scene(fps, go=False)[:2], COURT, fps, 1000) == []          # ball just held
    assert events.find_serves(*_serve_scene(fps, server_y=9.5)[:2], COURT, fps, 1000) == []      # hit from mid-court
    assert events.find_serves(*_serve_scene(fps, incoming=True)[:2], COURT, fps, 1000) == []     # volleyed straight back


def test_merge_serves_drops_ball_handling_and_starts_rally():
    fps = 30
    pos = {"1": (2.0, 13.8), "2": (4.5, 12.5), "3": (2.0, 1.0)}
    players = [{pid: _player(*xy) for pid, xy in pos.items()} for _ in range(400)]
    bpx = lambda x, y: COURT.to_image([[x, y]])[0].tolist()
    evs = [
        {"frame": 10, "type": "hit", "player": "3", "px": [0, 0]},                # end of the previous point
        {"frame": 100, "type": "hit", "player": "1", "px": [0, 0]},               # bouncing the ball before serving
        {"frame": 120, "type": "bounce", "player": None, "px": bpx(2.0, 13.5)},
        {"frame": 150, "type": "hit", "player": "2", "px": [0, 0]},               # partner taps it back
        {"frame": 174, "type": "hit", "player": "3", "px": [0, 0]},               # served ball passing over a receiver on screen
        {"frame": 178, "type": "hit", "player": "1", "px": [0, 0]},               # late re-detection of the serve
        {"frame": 200, "type": "bounce", "player": None, "px": bpx(3.0, 3.0)},
        {"frame": 220, "type": "hit", "player": "3", "px": [0, 0]},
        {"frame": 260, "type": "bounce", "player": None, "px": bpx(3.0, 11.0)},
        {"frame": 280, "type": "hit", "player": "2", "px": [0, 0]},
    ]
    merged = events.merge_serves(evs, [{"frame": 170, "player": "1", "px": [5.0, 5.0]}], players, COURT, fps)
    hits = [(e["frame"], e["player"], bool(e.get("serve"))) for e in merged if e["type"] == "hit"]
    assert hits == [(10, "3", False), (170, "1", True), (220, "3", False), (280, "2", False)]
    shots, rallies = events.classify_shots(merged, players, COURT, fps, serves_detected=True)
    assert shots[0]["type"] != "serve"  # end of the previous point: no serve detected for it
    assert [s["type"] for s in shots][1:3] == ["serve", "return"]
    assert [r["shots"] for r in rallies] == [1, 3]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
