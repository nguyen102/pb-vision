"""Homography between the camera image and the flat court plane."""
import json

import cv2
import numpy as np

from .constants import COURT_L, COURT_W, FT, KITCHEN, NET_Y

# Players chase wide balls ~1 m past a sideline but serve and lob-chase up to ~3 m behind a baseline.
SIDE_MARGIN, END_MARGIN = 1.0, 3.0

# Order matters: far-left, far-right, near-right, near-left.
CORNERS_M = np.float32([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])

# Line intersections a user can click when some corners are out of frame or hidden by the net.
LANDMARKS = {
    "far-left": (0, 0), "far-right": (COURT_W, 0), "near-right": (COURT_W, COURT_L), "near-left": (0, COURT_L),
    "near-baseline-center": (COURT_W / 2, COURT_L), "far-baseline-center": (COURT_W / 2, 0),
    "near-kitchen-left": (0, NET_Y + KITCHEN), "near-kitchen-center": (COURT_W / 2, NET_Y + KITCHEN),
    "near-kitchen-right": (COURT_W, NET_Y + KITCHEN),
    "far-kitchen-left": (0, NET_Y - KITCHEN), "far-kitchen-center": (COURT_W / 2, NET_Y - KITCHEN),
    "far-kitchen-right": (COURT_W, NET_Y - KITCHEN),
}

# Kitchen lines as y = a + b * x in court metres, per side; the nominal ones are 7 ft from the net.
NOMINAL_KITCHEN = {"near": (NET_Y + KITCHEN, 0.0), "far": (NET_Y - KITCHEN, 0.0)}


def kitchen_line_zone(p, depth=3 * FT, lines=NOMINAL_KITCHEN, inside=False):
    """Up at the net: court point p within depth behind its side's kitchen line, not in the kitchen (the line
    is the kitchen), between the sidelines. lines: Court.kitchen_lines (the painted lines), else nominal.
    inside: the kitchen itself counts too (someone who has got up to the net, rather than one standing there)."""
    if not 0 <= p[0] <= COURT_W:
        return False
    side = "far" if p[1] < NET_Y else "near"
    a, b = lines[side]
    behind = p[1] - (a + b * p[0]) if side == "near" else (a + b * p[0]) - p[1]
    return (behind <= depth) if inside else (0 < behind <= depth)


class Court:
    # shortcut: hand-clicked landmarks; upgrade to a court-keypoint model + RANSAC
    # (as Badminton-VisionAI does) for automatic calibration.
    def __init__(self, points_px, points_m=CORNERS_M):
        """points_px[i] is where court point points_m[i] (metres) appears in the image; 4+ points."""
        self.points_px = np.float32(points_px).reshape(-1, 2)
        self.points_m = np.float32(points_m).reshape(-1, 2)
        if len(self.points_px) < 4 or len(self.points_px) != len(self.points_m):
            raise ValueError("need 4+ matching image/court points")
        self.H, _ = cv2.findHomography(self.points_px, self.points_m)
        if self.H is None:
            raise ValueError("court points are degenerate (3 or more on one line?)")
        self.H_inv = np.linalg.inv(self.H)
        self.corners_px = self.to_image(CORNERS_M)  # court outline, even if corners are off-screen
        self.kitchen_lines = self._painted_kitchen_lines()

    def _painted_kitchen_lines(self):
        """Each kitchen line where the clicked points say it is, mapped through H: a line through the clicked
        kitchen points (2+), or level through the one clicked point. The homography is a best fit over all the
        clicks, so it can put a clicked kitchen point several cm off the nominal 7 ft; a foot on the painted
        line should count as on the line. A side with no kitchen clicks keeps the nominal line."""
        lines = dict(NOMINAL_KITCHEN)
        mapped = self.to_court(self.points_px)
        for side, (y_nominal, _) in NOMINAL_KITCHEN.items():
            on = np.abs(self.points_m[:, 1] - y_nominal) < 1e-3
            pts = mapped[on]
            if len(pts) >= 2 and np.ptp(pts[:, 0]) > 0.5:
                b, a = np.polyfit(pts[:, 0], pts[:, 1], 1)
                lines[side] = (float(a), float(b))
            elif len(pts):
                lines[side] = (float(pts[:, 1].mean()), 0.0)
        return lines

    def at_kitchen_line(self, p, depth=3 * FT, inside=False):
        """kitchen_line_zone measured from this court's painted kitchen lines."""
        return kitchen_line_zone(p, depth, self.kitchen_lines, inside)

    @classmethod
    def from_landmarks(cls, named_px):
        """{"near-left": (x, y), "near-kitchen-center": (x, y), ...} -> Court."""
        unknown = set(named_px) - set(LANDMARKS)
        if unknown:
            raise ValueError(f"unknown landmarks {sorted(unknown)}; choose from {list(LANDMARKS)}")
        return cls(list(named_px.values()), [LANDMARKS[n] for n in named_px])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["corners"]) if "corners" in d else cls(d["points_px"], d["points_m"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"points_px": self.points_px.tolist(), "points_m": self.points_m.tolist()}, f)

    @staticmethod
    def _apply(H, pts):
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    def to_court(self, pts_px):
        """Image pixels -> court metres. Only valid for points on the ground (feet, bounces)."""
        return self._apply(self.H, pts_px)

    def to_image(self, pts_m):
        return self._apply(self.H_inv, pts_m)

    @staticmethod
    def inside(p, margin=0.0):
        return -margin <= p[0] <= COURT_W + margin and -margin <= p[1] <= COURT_L + margin

    @staticmethod
    def in_play_area(p, side_margin=SIDE_MARGIN, end_margin=END_MARGIN):
        """Where a player can be: a little past the sidelines, well behind the baselines.
        Refs, line judges and benches stand beside the court, so the sideline margin is tight."""
        return -side_margin <= p[0] <= COURT_W + side_margin and -end_margin <= p[1] <= COURT_L + end_margin

    @staticmethod
    def outside_distance(p):
        """Metres from p to the court rectangle (0 when on the court)."""
        dx = max(-p[0], 0.0, p[0] - COURT_W)
        dy = max(-p[1], 0.0, p[1] - COURT_L)
        return float(np.hypot(dx, dy))

    @staticmethod
    def in_kitchen(p):
        return 0 <= p[0] <= COURT_W and abs(p[1] - NET_Y) <= KITCHEN

    @staticmethod
    def side(p):
        return "far" if p[1] < NET_Y else "near"
