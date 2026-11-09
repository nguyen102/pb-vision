"""Clothing-colour descriptors, so a player who leaves the frame gets their own id back on return.

A person box is split into head (cap / hair), torso (shirt) and legs (shorts / skirt). Each part
gets a small HSV colour histogram from the middle of the box, where there is least background.
"""
import cv2
import numpy as np

# (top, bottom) of each body part as a fraction of the box height
PARTS = ((0.0, 0.18), (0.18, 0.5), (0.5, 0.8))
BINS = (8, 2, 3)  # hue, saturation, value: white, black and grey kit differ in S/V, not hue
DIM = int(np.prod(BINS))


def describe(frame, bbox):
    """-> uint8 vector (len(PARTS) * DIM): per-part HSV histograms, each scaled to sum to ~255.

    Returns None when the box is too small to say anything about clothes (far side, < 24 px tall).
    """
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, x2 = max(x1, 0), min(x2, w_img)
    y1, y2 = max(y1, 0), min(y2, h_img)
    h, w = y2 - y1, x2 - x1
    if h < 24 or w < 6:
        return None
    cx1, cx2 = x1 + w // 4, x2 - w // 4  # middle half of the width: body, not background
    hsv = cv2.cvtColor(frame[y1:y2, cx1:max(cx2, cx1 + 1)], cv2.COLOR_BGR2HSV)
    out = []
    for top, bottom in PARTS:
        part = hsv[int(top * h):max(int(bottom * h), int(top * h) + 1)]
        hist = cv2.calcHist([part], [0, 1, 2], None, list(BINS), [0, 180, 0, 256, 0, 256]).ravel()
        out.append(hist / max(hist.sum(), 1e-9) * 255)
    return np.concatenate(out).round().astype(np.uint8)


def distance(a, b):
    """Mean Bhattacharyya distance over body parts: 0 = same colours, 1 = nothing in common.
    A missing descriptor (None) is neutral (0), so people without one are matched on position only."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, float).reshape(len(PARTS), DIM)
    b = np.asarray(b, float).reshape(len(PARTS), DIM)
    a /= np.maximum(a.sum(1, keepdims=True), 1e-9)
    b /= np.maximum(b.sum(1, keepdims=True), 1e-9)
    bc = np.sqrt(a * b).sum(1)
    return float(np.sqrt(np.clip(1 - bc, 0, 1)).mean())


def add_appearance(video, people, start=0):
    """Fixed mode: read the video once more and attach a "look" descriptor to every tracked person.

    -> uint8 array (n_people, len(PARTS) * DIM) in frame order (zeros where the box was too small),
    which is what app.py caches as appearance.npy.
    """
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rows = []
    for fp in people:
        ok, frame = cap.read()
        for p in fp:
            d = describe(frame, p["bbox"]) if ok else None
            p["look"] = d
            rows.append(np.zeros(len(PARTS) * DIM, np.uint8) if d is None else d)
    return np.array(rows, np.uint8).reshape(-1, len(PARTS) * DIM)


def attach_appearance(people, looks):
    """Attach cached descriptors (from add_appearance) back onto people; all-zero rows mean None."""
    k = 0
    for fp in people:
        for p in fp:
            p["look"] = looks[k] if looks[k].any() else None
            k += 1
