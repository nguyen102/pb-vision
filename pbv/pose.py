"""Pose keypoints for every tracked player in every frame, for the hit detector (hits_model.pose_features).

Each player's box is cropped as a square (padded 30%) and run through the YOLO pose model, so the far players
(about 120 px tall) are seen at a usable size. The pose nearest the crop centre is kept. About 40-60 frames a
second on an RTX 3060.
"""
import cv2
import numpy as np

CROP = 256


def crop_square(frame, bbox, pad=0.3):
    """-> (CROP x CROP image, crop's left, top, frame pixels per crop pixel)."""
    x1, y1, x2, y2 = bbox
    s = max(x2 - x1, y2 - y1) * (1 + pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax, ay = int(round(cx - s / 2)), int(round(cy - s / 2))
    n = max(int(round(s)), 8)
    H, W = frame.shape[:2]
    out = np.zeros((n, n, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(ax, 0), max(ay, 0), min(ax + n, W), min(ay + n, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - ay:sy1 - ay, sx0 - ax:sx1 - ax] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (CROP, CROP)), ax, ay, n / CROP


def keypoints(video, players, start=0, model="yolov8n-pose.pt", batch=64, progress=None):
    """-> (kp [T, P, 17, 3] float16: x, y in frame pixels and confidence, zeros when missing; ids [P]).
    players: per-frame players as tracked (hidden ones, filled in, are skipped); start: the first frame's index
    in the video. progress: wraps the frame range (e.g. tqdm)."""
    from ultralytics import YOLO
    net = YOLO(str(model))
    T = len(players)
    ids = sorted({pid for fp in players for pid in fp}, key=lambda s: (len(s), s))
    col = {pid: i for i, pid in enumerate(ids)}
    kp = np.zeros((T, len(ids), 17, 3), np.float16)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    pending = []  # (frame, pid, left, top, scale, crop)

    def flush():
        if not pending:
            return
        res = net.predict([p[5] for p in pending], imgsz=CROP, verbose=False, conf=0.1)
        for (f, pid, ax, ay, sc, _), r in zip(pending, res):
            if r.keypoints is None or not len(r.boxes):
                continue
            c = r.boxes.xywh[:, :2].cpu().numpy()
            j = int(np.argmin(np.linalg.norm(c - CROP / 2, axis=1)))
            xy = r.keypoints.xy[j].cpu().numpy()
            conf = r.keypoints.conf[j].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
            kp[f, col[pid], :, 0] = xy[:, 0] * sc + ax
            kp[f, col[pid], :, 1] = xy[:, 1] * sc + ay
            kp[f, col[pid], :, 2] = conf
        pending.clear()

    for f in (progress or iter)(range(T)):
        ok, frame = cap.read()
        if not ok:
            break
        for pid, p in players[f].items():
            if p.get("hidden"):
                continue
            crop, ax, ay, sc = crop_square(frame, p["bbox"])
            pending.append((f, pid, ax, ay, sc, crop))
        if len(pending) >= batch:
            flush()
    flush()
    return kp, np.array(ids)


def load_or_extract(path, video, players, start=0, model="yolov8n-pose.pt", progress=None):
    """Keypoints cached at path (pose.npz), extracted when missing or for a different number of frames."""
    from pathlib import Path
    path = Path(path)
    if path.exists():
        z = np.load(path)
        if len(z["kp"]) == len(players):
            return z["kp"], z["ids"]
    kp, ids = keypoints(video, players, start, model, progress=progress)
    np.savez_compressed(path, kp=kp, ids=ids)
    return kp, ids
