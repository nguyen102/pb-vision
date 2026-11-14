"""YOLO-based detectors: players (ByteTrack IDs), ball, and pose."""
import cv2
import numpy as np
import torch
from ultralytics import YOLO

PERSON, SPORTS_BALL, RACKET = 0, 32, 38  # COCO class ids; pickleball paddles detect as "tennis racket"
DEVICE = "mps" if torch.backends.mps.is_available() else None  # None = ultralytics picks CUDA/CPU
QUANT = 16 if torch.cuda.is_available() else None  # fp16 on NVIDIA: ~1.6x faster, same detections


def is_own_body(box, w, h, margin=4):
    """POV: the wearer's arms/legs/hands show up as person boxes pinned to 2+ frame edges,
    or as short boxes sitting on the bottom edge (a hand holding the paddle)."""
    x1, y1, x2, y2 = box
    edges = (x1 <= margin) + (y1 <= margin) + (x2 >= w - margin) + (y2 >= h - margin)
    return edges >= 2 or (y2 >= h - margin and y2 - y1 < 0.25 * h)


class PlayerTracker:
    def __init__(self, model="yolov8m.pt", conf=0.3):
        self.model, self.conf = YOLO(model), conf
        self.person = class_id(self.model, "person")

    def _track_boxes(self, frame):
        """-> (Nx4 xyxy boxes, N ByteTrack ids) of the people in frame."""
        r = self.model.track(frame, persist=True, classes=[self.person], conf=self.conf,
                             tracker="bytetrack.yaml", device=DEVICE, verbose=False)[0]
        if r.boxes.id is None:
            return np.zeros((0, 4)), []
        return r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()

    def people_near_court(self, frame, court, margin=4.0):
        """Fixed mode: [{"id", "bbox", "court"}] for everyone whose feet are within margin m of the court.

        Everyone is kept (refs and spectators too) so pbv.players.select_players can pick the
        4 players afterwards and be retuned without re-running YOLO.
        """
        boxes, ids = self._track_boxes(frame)
        if not ids:
            return []
        feet = court.to_court(np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
        return [{"id": str(ids[i]), "bbox": boxes[i].round(1).tolist(), "court": feet[i].round(3).tolist()}
                for i in range(len(ids)) if court.inside(feet[i], margin=margin)]

    def track_pov(self, frame, max_players=3):
        """POV: {player_id: {"bbox"}} for the biggest people (partner + opponents), the wearer's own body dropped."""
        boxes, ids = self._track_boxes(frame)
        h, w = frame.shape[:2]
        keep = [i for i in range(len(ids)) if not is_own_body(boxes[i], w, h)]
        keep.sort(key=lambda i: -(boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
        return {str(ids[i]): {"bbox": boxes[i].round(1).tolist()} for i in keep[:max_players]}


def class_id(model, *names):
    """Index of the first of names in the model's classes (COCO and trained models name them differently)."""
    ids = {v: k for k, v in model.names.items()}
    return next((ids[n] for n in names if n in ids), None)


class BallDetector:
    """Ball candidates per frame: a full-frame pass plus a full-resolution crop around where the ball
    is predicted to be (a far ball is a few pixels; downscaling the whole frame loses it).

    COCO "sports ball" misses many small/blurred pickleballs; a model trained with training/
    (class "pickleball") is picked up automatically via --ball-model and runs at native resolution.
    """

    def __init__(self, model="yolov8m.pt", cls=None, conf=0.05, imgsz=None, crop=640, top_k=8):
        self.model = YOLO(model)
        if cls is None:
            cls = class_id(self.model, "pickleball", "sports ball")
        self.trained = "pickleball" in self.model.names.values()
        self.paddle_cls = None if cls is not None and cls < 0 else class_id(self.model, "paddle", "tennis racket")
        self.paddles = self.paddle_cls is not None  # paddles come from the same pass
        self.classes = None if cls < 0 else [cls] + ([self.paddle_cls] if self.paddles else [])
        # A trained model learned the ball at native scale (training/ crops full-res frames): feed it native.
        self.conf, self.imgsz = conf, imgsz or (1920 if self.trained else 1280)
        self.crop, self.top_k = crop, top_k
        self.reset()

    def reset(self):
        self.prev_gray, self.history = None, []  # history: (frame index, x, y) of the online best guess
        self.t = 0

    def _predict(self, img, imgsz, dx=0, dy=0):
        r = self.model.predict(img, classes=self.classes, conf=self.conf, imgsz=imgsz, device=DEVICE,
                               quantize=QUANT, verbose=False)[0]
        balls, paddles = [], []
        for (x, y, w, h), k, c in zip(r.boxes.xywh.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            if self.paddles and k == self.paddle_cls:
                paddles.append([x - w / 2 + dx, y - h / 2 + dy, x + w / 2 + dx, y + h / 2 + dy])
            else:
                balls.append([x + dx, y + dy, w, h, c])
        return balls, paddles

    def _predicted_xy(self):
        """Where the ball should be now from the last two online guesses, or None if it's been lost."""
        if not self.history or self.t - self.history[-1][0] > 6:
            return None
        j, x, y = self.history[-1]
        if len(self.history) >= 2 and j - self.history[-2][0] <= 3:
            j2, x2, y2 = self.history[-2]
            vx, vy = (x - x2) / (j - j2), (y - y2) / (j - j2)
            return x + vx * (self.t - j), y + vy * (self.t - j)
        return x, y

    def candidates(self, frame):
        """-> ([[x, y, w, h, conf, motion], ...] best first, tallest paddle box or None)."""
        from .balltrack import merge_candidates, motion_score

        H, W = frame.shape[:2]
        balls, paddles = self._predict(frame, self.imgsz)
        pred = self._predicted_xy()
        if self.crop and pred is not None:
            x0 = int(np.clip(pred[0] - self.crop / 2, 0, max(W - self.crop, 0)))
            y0 = int(np.clip(pred[1] - self.crop / 2, 0, max(H - self.crop, 0)))
            more, _ = self._predict(frame[y0:y0 + self.crop, x0:x0 + self.crop], self.crop, x0, y0)
            balls += more
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cands = [[round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(c, 3),
                  round(motion_score(self.prev_gray, gray, (x, y, w, h)), 2)]
                 for x, y, w, h, c in merge_candidates(balls, k=self.top_k)]
        self.prev_gray = gray
        # Online guess (only steers the crop; the final track is chosen offline by balltrack.select_path):
        # the candidate nearest the prediction, else the most confident moving one.
        if cands:
            if pred is not None:
                near = [c for c in cands if np.hypot(c[0] - pred[0], c[1] - pred[1]) < 0.06 * W]
            best = min(near, key=lambda c: np.hypot(c[0] - pred[0], c[1] - pred[1])) if pred is not None and near \
                else max(cands, key=lambda c: c[4] * (0.3 + c[5]))
            if best[4] * (0.3 + best[5]) > 0.05:
                self.history = (self.history + [(self.t, best[0], best[1])])[-2:]
        self.t += 1
        paddle = max(paddles, key=lambda b: b[3] - b[1], default=None)
        return cands, None if paddle is None else [round(v, 1) for v in paddle]

    def detect(self, frame):
        """-> ([x, y] centre of the most confident ball or None, tallest paddle box or None)."""
        cands, paddle = self.candidates(frame)
        return (cands[0][:2] if cands else None), paddle


class PoseEstimator:
    def __init__(self, model="yolov8n-pose.pt"):
        self.model = YOLO(model)

    def keypoints_for(self, frame, bbox):
        """-> 17x2 COCO keypoints of the pose closest to bbox, or None."""
        r = self.model.predict(frame, device=DEVICE, verbose=False)[0]
        if r.keypoints is None or not len(r.boxes):
            return None
        centres = r.boxes.xywh[:, :2].cpu().numpy()
        c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        return r.keypoints.xy[int(np.argmin(np.linalg.norm(centres - c, axis=1)))].cpu().numpy()
