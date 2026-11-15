"""Learned serve / return contact detector (trained by training/train_openings.py on hand labels from
training/annotate_openings.py).

Each frame becomes a feature vector: the ball (position, seen or filled in, motion, height relative to the
net) and the four players in fixed slots (near-left, near-right, far-left, far-right by court x): court
position, distance in from their baseline, box, hidden or not (players.fill_hidden), how far the ball is from
them in box heights, where the ball is up their box, and how much their box moved. A small network (temporal
convolutions + a bidirectional GRU over a few seconds) scores every frame as serve contact, return contact or
neither, with the hitter's side of the net for contacts. decode() turns the scores into serves (peaks of the
serve score) and their returns (the peak of the return score shortly after each serve), each with a side.
"""
import numpy as np
import torch
from torch import nn

from .constants import COURT_L, COURT_W, NET_Y
from .events import _box_distance

CLASSES = ("none", "serve-near", "serve-far", "return-near", "return-far")
SLOT_FEATS = 10


def frame_features(ball, detected, players, court, width, height):
    """Per-frame features -> float32 [T, F]. ball: cleaned [T, 2] (NaN = unknown); detected: real detections;
    players: per-frame players with hidden ones filled in (players.fill_hidden)."""
    ball = np.asarray(ball, float)
    T = len(ball)
    known = ~np.isnan(ball[:, 0])
    b = np.nan_to_num(ball)
    net = court.to_image([[COURT_W / 2, NET_Y]])[0][1]
    d = np.zeros_like(b)
    ok = known[1:] & known[:-1]
    d[1:][ok] = (b[1:] - b[:-1])[ok]
    feats = [b[:, 0] / width, b[:, 1] / height, known, np.asarray(detected, float)[:T],
             d[:, 0] * 10 / width, d[:, 1] * 10 / width, np.where(known, (b[:, 1] - net) / height, 0)]
    slots = np.zeros((T, 4, SLOT_FEATS), np.float32)
    prev_top = {}
    for f in range(T):
        fp = players[f]
        for side_i, side in enumerate(("near", "far")):
            mine = sorted(((pid, p) for pid, p in fp.items() if court.side(p["court"]) == side),
                          key=lambda kv: kv[1]["court"][0])[:2]
            if len(mine) == 1:  # one player on that side: left or right slot by which half they are on
                mine = [mine[0], None] if mine[0][1]["court"][0] < COURT_W / 2 else [None, mine[0]]
            for j, item in enumerate(mine):
                if item is None:
                    continue
                pid, p = item
                x1, y1, x2, y2 = p["bbox"]
                h = max(y2 - y1, 1.0)
                cx, cy = p["court"]
                depth = COURT_L - cy if side == "near" else cy
                if known[f]:
                    dx = max(x1 - b[f, 0], 0, b[f, 0] - x2)
                    dy = max(y1 - b[f, 1], 0, b[f, 1] - y2)
                    dist, rel = min(np.hypot(dx, dy) / h, 5.0), np.clip((b[f, 1] - y1) / h, -2, 3)
                else:
                    dist, rel = 5.0, 0.0
                top = ((x1 + x2) / 2, y1)
                pt = prev_top.get(pid, top)
                move = min(np.hypot(top[0] - pt[0], top[1] - pt[1]) / h, 1.0)
                prev_top[pid] = top
                slots[f, side_i * 2 + j] = (1, float(p.get("hidden", False)), cx / COURT_W, depth / (COURT_L / 2),
                                            (x1 + x2) / 2 / width, y1 / height, h / height, dist, rel, move * 10)
    return np.concatenate([np.stack(feats, 1).astype(np.float32), slots.reshape(T, -1)], 1)


class ContactNet(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.inp = nn.Conv1d(n_in, hidden, 5, padding=2)
        self.blocks = nn.ModuleList(nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d) for d in (1, 2, 4, 8, 16))
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, len(CLASSES))

    def forward(self, x):  # x [B, T, F] -> logits [B, T, C]
        h = torch.relu(self.inp(x.transpose(1, 2)))
        for conv in self.blocks:
            h = h + torch.relu(conv(h))
        h, _ = self.gru(h.transpose(1, 2))
        return self.out(h)


def predict(model, feats, chunk=3000, overlap=300, device="cpu"):
    """Per-frame class probabilities [T, C] for a whole clip, in overlapping chunks."""
    model.eval()
    T = len(feats)
    out = np.zeros((T, len(CLASSES)), np.float32)
    with torch.no_grad():
        start = 0
        while start < T:
            a, z = max(0, start - overlap), min(T, start + chunk + overlap)
            x = torch.from_numpy(feats[a:z]).unsqueeze(0).to(device)
            p = torch.softmax(model(x), -1)[0].cpu().numpy()
            out[start:min(T, start + chunk)] = p[start - a:start - a + min(chunk, T - start)]
            start += chunk
    return out


def decode(prob, fps, serve_thr=0.3, return_thr=0.15, nms_s=2.0, ret_s=(0.4, 2.5), smooth=2):
    """Serves (peaks of the serve score above serve_thr, at most one per nms_s) and, for each, its return (the
    peak of the return score ret_s after it, if above return_thr), with the hitter's side of the net (the more
    likely one at the peak) -> [((serve frame, side), (return frame, side) | None)]."""
    k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    sm = lambda v: np.convolve(v, k, "same")
    sn, sf, rn, rf = (sm(prob[:, i]) for i in (1, 2, 3, 4))
    ps, pr = sn + sf, rn + rf
    order = np.argsort(-ps)
    taken, serves = np.zeros(len(ps), bool), []
    for f in order:
        if ps[f] < serve_thr:
            break
        if taken[f]:
            continue
        serves.append(int(f))
        taken[max(0, f - int(nms_s * fps)):f + int(nms_s * fps)] = True
    out = []
    for s in sorted(serves):
        side = "near" if sn[s] >= sf[s] else "far"
        a, z = s + int(ret_s[0] * fps), min(len(pr), s + int(ret_s[1] * fps))
        r = a + int(np.argmax(pr[a:z])) if z > a else None
        ret = (r, "near" if rn[r] >= rf[r] else "far") if r is not None and pr[r] >= return_thr else None
        out.append(((s, side), ret))
    return out


def load(path, device="cpu"):
    """The trained ensemble -> list of models (average their predict() outputs)."""
    ck = torch.load(path, map_location=device)
    models = []
    for state in ck["states"]:
        m = ContactNet(ck["n_in"], ck.get("hidden", 64)).to(device)
        m.load_state_dict(state)
        models.append(m)
    return models


def assign(pairs, players, ball, court, diag_pen=1.0, baseline_m=1.5):
    """Hitters for decoded serves / returns (decode()) -> [{"frame", "role", "player", "hitter", "rally"}].

    The server is a player at their baseline (within baseline_m) on the serve's side, the returner one on the
    other side, chosen together: nearest the ball (in box heights) at their contact, plus diag_pen if they
    stand on the same half (serves go cross-court). players: with hidden ones filled in (players.fill_hidden).
    """
    ball = np.asarray(ball, float)
    half = lambda p: p["court"][0] < COURT_W / 2

    def cands(f, side, back=None):
        c = {k: p for k, p in players[f].items() if court.side(p["court"]) == side and (
            back is None or (COURT_L - p["court"][1] if side == "near" else p["court"][1]) <= back)}
        return c or ({k: p for k, p in players[f].items() if court.side(p["court"]) == side} if back else c)

    dist = lambda f, p: 0.0 if np.isnan(ball[f, 0]) else \
        _box_distance(ball[f], p["bbox"]) / max(p["bbox"][3] - p["bbox"][1], 1.0)
    out = []
    for n, ((s, side), ret) in enumerate(pairs, 1):
        cs = cands(s, side, baseline_m)
        if not cs:
            continue
        cr = cands(ret[0], "far" if side == "near" else "near") if ret else {}
        if cr:
            a, b = min(((a, b) for a in cs for b in cr), key=lambda ab: dist(s, cs[ab[0]]) + dist(ret[0], cr[ab[1]])
                       + (diag_pen if half(cs[ab[0]]) == half(cr[ab[1]]) else 0))
        else:
            a, b = min(cs, key=lambda k: dist(s, cs[k])), None
        out.append({"frame": s, "role": "serve", "player": a, "hitter": players[s][a]["court"], "rally": n})
        if b is not None:
            out.append({"frame": ret[0], "role": "return", "player": b, "hitter": players[ret[0]][b]["court"],
                        "rally": n})
    return out


def detect(models, ball, detected, players, court, fps, width, height, device="cpu", **dec):
    """Serves and returns from the trained ensemble -> (serves for events.merge_serves, return hit events).
    players: with hidden ones filled in."""
    feats = frame_features(ball, detected, players, court, width, height)
    prob = np.mean([predict(m, feats, device=device) for m in models], 0)
    shots = assign(decode(prob, fps, **dec), players, ball, court)
    px = lambda f: [round(float(v), 1) for v in np.nan_to_num(np.asarray(ball[f], float))]
    serves = [{"frame": s["frame"], "player": s["player"], "px": px(s["frame"])} for s in shots if s["role"] == "serve"]
    returns = [{"frame": s["frame"], "type": "hit", "player": s["player"], "px": px(s["frame"]), "model_return": True}
               for s in shots if s["role"] == "return"]
    return serves, returns
