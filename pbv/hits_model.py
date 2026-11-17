"""Learned hit detector for rally shot counts (trained by training/train_hits.py on hand labels from
training/annotate_rallies.py, every shot of whole rallies, and training/annotate_openings.py, the first three).

Same per-frame features and network as the serve / return detector (openings_model.frame_features,
ContactNet), with three classes: no hit, hit from the near side, hit from the far side. decode() takes the
peaks of the hit score; rally_hits() counts a rally as its serve plus the run of hits after it, stopping
at the next serve, when no hit follows for max_gap_s, or when the ball is out of play: a second output
scores each frame as ball in play or not (trained on the moments after the last marked shot, so later
ball handling isn't counted as a shot).
"""
import numpy as np

from .openings_model import ContactNet, frame_features  # noqa: F401  (re-exported)

CLASSES = ("none", "hit-near", "hit-far")
N_OUT = len(CLASSES) + 2  # + ball in play / not (the point is over, or not started)


def load(path, device="cpu"):
    """-> (ensemble, decode settings, whether its input includes pbv.audio features after the frame features)."""
    import torch
    ck = torch.load(path, map_location=device)
    models = []
    for state in ck["states"]:
        m = ContactNet(ck["n_in"], ck.get("hidden", 64), N_OUT).to(device)
        m.load_state_dict(state)
        models.append(m)
    return models, ck.get("decode", {}), ck.get("audio", False)


def predict(model, feats, chunk=3000, overlap=300, device="cpu"):
    """-> [T, 4]: P(no hit), P(hit from the near side), P(hit from the far side), P(ball in play)."""
    import torch
    model.eval()
    T = len(feats)
    out = np.zeros((T, 4), np.float32)
    with torch.no_grad():
        for start in range(0, T, chunk):
            a, z = max(0, start - overlap), min(T, start + chunk + overlap)
            logits = model(torch.from_numpy(feats[a:z]).unsqueeze(0).to(device))[0]
            p = torch.cat([torch.softmax(logits[:, :3], -1), torch.softmax(logits[:, 3:], -1)[:, 1:]], -1).cpu().numpy()
            n = min(chunk, T - start)
            out[start:start + n] = p[start - a:start - a + n]
    return out


def decode(prob, fps, thr=0.3, nms_s=0.3, smooth=1):
    """Peaks of the hit score above thr, at most one per nms_s -> [(frame, side, score)] in frame order."""
    k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    near, far = (np.convolve(prob[:, i], k, "same") for i in (1, 2))
    p = near + far
    taken, out = np.zeros(len(p), bool), []
    for f in np.argsort(-p):
        if p[f] < thr:
            break
        if taken[f]:
            continue
        out.append((int(f), "near" if near[f] >= far[f] else "far", float(p[f])))
        taken[max(0, f - int(nms_s * fps)):f + int(nms_s * fps) + 1] = True
    return sorted(out)


def rally_hits(start, until, peaks, fps, serve_s=0.3, max_gap_s=3.0, play=None, play_thr=0.0):
    """Hit frames of the rally served at frame start (the next serve at until): the serve, then each peak
    after it while the gap to the previous hit is at most max_gap_s and the ball is in play (play: the
    in-play score per frame, smoothed over the stretch since the last hit)."""
    hits = [start]
    for f, _, _ in peaks:
        if f <= start + serve_s * fps:
            continue
        if f >= until or f - hits[-1] > max_gap_s * fps:
            break
        if play is not None and play_thr > 0 and play[hits[-1]:f + 1].mean() < play_thr:
            break
        hits.append(f)
    return hits


def detect(models, ball, detected, players, court, width, height, device="cpu", feats=None):
    """Per-frame scores [T, 4] (predict()) from the trained ensemble. players: with hidden ones filled in."""
    if feats is None:
        feats = frame_features(ball, detected, players, court, width, height)
    return np.mean([predict(m, feats, device=device) for m in models], 0)
