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

from .openings_model import SLOT_FEATS, ContactNet, frame_features  # noqa: F401  (re-exported)

CLASSES = ("none", "hit-near", "hit-far")
N_OUT = len(CLASSES) + 2  # + ball in play / not (the point is over, or not started)


def load(path, device="cpu"):
    """-> (ensemble, decode settings, inputs): inputs says which extra features follow the frame features, in
    this order: {"formation": serve_formation(), "audio": pbv.audio features}."""
    import torch
    ck = torch.load(path, map_location=device)
    models = []
    for state in ck["states"]:
        m = ContactNet(ck["n_in"], ck.get("hidden", 64), N_OUT, gru_layers=ck.get("gru_layers", 1),
                       dilations=tuple(ck.get("dilations", (1, 2, 4, 8, 16)))).to(device)
        m.load_state_dict(state)
        models.append(m)
    return models, ck.get("decode", {}), {"formation": ck.get("formation", False), "audio": ck.get("audio", False)}


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


def rally_hits(start, until, peaks, fps, serve_s=0.3, max_gap_s=3.0, play=None, play_thr=0.0, play_at=0.0,
               alternate=False, serve_side=None):
    """Hit frames of the rally served at frame start (the next serve at until): the serve, then each peak
    after it while the gap to the previous hit is at most max_gap_s and the ball is in play (play: the
    in-play score per frame; its mean over the stretch since the last hit must reach play_thr, and its value
    at the hit itself play_at: after the point, ball handling peaks come with a low in-play score).
    alternate: shots alternate sides of the net, so a peak on the same side as the previous hit replaces it (the
    later of two same-side peaks is kept) instead of adding a shot; serve_side: the serve's side, if known."""
    hits, sides = [start], [serve_side]
    for f, side, _ in peaks:
        if f <= start + serve_s * fps:
            continue
        if f >= until or f - hits[-1] > max_gap_s * fps:
            break
        if play is not None and play_thr > 0 and play[hits[-1]:f + 1].mean() < play_thr:
            break
        if play is not None and play_at > 0 and play[f] < play_at:
            break
        if alternate and side == sides[-1] and len(hits) > 1:
            hits[-1] = f
            continue
        hits.append(f)
        sides.append(side)
    return hits


def detect(models, ball, detected, players, court, width, height, device="cpu", feats=None):
    """Per-frame scores [T, 4] (predict()) from the trained ensemble. players: with hidden ones filled in."""
    if feats is None:
        feats = frame_features(ball, detected, players, court, width, height)
    return np.mean([predict(m, feats, device=device) for m in models], 0)


FORMATION_FEATS = 2
BASE_M, KITCHEN_M = 1.5, 1.5  # "at the baseline" / "at the kitchen line": within this many metres of it


def serve_formation(players, fps, hold_s=1.0):
    """Per frame -> float32 [T, 2]: whether the players stand as they do before a serve (one team both at their
    baseline, the other with one player at the kitchen line and one at the baseline), and how long that has held
    (up to hold_s, as a share of it). Before a serve this holds in ~90% of the labelled rallies; it also comes
    back once a point is over and players get ready for the next, which tells the in-play score the point ended.
    players: with hidden ones filled in."""
    from .constants import COURT_L, KITCHEN, NET_Y
    kitchen = NET_Y - KITCHEN  # the kitchen line's distance in from a baseline
    base = lambda d: d <= BASE_M
    kit = lambda d: abs(d - kitchen) <= KITCHEN_M
    out = np.zeros((len(players), FORMATION_FEATS), np.float32)
    run = 0
    for f, fp in enumerate(players):
        near = sorted(COURT_L - p["court"][1] for p in fp.values() if p["court"][1] > NET_Y)
        far = sorted(p["court"][1] for p in fp.values() if p["court"][1] <= NET_Y)
        ok = len(near) == 2 and len(far) == 2 and (
            (base(near[0]) and base(near[1]) and base(far[0]) and kit(far[1])) or
            (base(far[0]) and base(far[1]) and base(near[0]) and kit(near[1])))
        run = run + 1 if ok else 0
        out[f] = (float(ok), min(run / (hold_s * fps), 1.0))
    return out
