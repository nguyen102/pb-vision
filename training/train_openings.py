"""Train the learned serve / return contact detector (pbv/openings_model.py) on hand labels from
training/annotate_openings.py, and measure it honestly.

  --cv     leave-one-clip-out: for each labelled clip, train on the others, predict the held-out clip and
           score it like training/eval_openings.py (serve / return found within --tol frames, by the right
           person). Reports per clip and in total.
  (default) train on every labelled clip and save models/openings.pt.

Training uses only the stretches of video that were labelled (each rally from 3 s before its serve to 3 s
after its last labelled shot, and "not a serve" moments), since the rest of a clip has serves nobody marked.
Per-frame features are cached next to each clip as openings_feats.npy.

Usage: python training/train_openings.py training/data/openings-* [--cv] [--epochs 60]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from pbv import events  # noqa: E402
from pbv import openings_model as om  # noqa: E402
from pbv.constants import COURT_L, COURT_W, NET_Y  # noqa: E402
from pbv.court import Court  # noqa: E402
from pbv.players import fill_hidden  # noqa: E402
import eval_openings as EO  # noqa: E402

ROLE_CLASS = {("serve", "near"): 1, ("serve", "far"): 2, ("return", "near"): 3, ("return", "far"): 4}


def load_clip(label_dir):
    """(features [T, F], players (filled), seen players, court, fps, labels, ball, net_px) for a labelled clip."""
    doc = json.loads((label_dir / "openings.json").read_text())
    run = Path(doc["meta"]["run"])
    data = json.loads((run / "tracking.json").read_text())
    court = Court.load(run / "court.json")
    off = data.get("ball_offscreen") or [False] * len(data["ball"])
    ball = events.clean_ball(data["ball"], max_jump=0.08 * data["width"], offscreen=off)
    players = fill_hidden(data["players"])
    cache = run / "openings_feats.npy"
    if cache.exists():
        feats = np.load(cache)
    else:
        detected = events.real_detections(data["ball"], ball)
        feats = om.frame_features(ball, detected, players, court, data["width"], data["height"])
        np.save(cache, feats)
    net_px = court.to_image([[COURT_W / 2, NET_Y]])[0][1]
    return dict(feats=feats, players=players, seen=data["players"], court=court, fps=data["fps"],
                labels=doc["labels"], ball=np.asarray(ball, float), net_px=net_px, name=label_dir.name)


def shot_side(clip, lab, i):
    """Side of the net a labelled shot was hit from: the hitter's, else (no box: "m") across from the other
    labelled shot of the rally (serve and return are on opposite sides), else where the ball was."""
    court, shots = clip["court"], lab["shots"]
    s = shots[i]
    if "court" in s:
        return court.side(s["court"])
    for j, o in enumerate(shots[:2]):
        if j != i and "court" in o:
            return "far" if court.side(o["court"]) == "near" else "near"
    b = clip["ball"][s["contact"]]
    return "far" if not np.isnan(b[1]) and b[1] < clip["net_px"] else "near"


def windows_and_targets(clip):
    """Labelled stretches [(a, z)] and per-frame targets (0 none / 1 serve / 2 return) over the clip."""
    T, fps = len(clip["feats"]), clip["fps"]
    y = np.zeros(T, np.int64)
    wins = []
    for key, lab in clip["labels"].items():
        k = int(key)
        if lab["kind"] == "none":
            wins.append((max(0, k - int(2 * fps)), min(T, k + int(4 * fps))))
            continue
        cs = [s["contact"] for s in lab["shots"]]
        wins.append((max(0, min(cs + [k]) - int(3 * fps)), min(T, max(cs) + int(3 * fps))))
        for i, s in enumerate(lab["shots"]):
            if s["role"] not in ("serve", "return"):
                continue
            side = shot_side(clip, lab, i)
            y[max(0, s["contact"] - 2):s["contact"] + 3] = ROLE_CLASS[(s["role"], side)]
    return wins, y


def train(clips, epochs=60, crop=240, batch=32, device="cuda", seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    items = []
    for c in clips:
        wins, y = windows_and_targets(c)
        items += [(c["feats"], y, a, z) for a, z in wins]
    n_in = clips[0]["feats"].shape[1]
    model = om.ContactNet(n_in).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0] + [15.0] * 4, device=device))
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(items))
        for i in range(0, len(order), batch):
            xs, ys = [], []
            for j in order[i:i + batch]:
                feats, y, a, z = items[j]
                # a crop of the window, shifted at random (the model mustn't learn where the window starts)
                lo = rng.integers(max(0, a - crop // 4), max(a - crop // 4, z - crop // 2) + 1)
                lo = min(max(0, lo), max(0, len(feats) - crop))
                xs.append(feats[lo:lo + crop])
                yy = y[lo:lo + crop].copy()
                # frames outside the labelled stretch are unknown: ignored by the loss
                yy[:max(0, a - lo)] = -100
                yy[max(0, z - lo):] = -100
                ys.append(yy)
            x = torch.from_numpy(np.stack(xs)).to(device)
            t = torch.from_numpy(np.stack(ys)).to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), t.reshape(-1),
                                               weight=loss_fn.weight, ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def hitters(clip, pairs):
    return om.assign(pairs, clip["players"], clip["ball"], clip["court"])


def score_clip(clip, models, tol, device, **dec):
    prob = np.mean([om.predict(m, clip["feats"], device=device) for m in models], 0)
    pairs = om.decode(prob, clip["fps"], **dec)
    opening = hitters(clip, pairs)
    return EO.score(clip["labels"], clip["seen"], opening, [], tol=tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path, nargs="+")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tol", type=int, default=8)
    ap.add_argument("--ensemble", type=int, default=3, help="models trained with different seeds, scores averaged")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "openings.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clips = [load_clip(d) for d in args.labels]
    print("clips:", ", ".join(f"{c['name']} {c['feats'].shape}" for c in clips), flush=True)
    if args.cv:
        import collections
        total = {r: collections.Counter() for r in ("serve", "return")}
        tf = tn = 0
        for i, held in enumerate(clips):
            models = [train([c for j, c in enumerate(clips) if j != i], args.epochs, device=device, seed=k)
                      for k in range(args.ensemble)]
            per, fs, n, _ = score_clip(held, models, args.tol, device)
            for r in total:
                total[r] += per[r]
            tf, tn = tf + fs, tn + n
            print(held["name"], {r: f"{per[r]['right']}/{sum(per[r].values())}" for r in total}, f"false {fs}/{n}",
                  flush=True)
        print("TOTAL (held-out)")
        EO.report({**total, "third": collections.Counter()}, tf, tn, "  ")
        return
    models = [train(clips, args.epochs, device=device, seed=k) for k in range(args.ensemble)]
    torch.save({"states": [m.state_dict() for m in models], "n_in": clips[0]["feats"].shape[1], "hidden": 64,
                "classes": om.CLASSES}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
