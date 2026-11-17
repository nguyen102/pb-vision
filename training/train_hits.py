"""Train the learned hit detector (pbv/hits_model.py) and measure rally shot counts honestly.

Labels: rallies-* dirs (training/annotate_rallies.py: every shot of a rally, and the moments after the point
that aren't shots) and openings-* dirs (training/annotate_openings.py: serve, return and third shot only, so
only the stretch up to the last of them is used). Both can label the same clip.

  --cv K   K-fold over the fully labelled rallies: each fold's rallies are left out of training (and any
           openings label overlapping them), then counted with the real pipeline's serves (app.analyse_fixed)
           like training/eval_rallies.py. The decode settings (score threshold, longest gap between shots)
           for each fold are picked on the other folds' held-out rallies. Hand counts (counts.json) with no
           marked shots are only ever tested.
  (default) train on everything and save models/hits.pt, with decode settings picked by --cv if given.

Usage: python training/train_hits.py training/data/rallies-* training/data/openings-* [--cv 5]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
import app  # noqa: E402
from pbv import audio, events  # noqa: E402
from pbv import hits_model as hm  # noqa: E402
from pbv.constants import COURT_W, NET_Y  # noqa: E402
from pbv.court import Court  # noqa: E402
from pbv.players import fill_hidden  # noqa: E402

PREFIXES = (("serve",), ("serve", "return"), ("serve", "return", "third shot"))
GRID = dict(thr=(0.4, 0.5, 0.6), max_gap_s=(2.0, 2.5, 3.0, 3.5), play_thr=(0.0, 0.3, 0.5),
            play_at=(0.0, 0.3, 0.5, 0.7))


class Run:
    """An analysed clip: features, the pipeline's rally starts, and every label on it."""

    def __init__(self, path):
        self.path = path
        data = json.loads((path / "tracking.json").read_text())
        data.setdefault("ball_offscreen", [False] * len(data["ball"]))
        self.fps, self.court = data["fps"], Court.load(path / "court.json")
        cache = path / "openings_feats.npy"
        if cache.exists():
            self.feats = np.load(cache)
        else:
            ball = events.clean_ball(data["ball"], max_jump=0.08 * data["width"], offscreen=data["ball_offscreen"])
            players = fill_hidden(data["players"])
            self.feats = hm.frame_features(ball, events.real_detections(data["ball"], ball), players, self.court,
                                           data["width"], data["height"])
            np.save(cache, self.feats)
        ball = events.clean_ball(data["ball"], max_jump=0.08 * data["width"], offscreen=data["ball_offscreen"])
        self.ball = np.asarray(ball, float)
        # paddle pops (pbv/audio.py), synced on the ball track's own hits as app.py does; zeros without sound
        acache = path / "audio_feats.npy"
        if acache.exists():
            self.audio = np.load(acache)
        else:
            players = fill_hidden(data["players"])
            hits = [e["frame"] for e in events.filter_hits(
                events.find_events(ball, players), self.fps,
                side=lambda e: self.court.side(players[e["frame"]][e["player"]]["court"])) if e["type"] == "hit"]
            video = json.loads((path / "stats.json").read_text())["video"]
            start_s = data.get("params", {}).get("start", 0) / self.fps
            self.audio, off = audio.features(video, start_s, len(self.feats), self.fps, hits)
            print(f"  {path.name}: " + (f"sound, offset {off:+.2f} s" if off is not None else "no sound"), flush=True)
            np.save(acache, self.audio)
        self.has_audio = bool(self.audio[:, -1].any())
        self.formation = hm.serve_formation(fill_hidden(data["players"]), self.fps)
        self.net_px = self.court.to_image([[COURT_W / 2, NET_Y]])[0][1]
        _, _, _, rallies, _ = app.analyse_fixed(data, self.court, SimpleNamespace(drive_speed=10.0))
        self.starts = [r["start"] for r in rallies]  # the pipeline's serves
        self.T = len(self.feats)
        self.full, self.partial, self.counts = [], [], []  # rallies: dicts with window, contacts, sides, n

    def sides(self, shots):
        """Side of the net each shot was hit from: sides alternate, anchored on any shot whose hitter has a box."""
        known = [(i, self.court.side(s["court"])) for i, s in enumerate(shots) if "court" in s]
        if known:
            j, side = known[0]
            flip = {"near": "far", "far": "near"}
            return [side if (i - j) % 2 == 0 else flip[side] for i in range(len(shots))]
        b = self.ball[shots[0]["contact"]]
        first = "far" if not np.isnan(b[1]) and b[1] < self.net_px else "near"
        return [first if i % 2 == 0 else {"near": "far", "far": "near"}[first] for i in range(len(shots))]

    def add_rallies(self, doc):
        fps, starts = self.fps, sorted(self.starts)
        for key, lab in doc["labels"].items():
            if lab["kind"] != "rally":
                continue
            k, cs = int(key), [s["contact"] for s in lab["shots"]]
            nxt = next((s for s in starts if s > k + int(0.5 * fps)), self.T)
            a = max(0, min(k, cs[0]) - int(2 * fps))
            z = min(self.T, max(min(nxt, k + int(15 * fps)), cs[-1] + int(0.5 * fps)))
            self.full.append(dict(win=(a, z), contacts=cs, sides=self.sides(lab["shots"]), serve=cs[0], n=len(cs)))

    def add_openings(self, doc):
        fps = self.fps
        for lab in doc["labels"].values():
            if lab["kind"] != "rally" or tuple(s["role"] for s in lab["shots"]) not in PREFIXES:
                continue
            cs = [s["contact"] for s in lab["shots"]]
            win = (max(0, cs[0] - int(3 * fps)), min(self.T, cs[-1] + int(0.3 * fps)))
            self.partial.append(dict(win=win, contacts=cs, sides=self.sides(lab["shots"])))

    def add_counts(self, rallies):
        self.counts += [dict(serve=c["serve_frame"], n=c["shots"]) for c in rallies
                        if all(abs(c["serve_frame"] - r["serve"]) > 15 for r in self.full)]


def load_runs(dirs):
    runs = {}
    for d in dirs:
        for name in ("rallies.json", "openings.json"):
            if (d / name).exists():
                doc = json.loads((d / name).read_text())
                path = Path(doc["meta"]["run"])
                if not (path / "tracking.json").exists():
                    print(f"skipping {d.name}: {path} is gone", flush=True)
                    continue
                if path not in runs:
                    print("analysing", path.name, flush=True)
                    runs[path] = Run(path)
                (runs[path].add_rallies if name == "rallies.json" else runs[path].add_openings)(doc)
    for d in dirs:
        if (d / "counts.json").exists():
            doc = json.loads((d / "counts.json").read_text())
            runs[Path(doc["meta"]["run"])].add_counts(doc["rallies"])
    for r in runs.values():  # openings labels inside a fully labelled rally add nothing
        r.partial = [p for p in r.partial if not any(overlap(p["win"], f["win"]) for f in r.full)]
    return list(runs.values())


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def targets(run, items):
    """Per-frame hit class, and ball in play (1), not (0) or unknown (-100): in play from the serve to just
    after the last shot; not before the serve or, in a fully labelled rally, from 1 s after the last shot."""
    y = np.zeros(run.T, np.int64)
    play = np.full(run.T, -100, np.int64)
    fps = run.fps
    for it in items:
        cs, (a, z) = it["contacts"], it["win"]
        for c, side in zip(cs, it["sides"]):
            y[max(0, c - 2):c + 3] = 1 if side == "near" else 2
        play[a:max(a, cs[0] - int(0.3 * fps))] = 0
        play[cs[0]:cs[-1] + int(0.3 * fps)] = 1
        if "n" in it:
            play[cs[-1] + int(1.0 * fps):z] = 0
    return np.stack([y, play], 1)


def train(runs, keep, epochs=60, crop=240, batch=32, device="cuda", seed=0, hidden=64, frac=1.0, drop_audio=0.3):
    """keep(run, item) -> whether a labelled rally is trained on."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    items = []
    for r in runs:
        mine = [it for it in r.full + r.partial if keep(r, it)]
        if frac < 1:  # learning curve: a random share of the training rallies
            mine = [it for it in mine if np.random.default_rng(hash(it["win"]) % 2**32).random() < frac]
        y = targets(r, mine)
        items += [(r.inputs, y, *it["win"], r.has_audio) for it in mine]
    hm_audio_n = audio.N_FEATS
    model = hm.ContactNet(runs[0].inputs.shape[1], hidden, n_out=hm.N_OUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    w = torch.tensor([1.0, 10.0, 10.0], device=device)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(items))
        for i in range(0, len(order), batch):
            xs, ys = [], []
            for j in order[i:i + batch]:
                feats, y, a, z, has_audio = items[j]
                lo = rng.integers(max(0, a - crop // 4), max(a - crop // 4, z - crop // 2) + 1)
                lo = min(max(0, lo), max(0, len(feats) - crop))
                x = feats[lo:lo + crop]
                if has_audio and rng.random() < drop_audio:  # as if the video had no sound
                    x = x.copy()
                    x[:, -hm_audio_n:] = 0
                xs.append(x)
                yy = y[lo:lo + crop].copy()
                yy[:max(0, a - lo)] = -100  # outside the labelled stretch: unknown
                yy[max(0, z - lo):] = -100
                ys.append(yy)
            logits = model(torch.from_numpy(np.stack(xs)).to(device))
            t = torch.from_numpy(np.stack(ys)).to(device)
            loss = nn.functional.cross_entropy(logits[..., :3].reshape(-1, 3), t[..., 0].reshape(-1),
                                               weight=w, ignore_index=-100) + \
                nn.functional.cross_entropy(logits[..., 3:].reshape(-1, 2), t[..., 1].reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


_PEAKS = {}


def peaks(prob, fps, thr):
    """hm.decode, cached: the settings search decodes the same scores many times."""
    key = (id(prob), thr)
    if key not in _PEAKS:
        _PEAKS[key] = hm.decode(prob, fps, thr=thr)
    return _PEAKS[key]


def count(run, prob, serve, thr, max_gap_s, play_thr, play_at=0.0, tol=15):
    """The pipeline's count for the rally served at frame serve (0: no detected serve within tol)."""
    starts = sorted(run.starts)
    i = min(range(len(starts)), key=lambda i: abs(starts[i] - serve), default=None)
    if i is None or abs(starts[i] - serve) > tol:
        return 0
    until = starts[i + 1] if i + 1 < len(starts) else run.T
    return len(hm.rally_hits(starts[i], until, peaks(prob, run.fps, thr), run.fps, max_gap_s=max_gap_s,
                             play=prob[:, 3], play_thr=play_thr, play_at=play_at))


def score(cases, params):
    """cases: [(run, prob, serve, n)] -> exact count."""
    return sum(count(r, p, s, **params) == n for r, p, s, n in cases)


def best_params(cases):
    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    return max(grid, key=lambda g: (score(cases, g), -abs(g["thr"] - 0.4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path, nargs="+")
    ap.add_argument("--cv", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--crop", type=int, default=360, help="training crop, frames (12 s: better than 8 s held-out)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--frac", type=float, default=1.0, help="share of training rallies used (learning curve)")
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "hits.pt")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--audio", choices=["on", "off"], default="on", help="off: no sound features (to compare)")
    ap.add_argument("--formation", choices=["on", "off"], default="off",
                    help="serve formation features (hits_model.serve_formation); off: without (to compare)")
    ap.add_argument("--dump", type=Path, help="save each fold's held-out scores here (.npz) for diagnosis")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = load_runs(args.labels)
    for r in runs:  # the model's input: frame features, then the sound's (zeros with --audio off)
        r.inputs = np.concatenate([r.feats] + ([r.formation] if args.formation == "on" else [])
                                  + [r.audio if args.audio == "on" else np.zeros_like(r.audio)], 1)
    for r in runs:
        print(f"{r.path.name}: {len(r.full)} full rallies, {len(r.partial)} openings, {len(r.counts)} counts only")
    params = dict(thr=0.6, max_gap_s=3.0, play_thr=0.0, play_at=0.5)
    if args.cv:
        rng = np.random.default_rng(0)
        tests = [(r, it) for r in runs for it in r.full + r.counts]
        fold = {id(it): f for (_, it), f in zip(tests, rng.permutation(len(tests)) % args.cv)}
        held, dump = [], {}  # (fold, run, prob, serve, n)
        for k in range(args.cv):
            test_wins = [(r, it) for r, it in tests if fold[id(it)] == k and "win" in it]
            keep = lambda r, it: (fold.get(id(it), -1) != k and
                                  not any(r is tr and overlap(it["win"], t["win"]) for tr, t in test_wins))
            models = [train(runs, keep, args.epochs, args.crop, device=device, seed=s, hidden=args.hidden, frac=args.frac) for s in range(args.ensemble)]
            probs = {id(r): np.mean([hm.predict(m, r.inputs, device=device) for m in models], 0) for r in runs}
            held += [(k, r, probs[id(r)], it["serve"], it["n"]) for r, it in tests if fold[id(it)] == k]
            if args.dump:
                dump.update({f"{k}|{r.path}": probs[id(r)] for r in runs})
            print(f"fold {k + 1}/{args.cv} trained", flush=True)
        if args.dump:
            np.savez_compressed(args.dump, folds=json.dumps({str(r.path) + "|" + str(it["serve"]): int(fold[id(it)])
                                                              for r, it in tests}), **dump)
        exact = close = err = 0
        rows = []
        for k in range(args.cv):
            p = best_params([c[1:] for c in held if c[0] != k])
            for _, r, prob, s, n in (c for c in held if c[0] == k):
                got = count(r, prob, s, **p)
                exact, close, err = exact + (got == n), close + (abs(got - n) <= 1), err + abs(got - n)
                rows.append((r.path.name, s / r.fps, n, got, p))
        N = len(held)
        if args.verbose:
            for name, t, n, got, p in sorted(rows, key=lambda x: (x[0], x[1])):
                print(f"  {name[:20]:20} {int(t // 60)}:{t % 60:04.1f}  true {n:2}  model {got:2}  {'' if got == n else '<-'}")
        for name in sorted({x[0] for x in rows}):
            sub = [x for x in rows if x[0] == name]
            print(f"{name}: exact {sum(x[2] == x[3] for x in sub)}/{len(sub)}  "
                  f"within 1 {sum(abs(x[2] - x[3]) <= 1 for x in sub)}/{len(sub)}")
        print(f"HELD-OUT: exact {exact}/{N} ({exact / N:.0%})  within 1 {close}/{N}  mean abs error {err / N:.2f}")
        params = best_params([c[1:] for c in held])
        print("decode settings for the saved model:", params)
    models = [train(runs, lambda r, it: True, args.epochs, args.crop, device=device, seed=s, hidden=args.hidden) for s in range(args.ensemble)]
    torch.save({"states": [m.state_dict() for m in models], "n_in": runs[0].inputs.shape[1], "hidden": args.hidden,
                "classes": hm.CLASSES, "decode": params, "audio": args.audio == "on",
                "formation": args.formation == "on"}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
