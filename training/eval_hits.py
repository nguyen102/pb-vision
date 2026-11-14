"""Score hitter attribution against hand labels from training/annotate_hits.py.

Re-runs the analysis on the labelled clip with the current code, then for each label:
  hit        the predicted hit nearest the labelled contact frame (within --tol frames) must be by the same
             person: the predicted hitter stands within --same-m metres of where the labelled hitter stood.
             Positions, not ids, so labels survive a re-track that renumbers players.
  none       no predicted hit within --tol-none frames of the labelled event (it was rejected)
  untracked  reported only: the hitter had no box, so no attribution could be right

Usage: python training/eval_hits.py training/data/hits-1011 [-v]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_hits import load_analysis  # noqa: E402


def score(labels, players, court, evs, tol=8, tol_none=3, same_m=1.2):
    """-> (counts dict, [(event frame, label, outcome, detail)])."""
    hits = [e for e in evs if e["type"] == "hit"]
    counts = dict.fromkeys(["right", "wrong_player", "wrong_side", "missed", "rejected", "kept", "untracked"], 0)
    rows = []
    for f, lab in sorted(labels.items()):
        if lab["kind"] == "untracked":
            counts["untracked"] += 1
            continue
        if lab["kind"] == "none":
            near = [e for e in hits if abs(e["frame"] - f) <= tol_none]
            out = "kept" if near else "rejected"
            counts[out] += 1
            rows.append((f, "none", out, f"P{near[0]['player']} f{near[0]['frame']}" if near else ""))
            continue
        c = lab["contact"]
        near = [e for e in hits if abs(e["frame"] - c) <= tol]
        if not near:
            counts["missed"] += 1
            rows.append((f, f"P{lab['hitter']}", "missed", ""))
            continue
        e = min(near, key=lambda e: abs(e["frame"] - c))
        got = players[e["frame"]][e["player"]]["court"]
        if np.hypot(*np.subtract(got, lab["court"])) <= same_m:
            out = "right"
        else:
            out = "wrong_side" if court.side(got) != court.side(lab["court"]) else "wrong_player"
        counts[out] += 1
        rows.append((f, f"P{lab['hitter']}", out, f"got P{e['player']} f{e['frame']} (contact f{c})"))
    return counts, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path, help="dir with hits.json (or the file)")
    ap.add_argument("--tol", type=int, default=8, help="frames between labelled contact and predicted hit")
    ap.add_argument("--tol-none", type=int, default=3)
    ap.add_argument("--same-m", type=float, default=1.2)
    ap.add_argument("-v", "--verbose", action="store_true", help="list every mistake")
    args = ap.parse_args()

    path = args.labels / "hits.json" if args.labels.is_dir() else args.labels
    doc = json.loads(path.read_text())
    labels = {int(k): v for k, v in doc["labels"].items()}
    data, court, evs = load_analysis(Path(doc["meta"]["run"]))
    c, rows = score(labels, data["players"], court, evs, args.tol, args.tol_none, args.same_m)

    n_hit = c["right"] + c["wrong_player"] + c["wrong_side"] + c["missed"]
    pct = lambda a, b: f"{a}/{b} ({a / b:.0%})" if b else "0/0"
    print(f"hitter right       {pct(c['right'], n_hit)}")
    print(f"  wrong side       {c['wrong_side']}   wrong partner {c['wrong_player']}   no hit found {c['missed']}")
    print(f"non-hits rejected  {pct(c['rejected'], c['rejected'] + c['kept'])}")
    print(f"hitter untracked   {c['untracked']} (not scored)")
    if args.verbose:
        for f, lab, out, detail in rows:
            if out not in ("right", "rejected"):
                print(f"  f{f:<6} label {lab:5} {out:12} {detail}")


if __name__ == "__main__":
    main()
