"""Score serve / return / third-shot detection against hand labels from training/annotate_openings.py.

Re-runs the analysis on the labelled clip with the current code. For each labelled shot (serve, return,
third shot) it looks for the reported shot with that role within --tol frames of the labelled contact:
  right          found, by the same person (the reported hitter stood within --same-m metres of the
                 labelled hitter; positions, not ids, so labels survive a re-track)
  wrong_player   found, by someone else
  wrong_role     no shot with that role there, but a reported shot with another role is (e.g. the
                 return reported as a serve)
  hit_only       not reported, but the pipeline did detect a hit there (lost when picking the
                 opening shots: the rally had no detected serve, the return was from the wrong side, ...)
  missed         no hit detected near the contact at all
Untracked hitters (labelled with no box) count only for detection, not for who.
"Not a serve" labels: is a serve still reported within --tol frames of it (false serve)?

Usage: python training/eval_openings.py training/data/openings-jb [more label dirs...] [--run outputs/<clip>] [-v]
"""
import argparse
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402
from pbv import events  # noqa: E402
from pbv.court import Court  # noqa: E402

ROLE = {"serve": "serve", "return": "return", "third shot": "third"}
OUTCOMES = ("right", "wrong_player", "wrong_role", "hit_only", "missed")


def analyse(run):
    """(players, court, opening shots, every detected hit event) for an analysed clip, with the current code."""
    data = json.loads((run / "tracking.json").read_text())
    data.setdefault("ball_offscreen", [False] * len(data["ball"]))
    court = Court.load(run / "court.json")
    _, evs, shots, rallies, _ = app.analyse_fixed(data, court, SimpleNamespace(drive_speed=10.0), openings_only=False)
    opening, _, _ = events.opening_shots(shots, rallies, evs, data["players"], court, data["fps"])
    return data["players"], court, opening, [e for e in evs if e["type"] == "hit"]


def score(labels, players, opening, hits, tol=8, same_m=1.2):
    """-> ({role: Counter(outcome)}, false serves, not-a-serve labels, [(rally key, role, outcome, detail)])."""
    per = {r: collections.Counter() for r in ROLE.values()}
    rows, false_serves, nones = [], 0, 0
    for key, lab in sorted(labels.items(), key=lambda kv: int(kv[0])):
        if lab["kind"] == "none":
            nones += 1
            bad = [s for s in opening if s["role"] == "serve" and abs(s["frame"] - int(key)) <= tol]
            false_serves += bool(bad)
            if bad:
                rows.append((key, "not a serve", "false serve", f"P{bad[0]['player']} f{bad[0]['frame']}"))
            continue
        for shot in lab["shots"]:
            role, c = ROLE[shot["role"]], shot["contact"]
            near = [s for s in opening if abs(s["frame"] - c) <= tol]
            same = [s for s in near if s["role"] == role]
            if same:
                s = min(same, key=lambda s: abs(s["frame"] - c))
                if shot.get("untracked"):
                    out = "right"
                else:
                    got = players[c][s["player"]]["court"] if s["player"] in players[c] else s["hitter"]
                    out = "right" if np.hypot(*np.subtract(got, shot["court"])) <= same_m else "wrong_player"
                detail = f"P{s['player']} {s['frame'] - c:+d}f"
            elif near:
                out, detail = "wrong_role", "reported as " + ", ".join(s["role"] for s in near)
            elif any(abs(e["frame"] - c) <= tol for e in hits):
                out, detail = "hit_only", ""
            else:
                out, detail = "missed", ""
            per[role][out] += 1
            rows.append((key, role, out, f"contact f{c} " + detail))
    return per, false_serves, nones, rows


def report(per, false_serves, nones, indent=""):
    for role, c in per.items():
        n = sum(c.values())
        print(f"{indent}{role:7s} right {c['right']}/{n} ({c['right'] / max(n, 1):.0%})  "
              + "  ".join(f"{o} {c[o]}" for o in OUTCOMES[1:]))
    print(f"{indent}false serves: {false_serves}/{nones} 'not a serve' labels still reported as serves")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path, nargs="+", help="dir(s) with openings.json; several are scored and totalled")
    ap.add_argument("--run", type=Path, help="analysed clip to score (default: the one labelled; one labels dir only)")
    ap.add_argument("--tol", type=int, default=8, help="frames between labelled and reported contact")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every labelled shot that isn't right")
    args = ap.parse_args()
    total = {r: collections.Counter() for r in ROLE.values()}
    tf = tn = 0
    for d in args.labels:
        doc = json.loads((d / "openings.json").read_text())
        players, _, opening, hits = analyse(args.run or Path(doc["meta"]["run"]))
        per, false_serves, nones, rows = score(doc["labels"], players, opening, hits, tol=args.tol)
        for r in per:
            total[r] += per[r]
        tf, tn = tf + false_serves, tn + nones
        if len(args.labels) > 1:
            print(d.name)
        report(per, false_serves, nones, "  " if len(args.labels) > 1 else "")
        if args.verbose:
            for key, role, out, detail in rows:
                if out != "right":
                    print(f"  rally@{key:>6} {role:11s} {out:12s} {detail}")
    if len(args.labels) > 1:
        print("TOTAL")
        report(total, tf, tn, "  ")


if __name__ == "__main__":
    main()
