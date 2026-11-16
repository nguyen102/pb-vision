"""Score rally shot counts against hand counts.

Re-runs the analysis on the labelled clip with the current code and compares each hand-counted rally's
shot count (serve included) with the reported rally starting within --tol frames of its serve.

Usage: python training/eval_rallies.py training/data/rallies-1011 [--run outputs/<clip>] [-v]
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app  # noqa: E402
from pbv.court import Court  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path, help="dir with counts.json (or the file)")
    ap.add_argument("--run", type=Path, help="analysed clip to score (default: the one labelled)")
    ap.add_argument("--tol", type=int, default=15, help="frames between labelled and reported serve")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every rally")
    args = ap.parse_args()
    doc = json.loads((args.labels / "counts.json" if args.labels.is_dir() else args.labels).read_text())
    run = args.run or Path(doc["meta"]["run"])
    data = json.loads((run / "tracking.json").read_text())
    data.setdefault("ball_offscreen", [False] * len(data["ball"]))
    _, _, _, rallies, _ = app.analyse_fixed(data, Court.load(run / "court.json"), SimpleNamespace(drive_speed=10.0))

    truth, got = [], []
    for lab in doc["rallies"]:
        r = min(rallies, key=lambda r: abs(r["start"] - lab["serve_frame"]), default=None)
        n = r["shots"] if r is not None and abs(r["start"] - lab["serve_frame"]) <= args.tol else 0
        truth.append(lab["shots"])
        got.append(n)
        if args.verbose:
            print(f"  {lab['time']:>7}  true {lab['shots']:2}  reported {n:2}  {'' if n else '(no rally found)'}")
    truth, got = np.array(truth), np.array(got)
    err = np.abs(got - truth)
    print(f"exact {int((err == 0).sum())}/{len(truth)}   within 1 {int((err <= 1).sum())}/{len(truth)}   "
          f"mean abs error {err.mean():.2f}   total {got.sum()} (true {truth.sum()})")


if __name__ == "__main__":
    main()
