"""Score how consistently players keep their ids, against hand labels of who is who.

Labels (training/data/ids-*/labels.json): for sampled frames, a string with one letter per person (one letter
per real person, e.g. "TBKP"; "?" = can't tell) for the raw tracker boxes inside the play area at that frame,
in left-to-right order of their left edge. Re-runs pbv.players.select_players on the clip's cached people and
clothing descriptors (tracking.json + appearance.npy, no video pass), maps each person to the id they carry
most often (one id per person), and reports how many labelled boxes carry their person's id.

Usage: python training/eval_ids.py training/data/ids-jb/labels.json [-v]
"""
import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbv.appearance import attach_appearance  # noqa: E402
from pbv.court import Court  # noqa: E402
from pbv.players import PLAYER_IDS, select_players  # noqa: E402


def load(run):
    """tracking.json of an analysed clip with its clothing descriptors attached to the people."""
    data = json.loads((ROOT / run / "tracking.json").read_text())
    attach_appearance(data["people"], np.load(ROOT / run / "appearance.npy"))
    return data


def score(labels, people, players):
    """-> (right, labelled, without an id, {person: id}, [(frame, person, wrong id)])."""
    pairs = []
    for f, who in labels.items():
        f = int(f)
        boxes = sorted((p for p in people[f] if Court.in_play_area(p["court"])), key=lambda p: p["bbox"][0])
        for w, p in zip(who, boxes):
            if w != "?":
                pid = next((k for k, v in players[f].items() if np.allclose(v["bbox"], p["bbox"])), None)
                pairs.append((w, pid, f))
    persons = sorted({w for w, _, _ in pairs})
    count = collections.Counter((w, i) for w, i, _ in pairs if i)
    best = max(itertools.permutations(PLAYER_IDS, len(persons)),
               key=lambda ids: sum(count[(w, i)] for w, i in zip(persons, ids)))
    mapping = dict(zip(persons, best))
    right = sum(i == mapping[w] for w, i, _ in pairs)
    wrong = [(f, w, i) for w, i, f in pairs if i is not None and i != mapping[w]]
    return right, len(pairs), sum(i is None for _, i, _ in pairs), mapping, wrong


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true", help="show the id mapping and every wrong box")
    args = ap.parse_args()
    doc = json.loads(args.labels.read_text())
    data = load(doc["run"])
    right, n, none, mapping, wrong = score(doc["labels"], data["people"], select_players(data["people"], data["fps"]))
    print(f"id right {right}/{n} ({right / n:.0%}), no id {none}")
    if args.verbose:
        print("  person -> id:", mapping)
        for f, w, i in wrong:
            print(f"  f{f}: {w} ({doc['people'].get(w, '')}) has id {i}")


if __name__ == "__main__":
    main()
