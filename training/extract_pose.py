"""Pose keypoints for every tracked player in every frame of an analysed run -> RUN/pose.npz (pbv/pose.py), used
by hits_model.pose_features (training/train_hits.py --pose on). app.py makes the same file itself when the hit
model needs it; this is for runs analysed before that.

  pose.npz: kp [T, P, 17, 3] float16 (x, y in frame pixels, confidence; zeros when missing), ids [P]

Usage: python training/extract_pose.py outputs/RUN [outputs/RUN ...] [--model yolov8n-pose.pt]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pbv import pose  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--model", default=str(ROOT / "yolov8n-pose.pt"))
    args = ap.parse_args()
    for run in args.runs:
        data = json.loads((run / "tracking.json").read_text())
        video = json.loads((run / "stats.json").read_text())["video"]
        progress = lambda r: tqdm(r, desc=run.name, mininterval=10)
        kp, _ = pose.load_or_extract(run / "pose.npz", video, data["players"], data.get("params", {}).get("start", 0),
                                     args.model, progress=progress)
        print(f"{run / 'pose.npz'}: {(kp[..., 2].astype(float).mean(-1) > 0.3).mean():.0%} of player-frames with a pose",
              flush=True)


if __name__ == "__main__":
    main()
