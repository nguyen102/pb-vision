# Ball-tracking training — 2026-11-13

## Goal

Track the pickleball more reliably in fixed-camera video. The original pipeline ran a generic COCO
"sports ball" detector and took the single best box per frame. It found the ball in about 62% of
frames, and about 44% of the points it tracked were not the ball.

## What we did

1. **Better tracking logic** (`pbv/balltrack.py`)
   - The detector keeps its top few candidates per frame instead of only the best one.
   - Candidates are linked into short tracks over time. The tracker picks the track that moves like a
     ball, rather than the most confident single box.
   - Spots where an object never moves (a rack of spare balls) are ignored.
   - A court check uses the ball's apparent size to work out its depth, which rejects balls on
     neighbouring courts.
   - The tracker detects a ball leaving the frame (lobs, balls hit out). That shows up as "off-screen",
     not as a lost ball, so it doesn't split a rally.

2. **Higher-resolution detection**
   - The detector runs at full 1920 px resolution. A far ball is only about 5 px wide, which is too
     small to find at lower resolution.
   - It also makes a second, full-resolution pass on a crop around where the ball is expected.
   - A simple motion cue from frame differencing helps tell a moving ball from static objects.

3. **Training data from our own video** (self-training, round 0)
   - Scanned a 2.5-hour training video (11-4 session) with the COCO model as a "teacher".
   - Turned its ball tracks into labels automatically, filling short gaps where it missed the ball.
   - Built a dataset of about 11,000 full-resolution 640×640 crops: balls, plus hard negatives such as
     heads, shoes and lights.

4. **Trained our own detector** (round 1, YOLOv8s)
   - Trained on that dataset. The run froze partway through, a GPU hang under WSL, so we used its best
     saved checkpoint (around epoch 22 of 40).

5. **Hand-labelled test set**
   - 150 random frames from a different day's video (10-11 session), labelled by hand with
     `training/annotate.py`. They measure accuracy honestly, including balls every model misses.

6. **Reliability fixes**
   - Long scans can be resumed and run under a watchdog that restarts them if they stall.

## Results so far (150 hand-labelled test frames)

| Setup | Ball tracked | Tracked points that are wrong |
|---|---|---|
| Original (COCO model, old tracker) | 62% | 44% |
| COCO model + new tracker | 72% | 27% |
| **Our round-1 model + new tracker** | **80%** | **23%** |

| Our round-2 model + new tracker | 76% | 29% |

The round-1 detector sees the ball in about 92% of test frames. The tracker keeps only about 80%, so
the tracker now loses more than the detector does.

### Round 2 (self-training, done the same evening)

- Rescanned the training video with the round-1 model. It found 2,196 ball tracks, up from 1,582.
- Built a bigger dataset (12,347 train / 976 val crops) and trained round 2 for 38 epochs, about 1.3 h,
  with no freezes. Training ran under a watchdog that resumes from the last checkpoint if it hangs,
  and its progress is written to `training/logs/STATUS.txt`.
- **Result: no clear gain over round 1.**
  - On the 150 hand-labelled frames it's slightly behind: 76% vs 80% tracked. That is only 5 frames,
    and 3 of them miss by just 10–25 px, so the gap is mostly noise.
  - On the larger automatic reference (all 12,000 test frames) it's slightly ahead: 90% vs 88%.
  - It's more confident about the real ball, but also about other ball-like objects. Many are balls on
    other courts, which the labels deliberately include. That raises the share of wrong tracked points.
- **Decision for now:** keep round 1 as the default model.

## Next steps (not done yet)

1. **Make round 2 pay off.** Two options:
   - Teach the model only balls on our court.
   - Leave the model as it is and make the tracker better at ignoring other courts.
   A bigger hand-labelled set (300+ frames) is also needed to tell close models apart.
2. **Close the detector-to-tracker gap.** Find out why the tracker drops detections the detector got
   right (track linking, gating and the court check), and tune it.
3. **Run the full app** on the 10-11 clip with the best model. Compare hits, bounces and rallies
   before and after.
4. **Update the README** with the measured accuracy.
5. **Optional follow-ups:**
   - Label the remaining 50 test frames.
   - Train a TrackNet-style model that looks at several frames at once, which helps with blurred
     balls.
   - Add more training videos, such as other courts and lighting.
