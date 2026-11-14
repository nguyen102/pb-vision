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

A tracked point counts as right when it's on the ball: within 10 px of the click, or within 0.6× the
ball's size for a close or blurred ball. Until the tracker work (below) the radius was a flat 10 px,
which counted points 12–15 px from the click on a 25 px ball as misses. That's why these numbers are
a few points higher than the first ones reported.

| Setup | Ball tracked | Tracked points that are wrong | Detector sees the ball |
|---|---|---|---|
| Original (COCO model, old tracker) | 66% | 41% | 78% |
| COCO model + new tracker | 77% | 22% | 78% |
| **Our round-1 model + new tracker (current default)** | **85%** | **18%** | **96%** |
| Our round-2 model + new tracker | 79% | 26% | 96% |

### Round 2 (self-training, done the same evening)

- Rescanned the training video with the round-1 model. It found 2,196 ball tracks, up from 1,582.
- Built a bigger dataset (12,347 train / 976 val crops) and trained round 2 for 38 epochs, about 1.3 h,
  with no freezes. Training ran under a watchdog that resumes from the last checkpoint if it hangs,
  and its progress is written to `training/logs/STATUS.txt`.
- **Result: no gain over round 1.**
  - On the hand-labelled frames it's behind: 79% vs 85% tracked.
  - On the larger automatic reference (all 12,000 test frames) it's slightly ahead: 90% vs 88%.
    That reference is partly built from round 2's own detections, so it's biased towards round 2.
  - It's more confident about the real ball, but also about other ball-like objects. Many are balls on
    other courts, which the labels deliberately include. That raises the share of wrong tracked points.
- **Decision:** round 1 is the default (`models/ball-r1.pt`). It was committed and pushed to `main`.
  Round 2 is kept locally as `models/ball-r2.pt`.

### Tracker work

`training/diagnose.py` follows the labelled ball through each tracker stage and reports where it's
lost. For the round-1 model on the 132 hand-labelled ball frames:

| Where the ball is lost | Frames |
|---|---|
| Tracked correctly | 112 (85%) |
| Another track took the frame | 9 (7%) |
| Dropped as a static hotspot | 7 (5%) |
| Detector missed it | 4 (3%) |

On the 18 frames with no ball in play, the tracker reports a ball in 8.

What the losses turned out to be:

- **Ball in the server's hand.** About half of the "another track" frames are the near player holding
  or dropping the ball to serve, sometimes holding two balls. The tracker picks the other ball, 15 px
  lower. Both are real balls, so this is a labelling ambiguity rather than a tracker bug.
- **Balls in the background.** The rest, and most of the false points on no-ball frames, are tracks of
  balls behind the far court, near the back wall or on other courts. From this low camera, those
  balls land on the same pixels as a ball about 1 m above our far baseline, at almost the same size. So
  the court check can't reject them: tightening its height limit (8 → 4.5 m) or size tolerance
  (2.6 → 2.0×) changed nothing. Only how far behind the baseline counts (3 → 1.5 m) helped slightly.
- **Static hotspots.** The ball crosses the middle of the court, over the net, so often that some of
  those spots are marked as fixed objects, and the real ball is dropped there. We tried three fixes:
  - Build hotspots only from still detections: worse.
  - Keep a moving detection inside a hotspot: recovers the ball, but on the 12,000-frame reference
    wrong points rise from 21% to 36%, because people moving near the ball rack come back.
  - Drop only tracks that stay mostly inside hotspots: +2% on the hand labels but −3.6% recall on the
    12,000-frame reference.
  None was a clear win, so the tracker code is unchanged (reverted).
- **Scoring fix kept:** the hit radius now scales with the ball's size (`evaluate.py`, `diagnose.py`).

**Conclusion:** simple settings changes don't move the tracker further. The remaining errors need a
structural change, plus a bigger test set to measure it. With 132 ball frames, one frame is 0.8%,
so the 2–3% changes above can't be told apart from noise.

## Next steps (not done yet)

1. **Label more test frames** (300–500) with `training/annotate.py`, so tracker changes of a few
   percent can be measured. Skip frames where the ball is in the server's hand, or add a label for them.
2. **Path selection with continuity.** Today the best-scoring track claims its frames first, one track
   at a time. A background ball's track can therefore win a frame even though it doesn't connect to
   our ball before or after. Instead, pick the chain of tracks where each one starts where the previous
   one ended (a Viterbi / shortest-path search over tracks). This should fix both the background-ball
   and the hotspot losses.
3. **Use the players.** A ball in play is hit by our players and crosses our net. Tracks that never come
   near our players could be scored down.
4. **Teach the model only our court's ball** (round 3), or add more training videos. The full 140-min
   10-11 session is available, but its section that became the test clip would have to be left out.
5. **Run the full app** on the 10-11 clip with the round-1 model, and compare hits, bounces and rallies
   before and after.
6. **Update the README** with the measured accuracy.
7. **Optional:** a TrackNet-style model that looks at several frames at once, which helps with blurred
   balls.
