"""Paddle-pop audio features for the hit detector (pbv/hits_model.py).

A paddle hitting the ball makes a sharp pop: a sudden rise in sound energy, strongest at high frequencies. The
soundtrack is cut into 5 ms steps; for each of four frequency bands we take the rise in log energy from one
step to the next (onset strength), scaled by the recording's own loud onsets so recordings of different loudness
compare. Each video frame gets the strongest onset per band within its 1/30 s, plus the high-band level and a
flag saying there is sound at all (videos without a soundtrack get zeros and flag 0, and the detector falls back
on the picture).

Pops from other courts are kept: the detector learns to trust a pop only when the picture also shows a player
who could be hitting the ball then.

The soundtrack can be out of sync with the picture (by about 0.26 s in 10-11-BBB-mb-tim-cut): estimate_offset()
finds the delay at which onsets line up best with hits seen in the ball track, no labels needed.
"""
import shutil
import subprocess

import numpy as np

SR, WIN, HOP = 24000, 512, 120  # 21 ms window, 5 ms hop: 200 onset steps a second
RATE = SR / HOP
BANDS = ((300, 1000), (1000, 2500), (2500, 5000), (5000, 11000))
N_FEATS = len(BANDS) + 2


def load(video, start_s=0.0):
    """Mono soundtrack at SR from start_s -> float32 samples, or None if the video has none (or no ffmpeg)."""
    if not shutil.which("ffmpeg"):
        return None
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                            "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    if not probe.stdout.strip():
        return None
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start_s), "-i", str(video), "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    x = np.frombuffer(out.stdout, np.float32)
    return x if len(x) > SR else None


def onsets(x):
    """-> (level [n, bands], onset [n, bands]) at RATE steps a second: log band energy and its rise."""
    n = (len(x) - WIN) // HOP
    win = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    masks = [(freqs >= a) & (freqs < b) for a, b in BANDS]
    level = np.empty((n, len(BANDS)), np.float32)
    for a in range(0, n, 20000):  # in chunks: the full spectrogram of a long match won't fit in memory
        idx = np.arange(WIN)[None, :] + HOP * np.arange(a, min(n, a + 20000))[:, None]
        spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        level[a:a + len(idx)] = np.log(np.stack([spec[:, m].sum(1) for m in masks], 1) + 1e-9)
    onset = np.maximum(np.diff(level, axis=0, prepend=level[:1]), 0)
    return level, onset


def estimate_offset(onset, hit_frames, fps, search_s=(-0.3, 0.6)):
    """Seconds the soundtrack runs late (positive) or early: the delay at which the high-band onset averaged over
    hit_frames (hits seen in the ball track, some of them wrong) is strongest."""
    hi = onset[:, -1]
    lags = np.arange(int(search_s[0] * RATE), int(search_s[1] * RATE) + 1)
    at = np.array([int(f / fps * RATE) for f in hit_frames])
    best, score = 0, -1.0
    for lag in lags:
        i = at + lag
        i = i[(i >= 0) & (i < len(hi))]
        if len(i) and hi[i].mean() > score:
            best, score = lag, hi[i].mean()
    return best / RATE


def frame_features(level, onset, n_frames, fps, offset_s=0.0):
    """Per video frame -> float32 [n_frames, N_FEATS]: the strongest onset per band within the frame (scaled by
    the recording's 99.5th percentile), the high band's level against its median, and 1 (sound present)."""
    scale = np.percentile(onset, 99.5, axis=0) + 1e-6
    hi = level[:, -1] - np.median(level[:, -1])
    out = np.zeros((n_frames, N_FEATS), np.float32)
    for f in range(n_frames):
        a = int((f / fps + offset_s) * RATE)
        z = int(((f + 1) / fps + offset_s) * RATE)
        a, z = max(a, 0), min(max(z, a + 1), len(onset))
        if a >= len(onset):
            break
        out[f, :len(BANDS)] = np.minimum(onset[a:z].max(0) / scale, 3.0)
        out[f, len(BANDS)] = np.clip(hi[a:z].max() / 5, -2, 2)
        out[f, -1] = 1.0
    return out


def features(video, start_s, n_frames, fps, hit_frames):
    """Audio features for a video's analysed frames ([n_frames, N_FEATS], zeros if it has no sound) and the
    soundtrack offset used (None without sound). hit_frames: hits from the ball track, to sync the sound."""
    x = load(video, start_s)
    if x is None:
        return np.zeros((n_frames, N_FEATS), np.float32), None
    level, onset = onsets(x)
    off = estimate_offset(onset, hit_frames, fps) if len(hit_frames) else 0.0
    return frame_features(level, onset, n_frames, fps, off), off
