"""Melody (f0) extraction from the isolated vocal stem.

Fartify picks a pitch tracker in order of quality, using whatever is
actually installed:

  1. CREPE (via torchcrepe) — a CNN-based monophonic pitch tracker. This is
     the project's preferred method: it's far more robust than classical
     trackers on breathy / vibrato-heavy / noisy singing, which is exactly
     what a post-Demucs vocal stem tends to sound like. It runs fine on CPU
     for typical song lengths.
  2. librosa's pYIN — a solid probabilistic classical tracker, used if
     torchcrepe/torch aren't installed. No GPU/ML runtime required.
  3. A pure numpy/scipy autocorrelation tracker — always available, lower
     quality on inharmonic or heavily reverberant vocals, but keeps Fartify
     fully functional with zero extra dependencies.

Why not RMVPE? RMVPE (from the RVC / singing-voice-conversion world) is
tuned for voiced-singing robustness in real-time conversion pipelines, but
it ships as a larger, less standardized checkpoint+inference stack outside
mainstream pip packaging, and its main advantage over CREPE is speed at
inference time in a streaming/real-time context — which Fartify, an
offline batch pipeline, doesn't need. CREPE has an off-the-shelf PyTorch
port (torchcrepe), a stable pip install, and published accuracy on par
with or better than RMVPE for offline monophonic melody extraction, so it
was chosen as the primary tracker.

All backends return the same shape: (times_sec, f0_hz, confidence[0..1]),
one value per hop.
"""
from __future__ import annotations

import numpy as np

HOP_LENGTH_SEC = 0.010  # 10ms hops
FMIN = 65.0   # ~C2, comfortably below the lowest sung notes we expect
FMAX = 1000.0  # ~B5, comfortably above typical melodic vocal range

# Search range per separated layer (see settings.LAYERS). The defaults above
# are tuned for singing; a bass line sits an octave or two lower (down to
# ~B0/E1 -- 33 Hz is also about as low as CREPE goes), and guitars/keys
# reach higher than a voice does. Drums aren't here: they're unpitched and
# go through notes.detect_hits instead.
LAYER_PITCH_RANGES = {
    "vocals": (FMIN, FMAX),
    "bass": (33.0, 400.0),
    "other": (FMIN, 1500.0),
}


def extract_pitch(
    y: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Returns (times, f0_hz, confidence, backend_name)."""
    for backend_fn, name in (
        (_extract_torchcrepe, "torchcrepe (CREPE)"),
        (_extract_pyin, "librosa pYIN"),
    ):
        try:
            times, f0, conf = backend_fn(y, sr, fmin, fmax)
            return times, f0, conf, name
        except ImportError:
            continue
        except Exception:
            continue

    times, f0, conf = _extract_autocorrelation(y, sr, fmin, fmax)
    return times, f0, conf, "autocorrelation (fallback)"


def _extract_torchcrepe(y: np.ndarray, sr: int, fmin: float, fmax: float):
    import torch
    import torchcrepe

    audio = torch.from_numpy(y).float().unsqueeze(0)
    hop_length = int(sr * HOP_LENGTH_SEC)
    f0, periodicity = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin=fmin,
        fmax=fmax,
        model="tiny",
        batch_size=2048,
        device="cpu",
        return_periodicity=True,
    )
    f0 = f0.squeeze(0).numpy()
    conf = periodicity.squeeze(0).numpy()
    times = np.arange(len(f0)) * HOP_LENGTH_SEC
    return times, f0, conf


def _extract_pyin(y: np.ndarray, sr: int, fmin: float, fmax: float):
    import librosa

    hop_length = int(sr * HOP_LENGTH_SEC)
    # pYIN needs a frame long enough to hold two periods of fmin; 4 hops
    # covers the vocal range, a lower fmin (bass) needs more.
    frame_length = max(hop_length * 4, int(2 * sr / fmin) + 4)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length, frame_length=frame_length
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    conf = np.nan_to_num(voiced_prob, nan=0.0)
    return times, f0, conf


def _extract_autocorrelation(y: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX):
    """Simple frame-wise autocorrelation pitch tracker. Dependency-free."""
    hop = int(sr * HOP_LENGTH_SEC)
    win = hop * 4
    n_frames = max(0, (len(y) - win) // hop + 1)

    times = np.zeros(n_frames)
    f0 = np.zeros(n_frames)
    conf = np.zeros(n_frames)

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    window = np.hanning(win)

    for i in range(n_frames):
        start = i * hop
        frame = y[start : start + win] * window
        times[i] = start / sr

        energy = np.sum(frame ** 2)
        if energy < 1e-6:
            continue

        corr = np.correlate(frame, frame, mode="full")[win - 1 :]
        corr = corr / (corr[0] + 1e-9)
        search = corr[min_lag : min(max_lag, len(corr) - 1)]
        if search.size == 0:
            continue

        peak_idx = int(np.argmax(search))
        peak_val = search[peak_idx]
        lag = min_lag + peak_idx
        if lag <= 0 or peak_val < 0.3:
            continue

        f0[i] = sr / lag
        conf[i] = float(np.clip(peak_val, 0.0, 1.0))

    return times, f0, conf
