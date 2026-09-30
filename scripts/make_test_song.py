#!/usr/bin/env python3
"""Build a tiny synthetic stereo test 'song' for exercising the pipeline
without needing a real MP3: a center-panned sung melody (sine + harmonics
with vibrato) over a side-panned 'instrumental' pad, mixed to stereo WAV.
"""
import os
import sys

import numpy as np
import scipy.io.wavfile as wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fartify.utils import TARGET_SR  # noqa: E402

sr = TARGET_SR


def note(freq, dur, sr=sr, vibrato_hz=5, vibrato_depth=0.01):
    t = np.arange(int(dur * sr)) / sr
    vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_hz * t)
    phase = 2 * np.pi * np.cumsum(freq * vibrato) / sr
    y = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    env = np.ones_like(t)
    a = int(0.02 * sr)
    env[:a] = np.linspace(0, 1, a)
    env[-a:] = np.linspace(1, 0, a)
    return (y * env).astype(np.float32)


# A simple ascending melody: C4 D4 E4 G4 E4 (Hz)
melody_freqs = [261.63, 293.66, 329.63, 392.00, 329.63]
melody = np.concatenate([note(f, 0.5) for f in melody_freqs] + [np.zeros(int(0.2 * sr))])

# Instrumental pad: a couple of sustained low tones, roughly same length
inst_len = len(melody)
t = np.arange(inst_len) / sr
instrumental = 0.3 * np.sin(2 * np.pi * 110 * t) + 0.15 * np.sin(2 * np.pi * 164.81 * t)
instrumental *= 0.5

vocal_gain = 0.9
left = melody * vocal_gain + instrumental  # center vocal + pad on both channels
right = melody * vocal_gain - instrumental * 0.2 + instrumental * 1.2  # slight stereo widening for pad

stereo = np.stack([left, right], axis=1)
stereo = np.clip(stereo, -1, 1)

out_path = os.path.join(os.path.dirname(__file__), "..", "uploads", "test_song.wav")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
wavfile.write(out_path, sr, (stereo * 32767).astype(np.int16))
print(f"wrote {out_path}  ({inst_len/sr:.2f}s)")
