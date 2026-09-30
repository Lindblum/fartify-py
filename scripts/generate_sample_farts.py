#!/usr/bin/env python3
"""Generate a starter library of synthetic placeholder fart samples.

Fartify's samples/ folder is meant to hold real recorded fart WAVs that the
user adds over time. To make the app runnable out of the box (and to keep
the repo free of any licensing questions around sourced sound effects),
this script procedurally *synthesizes* a small, varied set of placeholder
"fart" one-shots and drops them into samples/. Real recordings can replace
or sit alongside these at any time — the sample_library module re-analyzes
the folder automatically whenever it changes.

Synthesis approach: a low fundamental (sawtooth-ish buzz) run through a
wobbling resonant low-pass filter, layered with band-limited noise for
rasp, and shaped by a "sputtering" amplitude envelope built from smoothed
random noise. Duration, base pitch, and intensity are varied per sample so
the library spans a useful range for matching against real melody notes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fartify.sample_library import build_index  # noqa: E402
from fartify.utils import TARGET_SR, normalize_peak, save_audio  # noqa: E402

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def _smooth_noise(n_samples: int, rate_hz: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Low-passed random noise used for organic envelope/vibrato wobble."""
    raw = rng.normal(0, 1, n_samples)
    cutoff = max(1.0, rate_hz) / (sr / 2)
    cutoff = min(cutoff, 0.99)
    b, a = signal.butter(2, cutoff, btype="low")
    smoothed = signal.filtfilt(b, a, raw)
    m = np.max(np.abs(smoothed))
    return smoothed / m if m > 1e-9 else smoothed


def synthesize_fart(
    duration: float,
    base_freq: float,
    intensity: float,
    sr: int = TARGET_SR,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr

    # Organic pitch wobble (farts are not perfectly steady oscillators)
    vibrato = _smooth_noise(n, rate_hz=6, sr=sr, rng=rng) * 0.06
    instantaneous_freq = base_freq * (1.0 + vibrato)
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr

    # Buzzy fundamental: blend of sawtooth (rich harmonics = "brassy" rasp)
    # and a softer sine to keep the low end from being too harsh.
    saw = signal.sawtooth(phase)
    tone = 0.6 * saw + 0.4 * np.sin(phase)

    # Textural noise layer, band-limited around the fundamental's harmonics
    noise = rng.normal(0, 1, n)
    b, a = signal.butter(2, [max(base_freq * 0.5, 20) / (sr / 2), min(base_freq * 6, sr / 2 - 1) / (sr / 2)], btype="band")
    noise = signal.filtfilt(b, a, noise)
    noise = noise / (np.max(np.abs(noise)) + 1e-9)

    raw = 0.75 * tone + 0.25 * noise

    # Sputtering amplitude envelope: fast attack, "wobbly" sustain, tapering decay
    attack_n = max(1, int(0.02 * n))
    decay_n = max(1, int(0.35 * n))
    sustain_n = max(0, n - attack_n - decay_n)
    attack_env = np.linspace(0, 1, attack_n) ** 0.5
    sustain_wobble = 0.75 + 0.25 * _smooth_noise(sustain_n, rate_hz=10, sr=sr, rng=rng)
    decay_env = np.linspace(1, 0, decay_n) ** 1.5
    envelope = np.concatenate([attack_env, sustain_wobble, decay_env])[:n]

    y = raw * envelope * intensity

    # Gentle resonant low-pass so it reads as "fart" rather than "kazoo"
    cutoff = min(base_freq * 8, sr / 2 - 100) / (sr / 2)
    b, a = signal.butter(3, max(cutoff, 0.02), btype="low")
    y = signal.lfilter(b, a, y)

    return normalize_peak(y.astype(np.float32), peak=0.9 * intensity + 0.05)


# (duration_sec, base_freq_hz, intensity, name)
PRESETS = [
    (0.35, 130, 0.55, "toot_short_high"),
    (0.55, 95, 0.7, "parp_mid"),
    (0.9, 70, 0.8, "honk_low"),
    (1.3, 60, 0.9, "brraap_long"),
    (0.25, 160, 0.4, "squeak_tiny"),
    (1.8, 45, 0.95, "rumble_deep"),
    (0.7, 110, 0.65, "phhbt_mid"),
    (2.4, 55, 0.85, "extended_low"),
    (0.45, 140, 0.6, "quick_pop"),
    (1.1, 85, 0.75, "raspy_mid"),
]


def main() -> None:
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for i, (duration, base_freq, intensity, name) in enumerate(PRESETS):
        y = synthesize_fart(duration, base_freq, intensity, seed=1000 + i)
        path = os.path.join(SAMPLES_DIR, f"{i:02d}_{name}.wav")
        save_audio(path, y, sr=TARGET_SR)
        print(f"wrote {path}  ({duration}s, ~{base_freq}Hz)")

    print("\nBuilding samples index...")
    records = build_index(SAMPLES_DIR, force=True)
    for r in records:
        print(f"  {r.filename}: {r.duration_sec}s, {r.dbfs}dBFS, ~{r.frequency_hz}Hz (midi {r.midi_note}, pitch_conf {r.pitch_confidence})")
    print(f"\nWrote {len(records)} sample records to samples/samples_index.json")


if __name__ == "__main__":
    main()
