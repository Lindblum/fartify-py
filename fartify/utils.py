"""Shared audio I/O and DSP helpers used across the Fartify pipeline.

Design note: the only hard runtime dependencies for basic operation are
numpy, scipy, and the `ffmpeg` binary on PATH (used to decode/encode
whatever container format the user throws at us — mp3, wav, m4a, ...).
librosa/soundfile/pydub are used opportunistically when installed (better
resampling, etc.) but nothing here hard-fails without them. This keeps
Fartify runnable in "lite mode" even before the heavier optional deps
(torch, demucs, torchcrepe — see requirements.txt) are set up.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave

import numpy as np

TARGET_SR = 22050  # internal working sample rate for pitch/note analysis and synthesis

# Fixed resolution for the compact envelope descriptors used to compare a
# melody note's dynamics against a candidate fart sample's (see notes.py's
# Note.volume_envelope and sample_library.py's SampleRecord.envelope_shape).
# Both sides must use the same length to be compared directly.
ENVELOPE_SHAPE_POINTS = 12

FFMPEG_BIN = shutil.which("ffmpeg")


def _read_wav_pcm(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file (any bit depth ffmpeg-normalized to 16-bit) into float32 [-1, 1]."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels)

    return data, sr


def _ffmpeg_to_wav(src_path: str, sr: int, channels: int | None = None) -> str:
    """Decode src_path to a PCM WAV via ffmpeg. channels=None preserves the
    source's original channel count (so stereo-dependent code, like the
    naive center-channel separation fallback, can still see both channels);
    pass 1 to force a mono downmix."""
    if not FFMPEG_BIN:
        raise RuntimeError(
            "ffmpeg is required to decode this file but was not found on PATH. "
            "Install ffmpeg (e.g. `apt install ffmpeg` / `brew install ffmpeg`)."
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [FFMPEG_BIN, "-y", "-i", src_path]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    cmd += ["-ar", str(sr), "-sample_fmt", "s16", "-f", "wav", tmp.name]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {src_path}: {result.stderr.decode(errors='replace')[-800:]}")
    return tmp.name


def load_audio(path: str, sr: int = TARGET_SR, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load any common audio file as float32 samples at the target sample rate.

    Always goes through ffmpeg for decoding so mp3/wav/m4a/flac/etc all work
    identically without extra Python codec packages.
    """
    wav_path = _ffmpeg_to_wav(path, sr, channels=1 if mono else None)
    try:
        y, file_sr = _read_wav_pcm(wav_path)
    finally:
        import os

        try:
            os.unlink(wav_path)
        except OSError:
            pass

    if y.ndim > 1 and mono:
        y = y.mean(axis=1)

    return y.astype(np.float32), file_sr


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    try:
        import librosa

        return librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        return resample_poly(y, up, down).astype(np.float32)


def save_audio(path: str, y: np.ndarray, sr: int = TARGET_SR) -> None:
    """Write float32 [-1, 1] samples out as a 16-bit PCM WAV via the stdlib wave module."""
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def export_as(path: str, wav_path: str, fmt: str) -> str:
    """Transcode an internal WAV to another format (e.g. mp3) via ffmpeg. Returns output path."""
    if fmt.lower() == "wav":
        if path != wav_path:
            shutil.copy(wav_path, path)
        return path
    if not FFMPEG_BIN:
        raise RuntimeError("ffmpeg is required to export to this format but was not found on PATH.")
    cmd = [FFMPEG_BIN, "-y", "-i", wav_path, path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg export failed: {result.stderr.decode(errors='replace')[-800:]}")
    return path


def resample_1d(values: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample a 1D array to exactly `n` points, evenly spaced
    across its span. Used to reduce an envelope of arbitrary length/resolution
    (e.g. a note's per-frame loudness, or a sample's amplitude contour) down
    to a fixed-size descriptor that can be compared or interpolated
    regardless of the source's original duration or frame count.
    """
    values = np.asarray(values, dtype=np.float64)
    if n <= 0:
        return np.zeros(0)
    if values.size == 0:
        return np.zeros(n)
    if values.size == 1:
        return np.full(n, float(values[0]))
    src_positions = np.linspace(0.0, 1.0, values.size)
    dst_positions = np.linspace(0.0, 1.0, n)
    return np.interp(dst_positions, src_positions, values)


def rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y))))


def rms_to_dbfs(value: float) -> float:
    return 20 * np.log10(max(value, 1e-9))


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = np.max(np.abs(y)) if y.size else 0.0
    if m < 1e-9:
        return y
    return y * (peak / m)


def midi_to_hz(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def hz_to_midi(hz: float) -> float:
    hz = max(hz, 1e-6)
    return 69 + 12 * np.log2(hz / 440.0)


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_note: float) -> str:
    """Human-readable note name (e.g. "A4", "C#5") for display purposes --
    nearest semitone, rounded, with the standard octave numbering (A4 = 440Hz)."""
    rounded = int(round(midi_note))
    name = _NOTE_NAMES[rounded % 12]
    octave = rounded // 12 - 1
    return f"{name}{octave}"
