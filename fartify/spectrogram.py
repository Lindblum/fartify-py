"""Log-frequency spectrograms: amplitude over time, on a pitch-linear axis.

One spectrogram definition is shared by the fart samples (computed when a
sample is analyzed, and cached next to the library) and by the track layer
Spectral Synthesis is trying to rebuild (see spectral.py). Because both use
exactly the same parameters, a sample's spectrogram can be laid over the
track's like a stamp:

  - sliding it sideways is playing the sample at a different time,
  - sliding it up or down is pitch-shifting it -- the frequency axis is
    logarithmic, so a fixed number of rows is a fixed musical interval
    (BINS_PER_OCTAVE rows per octave) no matter where on the axis it is,
  - scaling its values is changing its volume, since the values are linear
    amplitude.

Layout: a float32 array of shape (N_BINS, n_frames). Row 0 is FMIN and
frequency rises with the row index; column t is centered on sample
t * HOP_LENGTH of the audio.

The PNG rendering (brightness = amplitude, in dB so quiet detail is still
visible; high frequencies at the top) is only for looking at -- matching
always uses the linear array.
"""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np

from .utils import TARGET_SR, load_audio

N_FFT = 2048  # ~93 ms window at 22.05 kHz: ~11 Hz resolution, enough to separate a low fart's harmonics
HOP_LENGTH = 512  # ~23 ms per column
BINS_PER_OCTAVE = 24  # quarter-tone rows
FMIN = 32.70  # C1 -- below the lowest fart fundamental in the library
N_OCTAVES = 8  # up to ~8.4 kHz, under the 11 kHz Nyquist limit
N_BINS = BINS_PER_OCTAVE * N_OCTAVES

# Dynamic range shown in a PNG: anything this far below the brightest value
# is black.
PNG_DYNAMIC_RANGE_DB = 70.0

# Where sample spectrograms are cached, inside the samples directory.
CACHE_DIRNAME = "spectrograms"
# Stored alongside each cached array; a cache written with different
# parameters is recomputed rather than trusted.
_PARAMS = np.array([TARGET_SR, N_FFT, HOP_LENGTH, BINS_PER_OCTAVE, FMIN, N_BINS], dtype=np.float64)

_FRAMES_PER_CHUNK = 1024

_filterbank_cache: dict[int, np.ndarray] = {}


def bin_frequencies() -> np.ndarray:
    """Center frequency (Hz) of each row."""
    return FMIN * 2.0 ** (np.arange(N_BINS) / BINS_PER_OCTAVE)


def frames_to_seconds(frames, sr: int = TARGET_SR):
    return np.asarray(frames) * HOP_LENGTH / sr


def _filterbank(sr: int) -> np.ndarray:
    """(N_BINS, N_FFT // 2 + 1) weights mapping linear FFT bins onto the
    log-frequency rows. Each row is a triangle centered on its frequency,
    as wide as its neighbours are far away -- but never narrower than the
    FFT's own bin spacing, so at the low end (where rows are much closer
    together than FFT bins) it degrades into interpolating between the two
    nearest FFT bins instead of selecting nothing. Rows sum to 1, so a
    row's value is an average amplitude, not a sum that grows with width."""
    if sr not in _filterbank_cache:
        fft_freqs = np.arange(N_FFT // 2 + 1) * sr / N_FFT
        centers = bin_frequencies()
        step = 2.0 ** (1.0 / BINS_PER_OCTAVE)
        half_widths = np.maximum(centers * (step - 1.0), sr / N_FFT)
        weights = np.maximum(0.0, 1.0 - np.abs(fft_freqs[None, :] - centers[:, None]) / half_widths[:, None])
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        _filterbank_cache[sr] = weights.astype(np.float32)
    return _filterbank_cache[sr]


def compute_spectrogram(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Log-frequency amplitude spectrogram of mono audio y: float32,
    shape (N_BINS, n_frames)."""
    y = np.asarray(y, dtype=np.float32)
    n_frames = len(y) // HOP_LENGTH + 1
    # Center each frame on its hop position (pad half a window either side)
    # so column t lines up with audio sample t * HOP_LENGTH.
    padded = np.pad(y, (N_FFT // 2, N_FFT // 2 + HOP_LENGTH))
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH][:n_frames]

    window = np.hanning(N_FFT).astype(np.float32)
    scale = 1.0 / window.sum()
    filterbank = _filterbank(sr)

    spec = np.empty((N_BINS, n_frames), dtype=np.float32)
    # Chunked so a whole song's STFT never has to sit in memory at once.
    for start in range(0, n_frames, _FRAMES_PER_CHUNK):
        chunk = frames[start : start + _FRAMES_PER_CHUNK] * window
        mag = np.abs(np.fft.rfft(chunk, axis=1)).astype(np.float32) * scale
        spec[:, start : start + len(chunk)] = (mag @ filterbank.T).T
    return spec


def save_png(spec: np.ndarray, path: str, reference: float | None = None) -> None:
    """Write a spectrogram as an 8-bit grayscale PNG: one pixel per
    (row, frame), brightness = amplitude in dB below `reference` (default:
    the spectrogram's own peak), high frequencies at the top. Pass the same
    `reference` for two spectrograms to make their brightness comparable.

    Written by hand (a PNG is just zlib-compressed rows plus a few chunk
    headers) rather than pulling in an imaging library for one function.
    """
    peak = float(reference) if reference else float(spec.max()) if spec.size else 0.0
    if peak <= 0:
        pixels = np.zeros(spec.shape, dtype=np.uint8)
    else:
        db = 20.0 * np.log10(np.maximum(spec, 1e-12) / peak)
        pixels = (np.clip(1.0 + db / PNG_DYNAMIC_RANGE_DB, 0.0, 1.0) * 255.0).astype(np.uint8)
    pixels = pixels[::-1]  # row 0 of the image is the top

    height, width = pixels.shape
    # Each scanline is prefixed with a filter-type byte (0 = none).
    raw = np.hstack([np.zeros((height, 1), dtype=np.uint8), pixels]).tobytes()

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", header))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# Sample spectrogram cache
# ---------------------------------------------------------------------------


def _cache_paths(samples_dir: str, filename: str) -> tuple[str, str]:
    base = os.path.join(samples_dir, CACHE_DIRNAME, filename)
    return base + ".npz", base + ".png"


def write_sample_spectrogram(samples_dir: str, filename: str, y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Compute a sample's spectrogram from its audio and cache both the
    array and a PNG of it. Called whenever a sample is (re)analyzed."""
    spec = compute_spectrogram(y, sr)
    npz_path, png_path = _cache_paths(samples_dir, filename)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, spec=spec, params=_PARAMS, source_mtime_ns=_source_mtime_ns(samples_dir, filename))
    save_png(spec, png_path)
    return spec


def _source_mtime_ns(samples_dir: str, filename: str) -> int:
    return os.stat(os.path.join(samples_dir, filename)).st_mtime_ns


def load_sample_spectrogram(samples_dir: str, filename: str) -> np.ndarray:
    """A sample's spectrogram, from the cache when it's there and still
    matches both the WAV on disk and the current parameters; otherwise
    computed (and cached) now. This is what lets samples analyzed before
    spectrograms existed pick one up the first time it's needed."""
    npz_path, png_path = _cache_paths(samples_dir, filename)
    if os.path.exists(npz_path) and os.path.exists(png_path):
        try:
            with np.load(npz_path) as cached:
                if np.array_equal(cached["params"], _PARAMS) and int(cached["source_mtime_ns"]) == _source_mtime_ns(
                    samples_dir, filename
                ):
                    return cached["spec"]
        except (OSError, KeyError, ValueError):
            pass  # unreadable/old-format cache -- fall through and rebuild it
    y, sr = load_audio(os.path.join(samples_dir, filename), sr=TARGET_SR)
    return write_sample_spectrogram(samples_dir, filename, y, sr)


def sample_spectrogram_png(samples_dir: str, filename: str) -> str:
    """Path to a sample's spectrogram PNG, generating it first if needed."""
    load_sample_spectrogram(samples_dir, filename)
    return _cache_paths(samples_dir, filename)[1]


def remove_sample_spectrogram(samples_dir: str, filename: str) -> None:
    for path in _cache_paths(samples_dir, filename):
        try:
            os.remove(path)
        except OSError:
            pass
