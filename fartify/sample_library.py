"""Maintains the internal record of the fart sample library.

For every WAV file in samples/, we keep track of:
  - duration (seconds) -- the raw file length
  - effective_duration_sec -- the length of the sample's *audible* content,
    found by trimming leading/trailing near-silence against a volume
    threshold. This is what actually gets used as "the note length" for
    duration matching and time-stretch calculations: a sample padded with
    a half-second of near-silent room tone isn't really a half-second
    longer as far as the ear (or the matcher) is concerned. Nothing on disk
    is ever trimmed -- trimming the stored audio would make playback sound
    choppy at the cut points -- this is purely a measurement.
  - volume (RMS, linear + dBFS)
  - main note frequency (Hz) and the nearest MIDI note
  - emphasis_sec: the timestamp (seconds from the start of the clip) of the
    sound's main accent -- its moment of peak short-term loudness. Most fart
    recordings aren't a flat honk; they have some wind-up/tail around a
    central "punch". synth.py uses this so a sample's punch lands on the
    melody note's onset instead of just starting the raw file there.
  - envelope_shape: a fixed-length, duration- AND loudness-normalized
    resample of the sample's amplitude contour over its audible span --
    e.g. "fast attack then quick decay" vs. "slow swell". find_best_sample
    compares this against a melody note's own envelope shape (see
    notes.py's Note.volume_envelope) so a note that swells picks a sample
    that itself swells, not just one whose scalar duration/volume/pitch
    averages happen to line up.

The record is cached in samples/samples_index.json and rebuilt whenever a
sample is added, removed, or changes on disk (mtime + size fingerprint).

Analyzing a sample also generates its log-frequency spectrogram (array +
PNG, see spectrogram.py), cached under samples/spectrograms/ rather than in
the index -- it's what Spectral Synthesis builds a track out of.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .spectrogram import load_sample_spectrogram, remove_sample_spectrogram, write_sample_spectrogram
from .utils import ENVELOPE_SHAPE_POINTS, TARGET_SR, hz_to_midi, load_audio, resample_1d, rms, rms_to_dbfs

INDEX_FILENAME = "samples_index.json"

# Farts are low-pitched, noisy, mostly-voiced signals. Restrict pitch search
# to a plausible low-brass-ish range so noise floor doesn't get picked up.
FART_FMIN = 40.0
FART_FMAX = 350.0

# How far below a sample's own peak (in dB) counts as "silent" when measuring
# its effective duration. Relative to the sample's own peak, not absolute,
# so it works the same for a quiet recording and a loud one.
EFFECTIVE_DURATION_THRESHOLD_DB = -40.0


@dataclass
class SampleRecord:
    filename: str
    duration_sec: float  # raw file length
    effective_duration_sec: float  # audible-content length after volume-threshold trim (measurement only, audio untouched)
    rms: float
    dbfs: float
    frequency_hz: float
    midi_note: float
    pitch_confidence: float  # 0..1 -- how much frequency_hz can be trusted as "a note" vs. noise
    emphasis_sec: float  # timestamp of the sound's main accent/punch, seconds from clip start
    envelope_shape: list[float]  # normalized (duration + loudness) amplitude contour, ENVELOPE_SHAPE_POINTS long
    fingerprint: str  # f"{mtime}:{size}" — used to detect staleness


def _fingerprint(path: str) -> str:
    st = os.stat(path)
    return f"{st.st_mtime_ns}:{st.st_size}"


def estimate_pitch(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Estimate a (possibly noisy) one-shot sample's main note frequency,
    plus a 0..1 confidence for how much that frequency can be trusted --
    i.e. how strongly the sample actually behaves like a pitched note
    versus a burst of noise/percussive texture that happens to have *a*
    dominant autocorrelation lag somewhere.

    A fart sample is not a musical instrument: many are noisy, inharmonic,
    or only briefly voiced. Reporting a frequency for those without any
    caveat made the sample matcher trust a meaningless number as much as a
    clean, sustained tone. Confidence lets callers (find_best_sample) favor
    samples that can actually be heard as "a note" for melodic passages.

    Prefers librosa's pYIN when installed: confidence there is how much of
    the clip came back voiced at all, scaled by pYIN's own per-frame voiced
    probability -- a clip that's confidently voiced start-to-finish scores
    near 1.0, one that only squeaks a pitch briefly (or not at all) scores
    low. Falls back to a normalized-autocorrelation clarity measure (the
    height of the dominant periodicity peak relative to zero-lag energy)
    so this works with zero extra dependencies and generalizes to any
    pitch detector: a clean repeating waveform peaks near 1.0, noise barely
    rises above the floor.
    """
    if y.size < sr * 0.02:
        return 0.0, 0.0

    try:
        import librosa

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=FART_FMIN, fmax=FART_FMAX, sr=sr, frame_length=2048
        )
        voiced_mask = ~np.isnan(f0)
        if voiced_mask.any():
            freq = float(np.median(f0[voiced_mask]))
            voiced_fraction = float(np.mean(voiced_mask))
            mean_voiced_prob = float(np.mean(voiced_prob[voiced_mask]))
            confidence = float(np.clip(voiced_fraction * mean_voiced_prob, 0.0, 1.0))
            return freq, confidence
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: autocorrelation-based pitch + clarity estimate over the
    # whole clip (dependency-free).
    windowed = y * np.hanning(len(y))
    corr = np.correlate(windowed, windowed, mode="full")[len(windowed) - 1 :]
    zero_lag = corr[0]
    if zero_lag <= 0:
        return 0.0, 0.0
    min_lag = int(sr / FART_FMAX)
    max_lag = int(sr / FART_FMIN)
    max_lag = min(max_lag, len(corr) - 1)
    if max_lag <= min_lag:
        return 0.0, 0.0
    segment = corr[min_lag:max_lag]
    if segment.size == 0 or np.max(segment) <= 0:
        return 0.0, 0.0
    peak_idx = int(np.argmax(segment))
    peak_lag = min_lag + peak_idx
    if peak_lag == 0:
        return 0.0, 0.0
    freq = float(sr / peak_lag)
    # Normalized periodicity strength at the winning lag: ~1.0 means the
    # signal repeats itself almost perfectly there (a clean tone); near 0
    # means that "peak" barely stands out from the noise floor, i.e. this
    # isn't really a pitched sound at all.
    confidence = float(np.clip(segment[peak_idx] / zero_lag, 0.0, 1.0))
    return freq, confidence


def estimate_emphasis(y: np.ndarray, sr: int, win_sec: float = 0.02) -> float:
    """Find the timestamp (seconds from clip start) of a one-shot sample's
    main emphasis -- the instant of peak short-term loudness.

    Computed as the center of the `win_sec`-wide window with the highest
    energy, via a moving-average of the squared signal. This is a simple,
    dependency-free stand-in for onset/transient detection: for a punchy,
    mostly-monophonic sound like a fart, "loudest instant" and "the accent
    a listener would tap their foot to" line up closely enough to be useful
    for beat alignment, without needing librosa's onset detector installed.
    """
    if y.size == 0:
        return 0.0

    win = max(1, min(int(round(win_sec * sr)), y.size))
    energy = y.astype(np.float64) ** 2

    if win >= y.size:
        # Whole clip fits in one window -- fall back to the single loudest sample.
        return float(np.argmax(energy) / sr)

    kernel = np.ones(win) / win
    envelope = np.convolve(energy, kernel, mode="valid")
    peak_idx = int(np.argmax(envelope))
    # peak_idx is the start of the peak window; report its center.
    return float((peak_idx + win / 2) / sr)


def _find_trim_bounds(
    y: np.ndarray,
    sr: int,
    win_sec: float = 0.02,
    threshold_db: float = EFFECTIVE_DURATION_THRESHOLD_DB,
) -> tuple[int, int]:
    """Sample-index bounds [start, end) of a clip's audible content, via a
    moving-RMS envelope thresholded relative to the clip's own peak. Shared
    by estimate_effective_duration (just the length) and
    estimate_envelope_shape (the shape *within* those bounds, so a shape
    descriptor isn't diluted by padding on either end). Returns (0, len(y))
    -- i.e. "don't trim anything" -- if the whole clip is at/below threshold.
    """
    if y.size == 0:
        return 0, 0

    win = max(1, min(int(round(win_sec * sr)), y.size))
    energy = y.astype(np.float64) ** 2

    if win >= y.size:
        return 0, y.size

    kernel = np.ones(win) / win
    rms_envelope = np.sqrt(np.convolve(energy, kernel, mode="valid"))

    peak = float(np.max(rms_envelope))
    if peak <= 1e-9:
        return 0, y.size

    threshold = peak * (10 ** (threshold_db / 20.0))
    above = np.where(rms_envelope >= threshold)[0]
    if above.size == 0:
        return 0, y.size

    # rms_envelope[i] covers samples [i, i+win) -- so the first above-threshold
    # window's start and the last one's end bound the audible region.
    start_sample = int(above[0])
    end_sample = min(int(above[-1]) + win, y.size)
    return start_sample, end_sample


def estimate_effective_duration(
    y: np.ndarray,
    sr: int,
    win_sec: float = 0.02,
    threshold_db: float = EFFECTIVE_DURATION_THRESHOLD_DB,
) -> float:
    """Measure the duration of a sample's audible content by trimming
    leading/trailing near-silence against a threshold relative to the
    sample's own peak loudness -- WITHOUT altering the audio itself.

    A recording with a beat of room tone before and after the actual fart
    isn't really that much longer as far as note-length matching or
    time-stretch amount should be concerned; this gives a truer "effective"
    length for that purpose while the stored WAV stays intact (trimming the
    file for real would risk an audible click/choppiness at the cut).

    Returns 0.0 for an effectively silent clip (caller should fall back to
    the raw duration in that case).
    """
    start_sample, end_sample = _find_trim_bounds(y, sr, win_sec, threshold_db)
    if end_sample <= start_sample:
        return 0.0
    return float((end_sample - start_sample) / sr)


def estimate_envelope_shape(
    y: np.ndarray,
    sr: int,
    num_points: int = ENVELOPE_SHAPE_POINTS,
    trim_win_sec: float = 0.02,
    shape_win_sec: float = 0.015,
    threshold_db: float = EFFECTIVE_DURATION_THRESHOLD_DB,
) -> list[float]:
    """A compact, duration- AND loudness-independent descriptor of a
    sample's amplitude contour -- "fast attack then quick decay" vs.
    "slow swell" -- for comparing against a melody note's own dynamic
    shape in find_best_sample. Silence-trims to the audible span first (via
    the same threshold as estimate_effective_duration) so the shape isn't
    diluted by leading/trailing near-silence, then resamples that span's
    smoothed RMS envelope to `num_points` and normalizes it to its own
    peak so absolute duration and loudness both drop out, leaving just
    the shape.
    """
    start_sample, end_sample = _find_trim_bounds(y, sr, trim_win_sec, threshold_db)
    audible = y[start_sample:end_sample] if end_sample > start_sample else y
    if audible.size == 0:
        return [0.0] * num_points

    win = max(1, min(int(round(shape_win_sec * sr)), audible.size))
    energy = audible.astype(np.float64) ** 2
    if win >= audible.size:
        rms_envelope = np.array([np.sqrt(np.mean(energy))])
    else:
        kernel = np.ones(win) / win
        rms_envelope = np.sqrt(np.convolve(energy, kernel, mode="valid"))

    resampled = resample_1d(rms_envelope, num_points)
    peak = float(np.max(resampled))
    if peak <= 1e-9:
        return [0.0] * num_points
    return (resampled / peak).tolist()


def normalize_envelope_shape(values) -> list[float]:
    """Normalize an amplitude envelope's *level* to its own peak (0..1),
    leaving pure shape -- attack/sustain/decay contour -- independent of
    absolute loudness. Used to bring a melody note's volume_envelope (raw
    RMS) into the same normalized space as SampleRecord.envelope_shape so
    the two can be compared directly.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    peak = float(np.max(arr))
    if peak <= 1e-9:
        return [0.0] * arr.size
    return (arr / peak).tolist()


def _shape_distance(a, b) -> float:
    """RMS distance between two same-length, already-normalized (0..1)
    envelope shapes. Returns 0.0 (no penalty) if either is missing or
    they're different lengths -- shouldn't normally happen since both sides
    use ENVELOPE_SHAPE_POINTS, but a mismatch shouldn't crash matching."""
    if not a or not b or len(a) != len(b):
        return 0.0
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a_arr - b_arr) ** 2)))


def analyze_sample(path: str, sr: int = TARGET_SR) -> SampleRecord:
    y, _ = load_audio(path, sr=sr)
    duration = len(y) / sr
    r = rms(y)
    freq, pitch_confidence = estimate_pitch(y, sr)
    emphasis = estimate_emphasis(y, sr)
    effective_duration = estimate_effective_duration(y, sr)
    if effective_duration <= 0:
        effective_duration = duration  # silent/edge-case clip -- fall back to raw length
    envelope_shape = estimate_envelope_shape(y, sr)
    write_sample_spectrogram(os.path.dirname(path), os.path.basename(path), y, sr)
    return SampleRecord(
        filename=os.path.basename(path),
        duration_sec=round(duration, 4),
        effective_duration_sec=round(effective_duration, 4),
        rms=round(r, 6),
        dbfs=round(rms_to_dbfs(r), 2),
        frequency_hz=round(freq, 2),
        midi_note=round(hz_to_midi(freq), 2) if freq > 0 else 0.0,
        pitch_confidence=round(pitch_confidence, 4),
        emphasis_sec=round(emphasis, 4),
        envelope_shape=[round(v, 4) for v in envelope_shape],
        fingerprint=_fingerprint(path),
    )


def _write_index(samples_dir: str, records: list[SampleRecord]) -> None:
    index_path = os.path.join(samples_dir, INDEX_FILENAME)
    with open(index_path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)


def add_sample_record(samples_dir: str, record: SampleRecord) -> list[SampleRecord]:
    """Merge one freshly analyzed sample into the cached index (replacing
    any existing entry for the same filename) and persist it -- without
    re-analyzing the rest of the library. Used when a single new sample is
    uploaded through the web UI, so adding one file to a large library stays
    fast.
    """
    records = load_index(samples_dir, rebuild_if_stale=False)
    records = [r for r in records if r.filename != record.filename]
    records.append(record)
    records.sort(key=lambda r: r.filename.lower())
    _write_index(samples_dir, records)
    return records


def remove_sample_record(samples_dir: str, filename: str) -> list[SampleRecord]:
    """Drop filename's entry from the cached index and persist it, without
    touching any other sample's cached analysis."""
    index_path = os.path.join(samples_dir, INDEX_FILENAME)
    if not os.path.exists(index_path):
        return []
    records = load_index(samples_dir, rebuild_if_stale=False)
    records = [r for r in records if r.filename != filename]
    _write_index(samples_dir, records)
    remove_sample_spectrogram(samples_dir, filename)
    return records


def build_index(samples_dir: str, force: bool = False) -> list[SampleRecord]:
    """(Re)build the metadata index for every WAV in samples_dir."""
    index_path = os.path.join(samples_dir, INDEX_FILENAME)
    existing: dict[str, dict] = {}
    if not force and os.path.exists(index_path):
        try:
            with open(index_path) as f:
                existing = {r["filename"]: r for r in json.load(f)}
        except Exception:
            existing = {}

    records: list[SampleRecord] = []
    for filename in sorted(os.listdir(samples_dir)):
        if not filename.lower().endswith(".wav"):
            continue
        path = os.path.join(samples_dir, filename)
        fp = _fingerprint(path)
        cached = existing.get(filename)
        if cached and cached.get("fingerprint") == fp:
            try:
                records.append(SampleRecord(**cached))
                # Unchanged sample, but make sure its spectrogram exists too
                # (it won't if the sample was indexed before those did).
                load_sample_spectrogram(samples_dir, filename)
                continue
            except TypeError:
                pass  # cached record predates a schema field -- re-analyze below
        records.append(analyze_sample(path))

    _write_index(samples_dir, records)

    return records


def load_index(samples_dir: str, rebuild_if_stale: bool = True) -> list[SampleRecord]:
    index_path = os.path.join(samples_dir, INDEX_FILENAME)
    wavs = {f for f in os.listdir(samples_dir) if f.lower().endswith(".wav")}

    if not os.path.exists(index_path):
        return build_index(samples_dir)

    with open(index_path) as f:
        raw = json.load(f)
    try:
        records = [SampleRecord(**r) for r in raw]
    except TypeError:
        # Index predates a schema field (e.g. emphasis_sec) -- rebuild fresh.
        return build_index(samples_dir, force=True)

    if rebuild_if_stale:
        indexed_names = {r.filename for r in records}
        stale = indexed_names != wavs
        if not stale:
            for r in records:
                p = os.path.join(samples_dir, r.filename)
                if os.path.exists(p) and _fingerprint(p) != r.fingerprint:
                    stale = True
                    break
        if stale:
            return build_index(samples_dir)

    return records


def find_best_sample(
    records: list[SampleRecord],
    target_duration: float,
    target_volume_rms: float,
    target_freq_hz: float,
    target_envelope_shape: list[float] | None = None,
    duration_weight: float = 1.0,
    volume_weight: float = 0.4,
    freq_weight: float = 0.6,
    repetition_weight: float = 0.15,
    shape_weight: float = 0.5,
    loud_sample_discount: float = 0.25,
    confidence_weight: float = 0.5,
    recent_usage: dict[str, float] | None = None,
) -> SampleRecord | None:
    """Pick the library sample that most closely matches a melody note.

    Distance is computed in a normalized log-ish space so that e.g. a 2x
    duration mismatch costs the same regardless of absolute note length.
    We prefer close duration/frequency matches because they need the least
    aggressive time-stretch / pitch-shift, which keeps the fart sounding
    natural instead of artifacted. Duration matching uses each sample's
    effective_duration_sec (audible content only, silence trimmed out of
    the measurement) rather than its raw file length, so a sample padded
    with room tone isn't penalized -- or falsely favored -- for length it
    doesn't actually sound like it has.

    Volume is compared the same way (a log2 ratio of RMS) so it sits on the
    same scale as duration/frequency, but *asymmetrically*: a sample louder
    than the target only costs `loud_sample_discount` (default 1/4) of what
    the same-sized mismatch would cost if it were too quiet. Turning a loud,
    characterful recording down is basically free; turning a quiet one up
    risks amplifying its noise floor/hiss. Without this, a real melody's
    volume range rarely reaches as high as a library's loudest, punchiest
    samples, so those samples were being scored as bad matches for nearly
    every note and effectively never picked -- symmetric matching was
    quietly discarding the best-sounding samples in the library.

    target_envelope_shape: optional normalized (0..1, ENVELOPE_SHAPE_POINTS
    long -- see normalize_envelope_shape) dynamics contour for the note,
    compared against each candidate's envelope_shape. Two samples can have
    identical duration/volume/pitch averages while one is a sharp instant
    burst and the other a slow swell; this lets that difference actually
    matter, instead of relying on scalar features alone to imply a good
    match. Pass None to skip (no shape cost applied).

    confidence_weight: scales a cost of `(1 - pitch_confidence)` per
    candidate, so samples whose "note" is really just noise/percussive
    texture (low confidence -- see estimate_pitch) are only reached for
    when nothing better fits, instead of being trusted exactly as much as
    a sample that actually sounds like the target pitch. Set to 0 to
    ignore pitch confidence entirely.

    recent_usage: optional {filename: heat} map, where a higher "heat"
    means the sample was used more recently. Adds `repetition_weight * heat`
    to that sample's cost so a long melody spreads across the library
    instead of looping the single best-fit sample for every similar note.
    Callers own this dict's lifecycle (see synth.py) -- pass None to disable
    and always take the strict nearest match.
    """
    if not records:
        return None
    recent_usage = recent_usage or {}

    best = None
    best_score = float("inf")
    for r in records:
        if r.duration_sec <= 0:
            continue
        effective_duration = r.effective_duration_sec if r.effective_duration_sec > 0 else r.duration_sec
        dur_cost = abs(np.log2(max(target_duration, 0.01) / effective_duration))
        vol_log_ratio = np.log2(max(target_volume_rms, 1e-6) / max(r.rms, 1e-6))
        if vol_log_ratio > 0:
            vol_cost = vol_log_ratio  # sample is quieter than target -- needs amplifying
        else:
            vol_cost = abs(vol_log_ratio) * loud_sample_discount  # sample is already loud enough -- just turn it down
        if r.frequency_hz > 0 and target_freq_hz > 0:
            freq_cost = abs(np.log2(target_freq_hz / r.frequency_hz))
        else:
            freq_cost = 0.5  # mild penalty for unpitched samples, not disqualifying
        shape_cost = _shape_distance(target_envelope_shape, r.envelope_shape)
        repeat_cost = recent_usage.get(r.filename, 0)
        confidence_cost = 1.0 - np.clip(r.pitch_confidence, 0.0, 1.0)
        score = (
            duration_weight * dur_cost
            + volume_weight * vol_cost
            + freq_weight * freq_cost
            + shape_weight * shape_cost
            + repetition_weight * repeat_cost
            + confidence_weight * confidence_cost
        )
        if score < best_score:
            best_score = score
            best = r
    return best
