"""Spectral Synthesis: rebuild a track layer's spectrogram out of fart
sample spectrograms, then render the audio that arrangement describes.

Where Classic reduces the layer to a list of notes and picks one sample per
note, this works on the sound itself. The layer's log-frequency spectrogram
(see spectrogram.py) is the picture to reproduce; every sample's
spectrogram is a stamp. A stamp can be placed at any time (horizontal
offset), any pitch shift within the allowed range (vertical offset -- the
frequency axis is logarithmic, so a vertical move is a pure transposition)
and any volume (brightness). The result is a list of Placements, which is
also exactly the recipe for the audio: play that sample, shifted by that
interval, at that gain, at that time.

The layer is cut at its beats (detected onsets), and each beat gets the one
sample, pitch shift and gain that best fit its slice of the spectrogram.
"Best fit" is a least-squares measure -- for a stamp A laid over the slice
R with overlap C = <R, A> and energy E = <A, A>, the best gain is C / E and
it removes C^2 / E of the slice's energy -- so a sample wins by matching the
slice's *shape* well, not just by being loud. The overlap at every pitch
shift comes from one cross-correlation per sample per beat.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import correlate

from . import spectrogram
from .notes import Note, detect_onsets
from .sample_library import SampleRecord
from .settings import PipelineSettings
from .synth import pitch_shift
from .utils import TARGET_SR, hz_to_midi, load_audio, normalize_peak

logger = logging.getLogger("fartify.spectral")

# The detail floor (settings.spectral_stop_percent) compares a placement
# with the strongest one possible within this many seconds either side.
STOP_WINDOW_SEC = 2.0

# A beat needs at least this many confident pitch-track frames (10 ms each)
# to count as having a melody note to shift samples onto.
MIN_NOTE_FRAMES = 3

# A sample cut off at the next beat fades out over this long instead of
# stopping dead (which clicks).
CUT_FADE_SEC = 0.015


@dataclass
class Placement:
    """One sample stamped into the track."""

    filename: str
    time_sec: float  # where the sample starts
    semitones: float  # pitch adjustment, + is up
    gain: float  # amplification (linear)
    # Where the sample is cut off (with a short fade), so it ends when the
    # next beat starts; None = it plays out in full.
    duration_sec: float | None = None


@dataclass
class Decomposition:
    placements: list[Placement]
    approximation: np.ndarray  # spectrogram built from the placements, same shape as the target
    explained_fraction: float  # share of the target spectrogram's energy the approximation accounts for


def _shift_rows(spec: np.ndarray, rows: int) -> np.ndarray:
    """Move a spectrogram up (rows > 0) or down the frequency axis, i.e.
    transpose it; whatever slides off the end is dropped."""
    if rows == 0:
        return spec
    shifted = np.zeros_like(spec)
    if rows > 0:
        shifted[rows:] = spec[:-rows]
    else:
        shifted[:rows] = spec[-rows:]
    return shifted


def fit_beats(
    target: np.ndarray,
    atoms: list[np.ndarray],
    beat_frames: list[int],
    max_shift_rows: int,
    stop_fraction: float,
    max_gain: float,
    stop_window: int = 0,
    progress_cb=None,
    pinned_rows=None,
) -> list[tuple[int, int, int, float, int | None]]:
    """One sample per beat. Each beat's window runs from its onset frame to
    the next beat's (the last one to the end of `target`); the sample,
    pitch shift and gain that best fit that window's slice of `target`
    are chosen by the least-squares measure described above. A sample
    longer than its window is judged -- and later played -- cut off at the
    window's end, so it never runs over the next beat.

    pinned_rows: optional pinned_rows(start frame, end frame) -> None (every
    sample shifts freely this beat) or an array with one row shift per
    atom: the shift that atom must use (fractional rows are fine; the fit
    is judged at the nearest whole row), NaN for one that shifts freely. An
    atom whose pinned shift is beyond max_shift_rows sits the beat out.

    Returns [(atom index, row shift, start frame, gain, cut length in
    frames or None)]. A beat whose best fit explains less than
    stop_fraction of the best within stop_window frames of it is left
    empty, as is one nothing fits at all.
    """
    n_rows, n_frames = target.shape
    starts = sorted({f for f in beat_frames if 0 <= f < n_frames})
    best: list[tuple[float, tuple]] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else n_frames
        width = end - start
        # Rows padded so every shift is a plain translation: a stamp shifted
        # past the top or bottom still pays for what it loses.
        window = np.pad(target[:, start:end], ((max_shift_rows, max_shift_rows), (0, 0)))
        pinned = pinned_rows(start, end) if pinned_rows else None
        choice, choice_score = None, 0.0
        for s, atom in enumerate(atoms):
            stamp = atom[:, :width]
            energy = float(np.sum(stamp.astype(np.float64) ** 2))
            if energy <= 1e-12:
                continue
            # overlap[i] = <window, stamp shifted by (i - max_shift_rows) rows>
            overlap = correlate(window[:, : stamp.shape[1]], stamp, mode="valid", method="fft")[:, 0]
            gain = np.clip(overlap / energy, 0.0, max_gain)
            scores = gain * (2.0 * overlap - gain * energy)
            if pinned is not None and not np.isnan(pinned[s]):
                if abs(pinned[s]) > max_shift_rows:
                    continue
                i, rows = int(round(pinned[s])) + max_shift_rows, float(pinned[s])
            else:
                i = int(scores.argmax())
                rows = i - max_shift_rows
            if scores[i] > choice_score:
                cut = width if atom.shape[1] > width else None
                choice, choice_score = (s, rows, start, float(gain[i]), cut), float(scores[i])
        if choice is not None:
            best.append((choice_score, choice))
        if progress_cb:
            progress_cb((n + 1) / len(starts))

    if not best:
        return []
    scores = np.array([sc for sc, _ in best])
    frames = np.array([c[2] for _, c in best])
    if stop_window > 0:
        nearby = np.array([scores[np.abs(frames - f) <= stop_window].max() for f in frames])
    else:
        nearby = np.full(len(scores), scores.max())
    return [c for (sc, c), ref in zip(best, nearby) if sc >= stop_fraction * ref]


def beat_note_shifts(melody, sample_records: list[SampleRecord], settings: PipelineSettings, frame_time):
    """A fit_beats pinned_rows function: for a beat with a melody note, the
    row shift that moves each pitched sample's main note (its analyzed
    pitch) exactly onto that note -- plus any transpose, since the target
    being matched is already transposed. The beat's note is the median
    pitch of the track's confident frames within it; a beat with fewer than
    MIN_NOTE_FRAMES of those has no note, and its samples shift freely, as
    do unpitched samples on every beat."""
    times, f0, conf = (np.asarray(a, dtype=np.float64) for a in melody)
    voiced = (conf >= settings.confidence_threshold) & (f0 > 0)
    times, midi = times[voiced], 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)
    sample_midi = np.array([hz_to_midi(r.frequency_hz) if r.frequency_hz > 0 else np.nan for r in sample_records])
    rows_per_semitone = spectrogram.BINS_PER_OCTAVE / 12.0

    def pinned_rows(start: int, end: int):
        lo, hi = np.searchsorted(times, [frame_time(start), frame_time(end)])
        if hi - lo < MIN_NOTE_FRAMES:
            return None
        note = float(np.median(midi[lo:hi])) + settings.transpose_semitones
        return (note - sample_midi) * rows_per_semitone

    return pinned_rows


def match_layer(
    layer_audio: np.ndarray,
    sr: int,
    sample_records: list[SampleRecord],
    samples_dir: str,
    settings: PipelineSettings,
    progress_cb=None,
    melody=None,
) -> tuple[np.ndarray, Decomposition]:
    """Work out which samples to place where to approximate the layer.
    Returns (the layer's spectrogram, the Decomposition).

    melody: optional pitch track of the layer, (times, f0 Hz, confidence)
    as pitch.extract_pitch returns it. Where a beat has a melody note,
    every pitched sample is shifted so its own main note lands exactly on
    that note (see beat_note_shifts); otherwise, and for unpitched
    samples, the shift is whichever fits the spectrogram best."""
    target = spectrogram.compute_spectrogram(layer_audio, sr)
    rows_per_semitone = spectrogram.BINS_PER_OCTAVE / 12.0

    # Transposing the output = matching against a transposed picture.
    match_target = _shift_rows(target, int(round(settings.transpose_semitones * rows_per_semitone)))

    atoms = [spectrogram.load_sample_spectrogram(samples_dir, r.filename) for r in sample_records]

    # Match on contrast-compressed pictures. (g * A) ** gamma = g ** gamma *
    # A ** gamma, so a gain found there is a gain ** (1 / gamma) in reality.
    gamma = settings.spectral_gamma
    # Samples go at the onsets themselves (10 ms resolution), not at the
    # spectrogram column they round to.
    onset_at_frame: dict[int, float] = {}

    def frame_time(frame: int) -> float:
        return onset_at_frame.get(frame, float(spectrogram.frames_to_seconds(frame, sr)))

    for t in detect_onsets(layer_audio, sr):
        onset_at_frame.setdefault(int(round(t * sr / spectrogram.HOP_LENGTH)), float(t))
    pinned_rows = None
    if melody is not None:
        pinned_rows = beat_note_shifts(melody, sample_records, settings, frame_time)
    raw = fit_beats(
        match_target**gamma,
        [a**gamma for a in atoms],
        sorted(onset_at_frame),
        max_shift_rows=int(round(settings.max_pitch_shift_semitones * rows_per_semitone)),
        stop_fraction=settings.spectral_stop_percent / 100.0,
        max_gain=settings.spectral_max_gain**gamma,
        stop_window=int(round(STOP_WINDOW_SEC * sr / spectrogram.HOP_LENGTH)),
        progress_cb=progress_cb,
        pinned_rows=pinned_rows,
    )
    raw = [(s, rows, frame, gain ** (1.0 / gamma), cut) for s, rows, frame, gain, cut in raw]

    # The approximation as it will actually sound: real gains, amplitudes
    # adding linearly (the compressed match only approximately adds up).
    n_rows, n_frames = match_target.shape
    canvas = np.zeros((n_rows, n_frames + max(a.shape[1] for a in atoms)), dtype=np.float32)
    for s, rows, frame, gain, cut in raw:
        stamp = atoms[s][:, :cut]
        canvas[:, frame : frame + stamp.shape[1]] += gain * _shift_rows(stamp, int(round(rows)))
    approximation = canvas[:, :n_frames]

    placements = sorted(
        (
            Placement(
                filename=sample_records[s].filename,
                time_sec=frame_time(frame),
                semitones=rows / rows_per_semitone,
                gain=gain,
                # a cut ends where the next beat starts
                duration_sec=None if cut is None else frame_time(frame + cut) - frame_time(frame),
            )
            for s, rows, frame, gain, cut in raw
        ),
        key=lambda p: p.time_sec,
    )

    total = float(np.sum(match_target.astype(np.float64) ** 2))
    missed = float(np.sum((match_target.astype(np.float64) - approximation) ** 2))
    explained = 1.0 - missed / total if total > 0 else 0.0
    logger.info("spectral match: %d placements explain %.1f%% of the layer", len(placements), explained * 100)
    return match_target, Decomposition(placements, approximation, explained)


def render_placements(
    placements: list[Placement],
    samples_dir: str,
    total_duration_sec: float,
    sr: int = TARGET_SR,
    progress_cb=None,
) -> np.ndarray:
    """Turn a list of Placements into audio: each sample pitch-shifted (its
    length unchanged), scaled, cut off with a short fade if the placement
    says so, and added in at its time."""
    output = np.zeros(max(int(round(total_duration_sec * sr)), 1), dtype=np.float32)
    raw_audio: dict[str, np.ndarray] = {}
    # The same sample at the same shift usually recurs many times over a
    # song; shifting is the expensive part, so do each combination once.
    shifted_audio: dict[tuple[str, float], np.ndarray] = {}

    for n, p in enumerate(placements):
        if progress_cb:
            progress_cb(n / len(placements))
        key = (p.filename, p.semitones)
        if key not in shifted_audio:
            if p.filename not in raw_audio:
                raw_audio[p.filename], _ = load_audio(os.path.join(samples_dir, p.filename), sr=sr)
            shifted_audio[key] = pitch_shift(raw_audio[p.filename], sr, 2.0 ** (p.semitones / 12.0))
        clip = shifted_audio[key]
        if p.duration_sec is not None and p.duration_sec * sr < len(clip):
            clip = clip[: max(int(round(p.duration_sec * sr)), 1)].copy()
            fade = min(int(CUT_FADE_SEC * sr), len(clip))
            clip[len(clip) - fade :] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        start = int(round(p.time_sec * sr))
        end = min(start + len(clip), len(output))
        if end > start:
            output[start:end] += p.gain * clip[: end - start]

    return normalize_peak(output, peak=0.95)


def placements_to_notes(placements: list[Placement], sample_records: list[SampleRecord]) -> list[Note]:
    """Express the placements as Notes, for the MIDI export: each pitched
    sample sounds at its own pitch moved by its placement's shift, for as
    long as the sample's audible content lasts. Unpitched samples have no
    note to write and are left out."""
    by_name = {r.filename: r for r in sample_records}
    notes = []
    for p in placements:
        record = by_name[p.filename]
        if record.frequency_hz <= 0:
            continue
        freq = record.frequency_hz * 2.0 ** (p.semitones / 12.0)
        length = record.effective_duration_sec if record.effective_duration_sec > 0 else record.duration_sec
        if p.duration_sec is not None:
            length = min(length, p.duration_sec)
        notes.append(
            Note(
                start_sec=p.time_sec,
                end_sec=p.time_sec + length,
                frequency_hz=freq,
                midi_note=round(hz_to_midi(freq), 2),
                volume_rms=record.rms * p.gain,
            )
        )
    return notes


def placements_as_dicts(placements: list[Placement]) -> list[dict]:
    return [asdict(p) for p in placements]
