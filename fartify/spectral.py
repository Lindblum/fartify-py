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

Choosing the placements is matching pursuit: repeatedly take the single
placement that explains the most of what's still unexplained, subtract it,
and look again. "Explains the most" is a least-squares measure -- for a
stamp A laid over the residual R with overlap C = <R, A> and energy
E = <A, A>, the best gain is C / E and it removes C^2 / E of the residual's
energy -- so a placement wins by matching the residual's *shape* well, not
just by being loud.

Two things keep that loop fast enough to run thousands of times:
  - the overlap C of every sample at every (pitch shift, time) is computed
    once up front by cross-correlation, and
  - subtracting a placement changes those overlaps by exactly (gain x the
    overlap between the two stamps involved), which is precomputed per pair
    of samples. So an iteration is a handful of small array subtractions
    rather than a fresh correlation against the whole track.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d
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


def decompose(
    target: np.ndarray,
    atoms: list[np.ndarray],
    max_shift_rows: int,
    max_placements: int,
    stop_fraction: float,
    max_gain: float,
    stop_window: int = 0,
    progress_cb=None,
) -> tuple[list[tuple[int, int, int, float]], np.ndarray]:
    """Matching pursuit of `target` (rows x frames) over `atoms` (each
    rows x its own frame count). Returns ([(atom index, row shift, start
    frame, gain)], approximation).

    stop_fraction: a placement is only made if it would explain at least
    this fraction of what the strongest placement within stop_window
    frames of it could have before matching began; matching stops when no
    such placement is left. Local rather than song-wide, so a quiet passage
    isn't dropped for being quiet next to a loud one.
    """
    n_rows, n_frames = target.shape
    shift_span = 2 * max_shift_rows + 1
    atom_frames = [a.shape[1] for a in atoms]
    longest = max(atom_frames)
    energies = np.array([float(np.sum(a.astype(np.float64) ** 2)) for a in atoms])
    usable = energies > 1e-12

    # The canvas is the target with room for a stamp to move: max_shift_rows
    # of empty rows above and below (so a shifted stamp never runs off the
    # edge -- every shift is a plain translation, which the precomputed
    # stamp-vs-stamp overlaps below rely on), and `longest` empty frames at
    # the end (so a stamp may start near the end and overhang it).
    canvas = np.zeros((n_rows + 2 * max_shift_rows, n_frames + longest), dtype=np.float32)
    canvas[max_shift_rows : max_shift_rows + n_rows, :n_frames] = target

    # overlap[s, i, t] = <canvas, atom s shifted by (i - max_shift_rows) rows, starting at frame t>
    overlap = np.zeros((len(atoms), shift_span, n_frames), dtype=np.float32)
    for s, atom in enumerate(atoms):
        if usable[s]:
            overlap[s] = correlate(canvas[:, : n_frames + atom_frames[s] - 1], atom, mode="valid", method="fft")
        if progress_cb:
            progress_cb(0.5 * (s + 1) / len(atoms))

    safe_energies = np.where(usable, energies, 1.0).astype(np.float32)[:, None, None]

    def scores(block: np.ndarray) -> np.ndarray:
        """Residual energy each placement in `block` (a slice of `overlap`
        along time) would remove, at its best gain within [0, max_gain]."""
        gain = np.clip(block / safe_energies, 0.0, max_gain)
        gained = gain * (2.0 * block - gain * safe_energies)
        gained[~usable] = 0.0
        return gained

    # Best placement starting at each frame, so finding the overall best is
    # a scan over frames and only the frames a subtraction touched need
    # re-evaluating.
    best_score = np.zeros(n_frames, dtype=np.float32)
    best_choice = np.zeros(n_frames, dtype=np.int64)  # flat index into (atom, shift)

    def refresh(t0: int, t1: int) -> None:
        flat = scores(overlap[:, :, t0:t1]).reshape(-1, t1 - t0)
        best_choice[t0:t1] = flat.argmax(axis=0)
        best_score[t0:t1] = flat.max(axis=0)

    for t0 in range(0, n_frames, 2048):
        refresh(t0, min(t0 + 2048, n_frames))

    # Per frame, the least a placement starting there must explain.
    if stop_window > 0:
        floor = stop_fraction * maximum_filter1d(best_score, size=2 * stop_window + 1)
    else:
        floor = np.full(n_frames, stop_fraction * float(best_score.max()), dtype=np.float32)

    # pair_overlap[s][s2][dp + 2 * max_shift_rows, dt + frames(s2) - 1] =
    # <atom s2, atom s moved by dp rows and dt frames>. Built per chosen
    # atom the first time it's used, since most runs lean on a subset.
    pair_overlap: dict[int, list[np.ndarray]] = {}

    def overlaps_with(s: int) -> list[np.ndarray]:
        if s not in pair_overlap:
            mid = n_rows - 1  # zero row-lag in a full correlation
            rows = slice(mid - 2 * max_shift_rows, mid + 2 * max_shift_rows + 1)
            pair_overlap[s] = [
                correlate(atoms[s], other, mode="full", method="fft")[rows].astype(np.float32) for other in atoms
            ]
        return pair_overlap[s]

    placements: list[tuple[int, int, int, float]] = []
    approx_canvas = np.zeros_like(canvas)

    for n in range(max_placements):
        t = int(np.where(best_score >= floor, best_score, 0.0).argmax())
        score = float(best_score[t])
        if score <= 0 or score < floor[t]:
            break

        s, i = divmod(int(best_choice[t]), shift_span)
        gain = float(min(overlap[s, i, t] / energies[s], max_gain))
        placements.append((s, i - max_shift_rows, t, gain))
        approx_canvas[i : i + n_rows, t : t + atom_frames[s]] += gain * atoms[s]

        # Take this placement's contribution out of every overlap it touches.
        for s2, pair in enumerate(overlaps_with(s)):
            if not usable[s2]:
                continue
            # frames t2 of atom s2 that overlap the placed stamp at all
            t2_start = max(t - (atom_frames[s2] - 1), 0)
            t2_end = min(t + atom_frames[s], n_frames)
            cols = slice(t2_start - t + atom_frames[s2] - 1, t2_end - t + atom_frames[s2] - 1)
            rows = slice(2 * max_shift_rows - i, 2 * max_shift_rows - i + shift_span)
            overlap[s2, :, t2_start:t2_end] -= gain * pair[rows, cols]

        refresh(max(t - (longest - 1), 0), min(t + atom_frames[s], n_frames))

        if progress_cb and n % 16 == 0:
            progress_cb(0.5 + 0.5 * n / max_placements)

    approximation = approx_canvas[max_shift_rows : max_shift_rows + n_rows, :n_frames]
    return placements, approximation


def fit_beats(
    target: np.ndarray,
    atoms: list[np.ndarray],
    beat_frames: list[int],
    max_shift_rows: int,
    stop_fraction: float,
    max_gain: float,
    stop_window: int = 0,
    progress_cb=None,
) -> list[tuple[int, int, int, float, int | None]]:
    """One sample per beat. Each beat's window runs from its onset frame to
    the next beat's (the last one to the end of `target`); the sample,
    pitch shift and gain that best fit that window's slice of `target`
    are chosen by the same least-squares measure as decompose. A sample
    longer than its window is judged -- and later played -- cut off at the
    window's end, so it never runs over the next beat.

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
        # Rows padded so every shift is a plain translation, as in decompose:
        # a stamp shifted past the top or bottom still pays for what it loses.
        window = np.pad(target[:, start:end], ((max_shift_rows, max_shift_rows), (0, 0)))
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
            i = int(scores.argmax())
            if scores[i] > choice_score:
                cut = width if atom.shape[1] > width else None
                choice, choice_score = (s, i - max_shift_rows, start, float(gain[i]), cut), float(scores[i])
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


def match_layer(
    layer_audio: np.ndarray,
    sr: int,
    sample_records: list[SampleRecord],
    samples_dir: str,
    settings: PipelineSettings,
    progress_cb=None,
) -> tuple[np.ndarray, Decomposition]:
    """Work out which samples to place where to approximate the layer.
    Returns (the layer's spectrogram, the Decomposition)."""
    target = spectrogram.compute_spectrogram(layer_audio, sr)
    rows_per_semitone = spectrogram.BINS_PER_OCTAVE / 12.0

    # Transposing the output = matching against a transposed picture.
    match_target = _shift_rows(target, int(round(settings.transpose_semitones * rows_per_semitone)))

    atoms = [spectrogram.load_sample_spectrogram(samples_dir, r.filename) for r in sample_records]
    duration_sec = len(layer_audio) / sr

    # Match on contrast-compressed pictures. (g * A) ** gamma = g ** gamma *
    # A ** gamma, so a gain found there is a gain ** (1 / gamma) in reality.
    gamma = settings.spectral_gamma
    common = dict(
        max_shift_rows=int(round(settings.max_pitch_shift_semitones * rows_per_semitone)),
        stop_fraction=settings.spectral_stop_percent / 100.0,
        max_gain=settings.spectral_max_gain**gamma,
        stop_window=int(round(STOP_WINDOW_SEC * sr / spectrogram.HOP_LENGTH)),
        progress_cb=progress_cb,
    )
    # Beat mode places samples at the onsets themselves (10 ms resolution),
    # not at the spectrogram column they round to.
    onset_at_frame: dict[int, float] = {}
    if settings.spectral_fit_beats:
        for t in detect_onsets(layer_audio, sr):
            onset_at_frame.setdefault(int(round(t * sr / spectrogram.HOP_LENGTH)), float(t))
        raw = fit_beats(
            match_target**gamma, [a**gamma for a in atoms], sorted(onset_at_frame), **common
        )
    else:
        raw = decompose(
            match_target**gamma,
            [a**gamma for a in atoms],
            max_placements=max(1, int(round(settings.spectral_density * duration_sec))),
            **common,
        )[0]
        raw = [(s, rows, frame, gain, None) for s, rows, frame, gain in raw]
    raw = [(s, rows, frame, gain ** (1.0 / gamma), cut) for s, rows, frame, gain, cut in raw]

    # The approximation as it will actually sound: real gains, amplitudes
    # adding linearly (the compressed match only approximately adds up).
    n_rows, n_frames = match_target.shape
    canvas = np.zeros((n_rows, n_frames + max(a.shape[1] for a in atoms)), dtype=np.float32)
    for s, rows, frame, gain, cut in raw:
        stamp = atoms[s][:, :cut]
        canvas[:, frame : frame + stamp.shape[1]] += gain * _shift_rows(stamp, rows)
    approximation = canvas[:, :n_frames]

    def frame_time(frame: int) -> float:
        return onset_at_frame.get(frame, float(spectrogram.frames_to_seconds(frame, sr)))

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
