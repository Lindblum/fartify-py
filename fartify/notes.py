"""Turn a raw pitch (f0) curve into a sequence of discrete melody notes.

A "note" is a run of voiced, confident frames that stay on (approximately)
the same pitch. For each note we record the fields the rest of the
pipeline (and the MIDI export) need: start/end time, duration, a
representative frequency, a representative volume (RMS) pulled from the
vocal stem's amplitude envelope over that span, and volume_envelope -- a
fixed-length resample of that same span's loudness contour over time
(not just its median). synth.py uses the single volume_rms figure for
sample *matching*, but volume_envelope for envelope-following gain at
render time, so a note that swells or fades has that reflected in the
fart's dynamics too, not just its average level.

Unpitched layers (drums) have no pitch curve to segment; detect_hits at the
bottom of this module builds the same Note list from onsets instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .utils import ENVELOPE_SHAPE_POINTS, hz_to_midi, resample_1d, rms


@dataclass
class Note:
    start_sec: float
    end_sec: float
    frequency_hz: float
    midi_note: float
    volume_rms: float
    # Fixed-length (ENVELOPE_SHAPE_POINTS) resample of this note's RMS
    # loudness over its own span -- absolute levels, same scale as the
    # vocal audio itself. Defaults to empty for any caller constructing a
    # Note without it (e.g. tests); render_note_as_fart falls back to the
    # old flat volume_rms match when it's empty.
    volume_envelope: list[float] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _frame_rms_envelope(y: np.ndarray, sr: int, times: np.ndarray, hop_sec: float) -> np.ndarray:
    hop = max(1, int(round(hop_sec * sr)))
    win = hop * 2
    env = np.zeros(len(times))
    for i, t in enumerate(times):
        start = int(t * sr)
        frame = y[start : start + win]
        env[i] = rms(frame)
    return env


def segment_notes(
    y: np.ndarray,
    sr: int,
    times: np.ndarray,
    f0: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.35,
    min_note_duration_sec: float = 0.0,
    semitone_tolerance: float = 0.75,
    max_gap_frames: int = 4,
    drop_quiet_notes: bool = False,
    quiet_note_relative_floor: float = 0.08,
) -> list[Note]:
    """Group a frame-wise pitch curve into discrete notes.

    confidence_threshold: frames below this (unvoiced/uncertain) break notes.
    semitone_tolerance: how far (in semitones) f0 may drift within one note
        before it's considered a new note (captures natural vibrato/glide
        without over-segmenting).
    max_gap_frames: allow this many low-confidence frames inside a note
        (covers brief dropouts in the tracker) before splitting it.

    By default every voiced run that survives the above becomes a real Note
    -- min_note_duration_sec=0.0 and drop_quiet_notes=False mean nothing
    detected here is discarded, so every note downstream gets mapped to
    whichever sample scores best for it (see sample_library.find_best_sample),
    even a short or quiet one, rather than being silently dropped. Pass a
    positive min_note_duration_sec and/or drop_quiet_notes=True to filter out
    likely tracker noise instead, at the cost of some real notes going
    unmapped.
    """
    if len(times) == 0:
        return []

    hop_sec = float(times[1] - times[0]) if len(times) > 1 else 0.01
    envelope = _frame_rms_envelope(y, sr, times, hop_sec)

    voiced = (confidence >= confidence_threshold) & (f0 > 0)

    notes: list[Note] = []
    cur_start_idx = None
    cur_midi_vals: list[float] = []
    cur_freq_vals: list[float] = []
    gap_count = 0

    def flush(end_idx: int):
        nonlocal cur_start_idx, cur_midi_vals, cur_freq_vals, gap_count
        if cur_start_idx is not None and cur_freq_vals:
            start_t = float(times[cur_start_idx])
            end_t = float(times[min(end_idx, len(times) - 1)]) + hop_sec
            if end_t - start_t >= min_note_duration_sec:
                seg_slice = slice(cur_start_idx, end_idx)
                env_seg = envelope[seg_slice]
                vol = float(np.median(env_seg)) if env_seg.size else 0.0
                vol_envelope = resample_1d(env_seg, ENVELOPE_SHAPE_POINTS).tolist()
                freq = float(np.median(cur_freq_vals))
                notes.append(
                    Note(
                        start_sec=start_t,
                        end_sec=end_t,
                        frequency_hz=freq,
                        midi_note=round(hz_to_midi(freq), 2),
                        volume_rms=vol,
                        volume_envelope=vol_envelope,
                    )
                )
        cur_start_idx = None
        cur_midi_vals = []
        cur_freq_vals = []
        gap_count = 0

    for i in range(len(times)):
        if not voiced[i]:
            if cur_start_idx is not None:
                gap_count += 1
                if gap_count > max_gap_frames:
                    flush(i - gap_count + 1)
            continue

        midi_val = hz_to_midi(f0[i])

        if cur_start_idx is None:
            cur_start_idx = i
            cur_midi_vals = [midi_val]
            cur_freq_vals = [f0[i]]
            gap_count = 0
            continue

        running_median = float(np.median(cur_midi_vals))
        if abs(midi_val - running_median) <= semitone_tolerance:
            cur_midi_vals.append(midi_val)
            cur_freq_vals.append(f0[i])
            gap_count = 0
        else:
            flush(i)
            cur_start_idx = i
            cur_midi_vals = [midi_val]
            cur_freq_vals = [f0[i]]
            gap_count = 0

    flush(len(times))

    if drop_quiet_notes:
        return _drop_spurious_quiet_notes(notes, quiet_note_relative_floor)
    return notes


def _drop_spurious_quiet_notes(notes: list[Note], relative_floor: float = 0.08) -> list[Note]:
    """Drop notes far quieter than the loudest note — usually tracker noise
    at track boundaries or during vocal fade-outs, not real melody content."""
    if len(notes) <= 1:
        return notes
    max_vol = max((n.volume_rms for n in notes), default=0.0)
    if max_vol <= 0:
        return notes
    return [n for n in notes if n.volume_rms >= relative_floor * max_vol]


# ---------------------------------------------------------------------------
# Unpitched layers (drums): hits instead of a pitch curve
# ---------------------------------------------------------------------------

HIT_HOP_SEC = 0.010
HIT_N_FFT = 512
# Two hits closer together than this are treated as one (a flam, or one hit's
# attack smeared across adjacent frames).
HIT_MIN_GAP_SEC = 0.06
# A hit lasts until the next one starts, but no longer than this -- a drum
# stem's tail after a hit is mostly decay/bleed, not something to sustain a
# fart over.
HIT_MAX_DURATION_SEC = 0.35
# A drum hit has no pitch of its own, so its brightness (spectral centroid)
# is mapped log-linearly onto the range fart samples actually live in: kicks
# land low, snares in the middle, hi-hats/cymbals at the top.
HIT_CENTROID_RANGE_HZ = (150.0, 6000.0)
HIT_PITCH_RANGE_HZ = (50.0, 300.0)


def detect_hits(
    y: np.ndarray,
    sr: int,
    drop_quiet_notes: bool = False,
    quiet_note_relative_floor: float = 0.08,
) -> list[Note]:
    """Turn an unpitched, percussive stem into Notes -- one per drum hit.

    The pitch trackers in pitch.py find almost nothing in a drum stem, so
    this goes by onsets instead: spectral flux (how much energy each
    frequency bin *gains* from one frame to the next) spikes at every hit,
    and peaks that stand clear of their local surroundings are taken as
    onsets. Each hit becomes a Note running until the next hit (capped at
    HIT_MAX_DURATION_SEC), with a stand-in frequency derived from how bright
    the hit is (see HIT_PITCH_RANGE_HZ) so the usual sample matching and
    pitch-shifting downstream works unchanged.
    """
    from scipy.ndimage import maximum_filter1d, median_filter
    from scipy.signal import stft

    hop = max(1, int(round(HIT_HOP_SEC * sr)))
    if y.size < HIT_N_FFT * 2:
        return []

    freqs, frame_times, spec = stft(
        y, fs=sr, nperseg=HIT_N_FFT, noverlap=HIT_N_FFT - hop, boundary=None, padded=False
    )
    mag = np.abs(spec)
    if mag.shape[1] < 3:
        return []

    # Log-compress so a quiet hi-hat's onset isn't drowned out by the kick's.
    log_mag = np.log1p(1000.0 * mag)
    flux = np.concatenate([[0.0], np.maximum(np.diff(log_mag, axis=1), 0.0).sum(axis=0)])

    # A frame is an onset if it's the biggest flux within +/- the minimum
    # gap AND clears an adaptive threshold: well above what's typical right
    # around it, and above a floor tied to the track's own stronger onsets
    # (so near-silent stretches of stem bleed don't sprout hits).
    gap_frames = max(1, int(round(HIT_MIN_GAP_SEC / HIT_HOP_SEC)))
    local_median = median_filter(flux, size=2 * 10 + 1, mode="nearest")
    threshold = 1.5 * local_median + 0.25 * float(np.percentile(flux, 95))
    is_peak = flux == maximum_filter1d(flux, size=2 * gap_frames + 1, mode="nearest")
    onset_frames = np.where(is_peak & (flux > threshold))[0]

    total_sec = len(y) / sr
    centroid_lo, centroid_hi = HIT_CENTROID_RANGE_HZ
    pitch_lo, pitch_hi = HIT_PITCH_RANGE_HZ

    notes: list[Note] = []
    for n, frame in enumerate(onset_frames):
        # frame_times are window centers; the rise that produced this flux
        # peak began about a hop earlier.
        start_t = max(float(frame_times[frame]) - HIT_HOP_SEC, 0.0)
        if n + 1 < len(onset_frames):
            next_t = float(frame_times[onset_frames[n + 1]]) - HIT_HOP_SEC
        else:
            next_t = total_sec
        end_t = min(next_t, start_t + HIT_MAX_DURATION_SEC, total_sec)
        seg = y[int(start_t * sr) : int(end_t * sr)]
        if seg.size < hop:
            continue

        attack = mag[:, frame : frame + 3].mean(axis=1)
        centroid = float(np.sum(freqs * attack) / max(float(np.sum(attack)), 1e-9))
        brightness = np.log(np.clip(centroid, centroid_lo, centroid_hi) / centroid_lo) / np.log(
            centroid_hi / centroid_lo
        )
        freq = float(pitch_lo * (pitch_hi / pitch_lo) ** brightness)

        env = np.array([rms(seg[i : i + hop]) for i in range(0, seg.size - hop + 1, hop)])
        notes.append(
            Note(
                start_sec=start_t,
                end_sec=end_t,
                frequency_hz=freq,
                midi_note=round(hz_to_midi(freq), 2),
                volume_rms=rms(seg),
                volume_envelope=resample_1d(env, ENVELOPE_SHAPE_POINTS).tolist(),
            )
        )

    if drop_quiet_notes:
        return _drop_spurious_quiet_notes(notes, quiet_note_relative_floor)
    return notes
