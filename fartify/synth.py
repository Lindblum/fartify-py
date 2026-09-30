"""Resynthesize a melody (list of Notes) as farts.

For every detected melody note we pick the closest-matching sample from
the library (by duration/volume/frequency — see sample_library), then:

  1. Pitch-shift that sample from its own recorded frequency to the note's
     target frequency.
  2. Time-stretch it to the note's duration.
  3. Boost its volume, if needed, so it ends up at least as loud as the
     note it's replacing. The note's own loudness curve is not used to
     reshape the fart's dynamics -- it's only used earlier (see
     sample_library.find_best_sample) as one factor in picking which
     sample to use in the first place.
  4. Overlap-add it into the output track, timed so the sample's own
     emphasis/accent (sample_library.SampleRecord.emphasis_sec, scaled into
     the rendered clip) lands on the note's onset -- not just the raw start
     of the sample file. A fart with wind-up before its main honk should
     have that honk hit the beat, not arrive late.

Uses librosa's phase-vocoder based pitch_shift/time_stretch when librosa
is installed (best quality); otherwise falls back to a small self-contained
phase-vocoder implementation so the whole pipeline stays dependency-light.
"""
from __future__ import annotations

import os

import numpy as np

from .notes import Note
from .sample_library import SampleRecord, find_best_sample, normalize_envelope_shape
from .settings import PipelineSettings
from .utils import TARGET_SR, load_audio, normalize_peak, rms


# ---------------------------------------------------------------------------
# Time-stretch / pitch-shift primitives
# ---------------------------------------------------------------------------


def _phase_vocoder_stretch(x: np.ndarray, rate: float, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Time-stretch x by `rate` (output length ~= len(x)/rate), preserving pitch."""
    if x.size < n_fft or abs(rate - 1.0) < 1e-3:
        return x.copy() if rate >= 1.0 else np.concatenate([x, np.zeros(int(len(x) * (1 / rate - 1)))])

    window = np.hanning(n_fft).astype(np.float32)
    hop_out = hop
    hop_in = hop * rate

    n_frames = max(1, int((len(x) - n_fft) / hop_in) + 1)
    out_len = int(n_frames * hop_out + n_fft)
    output = np.zeros(out_len, dtype=np.float32)
    window_sum = np.zeros(out_len, dtype=np.float32)

    n_bins = n_fft // 2 + 1
    prev_phase = np.zeros(n_bins)
    accum_phase = np.zeros(n_bins)
    expected_advance = 2 * np.pi * hop_out * np.arange(n_bins) / n_fft

    for i in range(n_frames):
        in_pos = i * hop_in
        i0 = int(np.floor(in_pos))
        frac = in_pos - i0
        if i0 + n_fft + 1 > len(x):
            break

        frame = (1 - frac) * x[i0 : i0 + n_fft] + frac * x[i0 + 1 : i0 + n_fft + 1]
        frame = frame * window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        phase = np.angle(spec)

        if i == 0:
            accum_phase = phase.copy()
        else:
            delta = phase - prev_phase - expected_advance
            delta = delta - 2 * np.pi * np.round(delta / (2 * np.pi))
            accum_phase = accum_phase + expected_advance + delta

        prev_phase = phase
        new_spec = mag * np.exp(1j * accum_phase)
        new_frame = np.fft.irfft(new_spec, n=n_fft).astype(np.float32) * window

        start = i * hop_out
        output[start : start + n_fft] += new_frame
        window_sum[start : start + n_fft] += window ** 2

    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]
    return output


def _resample_speed(x: np.ndarray, factor: float) -> np.ndarray:
    """Resample x as if played back `factor` times faster (changes pitch + duration together)."""
    if x.size == 0 or abs(factor - 1.0) < 1e-3:
        return x.copy()
    new_len = max(1, int(round(len(x) / factor)))
    from scipy.signal import resample

    return resample(x, new_len).astype(np.float32)


def pitch_shift_and_stretch(x: np.ndarray, sr: int, freq_ratio: float, duration_ratio: float) -> np.ndarray:
    """Transform x (duration D0, freq F0) -> (duration D0*duration_ratio, freq F0*freq_ratio)."""
    freq_ratio = max(freq_ratio, 0.05)
    duration_ratio = max(duration_ratio, 0.05)

    try:
        import librosa

        n_steps = 12 * np.log2(freq_ratio)
        y = librosa.effects.pitch_shift(x, sr=sr, n_steps=n_steps)
        rate = 1.0 / duration_ratio  # librosa: rate > 1 => faster/shorter
        y = librosa.effects.time_stretch(y, rate=rate)
        return y.astype(np.float32)
    except ImportError:
        pass
    except Exception:
        pass

    # Dependency-free fallback: phase-vocoder stretch + resample composed to
    # hit the target pitch while preserving duration, then a second stretch
    # to hit the target duration. See module docstring in synth.py history
    # / README for the derivation.
    x1 = _phase_vocoder_stretch(x, rate=1.0 / freq_ratio)
    x2 = _resample_speed(x1, factor=freq_ratio)
    x3 = _phase_vocoder_stretch(x2, rate=1.0 / duration_ratio)
    return x3.astype(np.float32)


def pitch_shift(x: np.ndarray, sr: int, freq_ratio: float) -> np.ndarray:
    """Transpose x by freq_ratio without changing its length."""
    if abs(freq_ratio - 1.0) < 1e-6:
        return x.astype(np.float32)

    try:
        import librosa

        y = librosa.effects.pitch_shift(x, sr=sr, n_steps=12 * np.log2(freq_ratio))
    except Exception:
        # Same dependency-free route as pitch_shift_and_stretch: stretch,
        # then resample back to the original length at the new pitch.
        y = _resample_speed(_phase_vocoder_stretch(x, rate=1.0 / freq_ratio), factor=freq_ratio)

    # Both routes can come back a few samples off; pin the length so a
    # shifted sample still lines up with its unshifted spectrogram.
    if len(y) >= len(x):
        return y[: len(x)].astype(np.float32)
    return np.pad(y, (0, len(x) - len(y))).astype(np.float32)


# ---------------------------------------------------------------------------
# Note -> fart rendering
# ---------------------------------------------------------------------------


def render_note_as_fart(
    note: Note,
    sample_records: list[SampleRecord],
    samples_dir: str,
    sr: int = TARGET_SR,
    recent_usage: dict[str, float] | None = None,
    settings: PipelineSettings | None = None,
) -> tuple[np.ndarray, float]:
    """Returns (rendered_audio, emphasis_offset_sec) -- the latter is how far
    into the clip its emphasis point falls, so the caller can shift playback
    earlier by that much and land the emphasis on the note's onset.

    recent_usage: optional shared {filename: heat} dict (see
    sample_library.find_best_sample) that this call both reads -- to steer
    away from whatever was just used -- and updates in place: every other
    entry's heat decays by one note, and the sample picked here is reset to
    full heat since it was just used again.

    settings: matching weights and stretch/shift limits (see settings.py);
    None uses the defaults.
    """
    settings = settings or PipelineSettings()
    target_shape = normalize_envelope_shape(note.volume_envelope) if note.volume_envelope else None
    best = find_best_sample(
        sample_records,
        target_duration=note.duration_sec,
        target_volume_rms=note.volume_rms,
        target_freq_hz=note.frequency_hz,
        target_envelope_shape=target_shape,
        duration_weight=settings.duration_weight,
        volume_weight=settings.volume_weight,
        freq_weight=settings.freq_weight,
        repetition_weight=settings.repetition_weight,
        shape_weight=settings.shape_weight,
        confidence_weight=settings.confidence_weight,
        recent_usage=recent_usage,
    )

    if recent_usage is not None:
        for fname in list(recent_usage.keys()):
            recent_usage[fname] -= 1
            if recent_usage[fname] <= 0:
                del recent_usage[fname]
        if best is not None and settings.repeat_cooldown > 0:
            recent_usage[best.filename] = settings.repeat_cooldown

    if best is None:
        return np.zeros(int(note.duration_sec * sr), dtype=np.float32), 0.0

    sample_audio, _ = load_audio(os.path.join(samples_dir, best.filename), sr=sr)

    # Stretch against the sample's effective (silence-trimmed) duration, not
    # its raw file length -- otherwise a sample with room tone padding would
    # get stretched as if that silence were part of the "note" too.
    best_duration = best.effective_duration_sec if best.effective_duration_sec > 0 else best.duration_sec
    freq_ratio = (note.frequency_hz / best.frequency_hz) if best.frequency_hz > 0 and note.frequency_hz > 0 else 1.0
    duration_ratio = note.duration_sec / best_duration if best_duration > 0 else 1.0
    # Guard rails: extreme stretch/shift ratios sound broken rather than "farty"
    # (see settings.py)
    max_freq_ratio = 2.0 ** (settings.max_pitch_shift_semitones / 12.0)
    freq_ratio = float(np.clip(freq_ratio, 1.0 / max_freq_ratio, max_freq_ratio))
    duration_ratio = float(np.clip(duration_ratio, 1.0 / settings.max_stretch, settings.max_stretch))

    rendered = pitch_shift_and_stretch(sample_audio, sr, freq_ratio, duration_ratio)

    # Trim/pad to the exact note duration *before* volume matching, so the
    # RMS measured below reflects the clip's final timeline, not the
    # pre-trim one.
    target_len = int(round(note.duration_sec * sr))
    if len(rendered) > target_len:
        rendered = rendered[:target_len]
    elif len(rendered) < target_len:
        rendered = np.pad(rendered, (0, target_len - len(rendered)))

    # Volume: the note's loudness curve is not used here to reshape the
    # fart over time -- it's only a scoring factor earlier, in picking
    # which sample to use (see target_shape / find_best_sample above). Here
    # we just make sure the rendered fart ends up at least as loud as the
    # note it's replacing: boost it up to the note's RMS if it's quieter,
    # but never turn a naturally louder sample down to match a quieter note.
    cur_rms = rms(rendered)
    if cur_rms > 1e-6 and note.volume_rms > 0:
        gain = max(note.volume_rms / cur_rms, 1.0)
        rendered = rendered * gain

    # Short fade in/out to avoid clicks when overlap-adding into the track
    fade_len = min(int(0.005 * sr), len(rendered) // 4)
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len)
        rendered[:fade_len] *= fade
        rendered[-fade_len:] *= fade[::-1]

    # Where the sample's emphasis lands inside the rendered clip. The pitch
    # shift / time stretch above don't preserve absolute timestamps, but a
    # phase-vocoder stretch is uniform in time, so the emphasis's *relative*
    # position within the clip is preserved; map that fraction onto the
    # clip's final (post trim/pad) duration to get its new timestamp.
    emphasis_fraction = (best.emphasis_sec / best.duration_sec) if best.duration_sec > 0 else 0.0
    emphasis_fraction = float(np.clip(emphasis_fraction, 0.0, 1.0))
    emphasis_offset_sec = emphasis_fraction * note.duration_sec

    return rendered.astype(np.float32), emphasis_offset_sec


def synthesize_track(
    notes: list[Note],
    sample_records: list[SampleRecord],
    samples_dir: str,
    total_duration_sec: float,
    sr: int = TARGET_SR,
    settings: PipelineSettings | None = None,
    progress_cb=None,
) -> np.ndarray:
    """progress_cb(fraction: float), if given, is called after each note
    with how much of the note list has been rendered so far."""
    total_len = max(int(round(total_duration_sec * sr)), 1)
    output = np.zeros(total_len, dtype=np.float32)
    recent_usage: dict[str, float] = {}

    for i, note in enumerate(notes):
        if progress_cb:
            progress_cb(i / len(notes))
        if note.duration_sec <= 0:
            continue
        rendered, emphasis_offset_sec = render_note_as_fart(
            note, sample_records, samples_dir, sr=sr, recent_usage=recent_usage, settings=settings
        )

        # Shift playback earlier by the emphasis offset so the sample's own
        # accent -- not its raw start -- lands on the note's onset.
        start = int(round((note.start_sec - emphasis_offset_sec) * sr))
        end = start + len(rendered)

        # Every note that made it this far was already mapped to a best-fit
        # sample above; clamp its placement into the track instead of ever
        # dropping it outright, so a mapped note is never silently missing
        # from the output. In normal use start/end already land in bounds --
        # this only bites at rounding edges (e.g. a pitch tracker's frame
        # timestamps running a hair past the audio's raw duration).
        if end <= 0:
            # whole clip would land at/before the track start -- push it
            # forward just enough for its last sample to land at t=0
            shift = 1 - end
            start += shift
            end += shift
        if start >= total_len:
            # whole clip would land at/after the track end -- pull it back
            # just enough for its first sample to land at the last valid index
            shift = start - (total_len - 1)
            start -= shift
            end -= shift

        src_start = 0
        if start < 0:
            # The lead-in before the emphasis would start before the track
            # itself does (e.g. the very first note) -- trim it instead of
            # dropping the whole clip, so the emphasis still lands on time.
            src_start = -start
            start = 0

        end = min(end, total_len)
        if end > start:
            seg = rendered[src_start : src_start + (end - start)]
            output[start:end] += seg

    return normalize_peak(output, peak=0.95)
