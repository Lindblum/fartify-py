"""End-to-end Fartify pipeline: audio in -> fart melody out (+ MIDI).

Separation and the final mix are shared; what happens in between depends on
settings.algorithm -- Classic goes through notes (pitch.py, notes.py,
synth.py), Spectral Synthesis through spectrograms (spectral.py)."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, replace

from . import midi_export, pitch, separation, spectral, spectrogram, synth
from .notes import Note, detect_hits, segment_notes
from .progress import ProgressTracker
from .sample_library import load_index
from .settings import ALGORITHMS, IMPLEMENTED_ALGORITHMS, LAYERS, PipelineSettings
from .utils import TARGET_SR, load_audio, normalize_peak, save_audio

logger = logging.getLogger("fartify.pipeline")

# What Spectral Synthesis leaves in a job's work dir besides the audio: the
# layer's spectrogram, the approximation of it built from sample
# spectrograms, and the list of placements that approximation consists of.
TARGET_SPECTROGRAM_PNG = "layer_spectrogram.png"
APPROX_SPECTROGRAM_PNG = "approx_spectrogram.png"
PLACEMENTS_JSON = "spectral_placements.json"


@dataclass
class PipelineResult:
    output_wav_path: str
    output_midi_path: str
    separation_method: str
    separation_degraded: bool
    pitch_backend: str
    note_count: int  # notes (Classic) or sample placements (Spectral Synthesis)
    duration_sec: float
    notes: list[dict]
    # Spectral Synthesis only: share of the layer's spectrogram energy that
    # the sample placements account for.
    explained_fraction: float | None = None


def run_pipeline(
    input_path: str,
    work_dir: str,
    samples_dir: str,
    settings: PipelineSettings | None = None,
    progress_cb=None,
    timings_path: str | None = None,
) -> PipelineResult:
    """Run the full pipeline.

    settings picks the resynthesis algorithm, the layer to run it on, and
    carries the calibration knobs (see settings.py); None runs Classic on
    the vocals with every default.

    progress_cb(state: dict) is optional; it's called whenever the run moves
    on, with a state dict that progress.snapshot() can turn into an overall
    fraction + ETA at any later moment (see progress.py). timings_path is
    where per-stage timings are kept so those estimates calibrate themselves
    to this machine; None just uses the built-in estimates.
    """
    settings = settings or PipelineSettings()

    # Checked up front rather than at the resynthesis step, so picking an
    # algorithm that's listed but not built yet fails immediately instead of
    # after minutes of separation.
    if settings.algorithm not in IMPLEMENTED_ALGORITHMS:
        name = ALGORITHMS.get(settings.algorithm, settings.algorithm)
        raise NotImplementedError(f"The {name} algorithm isn't implemented yet -- use Classic for now.")

    os.makedirs(work_dir, exist_ok=True)

    layer = settings.layer
    layer_name = LAYERS.get(layer, layer).lower()
    if settings.algorithm == "spectral":
        stages = [("spectral_match", "Matching spectrogram"), ("spectral_render", "Rendering farts")]
    elif layer in pitch.LAYER_PITCH_RANGES:
        stages = [("pitch", "Extracting melody"), ("synth", "Resynthesizing as farts")]
    else:
        stages = [("onsets", "Detecting hits"), ("synth", "Resynthesizing as farts")]

    # Every stage's expected duration scales with how much audio there is,
    # so that has to be known before the first (and longest) one starts.
    # Decoding the input just to measure it costs a second or two.
    if progress_cb:
        progress_cb({"stage": "Reading audio"})
    input_audio, input_sr = load_audio(input_path, sr=TARGET_SR)
    tracker = ProgressTracker(
        [("separate", f"Separating {layer_name}"), *stages, ("mix", "Mixing output")],
        audio_sec=len(input_audio) / input_sr,
        history_path=timings_path,
        on_update=progress_cb,
    )
    del input_audio

    def start(stage: str, units: float | None = None):
        logger.info("[%s]", stage)
        tracker.start(stage, units)

    start("separate")
    sep = separation.separate_layer(input_path, work_dir, layer=layer, sr=TARGET_SR)

    layer_audio, sr = load_audio(sep.layer_path, sr=TARGET_SR)
    duration_sec = len(layer_audio) / sr

    if settings.algorithm == "spectral":
        notes, fart_track, match = _resynthesize_spectral(
            layer_audio, sr, duration_sec, samples_dir, work_dir, settings, start, tracker
        )
        backend_name = "none (spectrogram matching)"
        note_count = len(match.placements)
        explained_fraction = match.explained_fraction
    else:
        notes, fart_track, backend_name = _resynthesize_classic(
            layer_audio, sr, duration_sec, samples_dir, settings, start, tracker
        )
        note_count = len(notes)
        explained_fraction = None
    fart_track = fart_track * settings.fart_gain

    start("mix")
    if settings.mix_with_instrumental and os.path.exists(sep.backing_path):
        backing_audio, _ = load_audio(sep.backing_path, sr=sr)
        n = max(len(fart_track), len(backing_audio))
        mix = _pad_to(fart_track, n) + _pad_to(backing_audio, n)
    else:
        mix = fart_track
    # Always peak-normalize last (regardless of branch above) so fart_gain
    # can't cause clipping -- it only shifts the fart/backing balance,
    # this brings the result back to a safe overall level.
    mix = normalize_peak(mix, peak=0.95)

    output_wav_path = os.path.join(work_dir, "fartify_output.wav")
    save_audio(output_wav_path, mix, sr=sr)

    output_midi_path = os.path.join(work_dir, "fartify_melody.mid")
    midi_export.write_midi(notes, output_midi_path)

    tracker.finish()

    return PipelineResult(
        output_wav_path=output_wav_path,
        output_midi_path=output_midi_path,
        separation_method=sep.method,
        separation_degraded=sep.degraded,
        pitch_backend=backend_name,
        note_count=note_count,
        duration_sec=duration_sec,
        notes=[asdict(n) | {"duration_sec": n.duration_sec} for n in notes],
        explained_fraction=explained_fraction,
    )


def _load_samples(samples_dir: str):
    sample_records = load_index(samples_dir)
    if not sample_records:
        raise RuntimeError(
            f"No .wav samples found in {samples_dir}. Add some fart sounds, or run "
            "scripts/generate_sample_farts.py to create placeholders."
        )
    return sample_records


def _resynthesize_classic(layer_audio, sr, duration_sec, samples_dir, settings, start, tracker):
    """Classic: reduce the layer to notes, then replay each note with its
    best-matching sample. Returns (notes, fart track, name of the
    note-extraction backend used)."""
    layer = settings.layer

    if layer in pitch.LAYER_PITCH_RANGES:
        start("pitch")
        fmin, fmax = pitch.LAYER_PITCH_RANGES[layer]
        times, f0, conf, backend_name = pitch.extract_pitch(layer_audio, sr, fmin=fmin, fmax=fmax)
        notes: list[Note] = segment_notes(
            layer_audio,
            sr,
            times,
            f0,
            conf,
            confidence_threshold=settings.confidence_threshold,
            min_note_duration_sec=settings.min_note_duration_sec,
            semitone_tolerance=settings.semitone_tolerance,
            max_gap_frames=settings.max_gap_frames,
            drop_quiet_notes=settings.drop_quiet_notes,
            quiet_note_relative_floor=settings.quiet_note_relative_floor,
        )
    else:
        start("onsets")
        backend_name = "onset detection (unpitched layer)"
        notes = detect_hits(
            layer_audio,
            sr,
            drop_quiet_notes=settings.drop_quiet_notes,
            quiet_note_relative_floor=settings.quiet_note_relative_floor,
        )
    logger.info("found %d notes in the %s layer using %s", len(notes), layer, backend_name)

    # Transposition only affects what gets resynthesized -- `notes` (and so
    # the MIDI export) stays the melody as it was actually performed.
    synth_notes = notes
    if settings.transpose_semitones:
        ratio = 2.0 ** (settings.transpose_semitones / 12.0)
        synth_notes = [
            replace(n, frequency_hz=n.frequency_hz * ratio, midi_note=n.midi_note + settings.transpose_semitones)
            for n in notes
        ]

    start("synth", units=len(synth_notes))
    sample_records = _load_samples(samples_dir)

    # This stage can count its own work (notes rendered so far), so it
    # reports real progress rather than leaning on the time estimate.
    fart_track = synth.synthesize_track(
        synth_notes, sample_records, samples_dir, duration_sec, sr=sr, settings=settings, progress_cb=tracker.advance
    )
    return notes, fart_track, backend_name


def _resynthesize_spectral(layer_audio, sr, duration_sec, samples_dir, work_dir, settings, start, tracker):
    """Spectral Synthesis: approximate the layer's spectrogram with sample
    spectrograms, then render the resulting placements (see spectral.py).
    Returns (notes for the MIDI export, fart track, the Decomposition).

    Also leaves the layer's spectrogram, the approximation built from
    samples, and the placement list (times, pitch shifts, gains) in
    work_dir for inspection."""
    start("spectral_match")
    sample_records = _load_samples(samples_dir)
    target, match = spectral.match_layer(
        layer_audio, sr, sample_records, samples_dir, settings, progress_cb=tracker.advance
    )

    # One brightness reference for both images, so they compare directly.
    reference = float(target.max()) if target.size else None
    spectrogram.save_png(target, os.path.join(work_dir, TARGET_SPECTROGRAM_PNG), reference=reference)
    spectrogram.save_png(match.approximation, os.path.join(work_dir, APPROX_SPECTROGRAM_PNG), reference=reference)
    with open(os.path.join(work_dir, PLACEMENTS_JSON), "w", encoding="utf-8") as f:
        json.dump(spectral.placements_as_dicts(match.placements), f, indent=1)

    start("spectral_render", units=len(match.placements))
    fart_track = spectral.render_placements(
        match.placements, samples_dir, duration_sec, sr=sr, progress_cb=tracker.advance
    )
    notes = spectral.placements_to_notes(match.placements, sample_records)
    return notes, fart_track, match


def _pad_to(y, n):
    import numpy as np

    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))
