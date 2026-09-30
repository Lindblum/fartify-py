"""User-tunable pipeline settings: which resynthesis algorithm to run, which
layer of the song to run it on, plus the calibration knobs exposed in the
web UI's input panel.

PipelineSettings is the single source of truth for every option's default,
allowed range and UI label -- the input panel is rendered from
settings_schema(), and whatever the browser posts back is validated against
the same field metadata by parse_settings(). Adding a knob here is enough to
make it show up in the UI; it then just needs reading wherever it applies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

ALGORITHMS = {
    "classic": "Classic",
    "spectral": "Spectral Synthesis",
}
DEFAULT_ALGORITHM = "classic"
# Algorithms that can actually run. Anything in ALGORITHMS but not in here is
# selectable in the UI but fails fast in run_pipeline.
IMPLEMENTED_ALGORITHMS = {"classic", "spectral"}

# Which part of the song gets isolated and replayed as farts. Keys are
# Demucs stem names (see separation.py); everything else in the song becomes
# the backing track the farts are mixed over.
LAYERS = {
    "vocals": "Vocals",
    "bass": "Bass",
    "drums": "Drums",
    "other": "Other instruments",
}
DEFAULT_LAYER = "vocals"

# Caveats shown under the input panel's dropdowns while that option is selected.
OPTION_NOTES = {
    "spectral": "Rebuilds the layer's spectrogram out of sample spectrograms, instead of "
    "extracting notes first. Samples keep their natural length; only timing, pitch and volume change.",
    "drums": "In Classic, drums are unpitched: hits are found by onset detection instead of pitch "
    "tracking, so Pitch confidence, Note tolerance, Min note length and Gap bridging don't apply.",
    "other": "Guitars, keys and whatever else isn't vocals, bass or drums. Usually several "
    "notes at once, so in Classic the melody follows whichever line is most prominent.",
}

# (key, title, algorithms the group applies to -- None means all of them).
# Groups tied to specific algorithms are only shown while one of those is
# selected.
GROUPS = [
    ("output", "Output", None),
    ("melody", "Melody extraction", ("classic",)),
    ("classic", "Sample matching & stretching", ("classic",)),
    ("spectral", "Spectrogram matching", ("spectral",)),
]


def _option(default, label: str, group: str, help: str, lo=None, hi=None, step=None):
    return field(
        default=default,
        metadata={"label": label, "group": group, "help": help, "min": lo, "max": hi, "step": step},
    )


@dataclass
class PipelineSettings:
    algorithm: str = DEFAULT_ALGORITHM
    layer: str = DEFAULT_LAYER

    # --- Output ---------------------------------------------------------
    # Extra gain for the fart track relative to the instrumental before
    # mixing. The fart track keeps the vocal's real dynamic range while a
    # mastered instrumental sits near peak throughout, so without a boost the
    # farts read as quiet for most of the song. The final mix is always
    # peak-normalized afterward, so this can't clip -- it only rebalances
    # fart vs. instrumental loudness.
    fart_gain: float = _option(
        2.5, "Fart volume", "output", "Fart loudness relative to the instrumental (1 = no boost).", 0.1, 10.0, 0.1
    )
    transpose_semitones: int = _option(
        0, "Transpose (semitones)", "output", "Shift the fart melody's pitch (-12 = an octave down).", -24, 24, 1
    )
    # Guard rail: an extreme shift sounds broken rather than "farty".
    max_pitch_shift_semitones: float = _option(
        12.0, "Max pitch shift (semitones)", "output", "Limit on pitch-shifting a sample, up or down.", 0.0, 36.0, 1.0
    )
    mix_with_instrumental: bool = _option(
        True, "Keep backing track", "output", "Mix the farts over the rest of the song; off = farts alone."
    )

    # --- Melody extraction (see notes.segment_notes) ---------------------
    confidence_threshold: float = _option(
        0.35, "Pitch confidence", "melody", "Pitch-tracker frames below this confidence count as silence.", 0.0, 1.0, 0.01
    )
    semitone_tolerance: float = _option(
        0.75, "Note tolerance (semitones)", "melody", "How far pitch may drift before a new note starts.", 0.1, 6.0, 0.05
    )
    min_note_duration_sec: float = _option(
        0.0, "Min note length (s)", "melody", "Shorter notes are discarded (0 keeps everything).", 0.0, 1.0, 0.01
    )
    max_gap_frames: int = _option(
        4, "Gap bridging (frames)", "melody", "Low-confidence 10 ms frames allowed inside one note.", 0, 50, 1
    )
    quiet_note_relative_floor: float = _option(
        0.08, "Quiet floor", "melody", "Fraction of the loudest note below which a note counts as quiet.", 0.0, 1.0, 0.01
    )
    drop_quiet_notes: bool = _option(
        False, "Drop quiet notes", "melody", "Discard notes below the quiet floor (usually tracker noise)."
    )

    # --- Classic: sample matching (see sample_library.find_best_sample) ---
    duration_weight: float = _option(
        1.0, "Duration weight", "classic", "Cost of a sample's length differing from the note's.", 0.0, 5.0, 0.05
    )
    freq_weight: float = _option(
        0.6, "Pitch weight", "classic", "Cost of a sample's pitch differing from the note's.", 0.0, 5.0, 0.05
    )
    volume_weight: float = _option(
        0.4, "Volume weight", "classic", "Cost of a sample's loudness differing from the note's.", 0.0, 5.0, 0.05
    )
    shape_weight: float = _option(
        0.5, "Envelope weight", "classic", "Cost of a sample's attack/decay shape differing from the note's.", 0.0, 5.0, 0.05
    )
    confidence_weight: float = _option(
        0.5, "Tonality weight", "classic", "Penalty for noisy samples with no clear pitch.", 0.0, 5.0, 0.05
    )
    repetition_weight: float = _option(
        0.15, "Variety weight", "classic", "Penalty for reusing a recently played sample.", 0.0, 5.0, 0.05
    )
    # How many notes a just-used sample stays penalized for, before it's
    # fully eligible again. Keeps a long melody from looping the single
    # best-fit fart on every similar note, without permanently ruling it out.
    repeat_cooldown: int = _option(
        6, "Repeat cooldown (notes)", "classic", "How many notes a just-used sample stays penalized for.", 0, 50, 1
    )
    max_stretch: float = _option(
        4.0, "Max stretch (x)", "classic", "Limit on time-stretching a sample, longer or shorter.", 1.0, 16.0, 0.5
    )

    # --- Spectral Synthesis: spectrogram matching (see spectral.decompose) --
    spectral_density: float = _option(
        6.0, "Max placements per second", "spectral", "Cap on how many samples are placed, per second of audio.",
        0.5, 40.0, 0.5,
    )
    # Matching stops early once the best remaining placement explains less
    # than this share of what the strongest one did -- past that point it's
    # adding faint samples to chase detail nobody will hear.
    spectral_stop_percent: float = _option(
        1.0, "Detail floor (%)", "spectral", "Stop once a placement would add less than this % of the strongest one.",
        0.0, 100.0, 0.1,
    )
    spectral_max_gain: float = _option(
        4.0, "Max amplification (x)", "spectral", "Limit on boosting a sample to match the track.", 0.1, 32.0, 0.1
    )


def _option_fields():
    return [f for f in fields(PipelineSettings) if f.metadata]


def settings_schema() -> list[dict]:
    """The option fields grouped for display, in the shape the input panel's
    template renders."""
    groups = []
    for key, title, algorithms in GROUPS:
        group_fields = []
        for f in _option_fields():
            if f.metadata["group"] != key:
                continue
            group_fields.append(
                {
                    "name": f.name,
                    "kind": "checkbox" if isinstance(f.default, bool) else "number",
                    "default": f.default,
                    "label": f.metadata["label"],
                    "help": f.metadata["help"],
                    "min": f.metadata["min"],
                    "max": f.metadata["max"],
                    "step": f.metadata["step"],
                }
            )
        groups.append({"title": title, "algorithms": algorithms or (), "fields": group_fields})
    return groups


def parse_settings(raw) -> PipelineSettings:
    """Build a PipelineSettings from untrusted input (the JSON the input
    panel posts). Missing keys keep their defaults and unknown keys are
    ignored; a wrong type or out-of-range value raises ValueError with a
    message fit to show the user."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("expected an object of settings")

    settings = PipelineSettings()

    algorithm = raw.get("algorithm", DEFAULT_ALGORITHM)
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    settings.algorithm = algorithm

    layer = raw.get("layer", DEFAULT_LAYER)
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}")
    settings.layer = layer

    for f in _option_fields():
        if f.name not in raw:
            continue
        value = raw[f.name]
        label = f.metadata["label"]
        if isinstance(f.default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be on or off")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{label} must be a number")
            if isinstance(f.default, int):
                if value != int(value):
                    raise ValueError(f"{label} must be a whole number")
                value = int(value)
            else:
                value = float(value)
            lo, hi = f.metadata["min"], f.metadata["max"]
            if not lo <= value <= hi:
                raise ValueError(f"{label} must be between {lo} and {hi}")
        setattr(settings, f.name, value)

    return settings
