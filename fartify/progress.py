"""Time-based progress estimation for a pipeline run.

Most of a run is spent inside a few long, opaque steps (Demucs, the pitch
tracker) that can't report how far along they are. So instead of jumping
the progress bar only when a step finishes, each step gets an *expected*
duration -- a fixed overhead plus a rate per unit of work (seconds of input
audio for most steps, notes for resynthesis) -- and the bar advances
through that step as wall-clock time passes.

The expected durations are self-calibrating: every finished step appends
(units of work, seconds taken) to a small history file, and later estimates
are the built-in model scaled by how this machine has actually performed.
The built-in numbers only have to be roughly right for the first run.

The pieces:
  - ProgressTracker is driven by the pipeline (start a stage, optionally
    report measured progress within it) and emits a flat, JSON-friendly
    state dict on every change.
  - snapshot() turns that state plus the current time into an overall
    fraction and an ETA. It's a pure function so whoever holds the state
    (the web app's job store) can evaluate it whenever it's asked.
"""
from __future__ import annotations

import json
import logging
import math
import os
import statistics
import threading
import time

logger = logging.getLogger("fartify.progress")

# Built-in cost model per stage: (fixed overhead sec, sec per unit of work).
# The unit is a second of input audio, except for resynthesis, whose cost
# tracks the number of notes far more closely than the song's length.
# Measured on a CPU-only desktop with Demucs + torchcrepe installed (2.3 and
# 5 minute songs); the history file corrects for anything faster or slower.
DEFAULT_STAGE_COSTS = {
    "separate": (5.0, 0.73),
    "pitch": (0.5, 0.125),
    "onsets": (0.0, 0.0016),
    "synth": (0.5, 0.068),
    "mix": (0.1, 0.0015),
    # Spectral Synthesis: matching the spectrogram, then rendering placements.
    "spectral_match": (0.2, 0.017),
    "spectral_render": (0.3, 0.005),
}

# Stages whose unit of work is notes (or sample placements) rather than
# audio seconds. The real count isn't known until the melody has been
# extracted, so until then the stage is sized with a typical note density
# (vocals run about 4-5 per second; bass and drums fewer).
NOTE_COSTED_STAGES = {"synth", "spectral_render"}
TYPICAL_NOTES_PER_SEC = 4.0

# How many past runs per stage feed the calibration. A median over a short
# window shrugs off one odd run (machine busy, two jobs at once) while still
# tracking a real change such as installing a GPU build of torch.
HISTORY_WINDOW = 10

# A time-estimated stage runs linearly up to this fraction at its expected
# duration, then creeps asymptotically toward (never reaching) the end -- so
# a step that overruns keeps visibly moving instead of sitting at "100%".
ON_TIME_FRACTION = 0.9
MAX_ESTIMATED_FRACTION = 0.99

_HISTORY_LOCK = threading.Lock()


def _default_estimate(stage: str, units: float) -> float:
    overhead, rate = DEFAULT_STAGE_COSTS[stage]
    return overhead + rate * units


def _load_history(path: str | None) -> dict[str, list]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("couldn't read timing history %s, using built-in estimates", path)
        return {}


def _record_timing(path: str | None, stage: str, units: float, elapsed_sec: float) -> None:
    if not path:
        return
    with _HISTORY_LOCK:
        history = _load_history(path)
        runs = history.get(stage, [])
        runs.append([round(units, 2), round(elapsed_sec, 2)])
        history[stage] = runs[-HISTORY_WINDOW:]
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(history, f)
            os.replace(tmp_path, path)
        except OSError:
            logger.exception("failed to save timing history %s", path)


def estimate_stage_seconds(stage: str, units: float, history: dict[str, list]) -> float:
    """Expected duration of `stage` for this much work: the built-in model,
    scaled by the median of (actual / built-in) over this machine's recent
    runs of that stage."""
    estimate = _default_estimate(stage, units)
    ratios = []
    for run in history.get(stage, []):
        try:
            past_units, past_elapsed_sec = float(run[0]), float(run[1])
        except (TypeError, ValueError, IndexError):
            continue
        baseline = _default_estimate(stage, past_units)
        if baseline > 0 and past_elapsed_sec > 0:
            ratios.append(past_elapsed_sec / baseline)
    if ratios:
        estimate *= statistics.median(ratios)
    return max(estimate, 0.1)


class ProgressTracker:
    """Tracks one run through an ordered list of stages.

    stages: [(key, label)] in the order they'll run; key picks the cost
    model (DEFAULT_STAGE_COSTS) and label is what the user sees.
    on_update: called with the state dict (see state()) on every change.
    history_path: where timings are read from / recorded to; None disables
    calibration and just uses the built-in model.
    """

    def __init__(self, stages, audio_sec: float, history_path: str | None = None, on_update=None):
        self._history_path = history_path
        self._on_update = on_update
        self._history = _load_history(history_path)
        self._order = [key for key, _ in stages]
        self._labels = dict(stages)
        self._units = {
            key: audio_sec * TYPICAL_NOTES_PER_SEC if key in NOTE_COSTED_STAGES else audio_sec for key in self._order
        }
        self._estimates = {key: estimate_stage_seconds(key, self._units[key], self._history) for key in self._order}
        self._current: str | None = None
        self._started_at = 0.0
        self._started_clock = 0.0
        self._fraction: float | None = None
        self._fraction_floor = 0.0
        self._span = (0.0, 0.0)  # share of the overall bar the current stage covers
        self._later_est = 0.0  # expected seconds for every stage after the current one
        self._last_emit_clock = 0.0

    def start(self, stage: str, units: float | None = None) -> None:
        """Finish whatever stage was running and begin `stage`. Pass `units`
        when the stage's real amount of work is now known (e.g. the note
        count, for resynthesis) to replace the up-front guess."""
        self._close_current()
        if units is not None:
            self._units[stage] = units
            self._estimates[stage] = estimate_stage_seconds(stage, units, self._history)
        # The stage takes over where the last one's span ended and claims its
        # share of whatever bar is left, by expected time. Spans are handed
        # out one stage at a time like this (rather than fixed up front) so
        # a revised estimate can't make the bar run backwards.
        self._later_est = sum(self._estimates[key] for key in self._order[self._order.index(stage) + 1 :])
        begin = self._span[1]
        share = self._estimates[stage] / (self._estimates[stage] + self._later_est)
        self._span = (begin, begin + (1.0 - begin) * share)
        self._current = stage
        self._started_at = time.time()
        self._started_clock = time.monotonic()
        self._fraction = None
        self._emit()

    def advance(self, fraction: float) -> None:
        """Report *measured* progress (0..1) within the current stage, for
        steps that can actually count their work. Overrides the time-based
        guess for that stage. Emits at most about once a second."""
        fraction = float(min(max(fraction, 0.0), 1.0))
        if self._fraction is None:
            # Until now the bar has been creeping along on the time-based
            # guess. Measured progress picks up from wherever that got to
            # and covers the rest of the stage, so the bar can't step back
            # if the stage had setup work before it started counting.
            elapsed = time.monotonic() - self._started_clock
            self._fraction_floor = _estimated_fraction(elapsed, self._estimates[self._current])
        self._fraction = self._fraction_floor + (1.0 - self._fraction_floor) * fraction
        if time.monotonic() - self._last_emit_clock >= 1.0:
            self._emit()

    def finish(self) -> None:
        """Close out the last stage (recording its timing)."""
        self._close_current()
        self._current = None

    def state(self) -> dict:
        """Flat, JSON-serializable progress state -- everything snapshot()
        needs to work out the overall fraction/ETA at any later moment."""
        return {
            "stage": self._labels.get(self._current, ""),
            "stage_started_at": self._started_at,
            "stage_est_sec": self._estimates.get(self._current, 0.0),
            "stage_fraction": self._fraction,
            "stage_span": list(self._span),
            "later_est_sec": self._later_est,
        }

    def _close_current(self) -> None:
        if self._current is None:
            return
        elapsed = time.monotonic() - self._started_clock
        _record_timing(self._history_path, self._current, self._units[self._current], elapsed)

    def _emit(self) -> None:
        self._last_emit_clock = time.monotonic()
        if self._on_update:
            self._on_update(self.state())


def _estimated_fraction(elapsed_sec: float, stage_est_sec: float) -> float:
    """Time-based guess at how far through a stage we are."""
    x = elapsed_sec / stage_est_sec if stage_est_sec > 0 else 1.0
    if x < 1.0:
        return ON_TIME_FRACTION * x
    tail = MAX_ESTIMATED_FRACTION - ON_TIME_FRACTION
    return ON_TIME_FRACTION + tail * (1.0 - math.exp(-(x - 1.0)))


def snapshot(state: dict, now: float | None = None) -> tuple[float, float | None]:
    """(overall fraction 0..1, estimated seconds remaining or None) for a
    ProgressTracker state dict as of `now` (defaults to the current time)."""
    span = state.get("stage_span")
    if not span:
        return float(state.get("progress") or 0.0), None

    now = time.time() if now is None else now
    stage_est = state.get("stage_est_sec") or 0.0
    elapsed = max(now - (state.get("stage_started_at") or now), 0.0)
    fraction = state.get("stage_fraction")
    if fraction is None:
        fraction = _estimated_fraction(elapsed, stage_est)
        stage_remaining = stage_est * (1.0 - fraction)
    elif fraction >= 0.05:
        # Measured progress: the pace so far is a better guide than the estimate.
        stage_remaining = elapsed * (1.0 - fraction) / fraction
    else:
        stage_remaining = stage_est * (1.0 - fraction)

    begin, end = span
    progress = begin + fraction * (end - begin)
    eta_sec = stage_remaining + (state.get("later_est_sec") or 0.0)
    return progress, eta_sec
