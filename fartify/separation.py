"""Layer / backing-track separation.

Splits a song into the one layer Fartify is going to replay as farts
(vocals by default, or bass / drums / other -- see settings.LAYERS) and a
backing track of everything else.

Primary method: Demucs (htdemucs, via the `demucs` package's two-stems
mode). Demucs is the pragmatic choice here over alternatives like Spleeter
or Open-Unmix: it's actively maintained, has state-of-the-art separation
quality (it's what most modern karaoke/stem-splitting tools use under the
hood), ships a simple two-stems=<stem> CLI mode that does exactly the
"one layer vs. everything else" split Fartify needs, and its pretrained
weights download automatically on first use.

Demucs (and its torch dependency) is a large, optional install — see
requirements.txt. When it isn't available, Fartify falls back to a classic
"center-channel" trick (mid = (L+R)/2, side = (L-R)/2) so the app still
runs end to end on a lightweight install. This fallback is clearly a
degraded approximation (it captures any center-panned instrument along
with the vocal, and does nothing useful on mono input) and is logged as
such — it exists to keep the pipeline demoable, not as a real substitute
for Demucs. It can only approximate vocals: asking for any other layer
without a working Demucs is an error rather than a silently wrong result.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

from .utils import TARGET_SR, load_audio, save_audio

logger = logging.getLogger("fartify.separation")

# What separation leaves in a job's work dir. All of it is scratch: once the
# run has produced its output nothing reads these again, and they're by far
# the bulk of a job's disk use (Demucs writes full-quality stereo stems), so
# the app deletes them after a while. The last two are what layer.wav /
# backing.wav were called before layers other than vocals existed.
DEMUCS_OUT_DIRNAME = "demucs_out"
LAYER_FILENAME = "layer.wav"
BACKING_FILENAME = "backing.wav"
INTERMEDIATE_NAMES = (DEMUCS_OUT_DIRNAME, LAYER_FILENAME, BACKING_FILENAME, "vocals.wav", "instrumental.wav")


class SeparationResult:
    def __init__(self, layer_path: str, backing_path: str, method: str, degraded: bool):
        self.layer_path = layer_path  # the isolated layer
        self.backing_path = backing_path  # everything else
        self.method = method
        self.degraded = degraded


def separate_layer(input_path: str, work_dir: str, layer: str = "vocals", sr: int = TARGET_SR) -> SeparationResult:
    """Isolate `layer` (a Demucs stem name: vocals/bass/drums/other) from
    everything else in the song."""
    os.makedirs(work_dir, exist_ok=True)

    try:
        return _separate_demucs(input_path, work_dir, layer, sr)
    except ImportError:
        if layer != "vocals":
            raise RuntimeError(
                f"Isolating the {layer} layer needs Demucs, which isn't installed "
                "(the lite fallback can only approximate vocals)."
            )
        logger.info("demucs not installed — using degraded center-channel fallback")
    except Exception as exc:
        if layer != "vocals":
            raise RuntimeError(f"Demucs couldn't isolate the {layer} layer: {exc}") from exc
        logger.warning("demucs separation failed (%s) — using degraded center-channel fallback", exc)

    return _separate_naive(input_path, work_dir, sr)


def _separate_demucs(input_path: str, work_dir: str, layer: str, sr: int) -> SeparationResult:
    import demucs.separate  # noqa: F401 — just to trigger ImportError early if absent

    out_dir = os.path.join(work_dir, DEMUCS_OUT_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        layer,
        "-o",
        out_dir,
        input_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"demucs failed: {result.stderr.decode(errors='replace')[-800:]}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    # demucs writes <out_dir>/<model_name>/<base>/{<layer>,no_<layer>}.wav
    model_dirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    if not model_dirs:
        raise RuntimeError("demucs produced no output directory")
    track_dir = os.path.join(out_dir, model_dirs[0], base)

    layer_src = os.path.join(track_dir, f"{layer}.wav")
    backing_src = os.path.join(track_dir, f"no_{layer}.wav")
    if not os.path.exists(layer_src):
        raise RuntimeError(f"expected demucs output not found at {layer_src}")

    layer_path = os.path.join(work_dir, LAYER_FILENAME)
    backing_path = os.path.join(work_dir, BACKING_FILENAME)

    y, _ = load_audio(layer_src, sr=sr)
    save_audio(layer_path, y, sr=sr)
    if os.path.exists(backing_src):
        y2, _ = load_audio(backing_src, sr=sr)
        save_audio(backing_path, y2, sr=sr)
    else:
        save_audio(backing_path, y * 0, sr=sr)

    return SeparationResult(layer_path, backing_path, method="demucs (htdemucs)", degraded=False)


def _separate_naive(input_path: str, work_dir: str, sr: int) -> SeparationResult:
    y, _ = load_audio(input_path, sr=sr, mono=False)

    layer_path = os.path.join(work_dir, LAYER_FILENAME)
    backing_path = os.path.join(work_dir, BACKING_FILENAME)

    if y.ndim == 2 and y.shape[1] == 2:
        left, right = y[:, 0], y[:, 1]
        mid = (left + right) * 0.5  # vocals + center-panned instruments
        side = (left - right) * 0.5  # everything panned away from center
        save_audio(layer_path, mid, sr=sr)
        save_audio(backing_path, side, sr=sr)
    else:
        mono = y if y.ndim == 1 else y.mean(axis=1)
        logger.warning("input is mono — cannot isolate a center channel; using full mix as vocal estimate")
        save_audio(layer_path, mono, sr=sr)
        save_audio(backing_path, mono * 0, sr=sr)

    return SeparationResult(
        layer_path, backing_path, method="center-channel fallback (no Demucs)", degraded=True
    )
