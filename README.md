# Fartify 💨

Upload a song. Fartify isolates the vocals, extracts the melody as a
sequence of notes, and resynthesizes that melody using fart samples —
pitch-shifted and time-stretched to match the original notes — mixed back
over the instrumental.

## How it works

```
input audio ──▶ vocal separation ──▶ pitch/melody extraction ──▶ note segmentation
                      │                                                  │
                      ▼                                                  ▼
              instrumental stem                          match each note to a fart sample,
                      │                                   pitch-shift + time-stretch it
                      │                                                  │
                      └─────────────────▶  mix  ◀───────────────────────┘
                                            │
                                  fartify_output.wav + fartify_melody.mid
```

1. **Layer separation** — splits the upload into the chosen layer (vocals
   by default) and a backing track of everything else.
2. **Melody extraction** — tracks the fundamental frequency (pitch) of the
   vocal stem over time.
3. **Note segmentation** — groups the frame-by-frame pitch curve into
   discrete notes (start time, duration, frequency, volume).
4. **Sample matching & resynthesis** — for each note, picks the closest
   sample from `samples/` (by duration/volume/frequency), pitch-shifts and
   time-stretches it to fit the note exactly, and places it in the output
   track.
5. **Mixdown & export** — the fart melody is mixed with the instrumental
   stem and exported as WAV, alongside a MIDI file of the extracted notes.

## Architecture decisions

### Vocal separation: Demucs

[Demucs](https://github.com/facebookresearch/demucs) (`htdemucs`, two-stems
mode) is the primary separation method. It's the pragmatic choice over
alternatives like Spleeter or Open-Unmix: actively maintained,
state-of-the-art quality, and its `--two-stems=vocals` mode does exactly
the "vocals vs. everything else" split this project needs with one CLI
call. Pretrained weights download automatically on first use.

Demucs (and its `torch` dependency) is a large, optional install. When it
isn't available, Fartify falls back to a classic **center-channel trick**
(`mid = (L+R)/2`, `side = (L-R)/2`) so the app still runs end to end on a
lightweight install. This fallback is clearly logged and surfaced in the
UI as degraded — it captures any center-panned instrument along with the
vocal, and can't do anything useful on mono input. It exists to keep the
pipeline demoable without a multi-GB install, not as a real substitute for
Demucs.

### Melody extraction: CREPE (via torchcrepe), not RMVPE

**CREPE** was chosen over RMVPE for the pitch tracker:

- CREPE has a standard, stable PyTorch port (`torchcrepe`) that installs
  with one `pip install` and needs no extra checkpoint wrangling.
- RMVPE comes out of the RVC (singing voice conversion) ecosystem, tuned
  for robustness in real-time voice-conversion pipelines. Its main edge
  over CREPE is inference speed in a streaming context — irrelevant here,
  since Fartify is an offline batch pipeline.
- CREPE's published accuracy on monophonic melody extraction is on par
  with or better than RMVPE for this offline use case, with much simpler
  packaging.

Like Demucs, `torchcrepe` is optional. Pitch extraction falls back through
three tiers automatically (see `fartify/pitch.py`):

1. **CREPE** (`torchcrepe`) — best quality, used if installed.
2. **librosa pYIN** — a solid classical probabilistic tracker, no ML
   runtime required.
3. **Autocorrelation** — a small dependency-free tracker (pure
   numpy/scipy) so pitch extraction always works, even with zero optional
   packages installed.

### Time-stretch / pitch-shift: librosa, with a built-in fallback

Resynthesizing a fart sample to match a melody note requires both
pitch-shifting and time-stretching. When `librosa` is installed, its
phase-vocoder-based `pitch_shift`/`time_stretch` is used directly. When
it isn't, `fartify/synth.py` includes a small self-contained phase-vocoder
implementation (STFT-based, ~50 lines) composed with resampling to hit
the same target duration + frequency — so the whole pipeline, including
resynthesis, works without any optional dependency installed.

### Everything else is dependency-light on purpose

- **Audio I/O** goes through the `ffmpeg` binary (already required for
  MP3 support) rather than `soundfile`/`pydub`, so any input format
  ffmpeg understands (MP3, WAV, M4A, FLAC, OGG, AAC, ...) just works.
- **MIDI export** is a from-scratch ~80-line Standard MIDI File writer
  (`fartify/midi_export.py`) instead of a `pretty_midi`/`mido` dependency.
- **Web UI** is a single-page Flask app with vanilla JS — file upload,
  a progress bar polling `/status/<job>`, an HTML5 `<audio>` player, and
  download buttons for the WAV and MIDI outputs.

This means Fartify runs in a useful "lite mode" with just
`numpy`, `scipy`, `flask`, and `ffmpeg` on PATH. Installing `torch`,
`demucs`, and `torchcrepe` (see `requirements.txt`) upgrades separation
and pitch-tracking quality without changing how you use the app.

## Algorithms and calibration

The input panel has **Algorithm** and **Layer** dropdowns and a collapsible
**Calibration** section. All are defined in `fartify/settings.py`:
`PipelineSettings` holds every option's default, range and label, the panel
is rendered from it, and each job records the settings it ran with.

- **Classic** (default) — the note-based pipeline described above.
- **Spectral Synthesis** — skips note extraction and rebuilds the layer's
  spectrogram directly out of sample spectrograms; see below.

**Layer** picks which part of the song is isolated and replayed as farts;
everything else becomes the backing track:

- **Vocals** (default), **Bass**, **Other instruments** — pitch-tracked
  into a melody, each with its own search range (`pitch.LAYER_PITCH_RANGES`).
- **Drums** — unpitched, so hits are found by onset detection
  (`notes.detect_hits`) and each hit's brightness stands in for pitch: kicks
  become low farts, hi-hats high ones.

Any layer other than vocals needs Demucs; the center-channel fallback can
only approximate vocals.

Calibration fields cover the output (fart volume, transpose, pitch-shift
limit, keep backing track), melody extraction (pitch confidence, note tolerance, minimum
note length, gap bridging, quiet-note filtering) and, for Classic only, the
sample-matching weights and the pitch-shift / time-stretch limits.

## Spectrograms and Spectral Synthesis

Every sample gets a **log-frequency spectrogram** when it's analyzed
(`fartify/spectrogram.py`): time left to right, pitch bottom to top on a
logarithmic axis (24 rows per octave, C1 to about 8.4 kHz), value =
amplitude. The array and a grayscale PNG (brightness = amplitude, in dB) are
cached in `samples/spectrograms/` and shown on the sample library page.

Spectral Synthesis (`fartify/spectral.py`) computes the same spectrogram,
with the same parameters, for the isolated layer, then approximates it by
stamping sample spectrograms onto it. Each stamp — a *placement* — has:

- a **time** (horizontal position),
- a **pitch adjustment** (vertical position; on a log axis, moving a stamp
  up or down is exactly a transposition),
- an **amplification** (brightness).

Placements are chosen by matching pursuit: take the one placement that
explains the most of what's still unexplained, subtract it, repeat — until
the per-second cap is reached or the next placement would add too little.
The audio is then rendered straight from the list: each sample pitch-shifted
(length unchanged), scaled, and added at its time. Samples are never
time-stretched in this mode.

A finished job shows the layer's spectrogram next to the one rebuilt from
samples, and how much of the layer's spectrogram energy the placements
account for. Its work directory also keeps `spectral_placements.json` (the
full list of times, pitch shifts and gains). The MIDI download lists each
pitched sample's placement as a note.

## Progress estimates

Most of a run is spent inside steps that can't report how far along they
are (Demucs, the pitch tracker). `fartify/progress.py` gives each stage an
expected duration — a fixed overhead plus a rate per second of input audio
— and the progress bar advances through the stage as time passes, along
with a rough time-remaining figure. Resynthesis reports real progress
(notes rendered so far) instead.

The estimates calibrate themselves: each finished stage is logged to
`timings.json`, and later estimates are scaled by how this machine has
actually performed over its last few runs.

## The sample library

`samples/` holds the WAV fart sounds used for resynthesis, plus a
`samples_index.json` cache of each sample's analyzed **duration**,
**volume** (RMS + dBFS), and **main note frequency** (Hz + nearest MIDI
note). The index is rebuilt automatically whenever a file is added,
removed, or changed — no manual step needed after dropping in new
samples.

To start, `scripts/generate_sample_farts.py` procedurally synthesizes a
small varied set of placeholder fart one-shots (different pitches,
durations, and intensities) so the app is runnable out of the box without
sourcing real recordings up front:

```bash
python scripts/generate_sample_farts.py
```

Add your own recorded samples to `samples/` at any time — just drop WAV
files in; they'll be picked up and analyzed on the next run.

## Setup

```bash
pip install -r requirements.txt      # torch/demucs/torchcrepe are optional
                                       # but recommended — see architecture notes above
python scripts/generate_sample_farts.py   # only needed once, to seed samples/
python app.py
```

Then open http://localhost:5000, upload a song, and wait for the progress
bar. When it's done you get an inline player plus download buttons for the
generated WAV and the extracted melody's MIDI file.

`ffmpeg` must be installed and on PATH (`apt install ffmpeg` /
`brew install ffmpeg` / etc.) — it's used for all audio format handling.

## Project layout

```
fartify/
  app.py                  Flask web app (upload, job status, downloads)
  fartify/
    utils.py              Audio I/O (via ffmpeg) + DSP helpers
    separation.py          Demucs wrapper + center-channel fallback
    pitch.py               CREPE / pYIN / autocorrelation pitch tracking
    notes.py                Pitch curve -> discrete Note segmentation
    sample_library.py      Sample analysis, index, best-match selection
    synth.py                Pitch-shift/time-stretch + track resynthesis
    midi_export.py          Dependency-free Standard MIDI File writer
    spectrogram.py          Log-frequency spectrograms (+ sample cache, PNGs)
    spectral.py             Spectral Synthesis: spectrogram matching + rendering
    pipeline.py              Orchestrates the full run
    progress.py              Per-stage time estimates for the progress bar
    settings.py              Algorithm choice + calibration options
  samples/                Fart sample WAVs + samples_index.json
  scripts/
    generate_sample_farts.py   Seeds samples/ with synthetic placeholders
  static/, templates/     Web UI assets
  uploads/, output/        Runtime scratch space (gitignored)
```

Each job's uploaded original (`uploads/<job>_<name>`) and separated stems
(`output/<job>/demucs_out/`, `layer.wav`, `backing.wav`) are only needed
while it runs, and the stems are most of its disk use; the app deletes both
once they're 3 hours old (checked at startup and every 15 minutes). The
finished audio, MIDI and spectrogram images stay until the job is deleted.

## Known limitations

- The center-channel separation fallback (no Demucs installed) can't
  isolate vocals from mono input, and leaks center-panned instruments
  into the vocal stem from stereo input.
- The autocorrelation pitch fallback (no torchcrepe/librosa installed) is
  noticeably less robust on breathy, heavily reverberant, or
  harmony-heavy vocals than CREPE.
- Very short or very long melody notes are matched to the closest-fitting
  sample and stretched/shifted within a clamped range (0.5x–2x pitch,
  0.25x–4x duration) to avoid audibly broken artifacts — an extreme
  mismatch between a note and the available sample library will still
  sound more "stretched" than a natural fart.
