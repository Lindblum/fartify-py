# Fartify 💨

Upload a song and Fartify replays one of its layers (vocals by default) as
fart samples, mixed back over the rest of the track. You get a WAV and a
MIDI file of the extracted notes.

## Run it

**Docker** (bundles ffmpeg and CPU-only torch/Demucs/CREPE):

```bash
docker compose up --build
```

**Locally** (needs `ffmpeg` on PATH):

```bash
pip install -r requirements.txt
python scripts/generate_sample_farts.py   # once, seeds placeholder samples
python app.py
```

Then open http://localhost:5181.

## How it works

1. **Separate** the chosen layer (vocals, bass, drums, other) from the
   backing track with [Demucs](https://github.com/facebookresearch/demucs).
2. **Track** the layer's melody with CREPE (pYIN or autocorrelation as
   fallbacks). Drums use onset detection, with brightness standing in for pitch.
3. **Resynthesize** with farts using one of two algorithms:
   - **Spectral Synthesis** (default): at each detected beat, stamp the
     sample, pitch shift and gain whose log-frequency spectrogram best fits
     that slice of the layer. Pitched samples are pinned to the melody note.
   - **Classic**: split the melody into notes, then pitch-shift and
     time-stretch the closest-matching sample onto each note.
4. **Mix** the result over the backing track and export WAV + MIDI.

Calibration options (volume, transpose, pitch confidence, contrast, etc.)
are defined in `fartify/settings.py` and exposed in the UI.

## Lite mode

`torch`, `demucs` and `torchcrepe` are optional. Without them, Fartify
still runs end to end on `numpy`, `scipy`, `flask` and `ffmpeg`, falling
back to a center-channel vocal split, classical pitch tracking, and a
built-in phase vocoder. Quality drops noticeably, and only the vocals layer
is available.

## Samples

Drop WAV files into `samples/`; they're analyzed (duration, volume, main
note, spectrogram) and indexed automatically on the next run.

## Layout

```
app.py              Flask web app
fartify/            Pipeline: separation, pitch, notes, synth, spectral, MIDI export
samples/            Fart WAVs + analysis cache
scripts/            Placeholder sample generator
static/, templates/ Web UI
```

Uploads and separated stems are deleted after 3 hours; finished outputs
stay until the job is deleted.
