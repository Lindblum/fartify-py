"""Fartify — Flask web app.

Simple, single-page interface: upload an audio file (mp3/wav/etc.),
watch it process (vocal separation -> melody extraction -> fart
resynthesis), then play and download the result (audio + MIDI of the
extracted note data).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename

from fartify.pipeline import APPROX_SPECTROGRAM_PNG, TARGET_SPECTROGRAM_PNG, run_pipeline
from fartify.progress import snapshot
from fartify.sample_library import (
    add_sample_record,
    analyze_sample,
    build_index,
    load_index,
    remove_sample_record,
)
from fartify.spectrogram import sample_spectrogram_png
from fartify.separation import INTERMEDIATE_NAMES
from fartify.settings import ALGORITHMS, LAYERS, OPTION_NOTES, PipelineSettings, parse_settings, settings_schema
from fartify.utils import midi_to_note_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
JOBS_FILE = os.path.join(BASE_DIR, "jobs_state.json")
# Per-stage timings from past runs, used to estimate how long each stage of
# the next one will take (see fartify/progress.py).
TIMINGS_FILE = os.path.join(BASE_DIR, "timings.json")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# A job's uploaded original and its intermediate files (the separated stems
# -- see separation.INTERMEDIATE_NAMES) are deleted once they're this old.
# The finished audio, MIDI and spectrogram images are kept until the job
# itself is deleted.
INTERMEDIATE_MAX_AGE_SEC = 3 * 60 * 60
INTERMEDIATE_CLEANUP_INTERVAL_SEC = 15 * 60

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "flac", "ogg", "aac"}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Job store: {job_id: {status, stage, progress, error, result, filename, settings, created_at}},
# plus, while a job is processing, the progress-tracker state (stage_est_sec
# etc.) that _job_view turns into a live progress fraction + ETA.
# Kept in memory for speed, but every change is also written through to
# JOBS_FILE so the job list -- and the page's per-job panels -- survive a
# browser reload or an app restart, not just a full re-render.
JOBS_LOCK = threading.Lock()

# Guards read-modify-write access to samples/samples_index.json (and the
# sample files themselves) when a sample is added or deleted through the
# web UI, so two requests can't race and corrupt the index.
SAMPLES_LOCK = threading.Lock()


def _load_jobs() -> dict:
    if not os.path.exists(JOBS_FILE):
        return {}
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logging.exception("failed to load %s, starting with an empty job list", JOBS_FILE)
        return {}
    # A job that was still queued/processing when the process last stopped
    # can't be resumed -- its background thread is gone -- so mark it failed
    # instead of leaving it stuck "processing" forever.
    for job in data.values():
        if job.get("status") in ("queued", "processing"):
            job["status"] = "error"
            job["stage"] = "interrupted"
            job["error"] = "Interrupted by a server restart. Please re-upload."
    return data


def _save_jobs_locked():
    """Persist JOBS to disk. Caller must hold JOBS_LOCK."""
    tmp_path = JOBS_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(JOBS, f)
        os.replace(tmp_path, JOBS_FILE)
    except OSError:
        logging.exception("failed to save %s", JOBS_FILE)


JOBS: dict[str, dict] = _load_jobs()


_UPLOAD_JOB_ID = re.compile(r"[0-9a-f]{32}")


def _path_size(path: str) -> int:
    if not os.path.isdir(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)


def cleanup_intermediate_files(max_age_sec: float = INTERMEDIATE_MAX_AGE_SEC) -> int:
    """Delete files a finished job no longer needs once they're older than
    max_age_sec: the intermediate processing files in every job's work dir,
    and the uploaded originals. Returns the number of bytes freed.

    Jobs still queued or processing are skipped whatever their files' age,
    so a run is never pulled out from under itself."""
    cutoff = time.time() - max_age_sec
    with JOBS_LOCK:
        active = {job_id for job_id, job in JOBS.items() if job.get("status") in ("queued", "processing")}

    candidates = []
    for job_id in os.listdir(OUTPUT_DIR):
        work_dir = os.path.join(OUTPUT_DIR, job_id)
        if job_id in active or not os.path.isdir(work_dir):
            continue
        candidates += [os.path.join(work_dir, name) for name in INTERMEDIATE_NAMES]
    for fname in os.listdir(UPLOADS_DIR):
        # Only what /upload itself saved ("<32-hex job id>_<name>") -- not
        # .gitkeep, or anything else someone has put in the folder by hand.
        job_id, sep, _ = fname.partition("_")
        if sep and _UPLOAD_JOB_ID.fullmatch(job_id) and job_id not in active:
            candidates.append(os.path.join(UPLOADS_DIR, fname))

    freed = 0
    for path in candidates:
        try:
            if os.path.getmtime(path) > cutoff:
                continue
            size = _path_size(path)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            freed += size
        except FileNotFoundError:
            continue  # this job never had that file, or it's already gone
        except OSError:
            logging.exception("couldn't remove old file %s", path)
    if freed:
        logging.info(
            "removed %.1f MB of uploads and intermediate files older than %.0f h", freed / 1e6, max_age_sec / 3600
        )
    return freed


def _cleanup_loop():
    while True:
        try:
            cleanup_intermediate_files()
        except Exception:  # noqa: BLE001 -- a failed sweep must not kill the loop
            logging.exception("intermediate file cleanup failed")
        time.sleep(INTERMEDIATE_CLEANUP_INTERVAL_SEC)


# Sweeps once at startup, then every INTERMEDIATE_CLEANUP_INTERVAL_SEC.
threading.Thread(target=_cleanup_loop, daemon=True).start()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _set_job(job_id: str, **fields):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return  # job was deleted while its background thread was still running
        job.update(fields)
        _save_jobs_locked()


def _process_job(job_id: str, input_path: str, settings: PipelineSettings):
    work_dir = os.path.join(OUTPUT_DIR, job_id)

    def progress_cb(state: dict):
        _set_job(job_id, **state)

    try:
        _set_job(job_id, status="processing", stage="starting", progress=0.0)
        result = run_pipeline(
            input_path=input_path,
            work_dir=work_dir,
            samples_dir=SAMPLES_DIR,
            settings=settings,
            progress_cb=progress_cb,
            timings_path=TIMINGS_FILE,
        )
        job_result = {
            "audio_url": f"/download/{job_id}/audio",
            "midi_url": f"/download/{job_id}/midi",
            "separation_method": result.separation_method,
            "separation_degraded": result.separation_degraded,
            "pitch_backend": result.pitch_backend,
            "note_count": result.note_count,
            "duration_sec": round(result.duration_sec, 2),
        }
        if result.explained_fraction is not None:  # Spectral Synthesis
            job_result["explained_fraction"] = round(result.explained_fraction, 4)
            job_result["layer_spectrogram_url"] = f"/jobs/{job_id}/spectrogram/layer"
            job_result["approx_spectrogram_url"] = f"/jobs/{job_id}/spectrogram/approx"
        _set_job(job_id, status="done", progress=1.0, stage="done", result=job_result)
    except Exception as exc:  # noqa: BLE001
        logging.exception("job %s failed", job_id)
        _set_job(job_id, status="error", error=str(exc))


@app.route("/")
def index():
    records = load_index(SAMPLES_DIR)
    # Filenames are handed to the page so it can play a random sample as a
    # sound effect (e.g. on "Fartify it" or confirming a delete) without a
    # round trip just to ask "which one" -- only the audio bytes themselves
    # need fetching, via /samples/<filename> below.
    sample_files = [r.filename for r in records]
    def options(choices: dict) -> list[dict]:
        return [{"value": key, "label": label, "note": OPTION_NOTES.get(key, "")} for key, label in choices.items()]

    return render_template(
        "index.html",
        sample_count=len(records),
        sample_files=sample_files,
        algorithms=options(ALGORITHMS),
        layers=options(LAYERS),
        setting_groups=settings_schema(),
    )


def _describe_sample(r) -> dict:
    """Render a SampleRecord's analyzed metadata into the display strings
    the /samples page (and the JSON returned after an upload) shows."""
    length = f"{r.duration_sec:.2f}s"
    if abs(r.effective_duration_sec - r.duration_sec) >= 0.01:
        length += f" ({r.effective_duration_sec:.2f}s audible)"
    if r.frequency_hz > 0:
        pitch = f"{midi_to_note_name(r.midi_note)} (~{r.frequency_hz:.0f} Hz, {round(r.pitch_confidence * 100)}% confident)"
    else:
        pitch = "unpitched"
    return {
        "filename": r.filename,
        "length": length,
        "volume": f"{r.dbfs:.1f} dBFS",
        "pitch": pitch,
        "emphasis": f"{r.emphasis_sec:.2f}s",
        "spectrogram_url": url_for("sample_spectrogram", filename=r.filename),
    }


def _safe_sample_path(filename: str) -> str | None:
    """Resolve filename against SAMPLES_DIR, refusing anything that would
    escape it (e.g. "../../something") -- mirrors the containment check
    Flask's send_from_directory does for the GET route below, but this one
    also needs to gate a filesystem delete, not just a file read."""
    candidate = os.path.normpath(os.path.join(SAMPLES_DIR, filename))
    samples_root = os.path.normpath(SAMPLES_DIR)
    if candidate != samples_root and not candidate.startswith(samples_root + os.sep):
        return None
    return candidate


@app.route("/samples")
def samples_page():
    """Browse the fart sample library -- a player plus the analyzed
    metadata (from samples_index.json) for every sample, so Dave can see
    what the matcher sees when he adds new recordings."""
    records = load_index(SAMPLES_DIR)
    records = sorted(records, key=lambda r: r.filename.lower())
    samples = [_describe_sample(r) for r in records]
    return render_template("samples.html", samples=samples, sample_count=len(records))


@app.route("/samples", methods=["POST"])
def upload_sample():
    """Add a new fart sample to the library: save the WAV, analyze it, and
    merge it into samples_index.json -- without re-analyzing every other
    sample. Used by the drag-and-drop dropzone at the top of /samples."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV files are supported for samples"}), 400

    with SAMPLES_LOCK:
        dest_path = os.path.join(SAMPLES_DIR, filename)
        if os.path.exists(dest_path):
            # Don't clobber an existing sample with the same name -- add a
            # numeric suffix instead, same idea as a browser's own "file (2)".
            stem, ext = os.path.splitext(filename)
            i = 2
            while os.path.exists(os.path.join(SAMPLES_DIR, f"{stem} ({i}){ext}")):
                i += 1
            filename = f"{stem} ({i}){ext}"
            dest_path = os.path.join(SAMPLES_DIR, filename)

        file.save(dest_path)

        try:
            record = analyze_sample(dest_path)
        except Exception:
            logging.exception("failed to analyze uploaded sample %s", filename)
            os.remove(dest_path)
            return jsonify({"error": "Couldn't analyze this file -- is it a valid WAV?"}), 400

        add_sample_record(SAMPLES_DIR, record)

    return jsonify({"sample": _describe_sample(record)})


@app.route("/samples/<path:filename>")
def serve_sample(filename: str):
    as_attachment = request.args.get("download") is not None
    return send_from_directory(SAMPLES_DIR, filename, as_attachment=as_attachment)


@app.route("/spectrograms/<path:filename>")
def sample_spectrogram(filename: str):
    """A sample's log-frequency spectrogram as a PNG. Normally already on
    disk from when the sample was analyzed; generated on the spot for one
    that was indexed before spectrograms existed."""
    path = _safe_sample_path(filename)
    if path is None or not os.path.isfile(path) or not filename.lower().endswith(".wav"):
        return jsonify({"error": "unknown sample"}), 404
    with SAMPLES_LOCK:
        png_path = sample_spectrogram_png(SAMPLES_DIR, os.path.relpath(path, SAMPLES_DIR))
    return send_file(png_path, mimetype="image/png")


@app.route("/samples/<path:filename>", methods=["DELETE"])
def delete_sample(filename: str):
    path = _safe_sample_path(filename)
    if path is None:
        return jsonify({"error": "invalid filename"}), 400

    with SAMPLES_LOCK:
        existed = os.path.isfile(path)
        if existed:
            os.remove(path)
            remove_sample_record(SAMPLES_DIR, os.path.basename(path))

    if not existed:
        return jsonify({"error": "unknown sample"}), 404
    return jsonify({"status": "deleted"})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    # The input panel posts its algorithm + calibration fields as one JSON
    # form value; absent entirely (e.g. a bare curl upload) means defaults.
    try:
        settings = parse_settings(json.loads(request.form.get("settings") or "{}"))
    except ValueError as exc:  # also covers json.JSONDecodeError
        return jsonify({"error": f"Invalid settings: {exc}"}), 400

    job_id = uuid.uuid4().hex
    filename = secure_filename(file.filename)
    saved_path = os.path.join(UPLOADS_DIR, f"{job_id}_{filename}")
    file.save(saved_path)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "error": None,
            "result": None,
            "filename": file.filename,
            "settings": asdict(settings),
            "created_at": time.time(),
        }
        _save_jobs_locked()

    thread = threading.Thread(target=_process_job, args=(job_id, saved_path, settings), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


def _job_view(job: dict) -> dict:
    """A job as the page should see it right now. While it's processing,
    `progress` isn't a stored number: the pipeline only reports when it
    moves to a new stage, and this fills in the movement *within* that stage
    from how long it's been running vs. how long it was expected to take.
    Caller must hold JOBS_LOCK."""
    view = dict(job)
    if job.get("status") == "processing":
        view["progress"], view["eta_sec"] = snapshot(job)
    return view


@app.route("/jobs")
def list_jobs():
    """All known jobs (past and current), newest first -- used to repopulate
    the job panels on page load so they persist across a reload."""
    with JOBS_LOCK:
        jobs = [dict(_job_view(job), job_id=job_id) for job_id, job in JOBS.items()]
    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return jsonify(jobs)


@app.route("/status/<job_id>")
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        view = _job_view(job) if job is not None else None
    if view is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(view)


@app.route("/jobs/<job_id>", methods=["DELETE"])
def delete_job(job_id: str):
    with JOBS_LOCK:
        existed = JOBS.pop(job_id, None) is not None
        if existed:
            _save_jobs_locked()

    # Clean up its files regardless -- covers a job that was removed from
    # JOBS by an earlier request but whose files are still lying around.
    work_dir = os.path.join(OUTPUT_DIR, job_id)
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    for fname in os.listdir(UPLOADS_DIR):
        if fname.startswith(f"{job_id}_"):
            try:
                os.remove(os.path.join(UPLOADS_DIR, fname))
            except OSError:
                pass

    if not existed:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"status": "deleted"})


# Characters that are illegal in Windows filenames (this app runs on a
# Windows desktop) -- strip them but keep spaces/brackets so the
# "[Fartified]" suffix reads cleanly.
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _fartified_download_name(job: dict | None, ext: str) -> str:
    """Build a download filename that mirrors the original upload, e.g.
    "song.mp3" -> "song [Fartified].wav". Falls back to a generic name
    if the job or its original filename isn't available."""
    original = (job or {}).get("filename")
    if not original:
        return f"fartify_output{ext}"
    stem, _ = os.path.splitext(original)
    stem = _UNSAFE_FILENAME_CHARS.sub("", stem).strip()
    if not stem:
        return f"fartify_output{ext}"
    return f"{stem} [Fartified]{ext}"


@app.route("/download/<job_id>/<kind>")
def download_result(job_id: str, kind: str):
    work_dir = os.path.join(OUTPUT_DIR, job_id)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if kind == "audio":
        path = os.path.join(work_dir, "fartify_output.wav")
        download_name = _fartified_download_name(job, ".wav")
    elif kind == "midi":
        path = os.path.join(work_dir, "fartify_melody.mid")
        download_name = _fartified_download_name(job, ".mid")
    else:
        return jsonify({"error": "unknown kind"}), 404
    if not os.path.exists(path):
        return jsonify({"error": "not ready"}), 404
    return send_file(path, as_attachment=True, download_name=download_name)


@app.route("/jobs/<job_id>/spectrogram/<which>")
def job_spectrogram(job_id: str, which: str):
    """The spectrograms a Spectral Synthesis job leaves behind: the layer
    it was matching ("layer") and what it built from samples ("approx")."""
    names = {"layer": TARGET_SPECTROGRAM_PNG, "approx": APPROX_SPECTROGRAM_PNG}
    with JOBS_LOCK:
        known = job_id in JOBS
    if not known or which not in names:
        return jsonify({"error": "not found"}), 404
    path = os.path.join(OUTPUT_DIR, job_id, names[which])
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/samples/rebuild", methods=["POST"])
def rebuild_samples():
    records = build_index(SAMPLES_DIR, force=True)
    return jsonify({"count": len(records)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
