const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzone-text");
const fileInput = document.getElementById("file-input");
const submitBtn = document.getElementById("submit-btn");
const form = document.getElementById("upload-form");

const jobsContainer = document.getElementById("jobs");
const jobCardTemplate = document.getElementById("job-card-template");

const confirmOverlay = document.getElementById("confirm-overlay");
const confirmMessage = document.getElementById("confirm-message");
const confirmYesBtn = document.getElementById("confirm-yes");
const confirmNoBtn = document.getElementById("confirm-no");

let selectedFile = null;

// --- Sound effects: play a random library sample as UI feedback ----------

const SAMPLE_FILES = window.FARTIFY_SAMPLES || [];
const sfxAudio = new Audio(); // reused so back-to-back clicks restart cleanly instead of piling up

function playRandomSample() {
  if (!SAMPLE_FILES.length) return;
  const file = SAMPLE_FILES[Math.floor(Math.random() * SAMPLE_FILES.length)];
  sfxAudio.pause();
  sfxAudio.currentTime = 0;
  sfxAudio.src = `/samples/${encodeURIComponent(file)}`;
  sfxAudio.play().catch(() => {}); // ignore autoplay rejections -- this always fires from a user gesture anyway
}

// --- In-page confirmation panel (replaces window.confirm) -----------------

function showConfirmPanel(message) {
  return new Promise((resolve) => {
    confirmMessage.textContent = message;
    confirmOverlay.classList.remove("hidden");

    function cleanup(result) {
      confirmOverlay.classList.add("hidden");
      confirmYesBtn.removeEventListener("click", onYes);
      confirmNoBtn.removeEventListener("click", onNo);
      confirmOverlay.removeEventListener("click", onOverlayClick);
      document.removeEventListener("keydown", onKeydown);
      resolve(result);
    }
    function onYes() {
      cleanup(true);
    }
    function onNo() {
      cleanup(false);
    }
    function onOverlayClick(e) {
      if (e.target === confirmOverlay) cleanup(false);
    }
    function onKeydown(e) {
      if (e.key === "Escape") cleanup(false);
    }

    confirmYesBtn.addEventListener("click", onYes);
    confirmNoBtn.addEventListener("click", onNo);
    confirmOverlay.addEventListener("click", onOverlayClick);
    document.addEventListener("keydown", onKeydown);
  });
}

function pickFile(file) {
  selectedFile = file;
  dropzoneText.textContent = file ? file.name : "Drop an audio file here, or click to choose one";
  submitBtn.disabled = !file;
}

function resetInputPanel() {
  selectedFile = null;
  fileInput.value = "";
  dropzoneText.textContent = "Drop an audio file here, or click to choose one";
  submitBtn.disabled = true;
}

// --- Algorithm + calibration options --------------------------------------

const algorithmSelect = document.getElementById("algorithm");
const layerSelect = document.getElementById("layer");
const optionNotes = document.getElementById("option-notes");
const calibration = document.getElementById("calibration");
const resetSettingsBtn = document.getElementById("reset-settings");
const settingInputs = Array.from(document.querySelectorAll("[data-setting]"));

function settingValue(input) {
  return input.type === "checkbox" ? input.checked : Number(input.value);
}

function settingDefault(input) {
  return input.type === "checkbox" ? input.defaultChecked : Number(input.defaultValue);
}

// Groups tied to specific algorithms are hidden *and* disabled while another
// one is selected -- disabled so their fields are skipped by both validation
// and collectSettings (the server then just uses their defaults).
function syncOptionFields() {
  const algorithm = algorithmSelect.value;
  calibration.querySelectorAll("fieldset[data-algorithms]").forEach((group) => {
    const applies = group.dataset.algorithms.split(" ").includes(algorithm);
    group.disabled = !applies;
    group.classList.toggle("hidden", !applies);
  });

  // Any caveat attached to the selected algorithm / layer.
  optionNotes.replaceChildren();
  [algorithmSelect, layerSelect].forEach((select) => {
    const note = select.selectedOptions[0].dataset.note;
    if (!note) return;
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = note;
    optionNotes.appendChild(p);
  });
}

function optionLabel(select, value) {
  const option = Array.from(select.options).find((o) => o.value === value);
  return option ? option.textContent : value;
}

function collectSettings() {
  const settings = { algorithm: algorithmSelect.value, layer: layerSelect.value };
  settingInputs.forEach((input) => {
    if (input.matches(":disabled")) return;
    settings[input.name] = settingValue(input);
  });
  return settings;
}

// One-line summary of what a job was run with: its algorithm and layer,
// plus any calibration field that was moved off its default.
function describeSettings(settings) {
  if (!settings) return ""; // job predates the options panel
  const parts = [`Algorithm: ${optionLabel(algorithmSelect, settings.algorithm)}`];
  if (settings.layer) parts.push(`Layer: ${optionLabel(layerSelect, settings.layer)}`);
  settingInputs.forEach((input) => {
    const value = settings[input.name];
    if (value === undefined || value === settingDefault(input)) return;
    const shown = typeof value === "boolean" ? (value ? "on" : "off") : value;
    parts.push(`${input.dataset.label} ${shown}`);
  });
  return parts.join(" · ");
}

algorithmSelect.addEventListener("change", syncOptionFields);
layerSelect.addEventListener("change", syncOptionFields);
resetSettingsBtn.addEventListener("click", () => {
  settingInputs.forEach((input) => {
    if (input.type === "checkbox") input.checked = input.defaultChecked;
    else input.value = input.defaultValue;
  });
});
syncOptionFields();

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => pickFile(fileInput.files[0] || null));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) pickFile(file);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  // The form is novalidate so a bad value inside the collapsed Calibration
  // section can't silently block submission -- open it and point at the field.
  const invalid = settingInputs.find((input) => !input.checkValidity());
  if (invalid) {
    calibration.open = true;
    invalid.reportValidity();
    return;
  }

  playRandomSample();

  const filename = selectedFile.name;
  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("settings", JSON.stringify(collectSettings()));

  // A job panel is created immediately (queued state) and dropped in above
  // any earlier jobs, so the input panel stays put and every run — past and
  // in-flight — stacks up below it.
  const card = createJobCard(filename);
  jobsContainer.prepend(card);
  setJobState(card, { status: "queued", stage: "uploading…", progress: 0.02 });

  resetInputPanel();

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    card.dataset.jobId = data.job_id;
    pollStatus(card, data.job_id);
  } catch (err) {
    setJobState(card, { status: "error", error: err.message });
  }
});

function createJobCard(filename) {
  const node = jobCardTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".job-filename").textContent = filename;
  node.dataset.filename = filename;

  // Only one player should ever be audibly playing at once: when this
  // card's player starts, pause every other job's player.
  const player = node.querySelector(".player");
  player.addEventListener("play", () => {
    document.querySelectorAll("audio.player").forEach((other) => {
      if (other !== player && !other.paused) {
        other.pause();
      }
    });
  });

  attachPlayheads(player, node);

  node.querySelector(".delete-job").addEventListener("click", () => deleteJob(node));

  return node;
}

async function deleteJob(card) {
  const jobId = card.dataset.jobId;
  if (!jobId) return; // still uploading -- no job_id to delete yet

  const filename = card.dataset.filename || "this job";
  const confirmed = await showConfirmPanel(`Delete ${filename}? This can't be undone.`);
  if (!confirmed) return;

  playRandomSample();

  const deleteBtn = card.querySelector(".delete-job");
  deleteBtn.disabled = true;

  try {
    const res = await fetch(`/jobs/${jobId}`, { method: "DELETE" });
    if (!res.ok && res.status !== 404) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || "Failed to delete job");
    }
    if (card._pollTimeout) clearTimeout(card._pollTimeout);
    card.remove();
  } catch (err) {
    deleteBtn.disabled = false;
    window.alert(`Couldn't delete this job: ${err.message}`);
  }
}

// Repopulate the job panels from the server on page load, so past and
// in-flight jobs survive a reload instead of only living in this tab's
// memory. Jobs still queued/processing pick polling back up; finished ones
// just render their stored result/error.
async function loadExistingJobs() {
  try {
    const res = await fetch("/jobs");
    if (!res.ok) return;
    const jobs = await res.json();
    jobs.forEach((job) => {
      const card = createJobCard(job.filename || "Untitled upload");
      card.dataset.jobId = job.job_id;
      jobsContainer.appendChild(card);
      setJobState(card, job);
      if (job.status === "queued" || job.status === "processing") {
        pollStatus(card, job.job_id);
      }
    });
  } catch (err) {
    console.error("Failed to load job history", err);
  }
}

loadExistingJobs();

function pollStatus(card, jobId) {
  const poll = async () => {
    try {
      const res = await fetch(`/status/${jobId}`);
      const job = await res.json();
      if (!res.ok) throw new Error(job.error || "Unknown job");

      setJobState(card, job);

      if (job.status === "done") {
        // Only fires here (not from loadExistingJobs' initial render) so a
        // page reload full of already-finished jobs doesn't replay a fart
        // for every one of them -- just genuine completions during this visit.
        playRandomSample();
        return;
      }
      if (job.status === "error") {
        return;
      }
      card._pollTimeout = setTimeout(poll, 900);
    } catch (err) {
      setJobState(card, { status: "error", error: err.message });
    }
  };
  poll();
}

const STATUS_LABELS = {
  queued: "Queued",
  processing: "Processing…",
  done: "Done",
  error: "Failed",
};

function setJobState(card, job) {
  const statusEl = card.querySelector(".job-status");
  const progressWrap = card.querySelector(".job-progress");
  const progressFill = card.querySelector(".progress-fill");
  const progressStage = card.querySelector(".progress-stage");
  const resultWrap = card.querySelector(".job-result");
  const errorWrap = card.querySelector(".job-error");
  // The download links live in the persistent .job-actions row (alongside
  // delete, which stays available in every state), so they're shown/hidden
  // on their own rather than via their old parent .job-result wrapper.
  const downloadAudio = card.querySelector(".download-audio");
  const downloadMidi = card.querySelector(".download-midi");

  statusEl.textContent = STATUS_LABELS[job.status] || job.status;
  statusEl.className = `job-status status-${job.status}`;

  if (job.status === "done") {
    progressWrap.classList.add("hidden");
    errorWrap.classList.add("hidden");
    resultWrap.classList.remove("hidden");
    downloadAudio.classList.remove("hidden");
    downloadMidi.classList.remove("hidden");
    fillResult(card, job);
  } else if (job.status === "error") {
    progressWrap.classList.add("hidden");
    resultWrap.classList.add("hidden");
    errorWrap.classList.remove("hidden");
    downloadAudio.classList.add("hidden");
    downloadMidi.classList.add("hidden");
    card.querySelector(".error-message").textContent = job.error || "Processing failed";
  } else {
    resultWrap.classList.add("hidden");
    errorWrap.classList.add("hidden");
    downloadAudio.classList.add("hidden");
    downloadMidi.classList.add("hidden");
    progressWrap.classList.remove("hidden");
    progressFill.style.width = `${((job.progress || 0) * 100).toFixed(1)}%`;
    const eta = formatEta(job.eta_sec);
    progressStage.textContent = (job.stage || job.status) + (eta ? ` · ${eta}` : "");
  }
}

// "about 1m 40s left" -- deliberately coarse (5s steps), since it's an
// estimate and a countdown that flickers by the second reads as false precision.
function formatEta(etaSec) {
  if (etaSec == null || !isFinite(etaSec)) return "";
  if (etaSec < 5) return "almost done";
  const rounded = Math.round(etaSec / 5) * 5;
  const minutes = Math.floor(rounded / 60);
  const seconds = rounded % 60;
  if (!minutes) return `about ${seconds}s left`;
  return `about ${minutes}m${seconds ? ` ${seconds}s` : ""} left`;
}

function fillResult(card, job) {
  const result = job.result;
  const player = card.querySelector(".player");
  const downloadAudio = card.querySelector(".download-audio");
  const downloadMidi = card.querySelector(".download-midi");
  const resultMeta = card.querySelector(".result-meta");

  player.src = result.audio_url;
  downloadAudio.href = result.audio_url;
  downloadMidi.href = result.midi_url;

  const degradedNote = result.separation_degraded
    ? " (lite fallback — install Demucs for cleaner separation)"
    : "";

  // Spectral Synthesis results carry how well the spectrogram was matched
  // (and the two spectrograms to look at) instead of a note count.
  const spectral = result.explained_fraction != null;
  const summary = spectral
    ? `<div><strong>${result.note_count}</strong> sample placements over <strong>${result.duration_sec}s</strong>,
         matching <strong>${(result.explained_fraction * 100).toFixed(1)}%</strong> of the layer's spectrogram</div>`
    : `<div><strong>${result.note_count}</strong> melody notes extracted over <strong>${result.duration_sec}s</strong></div>`;
  resultMeta.innerHTML = `
    ${summary}
    <div>Separation: ${result.separation_method}${degradedNote}</div>
    ${spectral ? "" : `<div>Pitch tracker: ${result.pitch_backend}</div>`}
  `;

  const spectrograms = card.querySelector(".result-spectrograms");
  spectrograms.classList.toggle("hidden", !spectral);
  if (spectral) {
    [
      [".layer-spectrogram", result.layer_spectrogram_url],
      [".approx-spectrogram", result.approx_spectrogram_url],
    ].forEach(([selector, url]) => {
      const link = spectrograms.querySelector(selector);
      link.href = url;
      const img = link.querySelector("img");
      if (img.getAttribute("src") !== url) img.src = url; // don't reload on every status refresh
    });
  }

  const settingsSummary = describeSettings(job.settings);
  if (settingsSummary) {
    const line = document.createElement("div");
    line.textContent = settingsSummary;
    resultMeta.appendChild(line);
  }
}
