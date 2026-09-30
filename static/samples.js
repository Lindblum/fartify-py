const samplesContainer = document.getElementById("samples");
const sampleCardTemplate = document.getElementById("sample-card-template");
const emptyState = document.getElementById("empty-state");

const dropzone = document.getElementById("sample-dropzone");
const fileInput = document.getElementById("sample-file-input");
const uploadStatus = document.getElementById("sample-upload-status");

const confirmOverlay = document.getElementById("confirm-overlay");
const confirmMessage = document.getElementById("confirm-message");
const confirmYesBtn = document.getElementById("confirm-yes");
const confirmNoBtn = document.getElementById("confirm-no");

// --- In-page confirmation panel (same pattern as the main page) ----------

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

// --- Wiring shared by both server-rendered and freshly-uploaded cards ----

function wireSampleCard(card) {
  // Only one player audible at a time, same as the main page's job panels.
  const player = card.querySelector(".player");
  player.addEventListener("play", () => {
    document.querySelectorAll("audio.player").forEach((other) => {
      if (other !== player && !other.paused) {
        other.pause();
      }
    });
  });

  attachPlayheads(player, card);

  card.querySelector(".delete-sample").addEventListener("click", () => deleteSample(card));
}

document.querySelectorAll(".sample-card").forEach(wireSampleCard);

async function deleteSample(card) {
  const filename = card.dataset.filename;
  if (!filename) return;

  const confirmed = await showConfirmPanel(`Delete ${filename}? This can't be undone.`);
  if (!confirmed) return;

  const deleteBtn = card.querySelector(".delete-sample");
  deleteBtn.disabled = true;

  try {
    const res = await fetch(`/samples/${encodeURIComponent(filename)}`, { method: "DELETE" });
    if (!res.ok && res.status !== 404) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || "Failed to delete sample");
    }
    card.remove();
    if (!samplesContainer.querySelector(".sample-card")) {
      emptyState.classList.remove("hidden");
    }
  } catch (err) {
    deleteBtn.disabled = false;
    window.alert(`Couldn't delete this sample: ${err.message}`);
  }
}

// --- Upload (dropzone) ----------------------------------------------------

function createSampleCard(sample) {
  const node = sampleCardTemplate.content.firstElementChild.cloneNode(true);
  node.dataset.filename = sample.filename;
  node.querySelector(".sample-filename").textContent = sample.filename;
  const downloadUrl = `/samples/${encodeURIComponent(sample.filename)}`;
  node.querySelector(".player").src = downloadUrl;
  const spectrogram = node.querySelector(".spectrogram");
  spectrogram.src = sample.spectrogram_url;
  spectrogram.alt = `Spectrogram of ${sample.filename}`;
  node.querySelector(".download-sample").href = `${downloadUrl}?download=1`;
  node.querySelector(".meta-length").textContent = sample.length;
  node.querySelector(".meta-volume").textContent = sample.volume;
  node.querySelector(".meta-pitch").textContent = sample.pitch;
  node.querySelector(".meta-emphasis").textContent = `emphasis @ ${sample.emphasis}`;
  wireSampleCard(node);
  return node;
}

async function uploadSample(file) {
  uploadStatus.textContent = `Analyzing ${file.name}…`;
  uploadStatus.classList.remove("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/samples", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    const card = createSampleCard(data.sample);
    samplesContainer.prepend(card);
    emptyState.classList.add("hidden");
    uploadStatus.textContent = `Added ${data.sample.filename}.`;
  } catch (err) {
    uploadStatus.textContent = `Couldn't add ${file.name}: ${err.message}`;
  } finally {
    setTimeout(() => uploadStatus.classList.add("hidden"), 4000);
  }
}

function handleFiles(fileList) {
  Array.from(fileList).forEach((file) => uploadSample(file));
  fileInput.value = "";
}

document.getElementById("sample-upload-form").addEventListener("submit", (e) => e.preventDefault());

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFiles(fileInput.files));

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
  if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});
