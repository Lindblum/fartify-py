// Shared by the main page and the sample library: moves a vertical line
// across every spectrogram in a card (each `.playhead` inside `root`) in
// step with that card's audio player.
//
// A spectrogram image spans its audio's full duration left to right, so the
// line's position is simply currentTime / duration across the image.
function attachPlayheads(player, root) {
  const playheads = root.querySelectorAll(".playhead");
  if (!playheads.length) return;
  let frameRequest = null;

  function update() {
    const fraction = player.duration > 0 ? Math.min(player.currentTime / player.duration, 1) : 0;
    // Nothing to mark until playback has started (or been scrubbed) somewhere.
    const visible = !player.paused || player.currentTime > 0;
    playheads.forEach((el) => {
      el.style.left = `${fraction * 100}%`;
      el.classList.toggle("hidden", !visible);
    });
  }

  // `timeupdate` only fires a few times a second, which makes the line
  // visibly hop; while playing, follow the clock every animation frame.
  function tick() {
    update();
    frameRequest = requestAnimationFrame(tick);
  }

  player.addEventListener("play", () => {
    if (frameRequest === null) tick();
  });
  ["pause", "ended"].forEach((evt) =>
    player.addEventListener(evt, () => {
      cancelAnimationFrame(frameRequest);
      frameRequest = null;
      update();
    })
  );
  // Scrubbing while paused, or the source changing.
  ["seeking", "seeked", "timeupdate", "loadedmetadata", "emptied"].forEach((evt) =>
    player.addEventListener(evt, update)
  );
}
