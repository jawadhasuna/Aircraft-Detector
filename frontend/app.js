const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const workArea = document.getElementById("workArea");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const sweep = document.getElementById("sweep");
const resultsList = document.getElementById("resultsList");
const resultsCount = document.getElementById("resultsCount");
const statusLine = document.getElementById("statusLine");
const resetBtn = document.getElementById("resetBtn");

// one stable color per class, so the same airframe always reads the same color
const CLASS_COLORS = {
  A220: "#6dffb0",
  A320321: "#ffb445",
  A330: "#5fb4ff",
  ARJ21: "#ff6b9d",
  Boeing737: "#c98bff",
  Boeing787: "#ffe45f",
};

let currentImage = null;

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

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
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

resetBtn.addEventListener("click", () => {
  workArea.classList.add("hidden");
  resetBtn.classList.add("hidden");
  dropzone.classList.remove("hidden");
  statusLine.textContent = "";
  statusLine.className = "status-line";
  fileInput.value = "";
});

async function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    setStatus("That file doesn't look like an image.", "error");
    return;
  }

  dropzone.classList.add("hidden");
  workArea.classList.remove("hidden");
  resetBtn.classList.remove("hidden");
  sweep.classList.remove("hidden");
  setStatus("Running inference…", "");
  resultsList.innerHTML = "";
  resultsCount.textContent = "—";

  const img = new Image();
  const objectUrl = URL.createObjectURL(file);

  img.onload = async () => {
    currentImage = img;
    drawImageOnly(img);

    const formData = new FormData();
    formData.append("image", file);

    // The server scales to zero when idle, so the first request after a quiet
    // period has to start a container. Let the user know instead of looking stuck.
    const coldStartNotice = setTimeout(() => {
      setStatus("Waking up the server (first request takes ~20s)…", "");
    }, 4000);

    try {
      const res = await fetch("/detect", { method: "POST", body: formData });
      clearTimeout(coldStartNotice);
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Detection failed.");
      }

      sweep.classList.add("hidden");
      drawDetections(img, data.detections);
      renderResults(data.detections);

      setStatus(
        data.detections.length
          ? `Found ${data.detections.length} aircraft.`
          : "No aircraft detected above the confidence threshold.",
        "ok"
      );
    } catch (err) {
      clearTimeout(coldStartNotice);
      sweep.classList.add("hidden");
      setStatus(err.message, "error");
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  };

  img.onerror = () => {
    setStatus("Couldn't read that image file.", "error");
    sweep.classList.add("hidden");
  };

  img.src = objectUrl;
}

function drawImageOnly(img) {
  const maxW = 620;
  const scale = Math.min(1, maxW / img.width);
  canvas.width = img.width * scale;
  canvas.height = img.height * scale;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}

function drawDetections(img, detections) {
  drawImageOnly(img);
  const scaleX = canvas.width / img.width;
  const scaleY = canvas.height / img.height;

  detections.forEach((d) => {
    const [x1, y1, x2, y2] = d.box;
    const color = CLASS_COLORS[d.className] || "#6dffb0";
    const bx = x1 * scaleX;
    const by = y1 * scaleY;
    const bw = (x2 - x1) * scaleX;
    const bh = (y2 - y1) * scaleY;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(bx, by, bw, bh);

    const label = `${d.className} ${(d.score * 100).toFixed(0)}%`;
    ctx.font = "600 12px 'IBM Plex Mono', monospace";
    const textW = ctx.measureText(label).width;

    ctx.fillStyle = color;
    ctx.fillRect(bx, Math.max(0, by - 18), textW + 10, 18);

    ctx.fillStyle = "#0a0d0b";
    ctx.fillText(label, bx + 5, Math.max(12, by - 5));
  });
}

function renderResults(detections) {
  resultsCount.textContent = detections.length;

  if (!detections.length) {
    resultsList.innerHTML = `<li class="results-empty">No detections above threshold.</li>`;
    return;
  }

  resultsList.innerHTML = "";
  detections
    .slice()
    .sort((a, b) => b.score - a.score)
    .forEach((d) => {
      const li = document.createElement("li");
      li.className = "result-item";
      const color = CLASS_COLORS[d.className] || "#6dffb0";
      li.innerHTML = `
        <span class="result-swatch" style="background:${color}"></span>
        <span class="result-name">${d.className}</span>
        <span class="result-score">${(d.score * 100).toFixed(1)}%</span>
      `;
      resultsList.appendChild(li);
    });
}

function setStatus(text, kind) {
  statusLine.textContent = text;
  statusLine.className = "status-line" + (kind ? " " + kind : "");
}
