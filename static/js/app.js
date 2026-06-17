/* =========================================================
   Flask Voice Clone – Front-end logic
   ========================================================= */

// List of uploaded reference clips: [{ref_id, name, objectUrl}, ...]
let currentRefIds = [];
let modelReady    = false;
let runtimeDevice = null; // "cpu" | "cuda" | ...
let activeSrtJobId = null;

// ── Helpers ────────────────────────────────────────────────

function setStatus(el, msg, type = "") {
  el.innerHTML = msg;
  el.className = "status " + type;
}

function fmtBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
  return (b / 1024 / 1024).toFixed(1) + " MB";
}

async function apiFetch(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

// ── Model status polling ────────────────────────────────────

const banner = document.getElementById("modelBanner");
const deviceBadge = document.getElementById("deviceBadge");

function _shortModelLabel(modelType) {
  if (modelType === "turbo")        return "Turbo";
  if (modelType === "multilingual") return "Multilingual";
  if (modelType === "chatterbox")   return "Chatterbox";
  return modelType || "";
}

function updateDeviceBadge(device, modelType) {
  if (!deviceBadge) return;
  const d = (device || "").toString().toLowerCase();
  runtimeDevice = d || null;
  if (!d) {
    deviceBadge.textContent = "Detecting…";
    deviceBadge.className = "device-badge loading";
    return;
  }
  const deviceLabel = d.startsWith("cuda") ? "CUDA" : "CPU";
  const modelLabel  = _shortModelLabel(modelType);
  deviceBadge.textContent = modelLabel ? `${modelLabel} · ${deviceLabel}` : deviceLabel;
  deviceBadge.className   = "device-badge " + (deviceLabel === "CUDA" ? "cuda" : "cpu");
}

async function pollModelStatus() {
  try {
    const data = await fetch("/model-status").then(r => r.json());
    updateDeviceBadge(data.device, data.model_type);
    if (data.ready) {
      modelReady = true;
      const deviceLabel = runtimeDevice && runtimeDevice.startsWith("cuda") ? "CUDA" : "CPU";
      const modelLabel  = _shortModelLabel(data.model_type);
      const suffix      = modelLabel ? ` — ${modelLabel} on ${deviceLabel}` : ` (${deviceLabel})`;
      banner.textContent = "✓ Model ready" + suffix + " — you can start synthesizing";
      banner.className = "banner ready";
      updateSynthButtonState();
      updateSrtButtonState();
      loadSpeakers();
    } else {
      setTimeout(pollModelStatus, 2000);
    }
  } catch {
    setTimeout(pollModelStatus, 3000);
  }
}

pollModelStatus();

// ── Tabs ───────────────────────────────────────────────────

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    tab.classList.add("active");
    document.getElementById("tab-" + tab.dataset.tab).classList.remove("hidden");
    if (tab.dataset.tab === "history") refreshHistory();
  });
});

// ── Reference audio list ────────────────────────────────────

function renderRefList() {
  const list = document.getElementById("refList");
  if (!list) return;

  if (!currentRefIds.length) {
    list.innerHTML = "";
    return;
  }

  list.innerHTML = currentRefIds.map((item, idx) => `
    <div class="ref-item" id="ref-item-${idx}">
      <audio controls src="${item.objectUrl}" preload="none"></audio>
      <span class="ref-name" title="${item.name}">${item.name}</span>
      <button class="btn danger ref-remove" onclick="removeRefClip(${idx})">Remove</button>
    </div>
  `).join("");

  // Update the SRT tab clip count label
  const srtRefCount = document.getElementById("srtRefCount");
  if (srtRefCount) srtRefCount.textContent = currentRefIds.length;

  updateSynthButtonState();
  updateSrtButtonState();
}

function removeRefClip(idx) {
  const item = currentRefIds[idx];
  if (item && item.objectUrl) URL.revokeObjectURL(item.objectUrl);
  currentRefIds.splice(idx, 1);
  renderRefList();

  const status = document.getElementById("refStatus");
  if (currentRefIds.length === 0) {
    setStatus(status, "All reference clips removed.", "");
  } else {
    setStatus(status, `${currentRefIds.length} clip(s) ready.`, "ok");
  }
}

function updateSynthButtonState() {
  const btn = document.getElementById("synthesizeBtn");
  if (btn) btn.disabled = !(modelReady && currentRefIds.length > 0);
}

// ── Upload reference audio ─────────────────────────────────

document.getElementById("addRefBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("refFile");
  const status    = document.getElementById("refStatus");

  if (!fileInput.files.length) {
    setStatus(status, "Please select one or more audio files first.", "err");
    return;
  }

  const files = Array.from(fileInput.files);
  setStatus(status, `<span class="spin">⟳</span> Uploading ${files.length} file(s)…`, "loading");

  let uploaded = 0;
  let failed = 0;

  for (const file of files) {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res  = await fetch("/upload-ref", { method: "POST", body: formData });
      const data = await res.json();
      if (data.error) {
        failed++;
        console.warn("Upload failed for", file.name, ":", data.error);
      } else {
        currentRefIds.push({
          ref_id: data.ref_id,
          name: file.name,
          objectUrl: URL.createObjectURL(file),
        });
        uploaded++;
      }
    } catch (e) {
      failed++;
      console.warn("Upload error for", file.name, ":", e);
    }
  }

  // Clear the input so the same files can be re-added if needed
  fileInput.value = "";

  if (failed === 0) {
    setStatus(status, `✓ ${uploaded} clip(s) added. ${currentRefIds.length} total.`, "ok");
  } else {
    setStatus(status, `${uploaded} uploaded, ${failed} failed. ${currentRefIds.length} total.`, uploaded ? "loading" : "err");
  }

  renderRefList();
});

// ── Clone synthesis ────────────────────────────────────────

document.getElementById("synthesizeBtn").addEventListener("click", async () => {
  const text   = document.getElementById("cloneText").value.trim();
  const lang   = document.getElementById("cloneLang").value;
  const status = document.getElementById("synthStatus");

  if (!text)               { setStatus(status, "Please enter some text.", "err"); return; }
  if (!currentRefIds.length) { setStatus(status, "Please upload at least one reference audio clip first.", "err"); return; }

  setStatus(status, '<span class="spin">⟳</span> Synthesizing, please wait…', "loading");
  document.getElementById("synthesizeBtn").disabled = true;

  try {
    const data = await apiFetch("/synthesize", {
      text,
      ref_ids: currentRefIds.map(r => r.ref_id),
      lang,
    });

    if (data.error) {
      setStatus(status, "Error: " + data.error, "err");
      updateSynthButtonState();
      return;
    }

    setStatus(status, "✓ Done", "ok");

    const audioEl    = document.getElementById("cloneAudio");
    const downloadEl = document.getElementById("cloneDownload");
    audioEl.src = "/play/outputs/" + data.output_id;
    audioEl.load();
    audioEl.play();
    downloadEl.href = "/download/outputs/" + data.output_id;
    downloadEl.setAttribute("download", data.output_id);
    document.getElementById("cloneResultCard").classList.remove("hidden");
  } catch (e) {
    setStatus(status, "Request failed: " + e.message, "err");
  }

  updateSynthButtonState();
});

// ── Built-in speakers ──────────────────────────────────────

async function loadSpeakers() {
  try {
    const data = await fetch("/speakers").then(r => r.json());
    const speakers = (data && data.speakers) ? data.speakers : [];

    function fillSelect(selectEl) {
      if (!selectEl) return;
      selectEl.innerHTML = "";
      if (!speakers.length) {
        selectEl.innerHTML = "<option>(No built-in speakers)</option>";
        return;
      }
      speakers.forEach(sp => {
        const opt = document.createElement("option");
        opt.value = sp;
        opt.textContent = sp;
        selectEl.appendChild(opt);
      });
    }

    fillSelect(document.getElementById("speakerSelect"));
    fillSelect(document.getElementById("srtSpeakerSelect"));
    updateSrtButtonState();
  } catch { /* ignore */ }
}

document.getElementById("builtinSynthBtn").addEventListener("click", async () => {
  const text    = document.getElementById("builtinText").value.trim();
  const speaker = document.getElementById("speakerSelect").value;
  const lang    = document.getElementById("builtinLang").value;
  const status  = document.getElementById("builtinStatus");

  if (!text) { setStatus(status, "Please enter some text.", "err"); return; }

  setStatus(status, '<span class="spin">⟳</span> Synthesizing, please wait…', "loading");
  document.getElementById("builtinSynthBtn").disabled = true;

  try {
    const data = await apiFetch("/synthesize-builtin", { text, speaker, lang });

    if (data.error) {
      setStatus(status, "Error: " + data.error, "err");
      document.getElementById("builtinSynthBtn").disabled = false;
      return;
    }

    setStatus(status, "✓ Done", "ok");

    const audioEl    = document.getElementById("builtinAudio");
    const downloadEl = document.getElementById("builtinDownload");
    audioEl.src = "/play/outputs/" + data.output_id;
    audioEl.load();
    audioEl.play();
    downloadEl.href = "/download/outputs/" + data.output_id;
    downloadEl.setAttribute("download", data.output_id);
    document.getElementById("builtinResultCard").classList.remove("hidden");
  } catch (e) {
    setStatus(status, "Request failed: " + e.message, "err");
  }

  document.getElementById("builtinSynthBtn").disabled = false;
});

// ── Subtitles (.srt) UI ────────────────────────────────────

const srtFileInput   = document.getElementById("srtFile");
const srtModeInputs  = document.querySelectorAll('input[name="srtMode"]');
const srtBuiltinWrap = document.getElementById("srtBuiltinWrap");
const srtSpeakerSel  = document.getElementById("srtSpeakerSelect");
const srtLangSel     = document.getElementById("srtLang");
const srtFormatSel   = document.getElementById("srtFormat");
const srtBtn         = document.getElementById("srtGenerateBtn");
const srtJobControls = document.getElementById("srtJobControls");
const srtPauseResumeBtn = document.getElementById("srtPauseResumeBtn");
const srtCancelBtn = document.getElementById("srtCancelBtn");

function getSrtMode() {
  const el = document.querySelector('input[name="srtMode"]:checked');
  return el ? el.value : "clone";
}

function syncSrtModeUI() {
  if (!srtBuiltinWrap) return;
  const mode = getSrtMode();
  if (mode === "builtin") srtBuiltinWrap.classList.remove("hidden");
  else srtBuiltinWrap.classList.add("hidden");
  updateSrtButtonState();
}

function setSrtJobControlsState({ visible = false, paused = false, busy = false } = {}) {
  if (!srtJobControls || !srtPauseResumeBtn) return;
  srtJobControls.classList.toggle("hidden", !visible);
  srtPauseResumeBtn.disabled = busy;
  srtPauseResumeBtn.dataset.mode = paused ? "resume" : "pause";
  srtPauseResumeBtn.textContent = paused ? "Resume" : "Pause";
  if (srtCancelBtn) srtCancelBtn.disabled = busy;
}

function updateSrtButtonState() {
  if (!srtBtn) return;
  const hasFile   = !!(srtFileInput && srtFileInput.files && srtFileInput.files.length);
  const mode      = getSrtMode();
  const okClone   = mode === "clone"   && currentRefIds.length > 0;
  const okBuiltin = mode === "builtin" && srtSpeakerSel && srtSpeakerSel.value && !srtSpeakerSel.value.includes("Loading");
  srtBtn.disabled = !!activeSrtJobId || !(modelReady && hasFile && (okClone || okBuiltin));
}

if (srtModeInputs && srtModeInputs.length) {
  srtModeInputs.forEach(el => el.addEventListener("change", syncSrtModeUI));
}
if (srtFileInput) srtFileInput.addEventListener("change", updateSrtButtonState);
if (srtSpeakerSel) srtSpeakerSel.addEventListener("change", updateSrtButtonState);
syncSrtModeUI();

if (srtPauseResumeBtn) {
  srtPauseResumeBtn.addEventListener("click", async () => {
    if (!activeSrtJobId) return;
    const statusEl = document.getElementById("srtStatus");
    const paused = srtPauseResumeBtn.dataset.mode === "resume";
    const action = paused ? "resume" : "pause";
    setSrtJobControlsState({ visible: true, paused, busy: true });
    try {
      const res = await fetch(`/srt-job/${activeSrtJobId}/${action}`, { method: "POST" });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      const isPaused = data.status === "paused" || !!data.paused;
      setSrtJobControlsState({ visible: true, paused: isPaused, busy: false });
      setStatus(statusEl, isPaused ? "Paused — click Resume to continue." : "Resumed — synthesis will continue.", "loading");
    } catch (e) {
      setSrtJobControlsState({ visible: true, paused, busy: false });
      setStatus(statusEl, "Request failed: " + e.message, "err");
    }
  });
}

if (srtCancelBtn) {
  srtCancelBtn.addEventListener("click", async () => {
    if (!activeSrtJobId) return;
    const statusEl = document.getElementById("srtStatus");
    setSrtJobControlsState({ visible: true, paused: false, busy: true });
    try {
      const res = await fetch(`/srt-job/${activeSrtJobId}/cancel`, { method: "POST" });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      activeSrtJobId = null;
      setSrtJobControlsState({ visible: false });
      setStatus(statusEl, "Canceled.", "err");
      updateSrtButtonState();
    } catch (e) {
      setSrtJobControlsState({ visible: true, paused: false, busy: false });
      setStatus(statusEl, "Request failed: " + e.message, "err");
    }
  });
}

// ── SRT synthesis ──────────────────────────────────────────

if (srtBtn) {
  srtBtn.addEventListener("click", async () => {
    const status = document.getElementById("srtStatus");
    const fileEl = document.getElementById("srtFile");
    const mode   = getSrtMode();
    const lang   = srtLangSel ? srtLangSel.value : "en";
    const format = srtFormatSel ? srtFormatSel.value : "wav";
    const progWrap = document.getElementById("srtProgressWrap");
    const progBar  = document.getElementById("srtProgressBar");
    const progText = document.getElementById("srtProgressText");

    if (!fileEl || !fileEl.files.length) {
      setStatus(status, "Please select an .srt file first.", "err");
      return;
    }
    if (!modelReady) {
      setStatus(status, "Model is not ready yet — please wait.", "err");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileEl.files[0]);
    formData.append("lang", lang);
    formData.append("format", format);
    // Use async job mode so we can show progress.
    formData.append("async", "1");

    let endpoint = "/synthesize-srt";
    if (mode === "clone") {
      if (!currentRefIds.length) {
        setStatus(status, "Please upload at least one reference audio clip first (Clone mode).", "err");
        return;
      }
      // Append each ref_id as a separate form field (ref_ids[])
      currentRefIds.forEach(item => formData.append("ref_ids[]", item.ref_id));
    } else {
      endpoint = "/synthesize-srt-builtin";
      const speaker = srtSpeakerSel ? srtSpeakerSel.value : "";
      if (!speaker) {
        setStatus(status, "Please select a built-in speaker first.", "err");
        return;
      }
      formData.append("speaker", speaker);
    }

    setStatus(status, '<span class="spin">⟳</span> Synthesizing from SRT, please wait…', "loading");
    srtBtn.disabled = true;
    if (progWrap && progBar && progText) {
      progWrap.classList.remove("hidden");
      progBar.style.width = "0%";
      progText.textContent = "Starting…";
    }
    setSrtJobControlsState({ visible: false });

    try {
      const res  = await fetch(endpoint, { method: "POST", body: formData });
      const data = await res.json();
      if (data.error) {
        setStatus(status, "Error: " + data.error, "err");
        updateSrtButtonState();
        return;
      }

      // Async job mode: poll progress until done/error
      const jobId = data.job_id;
      if (!jobId) {
        setStatus(status, "Error: missing job_id from server.", "err");
        updateSrtButtonState();
        return;
      }
      activeSrtJobId = jobId;
      setSrtJobControlsState({ visible: true, paused: false, busy: false });
      updateSrtButtonState();

      const poll = async () => {
        const j = await fetch("/srt-job/" + jobId).then(r => r.json());
        if (j.error && j.status !== "error") {
          activeSrtJobId = null;
          setSrtJobControlsState({ visible: false });
          updateSrtButtonState();
          throw new Error(j.error);
        }
        const p = Math.max(0, Math.min(1, j.progress || 0));
        if (progWrap && progBar && progText) {
          progBar.style.width = Math.round(p * 100) + "%";
          const done = j.done || 0;
          const total = j.total || 0;
          const label = total ? `Synthesizing cues: ${done}/${total} (${Math.round(p * 100)}%)` : `Progress: ${Math.round(p * 100)}%`;
          progText.textContent = j.status === "paused" ? `${label} — paused` : label;
        }
        setSrtJobControlsState({ visible: j.status === "running" || j.status === "paused", paused: j.status === "paused", busy: false });

        if (j.status === "paused") {
          setStatus(status, "Paused — click Resume to continue.", "loading");
          setTimeout(poll, 1000);
          return;
        }
        if (j.status === "canceled") {
          activeSrtJobId = null;
          setSrtJobControlsState({ visible: false });
          setStatus(status, "Canceled.", "err");
          updateSrtButtonState();
          return;
        }

        if (j.status === "done" && j.output_id) {
          activeSrtJobId = null;
          setStatus(status, "✓ Done", "ok");
          if (progWrap && progBar && progText) {
            progBar.style.width = "100%";
            progText.textContent = "Completed.";
          }
          setSrtJobControlsState({ visible: false });
          const audioEl    = document.getElementById("srtAudio");
          const downloadEl = document.getElementById("srtDownload");
          audioEl.src = "/play/outputs/" + j.output_id;
          audioEl.load();
          audioEl.play();
          downloadEl.href = "/download/outputs/" + j.output_id;
          downloadEl.setAttribute("download", j.output_id);
          downloadEl.textContent = "Download " + format.toUpperCase();
          document.getElementById("srtResultCard").classList.remove("hidden");
          refreshHistory();
          updateSrtButtonState();
          return;
        }
        if (j.status === "error") {
          activeSrtJobId = null;
          setStatus(status, "Error: " + (j.error || "SRT synthesis failed"), "err");
          setSrtJobControlsState({ visible: false });
          updateSrtButtonState();
          return;
        }
        setTimeout(poll, 1000);
      };

      setTimeout(poll, 750);
    } catch (e) {
      activeSrtJobId = null;
      setSrtJobControlsState({ visible: false });
      setStatus(status, "Request failed: " + e.message, "err");
    }

    updateSrtButtonState();
  });
}

// ── History ────────────────────────────────────────────────

document.getElementById("refreshHistoryBtn").addEventListener("click", refreshHistory);

async function refreshHistory() {
  const list = document.getElementById("historyList");
  list.innerHTML = "<p style='color:#475569;font-size:.85rem'>Loading…</p>";
  try {
    const files = await fetch("/list-outputs").then(r => r.json());
    if (!files.length) {
      list.innerHTML = "<p style='color:#475569;font-size:.85rem'>No files yet.</p>";
      return;
    }
    list.innerHTML = files.map(f => `
      <div class="history-item" id="item-${f.name}">
        <audio controls src="/play/outputs/${f.name}" preload="none"></audio>
        <span class="hist-name" title="${f.name}">${f.name.slice(0, 12)}… (${fmtBytes(f.size)})</span>
        <div class="hist-actions">
          <a href="/download/outputs/${f.name}" download class="btn secondary">Download</a>
          <button class="btn danger" onclick="deleteOutput('${f.name}')">Delete</button>
        </div>
      </div>
    `).join("");
  } catch (e) {
    list.innerHTML = `<p style='color:#f87171'>Failed to load: ${e.message}</p>`;
  }
}

async function deleteOutput(name) {
  await fetch("/delete-output/" + name, { method: "DELETE" });
  const el = document.getElementById("item-" + name);
  if (el) el.remove();
}
