const state = {
  selectedRegime: null,
  galleries: {},
  overlaps: {},
  pollTimer: null,
  genPoll: null,
  activeRun: null,
  sources: [],
  pendingFiles: null,
  pendingMeta: null,
  /** @type {Record<string, Set<string>>} selected generated filenames per run */
  selectedByRun: {},
  sendingRun: null,
};

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

function selectedModelMeta() {
  const id = $("#model")?.value || "";
  return (state.imageModels || []).find((m) => m.id === id) || { id, provider: "openai" };
}

function isGeminiModel() {
  return selectedModelMeta().provider === "gemini";
}

function apiKey() {
  let v = ($("#apiKey")?.value || "").trim().replace(/^["']|["']$/g, "").replace(/^bearer\s+/i, "").trim();
  const store = isGeminiModel() ? "bs_gemini_key" : "bs_openai_key";
  if (v) {
    localStorage.setItem(store, v);
    $("#apiKey").value = v;
    return v;
  }
  return null;
}

// Rough Batch-API $/image (1024², incl. prompt + source tokens) — from test runs; real usage lands in run_manifest.json
const OPENAI_BATCH_PRICE = {
  "gpt-image-1": { low: 0.012, medium: 0.03, high: 0.09 },
  "gpt-image-1.5": { low: 0.01, medium: 0.032, high: 0.08 },
  "gpt-image-2": { low: 0.008, medium: 0.038, high: 0.11 },
  "gpt-image-1-mini": { low: 0.002, medium: 0.005, high: 0.018 },
};

function syncProviderUi() {
  const gem = isGeminiModel();
  const oa = $("#openaiOpts");
  if (oa) oa.style.display = gem ? "none" : "";
  const label = $("#keyLabel");
  if (label) label.textContent = gem ? "Gemini key" : "OpenAI key";
  const input = $("#apiKey");
  if (input) {
    const saved = localStorage.getItem(gem ? "bs_gemini_key" : "bs_openai_key") || "";
    input.placeholder = gem ? "AIza… (aistudio.google.com) — blank uses .env" : "sk-… (platform.openai.com) — blank uses .env";
    input.value = saved;
  }
  const finalBtn = $("#finalBtn");
  if (finalBtn) {
    finalBtn.textContent = gem
      ? "Final generate (Gemini batch · all sources)"
      : "Final generate (batch · all sources · chunks of 13)";
  }
  const testBtn = $("#testBtn");
  if (testBtn) testBtn.textContent = gem ? "Test generate (Gemini batch · 1 source)" : "Test generate (batch · 1 source)";
  updateOpenaiOpts();
}

function updateOpenaiOpts() {
  const model = $("#model")?.value || "gpt-image-2";
  const cost = $("#oaCost");
  if (!cost) return;
  const per = (OPENAI_BATCH_PRICE[model] || {})[$("#oaQuality").value];
  const perSrc = Number($("#perSource").value || 4);
  const nSrc = (state.sources || []).length;
  cost.textContent = per
    ? `≈ $${per.toFixed(3)}/img (batch) · test ${Math.min(perSrc, 4)} imgs ≈ $${(per * Math.min(perSrc, 4)).toFixed(2)}`
      + (nSrc ? ` · final ${nSrc}×${perSrc} ≈ $${(per * nSrc * perSrc).toFixed(2)} · ${Math.ceil(nSrc / 13)} chunk${Math.ceil(nSrc / 13) === 1 ? "" : "s"} of ≤13` : "")
    : "";
}

async function checkApiKey() {
  const status = $("#genStatus");
  try {
    const data = await api("/api/openai/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey() || null }),
    });
    status.className = data.ok ? "status ok" : "status error";
    status.textContent = data.ok
      ? `OpenAI key OK (${data.masked || "?"}${data.source ? `, from ${data.source}` : ""})`
      : `OpenAI key failed: ${data.message || "invalid"}`;
  } catch (e) {
    alert(e.message);
  }
}

async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: "same-origin", ...opts });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("Authentication required");
  }
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data));
  return data;
}

async function logout() {
  try {
    await api("/api/logout", { method: "POST" });
  } catch (_) { /* still leave */ }
  window.location.href = "/login";
}

function switchTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
  if (name === "generate") refreshGenerate();
}

function fmt(n, d = 3) {
  if (n == null || Number.isNaN(n)) return "—";
  return typeof n === "number" ? n.toFixed(d) : String(n);
}

function setUploadBanner(kind, phase, detail) {
  const box = $("#uploadStatus");
  if (!box) return;
  box.className = "upload-status show" + (kind ? ` ${kind}` : "");
  $("#uploadPhase").textContent = phase;
  $("#uploadDetail").textContent = detail;
}

async function loadHealth() {
  const h = await api("/api/health");
  const n = h.images != null ? ` · ${h.images} images` : "";
  $("#datasetPath").textContent = (h.dataset || "(no dataset)") + n;
  state.hasOpenaiKey = !!h.has_openai_key;
  state.hasGeminiKey = !!h.has_gemini_key;
  state.hasOpenrouterKey = !!h.has_openrouter_key;
  const od = h.openai_defaults || {};
  if (!state.oaDefaultsApplied) {
    state.oaDefaultsApplied = true;
    if (od.quality) $("#oaQuality").value = od.quality;
    if (od.input_fidelity != null) $("#oaFidelity").value = od.input_fidelity;
  }
  const hfBtn = $("#hfUploadBtn");
  const hfHint = $("#hfHint");
  if (hfBtn) {
    const ready = !!(h.has_hf_token && h.hf_repo);
    hfBtn.disabled = !ready;
    hfBtn.title = ready
      ? `Upload selected/latest final run → ${h.hf_repo}/runs/YYYYMMDD_HHMMSS/`
      : "Set hf_token and hf_basic in .env";
  }
  if (hfHint) {
    hfHint.textContent = h.hf_repo
      ? `${h.hf_repo} · runs/{date}/ · ${h.hf_private ? "private" : "public"}`
      : "One repo · dated runs/ folder (configure .env)";
  }
  state.genMode = h.gen_mode || "openai";
  state.detectors = h.detectors || {};
  state.switches = h.checkpoint_switches || {};
  renderSwitches();
  if (!state.genModeTouched) setGenMode(state.genMode);
  else renderGenModeHint();
  fillModelSelect(h);
  syncProviderUi();
  if (h.dataset && h.images) {
    setUploadBanner("ok", "ready", `Loaded ${h.images} images from ${h.dataset}`);
  }
}

function fillModelSelect(h) {
  const sel = $("#model");
  if (!sel) return;
  const all = Array.isArray(h.image_models) && h.image_models.length
    ? h.image_models
    : [
        { id: "gpt-image-2", label: "GPT Image 2", provider: "openai" },
      ];
  // Prompts were mined with this mode's detector (Analyze page) — only offer that provider's models
  const mode = h.gen_mode || "openai";
  const models = all.filter((m) => (m.provider || "openai") === mode).length
    ? all.filter((m) => (m.provider || "openai") === mode)
    : all;
  state.imageModels = models;
  const stored = localStorage.getItem("bs_image_model");
  const preferred = [stored, h.default_model].find((id) => models.some((m) => m.id === id)) || models[0].id;

  sel.innerHTML = "";
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    const prov = m.provider ? `[${m.provider}] ` : "";
    opt.textContent = `${prov}${m.label || m.id}`;
    opt.title = m.id + (m.hint ? ` — ${m.hint}` : "");
    if (m.id === preferred) {
      opt.selected = true;
    }
    sel.appendChild(opt);
  }
  const applyHint = () => {
    const cur = models.find((m) => m.id === sel.value);
    const hint = $("#modelHint");
    const d = (h.detectors || {})[mode];
    const modeNote = ` · mode ${mode} (Analyze page)${d ? (d.enabled === false ? " · detector off (source labels kept)" : ` · labels by ${d.weights} @ ${d.conf}`) : ""}`;
    if (hint) hint.textContent = (cur?.hint ? `${cur.id} · ${cur.hint}` : sel.value) + modeNote;
    syncProviderUi();
  };
  sel.onchange = () => {
    localStorage.setItem("bs_image_model", sel.value);
    applyHint();
  };
  applyHint();
}

function newUploadId() {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

function classifyUploadFiles(fileList) {
  const list = [...(fileList || [])];
  const useful = [];
  let nImg = 0;
  let nLbl = 0;
  let nYaml = 0;
  let hasImagesDir = false;
  let hasLabelsDir = false;
  for (const f of list) {
    const rel = (f.webkitRelativePath || f.name || "").replace(/\\/g, "/");
    const lower = rel.toLowerCase();
    const parts = lower.split("/").filter(Boolean);
    if (parts.includes("images")) hasImagesDir = true;
    if (parts.includes("labels")) hasLabelsDir = true;
    if (/\.(jpg|jpeg|png|webp|bmp)$/i.test(f.name)) {
      useful.push(f);
      nImg += 1;
    } else if (/\.txt$/i.test(f.name)) {
      useful.push(f);
      nLbl += 1;
    } else if (parts[parts.length - 1] === "data.yaml" || f.name.toLowerCase() === "data.yaml") {
      useful.push(f);
      nYaml += 1;
    }
  }
  const folderName = (useful[0]?.webkitRelativePath || useful[0]?.name || "upload")
    .split(/[/\\]/)[0] || "upload";
  const ok = nImg > 0 && (hasImagesDir || nImg > 0) && (hasLabelsDir || nLbl > 0);
  const errors = [];
  if (!nImg) errors.push("No images found");
  if (!hasImagesDir && nImg) errors.push("Tip: select the parent folder that contains images/");
  if (!hasLabelsDir) errors.push("No labels/ folder detected — YOLO labels recommended");
  if (!nLbl) errors.push("No .txt label files found");
  return {
    useful,
    nImg,
    nLbl,
    nYaml,
    hasImagesDir,
    hasLabelsDir,
    folderName,
    ok: nImg > 0 && nLbl > 0 && hasImagesDir && hasLabelsDir,
    softOk: nImg > 0,
    errors,
  };
}

function clearEvalSelection() {
  state.pendingFiles = null;
  state.pendingMeta = null;
  const input = $("#evalFolder");
  if (input) input.value = "";
  $("#evalFolderName").textContent = "No folder chosen yet";
  const title = $("#evalDropTitle");
  if (title) title.textContent = "Select a YOLO dataset folder";
  const okBtn = $("#evalOkBtn");
  if (okBtn) okBtn.disabled = true;
  $("#evalDropzone")?.classList.remove("dragover");
  setUploadBanner("", "idle", "Selection cleared. Drop or choose a folder, then press OK to upload into beverage_gt/.");
}

function stageEvalFolder(fileList) {
  const meta = classifyUploadFiles(fileList);
  state.pendingMeta = meta;
  state.pendingFiles = meta.useful;
  const okBtn = $("#evalOkBtn");
  const title = $("#evalDropTitle");
  if (!meta.useful.length) {
    clearEvalSelection();
    setUploadBanner("warn", "invalid", "No images or label .txt files in that selection.");
    return;
  }
  if (title) title.textContent = `Folder ready: ${meta.folderName}`;
  $("#evalFolderName").innerHTML =
    `<strong>${meta.folderName}</strong> — ${meta.nImg} images · ${meta.nLbl} labels`
    + (meta.nYaml ? " · data.yaml" : "")
    + (meta.ok ? " · valid YOLO layout" : " · check images/ + labels/");
  if (okBtn) okBtn.disabled = !meta.softOk;
  if (meta.ok) {
    setUploadBanner(
      "ok",
      "ready",
      `Valid YOLO folder. Press OK to upload into beverage_gt/ and start analyze (${meta.nImg} imgs / ${meta.nLbl} labels).`
    );
  } else if (meta.softOk) {
    setUploadBanner(
      "warn",
      "check",
      `${meta.errors.join(" · ")}. You can still press OK if images exist, but images/ + labels/ is preferred.`
    );
  } else {
    setUploadBanner("warn", "invalid", meta.errors.join(" · ") || "Invalid folder");
  }
}

function readEntryFile(entry, pathPrefix) {
  return new Promise((resolve, reject) => {
    entry.file((file) => {
      const rel = `${pathPrefix}${file.name}`;
      try {
        Object.defineProperty(file, "webkitRelativePath", { value: rel, configurable: true });
      } catch {
        file.webkitRelativePath = rel;
      }
      resolve(file);
    }, reject);
  });
}

function readDirectoryEntries(dirEntry) {
  const reader = dirEntry.createReader();
  return new Promise((resolve, reject) => {
    const all = [];
    const readBatch = () => {
      reader.readEntries((entries) => {
        if (!entries.length) return resolve(all);
        all.push(...entries);
        readBatch();
      }, reject);
    };
    readBatch();
  });
}

async function walkDirectoryEntry(dirEntry, pathPrefix = "") {
  const prefix = pathPrefix || `${dirEntry.name}/`;
  const entries = await readDirectoryEntries(dirEntry);
  const files = [];
  for (const entry of entries) {
    if (entry.isFile) {
      files.push(await readEntryFile(entry, prefix));
    } else if (entry.isDirectory) {
      files.push(...await walkDirectoryEntry(entry, `${prefix}${entry.name}/`));
    }
  }
  return files;
}

async function filesFromDataTransfer(dt) {
  const items = [...(dt.items || [])];
  const files = [];
  for (const item of items) {
    if (item.kind !== "file") continue;
    const entry = item.webkitGetAsEntry?.();
    if (entry?.isDirectory) {
      files.push(...await walkDirectoryEntry(entry));
    } else if (entry?.isFile) {
      files.push(await readEntryFile(entry, ""));
    } else {
      const f = item.getAsFile?.();
      if (f) files.push(f);
    }
  }
  if (!files.length && dt.files?.length) return [...dt.files];
  return files;
}

function wireEvalDropzone() {
  const zone = $("#evalDropzone");
  const input = $("#evalFolder");
  if (!zone || !input) return;

  // Label[for=evalFolder] opens the picker natively — don't intercept that click.
  zone.addEventListener("click", (ev) => {
    if (ev.target.closest("label[for='evalFolder'], #evalChooseBtn, a, button")) return;
    // Clicking empty dropzone area also opens picker
    input.click();
  });
  zone.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter" && ev.key !== " ") return;
    if (ev.target.closest("label[for='evalFolder'], #evalChooseBtn")) return;
    ev.preventDefault();
    input.click();
  });

  ["dragenter", "dragover"].forEach((type) => {
    zone.addEventListener(type, (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      zone.classList.add("dragover");
    });
  });
  ["dragleave", "dragend"].forEach((type) => {
    zone.addEventListener(type, (ev) => {
      ev.preventDefault();
      zone.classList.remove("dragover");
    });
  });
  zone.addEventListener("drop", async (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    zone.classList.remove("dragover");
    try {
      setUploadBanner("run", "reading", "Reading dropped folder…");
      const files = await filesFromDataTransfer(ev.dataTransfer);
      stageEvalFolder(files);
    } catch (e) {
      setUploadBanner("warn", "error", e.message || "Could not read dropped folder");
    }
  });

  input.addEventListener("change", (ev) => {
    stageEvalFolder(ev.target.files || []);
  });
}

async function uploadEvalFolder(files, { replace = true } = {}) {
  const meta = classifyUploadFiles(files);
  const useful = meta.useful;
  if (!useful.length) throw new Error("No images or label .txt files found in that folder.");
  if (!meta.nImg) throw new Error("No evaluation images found. Select the parent folder that contains images/ and labels/.");

  const folderName = meta.folderName;
  $("#evalFolderName").innerHTML =
    `<strong>${folderName}</strong> — uploading to beverage_gt (${meta.nImg} images, ${meta.nLbl} labels)`;
  setUploadBanner("run", "uploading", `Uploading into beverage_gt/ (${meta.nImg} images / ${meta.nLbl} labels)…`);
  $("#jobBar").classList.remove("hidden");
  $("#analyzeBtn").disabled = true;
  const okBtn = $("#evalOkBtn");
  if (okBtn) okBtn.disabled = true;

  const uploadId = "beveragegt";
  const chunkSize = 40;
  const chunks = Math.ceil(useful.length / chunkSize);
  for (let i = 0; i < chunks; i++) {
    const slice = useful.slice(i * chunkSize, (i + 1) * chunkSize);
    const fd = new FormData();
    for (const f of slice) {
      const rel = f.webkitRelativePath || f.name;
      fd.append("files", f, rel);
    }
    const qs = new URLSearchParams({
      upload_id: uploadId,
      workers: "8",
      target: "beverage_gt",
      replace: i === 0 && replace ? "true" : "false",
    });
    const body = await api(`/api/import-upload?${qs}`, { method: "POST", body: fd });
    const doneFiles = Math.min((i + 1) * chunkSize, useful.length);
    setUploadBanner(
      "run",
      "uploading",
      `beverage_gt: ${doneFiles}/${useful.length} files · ${body.images || "?"} imgs / ${body.labels || "?"} labels`
    );
    $("#jobFill").style.width = `${Math.round((doneFiles / useful.length) * 90)}%`;
    $("#jobMsg").textContent = `uploading ${i + 1}/${chunks} → beverage_gt`;
  }

  setUploadBanner("run", "analyzing", "Upload complete — analyzing beverage_gt…");
  $("#jobMsg").textContent = "finalizing beverage_gt…";
  const fin = await api(
    `/api/import-upload?upload_id=${uploadId}&finalize=true&workers=8&target=beverage_gt`,
    { method: "POST" }
  );
  $("#evalFolderName").innerHTML =
    `<strong>beverage_gt</strong> — ${fin.images || meta.nImg} images / ${fin.labels || meta.nLbl} labels`;
  setUploadBanner(
    "run",
    "analyzing",
    `Saved to beverage_gt (${fin.images || meta.nImg} imgs / ${fin.labels || meta.nLbl} labels) — clustering now…`
  );
  $("#jobFill").style.width = "92%";
  state.pendingFiles = null;
  state.pendingMeta = null;
  pollJob();
}

async function confirmUploadToBeverageGt() {
  if (!state.pendingFiles?.length) {
    setUploadBanner("warn", "idle", "Choose a folder first, then press OK.");
    return;
  }
  const meta = state.pendingMeta || classifyUploadFiles(state.pendingFiles);
  if (!meta.ok) {
    const proceed = confirm(
      `Folder may be incomplete:\n${(meta.errors || []).join("\n")}\n\nUpload to beverage_gt anyway?`
    );
    if (!proceed) return;
  } else {
    const proceed = confirm(
      `Upload ${meta.nImg} images + ${meta.nLbl} labels into beverage_gt/\n(this replaces existing files with the same names / clears on first chunk).\n\nContinue?`
    );
    if (!proceed) return;
  }
  try {
    await uploadEvalFolder(state.pendingFiles, { replace: true });
  } catch (e) {
    setUploadBanner("warn", "error", e.message);
    $("#analyzeBtn").disabled = false;
    const okBtn = $("#evalOkBtn");
    if (okBtn) okBtn.disabled = false;
    alert(e.message);
  }
}

function setGenMode(mode, touched = false) {
  const input = $("#genMode");
  if (input) input.value = mode;
  if (touched) state.genModeTouched = true;
  renderGenModeHint();
}

// Segmented OpenAI / Gemini picker: checkpoint + conf per option, "analyzed" badge on the mode the current
// analysis/prompts were built for, and a re-analyze nudge when the selection differs from it.
function renderGenModeHint() {
  const sel = $("#genMode")?.value || state.genMode || "openai";
  for (const btn of $$(".mode-opt")) {
    const m = btn.dataset.mode;
    const on = m === sel;
    btn.classList.toggle("on", on);
    btn.setAttribute("aria-checked", on ? "true" : "false");
    const d = (state.detectors || {})[m];
    const ck = btn.querySelector(".mode-ckpt");
    if (ck) {
      // "project__model.pt" → model name; keep the project only for generic names (best / final / last …)
      const [proj, model] = d ? d.weights.replace(/\.pt$/, "").split("__") : [];
      const name = !d ? "—" : !model ? proj : /^(best|final|last|weights|train|exp)\b/i.test(model) ? `${proj}/${model}` : model;
      const up = d && d.source === "uploaded";
      ck.innerHTML = d
        ? `${up ? '<span class="up-tag">uploaded</span>' : ""}${esc(name)} · conf ${d.conf}${d.enabled === false ? " · off" : d.exists ? "" : " · missing"}`
        : "—";
      ck.title = d
        ? `${d.weights} · ScoreVision conf ${d.conf}`
          + (d.sv ? ` · SV ${d.sv.composite} (mAP50 ${d.sv.map50}, FP/img ${d.sv.fp_per_image}) on ${d.gt_images} GT imgs` : "")
          + (d.uploaded_at ? ` · uploaded ${d.uploaded_at}` : "")
        : "";
      ck.classList.toggle("bad", Boolean(d && d.enabled !== false && !d.exists));
    }
    const badge = btn.querySelector(".mode-badge");
    if (badge) badge.hidden = m !== state.genMode;
  }
  const dirty = sel !== state.genMode || Boolean((state.ckptDirty || {})[sel]);
  const stale = $("#genModeStale");
  if (stale) {
    stale.hidden = !dirty;
    stale.textContent = sel !== state.genMode ? "re-analyze to switch" : "re-analyze to use new checkpoint";
  }
  const btn = $("#analyzeBtn");
  if (btn) btn.classList.toggle("pulse", dirty);
  const nm = $("#ckptModeName");
  if (nm) nm.textContent = sel === "gemini" ? "Gemini" : "OpenAI";
  const reset = $("#ckptResetBtn");
  if (reset) reset.hidden = !((state.detectors || {})[sel]?.source === "uploaded");
}

// Upload a .pt for the selected mode → server validates it and tunes its ScoreVision conf on the GT.
async function uploadCheckpoint(file) {
  const mode = $("#genMode")?.value || "openai";
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  showCkptStatus(`Uploading ${file.name} (${(file.size / 1048576).toFixed(1)} MB)…`, 0, "");
  $("#ckptUploadBtn").disabled = true;
  try {
    await api(`/api/detector/${mode}/upload`, { method: "POST", body: fd });
    await pollCkpt();
  } catch (e) {
    showCkptStatus(e.message || String(e), 0, "error");
    $("#ckptUploadBtn").disabled = false;
  }
}

function showCkptStatus(msg, progress, kind) {
  const box = $("#ckptStatus");
  if (!box) return;
  box.hidden = false;
  box.className = `ckpt-status ${kind || ""}`;
  $("#ckptMsg").textContent = msg;
  $("#ckptFill").style.width = `${Math.round((progress || 0) * 100)}%`;
}

async function pollCkpt() {
  for (;;) {
    let job;
    try { job = await api("/api/detector/job"); } catch { await new Promise((r) => setTimeout(r, 2000)); continue; }
    if (job.status === "running") {
      showCkptStatus(`${job.name}: ${job.message}`, job.progress, "");
      $("#ckptUploadBtn").disabled = true;
      await new Promise((r) => setTimeout(r, 1000));
      continue;
    }
    $("#ckptUploadBtn").disabled = false;
    if (job.status === "done") {
      showCkptStatus(job.message, 1, "ok");
      state.ckptDirty = { ...(state.ckptDirty || {}), [job.mode]: true };
      await loadHealth();
    } else if (job.status === "error") {
      showCkptStatus(job.message || job.error, 0, "error");
    }
    return job;
  }
}

function renderSwitches() {
  for (const box of $$("[data-switch]")) box.checked = Boolean((state.switches || {})[box.dataset.switch]);
}

// Checkpoint on / off → saved server-side (overrides .env). Prompts change only after the next Analyze & cluster.
async function toggleSwitch(box) {
  const name = box.dataset.switch;
  box.disabled = true;
  try {
    state.switches = await api(`/api/checkpoints/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: box.checked }),
    });
    // Checkpoint switches change the prompts (→ re-analyze); the CCTV look only affects new runs.
    const modes = name === "person_detector" ? ["openai", "gemini"]
      : name.startsWith("detector_") ? [name.replace("detector_", "")] : [];
    state.ckptDirty = { ...(state.ckptDirty || {}), ...Object.fromEntries(modes.map((m) => [m, true])) };
    await loadHealth();
  } catch (e) {
    showCkptStatus(e.message || String(e), 0, "error");
    renderSwitches();
  } finally {
    box.disabled = false;
  }
}

async function resetCheckpoint() {
  const mode = $("#genMode")?.value || "openai";
  if (!window.confirm(`Use the default ${mode} checkpoint again (drops the uploaded one)?`)) return;
  try {
    await api(`/api/detector/${mode}/reset`, { method: "POST" });
    state.ckptDirty = { ...(state.ckptDirty || {}), [mode]: true };
    showCkptStatus(`${mode}: back to the default checkpoint`, 1, "ok");
    await loadHealth();
  } catch (e) {
    showCkptStatus(e.message || String(e), 0, "error");
  }
}

async function startAnalyze() {
  $("#analyzeBtn").disabled = true;
  $("#jobBar").classList.remove("hidden");
  try {
    await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: $("#genMode")?.value || "openai" }),
    });
    pollJob();
  } catch (e) {
    alert(e.message);
    $("#analyzeBtn").disabled = false;
  }
}

function pollJob() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    const job = await api("/api/job");
    $("#jobMsg").textContent = `${job.status}: ${job.message || ""}`;
    $("#jobFill").style.width = `${Math.round((job.progress || 0) * 100)}%`;
    if (job.status === "done" || job.status === "error") {
      clearInterval(state.pollTimer);
      $("#analyzeBtn").disabled = false;
      if (job.status === "done") {
        state.genModeTouched = false;
        state.ckptDirty = {};
        setUploadBanner("ok", "ready", job.message || "Analysis complete");
        await loadHealth();
        await loadAnalysis();
      }
      if (job.status === "error") {
        setUploadBanner("warn", "error", job.error || job.message || "failed");
        alert(job.error || job.message);
      }
    }
  }, 700);
}

async function loadAnalysis() {
  const [summary, galleries, overlaps, mining] = await Promise.all([
    api("/api/summary"),
    api("/api/galleries"),
    api("/api/overlaps"),
    api("/api/mining-report"),
  ]);
  state.galleries = galleries || {};
  state.overlaps = overlaps || {};

  const kpis = [
    ["Images", summary.count ?? 0, ""],
    ["Objs / img", fmt(summary.mean_object_count, 2), ""],
    ["Brightness", fmt(summary.mean_brightness, 3), ""],
    ["Med bbox area", fmt(summary.median_bbox_area, 4), ""],
    ["Cups", summary.class_instance_totals?.cup ?? "—", ""],
    ["Cans", summary.class_instance_totals?.can ?? "—", ""],
    ["Bottles", summary.class_instance_totals?.bottle ?? "—", ""],
  ];
  $("#kpiRow").innerHTML = kpis.map(([l, v]) => `
    <div class="stat"><div class="label">${l}</div><div class="value">${v}</div></div>`).join("");

  renderGalleryCards("#sceneGalleries", galleries, "scene");
  renderGalleryCards("#hardGalleries", galleries, "class_hard");
  renderGalleryCards("#overlapGalleries", overlaps, "overlap");

  const recs = mining.recommendations || [];
  $("#recs").innerHTML = recs.length
    ? recs.map((r) => `<div class="rec ${r.priority || ""}"><strong>${r.title}</strong><div class="muted">${r.detail || ""}</div></div>`).join("")
    : '<p class="muted">Run analyze to get recommendations.</p>';

  await loadDetectorErrors();
  await loadSamples();
}

function esc(v) {
  return String(v ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// Thumbnails: server renders/caches a JPEG at width w (?w=); v pins a content version so the browser can keep it.
function sized(url, w, v) {
  const q = [w ? `w=${w}` : "", v ? `v=${encodeURIComponent(v)}` : ""].filter(Boolean).join("&");
  return q ? `${url}${url.includes("?") ? "&" : "?"}${q}` : url;
}

function short(v, n = 70) {
  const t = String(v ?? "").trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}

// near / mid / far / very far tag + where it sits in depth + size next to things at the same depth
function distanceCell(e) {
  const c = e.caption || {};
  if (!c.distance && !c.depth_detail) return '<span class="muted">—</span>';
  const d = String(c.distance || "").replace("_", " ");
  return `${d ? `<span class="dtag d-${esc(c.distance)}">${esc(d)}</span> ` : ""}${esc(short(c.depth_detail || "", 70))}`
    + (c.relative_size ? `<div class="muted">${esc(short(c.relative_size, 70))}</div>` : "");
}

// One compact table row per wrong drink: what went wrong · class · what it looks like · where it is.
function wrongRows(errors) {
  return `
    <table class="wrong-table">
      <thead><tr><th>#</th><th>Error</th><th>Distance from camera</th><th>Looks like</th><th>Where</th><th>Size</th></tr></thead>
      <tbody>
        ${errors.map((e) => {
          const c = e.caption || {};
          let tag;
          if (e.status === "false_positive") {
            tag = c.is_actually_drink
              ? `<span class="wtag gap" title="Gemini thinks this is a real drink the GT labels missed — not used as a hard negative">unlabelled ${esc(e.pred_class)}?</span>`
              : `<span class="wtag fa">false alarm: ${esc(e.pred_class)}</span>`;
          } else if (e.status === "wrong_class") {
            tag = `<span class="wtag wrong">${esc(e.gt_class)} → ${esc(e.pred_class)}</span>`;
          } else {
            tag = `<span class="wtag missed">missed ${esc(e.gt_class)}</span>`;
          }
          const why = [e.reason, c.likely_cause, c.confusers].filter(Boolean).join(" — ");
          return `<tr title="${esc(why)}">
            <td>${e.id}</td><td>${tag}</td>
            <td>${distanceCell(e)}</td>
            <td>${esc(short(c.object || "—"))}</td>
            <td>${esc(short(c.placement_detail || "—", 60))}</td>
            <td class="muted">${e.size_px?.[0]}×${e.size_px?.[1]}px${e.frame_pct ? `<br/>${e.frame_pct.h}% of frame h` : ""}</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
}

async function getModelErrors(force = false) {
  if (!state.modelErrors || force) {
    try { state.modelErrors = await api("/api/model-errors"); } catch { state.modelErrors = { ready: false }; }
  }
  return state.modelErrors;
}

async function renderWrongPanel(sourceImage) {
  const panel = $("#wrongPanel");
  if (!panel) return;
  $("#wrongSrcName").textContent = sourceImage || "—";
  const rep = await getModelErrors();
  if (!rep.ready) {
    panel.innerHTML = '<p class="muted">No detector errors yet — run Analyze &amp; cluster.</p>';
    return;
  }
  const src = (rep.sources || []).find((r) => r.filename === sourceImage);
  if (!src) {
    panel.innerHTML = '<p class="muted">Pick a source.</p>';
    return;
  }
  const k = src.counts || {};
  if (!src.errors.length) {
    panel.innerHTML = `<p class="muted">✓ No wrong beverages — the detector found all ${k.gt ?? 0} drinks here and raised no false alarms.</p>`;
    return;
  }
  const img = `/api/model-errors/${encodeURIComponent(src.filename)}/overlay`;
  panel.innerHTML = `
    <a class="wrong-img" href="${sized(img, 0, rep.version)}" target="_blank" rel="noopener"><img src="${sized(img, 800, rep.version)}" alt="" /></a>
    <div>
      <div class="wrong-sum"><strong>${k.missed}</strong> missed · <strong>${k.wrong_class}</strong> wrong class
        · <strong>${k.false_alarm_objects ?? 0}</strong> false alarms · ${k.hit}/${k.gt} found
        <span class="muted">(red = missed, magenta = wrong class, yellow = false alarm → hard negative; hover a row for why)</span></div>
      ${wrongRows(src.errors)}
    </div>`;
}

const DE_TYPES = [
  ["all", "All"], ["missed", "Missed"], ["wrong_class", "Wrong class"], ["false_positive", "False alarm"],
];
const DE_DIST = ["near", "mid", "far", "very_far"];

async function loadDetectorErrors() {
  const rep = await getModelErrors(true);
  const summary = $("#detErrSummary");
  const filters = $("#detErrFilters");
  const cards = $("#detErrCards");
  if (!rep.ready) {
    $("#detErrMeta").textContent = "";
    summary.innerHTML = "";
    filters.hidden = true;
    cards.innerHTML = '<p class="muted">Run Analyze &amp; cluster to mine the detector\'s errors.</p>';
    return;
  }
  state.de = state.de || { type: "all", cls: new Set(), dist: new Set(), q: "", sort: "errors", open: new Set() };
  $("#detErrMeta").textContent = `${rep.mode || "openai"} mode · ${String(rep.detector || "").replace(/\.pt$/, "")} · conf ${rep.conf} · ${rep.caption_model}`;
  renderDeSummary(rep);
  renderDeFilters(rep);
  filters.hidden = false;
  renderDeCards(rep);
}

function renderDeSummary(rep) {
  const s = rep.summary || {};
  const gt = s.gt || 0;
  const pct = (n, d) => (d ? Math.round((n / d) * 1000) / 10 : 0);
  const w = (n) => `${gt ? (n / gt) * 100 : 0}%`;
  const bc = s.by_class || {};
  const classes = ["cup", "can", "bottle"].map((c) => {
    const k = bc[c] || { gt: 0, missed: 0, wrong_class: 0, error_rate: 0 };
    const bad = (k.missed || 0) + (k.wrong_class || 0);
    const rate = Math.round((k.error_rate || 0) * 100);
    return `<div class="de-cls">
      <span class="de-cls-name">${c}</span>
      <span class="de-cls-bar"><i style="width:${rate}%" class="${rate >= 50 ? "hi" : rate >= 30 ? "mid" : "lo"}"></i></span>
      <span class="de-cls-val"><strong>${rate}%</strong> ${bad}/${k.gt}</span>
    </div>`;
  }).join("");
  const conf = Object.entries(s.confusions || {})
    .map(([k, v]) => `<span class="de-pill">${esc(k.replace("→", " → "))}<b>×${v}</b></span>`).join("")
    || '<span class="muted">none</span>';
  const fa = Object.entries(s.false_alarm_objects_seen || {}).slice(0, 4)
    .map(([k, v]) => `<span class="de-pill fa">${esc(k)}${v > 1 ? `<b>×${v}</b>` : ""}</span>`).join("");
  $("#detErrSummary").innerHTML = `
    <section class="de-hero">
      <div class="de-recall"><span>${pct(s.hit || 0, gt)}<small>%</small></span><em>recall</em></div>
      <div class="de-hero-body">
        <div class="de-hero-line"><strong>${s.hit ?? 0}</strong> of <strong>${gt}</strong> GT drinks found ·
          errors in <strong>${s.sources_with_errors ?? 0}</strong>/${(rep.sources || []).length} images</div>
        <div class="de-stack" title="found / missed / wrong class (share of GT drinks)">
          <i class="hit" style="width:${w(s.hit || 0)}"></i><i class="missed" style="width:${w(s.missed || 0)}"></i><i class="wrong" style="width:${w(s.wrong_class || 0)}"></i>
        </div>
        <div class="de-legend">
          <span><i class="hit"></i>found ${s.hit ?? 0}</span>
          <span><i class="missed"></i>missed ${s.missed ?? 0}</span>
          <span><i class="wrong"></i>wrong class ${s.wrong_class ?? 0}</span>
          <span><i class="fa"></i>false alarms ${s.false_alarm_objects ?? 0}<em> · ${s.hard_negatives ?? 0} hard negatives</em></span>
        </div>
      </div>
    </section>
    <section class="de-panel"><h4>Error rate by class</h4>${classes}</section>
    <section class="de-panel"><h4>Confusions</h4><div class="de-pills">${conf}</div>
      ${fa ? `<h4 class="de-sub">Fires on</h4><div class="de-pills">${fa}</div>` : ""}</section>`;
}

function deCounts(rep) {
  const c = { all: 0, missed: 0, wrong_class: 0, false_positive: 0 };
  for (const src of rep.sources || []) for (const e of src.errors) { c.all += 1; c[e.status] = (c[e.status] || 0) + 1; }
  return c;
}

function renderDeFilters(rep) {
  const st = state.de;
  const n = deCounts(rep);
  $("#deType").innerHTML = DE_TYPES.map(([k, label]) =>
    `<button type="button" role="tab" class="de-tab ${k} ${st.type === k ? "on" : ""}" data-k="${k}" aria-selected="${st.type === k}">${label}<b>${n[k] || 0}</b></button>`).join("");
  $("#deClass").innerHTML = ["cup", "can", "bottle"].map((c) =>
    `<button type="button" class="de-chip ${st.cls.has(c) ? "on" : ""}" data-k="${c}">${c}</button>`).join("");
  $("#deDist").innerHTML = DE_DIST.map((d) =>
    `<button type="button" class="de-chip d-${d} ${st.dist.has(d) ? "on" : ""}" data-k="${d}">${d.replace("_", " ")}</button>`).join("");
  $("#deSearch").value = st.q;
  $("#deSort").value = st.sort;
  const rerender = () => { renderDeFilters(rep); renderDeCards(rep); };
  $$("#deType .de-tab").forEach((b) => b.onclick = () => { st.type = b.dataset.k; rerender(); });
  const toggle = (set) => (b) => b.onclick = () => { set.has(b.dataset.k) ? set.delete(b.dataset.k) : set.add(b.dataset.k); rerender(); };
  $$("#deClass .de-chip").forEach(toggle(st.cls));
  $$("#deDist .de-chip").forEach(toggle(st.dist));
  $("#deSearch").oninput = (e) => { st.q = e.target.value.trim().toLowerCase(); renderDeCards(rep); };
  $("#deSort").onchange = (e) => { st.sort = e.target.value; renderDeCards(rep); };
}

function deMatch(e) {
  const st = state.de;
  if (st.type !== "all" && e.status !== st.type) return false;
  const cls = e.status === "false_positive" ? e.pred_class : e.gt_class;
  if (st.cls.size && !st.cls.has(cls)) return false;
  if (st.dist.size && !st.dist.has((e.caption || {}).distance)) return false;
  return true;
}

function deRow(src, e) {
  const c = e.caption || {};
  const key = `${src.filename}#${e.id}`;
  const open = state.de.open.has(key);
  const tag = e.status === "false_positive"
    ? (c.is_actually_drink ? `unlabelled ${esc(e.pred_class)}?` : `false alarm · ${esc(e.pred_class)}`)
    : e.status === "wrong_class" ? `${esc(e.gt_class)} → ${esc(e.pred_class)}` : `missed ${esc(e.gt_class)}`;
  const kind = e.status === "false_positive" && c.is_actually_drink ? "gap" : e.status;
  const dist = c.distance ? `<span class="dtag d-${esc(c.distance)}">${esc(String(c.distance).replace("_", " "))}</span>` : "";
  const size = e.frame_pct ? `${e.frame_pct.h}% h` : `${e.size_px?.[0]}×${e.size_px?.[1]}`;
  const detail = open ? `
    <dl class="de-detail">
      ${c.placement_detail ? `<dt>where</dt><dd>${esc(c.placement_detail)}</dd>` : ""}
      ${c.depth_detail ? `<dt>depth</dt><dd>${esc(c.depth_detail)}${c.relative_size ? ` — ${esc(c.relative_size)}` : ""}</dd>` : ""}
      <dt>size</dt><dd>${e.size_px?.[0]}×${e.size_px?.[1]} px${e.frame_pct ? ` · ${e.frame_pct.h}% of frame height` : ""}</dd>
      <dt>detector</dt><dd>${esc(e.reason || "—")}</dd>
      ${c.likely_cause ? `<dt>why</dt><dd>${esc(c.likely_cause)}</dd>` : ""}
      ${c.confusers ? `<dt>looks like</dt><dd>${esc(c.confusers)}</dd>` : ""}
    </dl>` : "";
  return `<li class="de-err ${kind} ${open ? "open" : ""}" data-key="${esc(key)}" tabindex="0">
    <div class="de-err-line">
      <span class="de-num">${e.id}</span>
      <span class="de-tag ${kind}">${tag}</span>${dist}
      <span class="de-txt">${esc(c.object || e.reason || "—")}</span>
      <span class="de-size">${size}</span>
    </div>${detail}
  </li>`;
}

function renderDeCards(rep) {
  const st = state.de;
  let srcs = (rep.sources || [])
    .filter((r) => !st.q || r.filename.toLowerCase().includes(st.q))
    .map((r) => ({ ...r, shown: r.errors.filter(deMatch) }))
    .filter((r) => r.shown.length);
  const recall = (r) => (r.counts.gt ? r.counts.hit / r.counts.gt : 1);
  srcs.sort(st.sort === "name" ? (a, b) => a.filename.localeCompare(b.filename)
    : st.sort === "recall" ? (a, b) => recall(a) - recall(b) || b.shown.length - a.shown.length
    : (a, b) => b.shown.length - a.shown.length || a.filename.localeCompare(b.filename));
  const cards = $("#detErrCards");
  if (!srcs.length) {
    cards.innerHTML = '<p class="muted de-empty">No errors match these filters.</p>';
    return;
  }
  cards.innerHTML = srcs.map((r) => {
    const k = r.counts || {};
    const img = `/api/model-errors/${encodeURIComponent(r.filename)}/overlay`;
    const more = st.open.has(`${r.filename}#*`);
    const rows = more ? r.shown : r.shown.slice(0, 4);
    const rec = Math.round(recall(r) * 100);
    return `<article class="de-card">
      <button type="button" class="de-thumb" data-img="${sized(img, 0, rep.version)}" data-cap="${esc(r.filename)}" title="Open full size">
        <img src="${sized(img, 640, rep.version)}" loading="lazy" decoding="async" alt="" />
        <span class="de-recall-badge ${rec < 50 ? "hi" : rec < 80 ? "mid" : "lo"}">${k.hit}/${k.gt} found</span>
      </button>
      <div class="de-card-body">
        <header class="de-card-head">
          <span class="de-name">${esc(r.filename.replace(/\.(png|jpe?g)$/i, ""))}</span>
          <span class="de-counts">
            ${k.missed ? `<i class="missed">${k.missed}</i>` : ""}${k.wrong_class ? `<i class="wrong_class">${k.wrong_class}</i>` : ""}${k.false_alarm_objects ? `<i class="false_positive">${k.false_alarm_objects}</i>` : ""}
          </span>
        </header>
        ${r.caption_error ? `<div class="muted de-warn">Gemini: ${esc(r.caption_error)}</div>` : ""}
        <ol class="de-list">${rows.map((e) => deRow(r, e)).join("")}</ol>
        ${r.shown.length > 4 ? `<button type="button" class="de-more" data-src="${esc(r.filename)}">${more ? "show less" : `+${r.shown.length - 4} more`}</button>` : ""}
      </div>
    </article>`;
  }).join("");
  cards.querySelectorAll(".de-err").forEach((li) => {
    const t = () => { const k = li.dataset.key; st.open.has(k) ? st.open.delete(k) : st.open.add(k); renderDeCards(rep); };
    li.onclick = t;
    li.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); t(); } };
  });
  cards.querySelectorAll(".de-more").forEach((b) => b.onclick = () => {
    const k = `${b.dataset.src}#*`; st.open.has(k) ? st.open.delete(k) : st.open.add(k); renderDeCards(rep);
  });
  cards.querySelectorAll(".de-thumb").forEach((b) => b.onclick = () => {
    $("#deViewerImg").src = b.dataset.img;
    $("#deViewerCap").textContent = `${b.dataset.cap} — red = missed · magenta = wrong class · yellow = false alarm · green = found`;
    $("#deViewer").classList.remove("hidden");
  });
}

function renderGalleryCards(sel, src, group) {
  const filtered = Object.values(src || {}).filter((g) => {
    if (group === "overlap") return g.group === "overlap";
    return g.group === group;
  });
  const root = $(sel);
  if (!filtered.length) {
    root.innerHTML = '<p class="muted">No galleries yet.</p>';
    return;
  }
  root.innerHTML = filtered.map((g) => `
    <article class="gcard ${state.selectedRegime === g.key ? "active" : ""}" data-key="${g.key}" style="border-top:3px solid ${g.color || "#888"}">
      <div class="label">${g.group || group} · ${g.pct ?? 0}%</div>
      <div class="title">${g.title || g.key}</div>
      <div class="value" style="font-family:Syne,sans-serif;font-size:1.15rem">${g.count ?? 0}</div>
      <div class="why">${g.why || g.short || ""}</div>
    </article>`).join("");
  root.querySelectorAll(".gcard").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedRegime = el.dataset.key;
      loadAnalysisCardsActive();
      loadSamples();
    });
  });
}

function loadAnalysisCardsActive() {
  $$(".gcard").forEach((el) => el.classList.toggle("active", el.dataset.key === state.selectedRegime));
}

async function loadSamples() {
  const q = $("#filterQ").value.trim();
  const params = new URLSearchParams({ limit: "120" });
  if (state.selectedRegime) params.set("regime", state.selectedRegime);
  if (q) params.set("q", q);
  const data = await api(`/api/records?${params}`);
  const root = $("#sampleGrid");
  const items = data.items || [];
  if (!items.length) {
    root.innerHTML = '<p class="muted">No samples.</p>';
    return;
  }
  root.innerHTML = items.map((r) => `
    <article class="thumb" data-file="${r.filename}">
      <img src="/api/image/${encodeURIComponent(r.filename)}/annotated?w=480" loading="lazy" decoding="async" alt="" />
      <div class="cap">${r.filename}<br/>n=${r.object_count} · ${fmt(r.brightness, 2)} · ${(r.regimes || []).slice(0, 2).join(",")}</div>
    </article>`).join("");
  root.querySelectorAll(".thumb").forEach((el) => {
    el.addEventListener("click", () => {
      const src = $("#lightboxSrc");
      const gen = $("#lightboxImg");
      const url = `/api/image/${encodeURIComponent(el.dataset.file)}/annotated`;
      if (src) src.src = url;
      if (gen) gen.src = url;
      $("#lightboxMeta").textContent = el.dataset.file;
      $("#lightbox").classList.remove("hidden");
    });
  });
}

function fillSourceSelect(sources) {
  state.sources = sources || [];
  const filter = (($("#sourceFilter") && $("#sourceFilter").value) || "").trim().toLowerCase();
  let list = state.sources.filter((s) => String(s.source_image || "").includes("block_"));
  if (filter) {
    list = list.filter((s) => String(s.source_image || "").toLowerCase().includes(filter));
  }
  const sel = $("#sourceImage");
  if (!sel) return;
  const prev = sel.value || localStorage.getItem("bs_source_image") || "";
  sel.innerHTML = list.length
    ? list.map((s) => {
      const mark = s.resolved ? "" : " (missing)";
      const full = `${s.source_image}${mark}`;
      const short = full.length > 28 ? `${full.slice(0, 12)}…${full.slice(-12)}` : full;
      return `<option value="${s.source_image}" title="${full}">${short}</option>`;
    }).join("")
    : '<option value="">No block_* sources — rebuild package</option>';
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  else if (list[0]) sel.value = list[0].source_image;
  const apply = () => {
    localStorage.setItem("bs_source_image", sel.value || "");
    const cur = list.find((s) => s.source_image === sel.value);
    const hint = $("#sourceHint");
    if (hint) {
      hint.textContent = cur
        ? `${cur.source_image} · regimes ${cur.regimes || "—"} · ×${cur.target_outputs} · ${cur.detector_errors || 0} detector errors targeted`
        : "Pick a block_* source by name";
    }
    $$("#sourceCards .source-card").forEach((c) => {
      c.classList.toggle("selected", c.dataset.source === sel.value);
    });
    renderWrongPanel(sel.value);
  };
  sel.onchange = apply;
  apply();

  const cards = $("#sourceCards");
  if (cards) {
    cards.innerHTML = list.slice(0, 48).map((s) => `
      <button type="button" class="source-card ${s.source_image === sel.value ? "selected" : ""}" data-source="${s.source_image}">
        <img src="/api/dataset/image/${encodeURIComponent(s.source_image)}?w=320" loading="lazy" decoding="async" alt="" />
        <div class="cap">${s.source_image}${s.detector_errors ? ` <span class="err-badge" title="best detector misses / misclassifies / falsely fires ${s.detector_errors}× here — prompts target them">${s.detector_errors} det. errors</span>` : ""}<br/><span class="muted">${(s.regimes || "").split("|").slice(0, 2).join(" · ") || "—"}</span></div>
      </button>`).join("") || '<p class="muted">No block_* sources in package.</p>';
    cards.querySelectorAll(".source-card").forEach((btn) => {
      btn.addEventListener("click", () => {
        sel.value = btn.dataset.source;
        apply();
      });
    });
  }
}

async function deleteSelectedPairs(runName, filenames) {
  if (!filenames.length) return;
  const status = $("#genStatus");
  if (!window.confirm(`Delete ${filenames.length} image${filenames.length === 1 ? "" : "s"} from ${runName}?\n`
    + "They are removed from the run (moved to its _deleted/ folder — not sent, not uploaded, not trained on).")) return;
  try {
    const res = await api("/api/gen/delete-images", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run: runName, filenames }),
    });
    const sel = selectedForRun(runName);
    filenames.forEach((f) => sel.delete(f));
    if (status) {
      status.className = "status ok";
      status.textContent = `Deleted ${res.deleted} image${res.deleted === 1 ? "" : "s"} from ${runName} · ${res.remaining} left`;
    }
    await refreshGenerate();
  } catch (err) {
    if (status) {
      status.className = "status error";
      status.textContent = err.message || String(err);
    }
  }
}

function selectedForRun(runName) {
  if (!state.selectedByRun[runName]) state.selectedByRun[runName] = new Set();
  return state.selectedByRun[runName];
}

function renderPairGrid(runName, items) {
  const root = $("#genGrid");
  if (!items?.length) {
    root.innerHTML = '<p class="muted">No generated images in this run yet.</p>';
    return;
  }
  const selected = selectedForRun(runName);
  const nSel = selected.size;
  root.innerHTML = `
    <div class="pair-toolbar">
      <span class="muted">${nSel ? `${nSel} selected → Send / Delete` : "none selected → Send ships all"}</span>
      <div class="pair-view" role="tablist" aria-label="Generated image version">
        <button type="button" class="pv ${state.pairView !== "raw" ? "on" : ""}" data-v="final"
          title="What is trained on: post-processed + stretched to 1024²">Final</button>
        <button type="button" class="pv ${state.pairView === "raw" ? "on" : ""}" data-v="raw"
          title="The generator's own output, before our post-processing and the square stretch">Raw (before post-processing)</button>
      </div>
      <div class="pair-toolbar-actions">
        ${items.some((it) => it.near_copy) ? `<button type="button" class="btn btn-sm ghost" id="selNearPairs"
          title="Select every image flagged as a near copy of its source">Select near copies (${items.filter((it) => it.near_copy).length})</button>` : ""}
        <button type="button" class="btn btn-sm ghost" id="selAllPairs">Select all</button>
        <button type="button" class="btn btn-sm ghost" id="selClearPairs">Clear</button>
        <button type="button" class="btn btn-sm danger" id="delPairs" ${nSel ? "" : "disabled"}
          title="Remove the selected images from this run (moved to the run's _deleted/ folder)">Delete selected${nSel ? ` (${nSel})` : ""}</button>
      </div>
    </div>
    <div class="pair-grid-inner">
    ${items.map((it) => {
    const fn = it.filename || "";
    const checked = selected.has(fn) ? "checked" : "";
    const active = selected.has(fn) ? "picked" : "";
    const srcUrl = it.source_url
      || `/api/gen/source/${encodeURIComponent(it.filename)}?run=${encodeURIComponent(runName)}`;
    const genUrl = state.pairView === "raw"
      ? `/api/gen/raw/${encodeURIComponent(it.filename)}?run=${encodeURIComponent(runName)}`
      : (it.url || `/api/gen/image/${encodeURIComponent(it.filename)}?run=${encodeURIComponent(runName)}`);
    return `<article class="pair ${active}" data-gen="${genUrl}" data-src="${srcUrl}" data-fn="${fn}">
      <label class="pair-pick" title="Select (for Send or Delete)">
        <input type="checkbox" class="pair-check" data-fn="${fn}" ${checked} />
        <span>select</span>
      </label>
      <div class="pair-labels"><span>source</span><span>${state.pairView === "raw" ? "generated · raw" : "generated"}</span></div>
      <div class="pair-imgs">
        <img src="${sized(srcUrl, 480)}" loading="lazy" decoding="async" alt="source" />
        <img src="${sized(genUrl, 480)}" loading="lazy" decoding="async" alt="generated"
          onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'raw-missing',textContent:'no raw output'}))" />
      </div>
      ${it.near_copy ? `<div class="cam-drift near" title="layout similarity to the source ${it.layout_similarity} (> 0.6) — kept, but close to a copy of a validation frame">≈ near copy · ${it.layout_similarity}</div>` : ""}
      ${(it.camera_drift || []).length ? `<div class="cam-drift" title="${esc(it.camera_drift.join(" · "))}">⚠ camera drift · ${esc(it.camera_drift.map((r) => r.split(":")[0]).join(" / "))}</div>` : ""}
      <div class="cap">${it.source_image || "source"} → ${it.filename}</div>
    </article>`;
  }).join("")}
    </div>`;

  root.querySelectorAll(".pair-view .pv").forEach((b) => b.addEventListener("click", () => {
    state.pairView = b.dataset.v;
    try { localStorage.setItem("bs_pair_view", state.pairView); } catch { /* private mode */ }
    renderPairGrid(runName, items);
  }));
  root.querySelectorAll(".pair-check").forEach((box) => {
    box.addEventListener("click", (e) => e.stopPropagation());
    box.addEventListener("change", () => {
      const fn = box.dataset.fn;
      if (!fn) return;
      if (box.checked) selected.add(fn);
      else selected.delete(fn);
      renderPairGrid(runName, items);
    });
  });
  root.querySelector("#selAllPairs")?.addEventListener("click", (e) => {
    e.preventDefault();
    items.forEach((it) => { if (it.filename) selected.add(it.filename); });
    renderPairGrid(runName, items);
  });
  root.querySelector("#selNearPairs")?.addEventListener("click", (e) => {
    e.preventDefault();
    items.forEach((it) => { if (it.near_copy && it.filename) selected.add(it.filename); });
    renderPairGrid(runName, items);
  });
  root.querySelector("#delPairs")?.addEventListener("click", (e) => {
    e.preventDefault();
    deleteSelectedPairs(runName, [...selected]);
  });
  root.querySelector("#selClearPairs")?.addEventListener("click", (e) => {
    e.preventDefault();
    selected.clear();
    renderPairGrid(runName, items);
  });
  root.querySelectorAll(".pair").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (e.target.closest(".pair-pick")) return;
      const src = $("#lightboxSrc");
      const gen = $("#lightboxImg");
      if (src) src.src = el.dataset.src || "";
      if (gen) gen.src = el.dataset.gen || "";
      $("#lightboxMeta").textContent = el.querySelector(".cap")?.textContent || "";
      $("#lightbox").classList.remove("hidden");
    });
  });
}

async function sendRunRemote(runName, ev) {
  if (ev) {
    ev.preventDefault();
    ev.stopPropagation();
  }
  if (!runName || state.sendingRun) return;
  const status = $("#genStatus");
  const selected = [...(state.selectedByRun[runName] || [])];
  const body = { run: runName };
  if (selected.length) body.filenames = selected;
  state.sendingRun = runName;
  refreshRunListButtons();
  if (status) {
    status.className = "status";
    status.textContent = selected.length
      ? `Sending ${selected.length} selected → labeler…`
      : `Sending all images from ${runName} → labeler…`;
  }
  try {
    await api("/api/gen/send-remote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await pollSend(selected.length ? " (selected only)" : " (all)");
  } catch (e) {
    if (status) {
      status.className = "status error";
      status.textContent = e.message || String(e);
    } else {
      alert(e.message || String(e));
    }
    state.sendingRun = null;
    refreshRunListButtons();
  }
}

// Send runs server-side in the background; poll its progress (also resumes after a page reload).
async function pollSend(suffix = "") {
  const status = $("#genStatus");
  for (;;) {
    let job;
    try {
      job = await api("/api/gen/send-job");
    } catch {
      await new Promise((r) => setTimeout(r, 2000));
      continue;
    }
    if (job.status === "running") {
      state.sendingRun = job.run;
      state.sendProgress = job.total ? `${job.done}/${job.total}` : "";
      refreshRunListButtons();
      if (status) {
        status.className = "status";
        status.textContent = `${job.run}: ${job.message}`;
      }
      await new Promise((r) => setTimeout(r, 1000));
      continue;
    }
    if (state.sendingRun && status) {
      const res = job.result || {};
      status.className = job.status === "done" ? "status ok" : "status error";
      status.textContent = job.status === "done"
        ? `Sent ${res.images} imgs @ ${res.size || 1024}×${res.size || 1024} stretch → ${res.host}:${res.dest}/images${suffix}`
        : (job.error || job.message || "Send failed");
    }
    state.sendingRun = null;
    state.sendProgress = "";
    refreshRunListButtons();
    return job;
  }
}

function refreshRunListButtons() {
  $$(".run-send").forEach((btn) => {
    const run = btn.dataset.run;
    const busy = state.sendingRun === run;
    btn.disabled = Boolean(state.sendingRun);
    const nSel = (state.selectedByRun[run] || new Set()).size;
    btn.textContent = busy ? `Sending ${state.sendProgress || ""}…` : `Send${nSel ? ` ${nSel}` : ""} →`;
  });
}

async function refreshGenerate() {
  const [dist, pkg, runs] = await Promise.all([
    api("/api/gen/distribution"),
    api("/api/gen/package?block_only=true"),
    api("/api/gen/runs"),
  ]);
  const rows = [...(dist.galleries || []), ...(dist.overlaps || [])];
  $("#quotaTable").innerHTML = rows.length
    ? rows.map((r) => `
      <div class="qrow">
        <div>${r.title || r.key}</div>
        <div>${r.eval_count}</div>
        <div>${r.eval_pct}%</div>
        <div class="bar"><i style="width:${Math.min(100, r.eval_pct)}%"></i></div>
      </div>`).join("")
    : '<p class="muted">Analyze first to see quotas.</p>';

  $("#pkgMeta").textContent = pkg.ready
    ? `Package ready\nblock_sources=${(pkg.sources || []).length} · per_source=${pkg.per_source} · total≈${pkg.total_targets}\nest. cost ≈ $${pkg.estimated_cost_usd} · model ${pkg.model_default}`
    : "No package yet — run Analyze first.";

  fillSourceSelect(pkg.sources || []);
  updateOpenaiOpts();

  const runList = runs.runs || [];
  $("#runCount").textContent = runList.length ? String(runList.length) : "";
  $("#runList").innerHTML = runList.map(runCard).join("") || '<p class="muted">No runs yet.</p>';
  $$(".run-item").forEach((el) => el.addEventListener("click", () => {
    state.activeRun = el.dataset.run;
    showRun(el.dataset.run);
    $$(".run-item").forEach((x) => x.classList.toggle("active", x.dataset.run === state.activeRun));
  }));
  $$(".run-send").forEach((btn) => {
    btn.addEventListener("click", (e) => sendRunRemote(btn.dataset.run, e));
  });
  $$(".run-collect").forEach((btn) => btn.addEventListener("click", (e) => runAction("collect", btn.dataset.run, e)));
  $$(".run-cancel").forEach((btn) => btn.addEventListener("click", (e) => runAction("cancel", btn.dataset.run, e)));
  $$(".run-autolabel").forEach((btn) => btn.addEventListener("click", (e) => runAction("autolabel", btn.dataset.run, e)));

  if (state.activeRun) {
    await showRun(state.activeRun);
  } else if ((runs.runs || [])[0]) {
    state.activeRun = runs.runs[0].name;
    await showRun(state.activeRun);
    $$(".run-item").forEach((x) => x.classList.toggle("active", x.dataset.run === state.activeRun));
  }
}

// "test_20261005_093122_179e27" (UTC stamp) → { when: "10-05 11:31" local, id: "179e27" }
function parseRunName(name) {
  const m = /^([a-z]+)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_?([0-9a-f]*)/i.exec(name || "");
  if (!m) return { when: "", id: name };
  const d = new Date(Date.UTC(+m[2], +m[3] - 1, +m[4], +m[5], +m[6], +m[7]));
  const p2 = (n) => String(n).padStart(2, "0");
  return { when: `${p2(d.getMonth() + 1)}-${p2(d.getDate())} ${p2(d.getHours())}:${p2(d.getMinutes())}`, id: m[8] || "" };
}

function runCard(r) {
  const { when, id } = parseRunName(r.name);
  const provider = r.provider || (String(r.model || "").startsWith("gemini") ? "gemini" : "openai");
  const nSel = (state.selectedByRun[r.name] || new Set()).size;
  const busy = state.sendingRun === r.name;
  const sendDisabled = r.count <= 0 || Boolean(state.sendingRun);
  const model = String(r.model || "").replace(/^gpt-image-/, "gpt-image ").replace(/^gemini-/, "");
  const labels = !r.count ? ""
    : r.autolabelled
      ? '<span class="rchip ok" title="labels replaced by the mode\'s detector">✓ detector labels</span>'
      : '<span class="rchip warn" title="still the source image\'s labels — press Re-label">⚠ source labels</span>';
  const pending = r.pending ? `<span class="rchip pend">● batch pending${r.chunks > 1 ? ` · ${r.chunks} chunks` : ""}</span>` : "";
  const sel = nSel ? `<span class="rchip sel">${nSel} selected</span>` : "";
  const nearc = r.near_copies ? `<span class="rchip warn" title="kept images whose layout is very close to their source">≈ ${r.near_copies} near cop${r.near_copies === 1 ? "y" : "ies"}</span>` : "";
  const drift = r.camera_drift ? `<span class="rchip warn" title="images whose camera height / tilt / distance to people differs too much from their source">⚠ ${r.camera_drift} camera drift</span>` : "";
  return `
    <article class="run-item ${state.activeRun === r.name ? "active" : ""}" data-run="${r.name}" title="${r.name}">
      <div class="run-top">
        <span class="rprov ${provider}">${provider === "gemini" ? "Gemini" : "OpenAI"}</span>
        <span class="rmode ${r.mode || ""}">${r.mode || "?"}</span>
        <span class="rwhen">${when}<em>#${id}</em></span>
      </div>
      <div class="run-meta">
        <span class="rmodel">${model || "—"}</span>
        <span class="rcount"><strong>${r.count}</strong> img${r.count === 1 ? "" : "s"}</span>
      </div>
      ${labels || pending || sel || drift || nearc ? `<div class="run-chips">${pending}${labels}${drift}${nearc}${sel}</div>` : ""}
      <div class="run-actions">
        ${r.pending ? `
          <button type="button" class="btn btn-sm run-collect" data-run="${r.name}"
            title="Finish the in-flight batch and resume any failed/missing chunks into this run">Collect</button>
          ${provider === "openai" ? `<button type="button" class="btn btn-sm ghost run-cancel" data-run="${r.name}"
            title="Cancel the active OpenAI batch and skip remaining chunks">Cancel</button>` : ""}` : `
          <button type="button" class="btn btn-sm ghost run-autolabel" data-run="${r.name}" ${r.count <= 0 ? "disabled" : ""}
            title="Replace this run's labels with the detector of its mode">Re-label</button>`}
        <button type="button" class="btn btn-sm run-send" data-run="${r.name}" ${sendDisabled ? "disabled" : ""}
          title="Send ${nSel ? "the selected" : "all"} images to Score Labeler Own local/images/">
          ${busy ? `Sending ${state.sendProgress || ""}…` : `Send${nSel ? ` ${nSel}` : ""} →`}</button>
      </div>
    </article>`;
}

async function runAction(action, run, e) {
  e?.stopPropagation();
  const status = $("#genStatus");
  if (action === "cancel" && !window.confirm(`Cancel OpenAI batch for ${run}?`)) return;
  try {
    const data = await api(`/api/gen/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run, openai_api_key: isGeminiModel() ? null : (apiKey() || null),
        gemini_api_key: isGeminiModel() ? (apiKey() || null) : null }),
    });
    status.className = "status";
    if (action === "collect" || action === "autolabel") {
      status.textContent = action === "collect" ? `Collecting ${run}…` : `Labelling ${run} with best detector…`;
      pollGen();
    } else {
      status.textContent = `Batch ${run}: ${data.status}`;
    }
  } catch (err) {
    status.className = "status error";
    status.textContent = err.message || String(err);
  }
}

async function showRun(name) {
  const data = await api(`/api/gen/images?run=${encodeURIComponent(name)}`);
  renderPairGrid(name, data.images || []);
}

async function preparePackage() {
  const per = Number($("#perSource").value || 10);
  const data = await api("/api/gen/prepare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ per_source: per }),
  });
  $("#pkgMeta").textContent = `Rebuilt package\nsources=${data.n_sources} · total≈${data.total_targets} · ~$${data.estimated_cost_usd}`;
  await refreshGenerate();
}

async function startGen(mode) {
  const status = $("#genStatus");
  status.className = "status";
  const sourceImage = ($("#sourceImage")?.value || "").trim();
  if (mode === "test" && !sourceImage) {
    status.className = "status error";
    status.textContent = "Pick a block_* source image first";
    return;
  }
  const model = $("#model").value;
  const gem = isGeminiModel();
  const provider = gem ? "Gemini" : "OpenAI";
  status.textContent = `Starting ${mode} (${provider} batch · ${model})…`;
  if (mode === "final") {
    const est = $("#oaCost")?.textContent || "";
    const nSrc = (state.sources || []).length;
    const perSrc = Number($("#perSource").value || 4);
    const totalImgs = nSrc * perSrc;
    let chunkLine = nSrc ? `${nSrc} sources · ≈${totalImgs} images` : "all package sources";
    if (nSrc) {
      const nChunks = Math.max(1, Math.ceil(nSrc / 13));
      chunkLine = `${nSrc} sources → ${nChunks} sequential batch${nChunks === 1 ? "" : "es"} of ≤13 sources (≈${totalImgs} images)`;
    }
    if (gem) chunkLine = nSrc ? `${nSrc} sources · ≈${totalImgs} images (16:9)` : "all package sources";
    if (!window.confirm(
      `Submit FINAL ${provider} batch with ${model}?\n${chunkLine}\n${gem ? "" : est}\nResults usually arrive in minutes–hours.`
    )) {
      status.textContent = "Cancelled";
      return;
    }
  }
  try {
    const key = apiKey() || null;
    const body = {
      openai_api_key: gem ? null : key,
      gemini_api_key: gem ? key : null,
      openrouter_api_key: null,
      quality: $("#oaQuality").value,
      input_fidelity: $("#oaFidelity").value,
      cctv_degrade: false,
      mode,
      model,
      per_source: Number($("#perSource").value || 4),
      source_image: mode === "test" ? sourceImage : null,
      source_index: null,
    };
    await api("/api/gen/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    pollGen();
  } catch (e) {
    status.className = "status error";
    status.textContent = e.message;
  }
}

function pollGen() {
  clearInterval(state.genPoll);
  state.genPoll = setInterval(async () => {
    const job = await api("/api/gen/job");
    const status = $("#genStatus");
    status.textContent = `${job.status}: ${job.message || ""} (${job.done || 0}/${job.total || 0})`;
    if (job.status === "done") {
      clearInterval(state.genPoll);
      status.className = "status ok";
      if (job.run_dir) {
        state.activeRun = job.run_dir.split(/[/\\]/).pop();
        renderPairGrid(state.activeRun, job.images || []);
      }
      const hf = job.hf_upload;
      if (hf && hf.ok && hf.url) {
        status.textContent = `${job.message || "done"} `;
        const a = document.createElement("a");
        a.href = hf.url;
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = "open HF";
        status.appendChild(a);
      }
      refreshGenerate();
    }
    if (job.status === "error") {
      clearInterval(state.genPoll);
      status.className = "status error";
      status.textContent = job.error || job.message;
    }
  }, 1000);
}

async function uploadToHuggingFace() {
  const status = $("#genStatus");
  const btn = $("#hfUploadBtn");
  status.className = "status";
  status.textContent = "Uploading to Hugging Face…";
  if (btn) btn.disabled = true;
  try {
    const body = state.activeRun ? { run: state.activeRun } : {};
    const data = await api("/api/gen/upload-hf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    status.className = "status ok";
    status.textContent = `HF ok: ${data.repo_id}/${data.path_in_repo} (${data.n_images} imgs)`;
    if (data.url) {
      const a = document.createElement("a");
      a.href = data.url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = " open";
      status.appendChild(a);
    }
  } catch (e) {
    status.className = "status error";
    status.textContent = e.message || String(e);
  } finally {
    loadHealth().catch(() => {});
  }
}

async function clearStudioData() {
  const status = $("#genStatus");
  let preview;
  try {
    preview = await api("/api/studio/clear-preview");
  } catch (e) {
    alert(e.message || String(e));
    return;
  }
  if (preview.gen_running || preview.analyze_running) {
    alert("A job is still running. Wait for it to finish before clearing.");
    return;
  }
  const lines = [
    "This will permanently delete:",
    `• beverage_gt: ${preview.beverage_gt_images} images + ${preview.beverage_gt_labels} labels`,
    `• generate history: ${preview.history_runs} runs`,
    `• analysis cache: ${preview.analysis_files} files`,
    "",
    'Type CLEAR below to confirm.',
  ];
  const typed = window.prompt(lines.join("\n"), "");
  if (typed == null) return;
  if (typed.trim().toUpperCase() !== "CLEAR") {
    alert("Cancelled — confirmation text did not match CLEAR.");
    return;
  }
  if (status) {
    status.className = "status";
    status.textContent = "Clearing beverage_gt + history…";
  }
  try {
    const data = await api("/api/studio/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        beverage_gt: true,
        history: true,
        analysis: true,
        confirm: "CLEAR",
      }),
    });
    state.activeRun = null;
    state.sources = [];
    $("#genGrid").innerHTML = "";
    $("#runList").innerHTML = "";
    $("#quotaTable").innerHTML = "";
    $("#sourceCards").innerHTML = "";
    $("#pkgMeta").textContent = "No package yet — run Analyze first.";
    if (status) {
      status.className = "status ok";
      const r = data.removed || {};
      status.textContent =
        `Cleared · gt ${r.beverage_gt_images || 0} imgs · ${r.history_runs || 0} runs · analysis wiped`;
    }
    await loadHealth();
    await refreshGenerate();
  } catch (e) {
    if (status) {
      status.className = "status error";
      status.textContent = e.message || String(e);
    } else {
      alert(e.message || String(e));
    }
  }
}

function boot() {
  try { state.pairView = localStorage.getItem("bs_pair_view") || "final"; } catch { state.pairView = "final"; }
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("#analyzeBtn").addEventListener("click", startAnalyze);
  $$(".mode-opt").forEach((b) => b.addEventListener("click", () => setGenMode(b.dataset.mode, true)));
  $("#deViewerClose")?.addEventListener("click", () => $("#deViewer").classList.add("hidden"));
  $("#deViewer")?.addEventListener("click", (e) => { if (e.target.id === "deViewer") $("#deViewer").classList.add("hidden"); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#deViewer")?.classList.add("hidden"); });
  $("#ckptUploadBtn")?.addEventListener("click", () => $("#ckptFile").click());
  $("#ckptFile")?.addEventListener("change", (e) => {
    uploadCheckpoint(e.target.files[0]);
    e.target.value = "";
  });
  $("#ckptResetBtn")?.addEventListener("click", resetCheckpoint);
  for (const box of $$("[data-switch]")) box.addEventListener("change", () => toggleSwitch(box));
  $(".mode-seg")?.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    e.preventDefault();
    const next = ($("#genMode").value === "openai") ? "gemini" : "openai";
    setGenMode(next, true);
    $(`.mode-opt[data-mode="${next}"]`)?.focus();
  });
  $("#checkKeyBtn")?.addEventListener("click", checkApiKey);
  ["#oaQuality", "#perSource"].forEach((id) => $(id)?.addEventListener("change", updateOpenaiOpts));
  $("#logoutBtn")?.addEventListener("click", () => logout());
  $("#filterQ").addEventListener("input", () => loadSamples().catch(console.error));
  $("#prepareBtn").addEventListener("click", () => preparePackage().catch((e) => alert(e.message)));
  $("#testBtn").addEventListener("click", () => startGen("test"));
  $("#finalBtn").addEventListener("click", () => startGen("final"));
  $("#hfUploadBtn")?.addEventListener("click", () => uploadToHuggingFace());
  $("#studioClearBtn")?.addEventListener("click", () => clearStudioData());
  $("#lightboxClose").addEventListener("click", () => $("#lightbox").classList.add("hidden"));
  $("#evalClearBtn")?.addEventListener("click", () => clearEvalSelection());
  $("#evalOkBtn")?.addEventListener("click", () => confirmUploadToBeverageGt());
  wireEvalDropzone();
  $("#sourceFilter")?.addEventListener("input", () => fillSourceSelect(state.sources));
  loadHealth()
    .then(loadAnalysis)
    .catch(() => {})
    .then(async () => {
      const job = await api("/api/job");
      if (job.status === "running") {
        $("#jobBar").classList.remove("hidden");
        $("#analyzeBtn").disabled = true;
        pollJob();
      }
      // Reattach to a long OpenAI batch after page reload.
      const gen = await api("/api/gen/job");
      if (gen.status === "running") pollGen();
      // ...and to a checkpoint being tuned.
      const ck = await api("/api/detector/job");
      if (ck.status === "running") pollCkpt();
      // ...and to a background Send.
      const send = await api("/api/gen/send-job");
      if (send.status === "running") {
        state.sendingRun = send.run;
        pollSend();
      }
    });
}

boot();
