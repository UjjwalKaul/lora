/**
 * Local Video Studio - Frontend State Machine & Controller
 */

const state = {
  currentProject: null,
  projects: [],
  styles: {},
  manifest: null,
  script: null,
  mockMode: false,
  activeJob: null,
  autoScroll: true,
  eventSource: null
};

// -----------------------------------------------------------------------------
// Initialization & Event Binding
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  initEventListeners();
  syncModeToggle();
  initSSE();
  await loadStyles();
  await loadProjects();
  startJobPolling();
});

function syncModeToggle() {
  const mockToggle = document.getElementById("mock-mode-toggle");
  const label = document.getElementById("mock-mode-label");
  if (!mockToggle || !label) return;
  // Intuitive: ON (checked) = GPU Active, OFF (unchecked) = Mock Mode (0 MB VRAM)
  if (mockToggle.checked) {
    state.mockMode = false;
    label.textContent = "GPU Active (~3.8 GB VRAM)";
    label.className = "badge badge-running";
  } else {
    state.mockMode = true;
    label.textContent = "Mock Mode (0 MB VRAM)";
    label.className = "badge badge-mock";
  }
}

function initEventListeners() {
  // Navigation & Project Select
  document.getElementById("project-select").addEventListener("change", (e) => {
    switchProject(e.target.value);
  });
  document.getElementById("style-select").addEventListener("change", async (e) => {
    if (state.currentProject) {
      await applyStyle(e.target.value);
    }
  });

  // Mode Toggle (Mock vs GPU)
  const mockToggle = document.getElementById("mock-mode-toggle");
  mockToggle.addEventListener("change", syncModeToggle);

  // Action Buttons
  document.getElementById("btn-cancel-job").addEventListener("click", cancelActiveJob);
  document.getElementById("btn-add-shot").addEventListener("click", addShot);
  document.getElementById("btn-save-script").addEventListener("click", saveScript);
  document.getElementById("btn-gen-keyframes").addEventListener("click", () => triggerStage("keyframes"));
  document.getElementById("btn-approve-all").addEventListener("click", approveAllShots);
  document.getElementById("btn-render-motion").addEventListener("click", () => triggerStage("motion"));
  document.getElementById("btn-assemble-master").addEventListener("click", () => triggerStage("assemble"));
  document.getElementById("btn-run-all").addEventListener("click", () => triggerStage("run-all"));

  // Terminal Controls
  document.getElementById("auto-scroll-toggle").addEventListener("change", (e) => {
    state.autoScroll = e.target.checked;
  });
  document.getElementById("btn-clear-logs").addEventListener("click", () => {
    document.getElementById("terminal-logs").innerHTML = "";
  });
  document.getElementById("btn-toggle-terminal").addEventListener("click", () => {
    const dock = document.querySelector(".terminal-dock");
    dock.classList.toggle("collapsed");
    const btn = document.getElementById("btn-toggle-terminal");
    btn.textContent = dock.classList.contains("collapsed") ? "Expand" : "Collapse";
  });

  // Tabs
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.getAttribute("data-tab"));
      if (target) target.classList.add("active");
    });
  });

  // Modals
  document.getElementById("btn-new-project").addEventListener("click", () => openModal("modal-new-project"));
  document.getElementById("btn-open-config").addEventListener("click", openConfigModal);
  document.getElementById("btn-submit-new-project").addEventListener("click", submitNewProject);
  document.getElementById("btn-save-config").addEventListener("click", submitConfig);
  document.getElementById("btn-submit-reroll").addEventListener("click", submitReroll);

  document.querySelectorAll(".modal-close, .modal-cancel").forEach(el => {
    el.addEventListener("click", () => closeModal());
  });
  document.querySelectorAll(".modal").forEach(modal => {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
  });
}

// -----------------------------------------------------------------------------
// SSE Log Stream & Terminal Handling
// -----------------------------------------------------------------------------
function initSSE() {
  if (state.eventSource) {
    state.eventSource.close();
  }

  const logsEl = document.getElementById("terminal-logs");
  state.eventSource = new EventSource("/api/jobs/logs");

  state.eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data && data.line) {
        appendTerminalLine(data.line);
        // If job completed or started, refresh state
        if (data.line.includes(">>> JOB") || data.line.includes(">>> STARTING")) {
          loadProjectDetails(state.currentProject);
        }
      }
    } catch (e) {
      // Heartbeat comment or non-JSON
    }
  };

  state.eventSource.onerror = () => {
    // Retry automatically handled by browser EventSource
  };
}

function appendTerminalLine(text) {
  const logsEl = document.getElementById("terminal-logs");
  const lineEl = document.createElement("div");
  lineEl.className = "terminal-line";

  if (text.includes("[ERROR]") || text.includes("Error") || text.includes("TERMINATED with exit code [^0]")) {
    lineEl.classList.add("error");
  } else if (text.includes("[STAGE") || text.includes(">>>")) {
    lineEl.classList.add("stage");
  } else if (text.includes("[INFO]")) {
    lineEl.classList.add("info");
  }

  lineEl.textContent = text;
  logsEl.appendChild(lineEl);

  if (state.autoScroll) {
    logsEl.scrollTop = logsEl.scrollHeight;
  }
}

// -----------------------------------------------------------------------------
// Active Job Polling & Hardware Video Unloading
// -----------------------------------------------------------------------------
function startJobPolling() {
  setInterval(async () => {
    try {
      const res = await fetch("/api/jobs/active");
      if (res.ok) {
        const job = await res.json();
        updateJobStatusUI(job);
      }
    } catch (e) {
      console.warn("Job polling failed:", e);
    }
  }, 2000);
}

function updateJobStatusUI(job) {
  state.activeJob = job;
  const pill = document.getElementById("job-status-pill");
  const cancelBtn = document.getElementById("btn-cancel-job");

  if (job.active) {
    pill.textContent = `Running: ${job.stage.toUpperCase()} (${job.elapsed_seconds ? job.elapsed_seconds.toFixed(0) + 's' : 'active'})`;
    pill.className = "badge badge-running";
    cancelBtn.disabled = false;

    // HARDWARE SAFETY: Unload video players to free Chromium NVDEC VRAM
    unloadVideoPlayers();
  } else {
    pill.textContent = job.status === "failed" ? "Failed" : (job.status === "completed" ? "Completed" : "Idle");
    pill.className = `badge badge-${job.status}`;
    cancelBtn.disabled = true;
  }
}

function unloadVideoPlayers() {
  const masterVideo = document.getElementById("master-video");
  if (masterVideo && !masterVideo.paused) {
    masterVideo.pause();
    masterVideo.removeAttribute("src");
    masterVideo.load();
  }
}

async function cancelActiveJob() {
  if (!confirm("Are you sure you want to abort the running pipeline job?")) return;
  try {
    const res = await fetch("/api/jobs/cancel", { method: "POST" });
    const data = await res.json();
    appendTerminalLine(`>>> ${data.message}`);
    await loadProjectDetails(state.currentProject);
  } catch (e) {
    alert(`Failed to cancel job: ${e.message}`);
  }
}

// -----------------------------------------------------------------------------
// API & Project Management
// -----------------------------------------------------------------------------
async function loadStyles() {
  try {
    const res = await fetch("/api/styles");
    if (!res.ok) throw new Error("Failed to fetch styles");
    state.styles = await res.json();

    const styleSelect = document.getElementById("style-select");
    const modalStyleSelect = document.getElementById("new-project-style");
    styleSelect.innerHTML = "";
    modalStyleSelect.innerHTML = "";

    Object.keys(state.styles).forEach(styleKey => {
      const opt1 = document.createElement("option");
      opt1.value = styleKey;
      opt1.textContent = styleKey;
      styleSelect.appendChild(opt1);

      const opt2 = document.createElement("option");
      opt2.value = styleKey;
      opt2.textContent = styleKey;
      modalStyleSelect.appendChild(opt2);
    });
  } catch (e) {
    console.error("Error loading styles:", e);
  }
}

async function loadProjects() {
  try {
    const res = await fetch("/api/projects");
    if (!res.ok) throw new Error("Failed to fetch projects");
    state.projects = await res.json();

    const select = document.getElementById("project-select");
    select.innerHTML = "";

    if (state.projects.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "No projects found";
      select.appendChild(opt);
      return;
    }

    state.projects.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = `${p.name} (${p.pipeline_stage})`;
      select.appendChild(opt);
    });

    const activeProject = state.projects[0].name;
    select.value = activeProject;
    await switchProject(activeProject);
  } catch (e) {
    console.error("Error loading projects:", e);
  }
}

async function switchProject(projectName) {
  state.currentProject = projectName;
  document.getElementById("project-select").value = projectName;
  await loadProjectDetails(projectName);
}

async function loadProjectDetails(projectName) {
  if (!projectName) return;
  try {
    const res = await fetch(`/api/projects/${projectName}`);
    if (!res.ok) throw new Error("Failed to fetch project details");
    const data = await res.json();
    state.manifest = data.manifest;
    state.script = data.script;

    // Sync Style dropdown
    const styleSelect = document.getElementById("style-select");
    const activeStyle = state.manifest.config?.style_profile || state.script.style_profile;
    if (activeStyle) styleSelect.value = activeStyle;

    renderScriptEditor();
    renderStoryboardGrid();
    renderScreeningRoom();
  } catch (e) {
    console.error(`Error loading details for ${projectName}:`, e);
  }
}

async function submitNewProject() {
  const nameInput = document.getElementById("new-project-name");
  const styleSelect = document.getElementById("new-project-style");
  const name = nameInput.value.trim();
  const style = styleSelect.value;

  if (!name || !/^[a-zA-Z0-9_\-]+$/.test(name)) {
    alert("Project name must contain only letters, numbers, hyphens, and underscores.");
    return;
  }

  try {
    const res = await fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_name: name, style_profile: style })
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || result.message);

    closeModal();
    nameInput.value = "";
    await loadProjects();
    await switchProject(name);
  } catch (e) {
    alert(`Failed to create project: ${e.message}`);
  }
}

async function applyStyle(styleName) {
  if (!state.currentProject) return;
  try {
    const res = await fetch(`/api/projects/${state.currentProject}/style`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ style_profile: styleName })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || err.message);
    }
    await loadProjectDetails(state.currentProject);
  } catch (e) {
    alert(`Failed to apply style: ${e.message}`);
  }
}

// -----------------------------------------------------------------------------
// Script & Shots Editor (Left Panel)
// -----------------------------------------------------------------------------
function renderScriptEditor() {
  if (!state.script) return;

  document.getElementById("global-style-prompt").value = state.script.global_style_prompt || "";
  document.getElementById("global-negative-prompt").value = state.script.negative_prompt || "";
  document.getElementById("target-fps").value = state.script.target_fps || 24;

  const container = document.getElementById("shots-container");
  container.innerHTML = "";
  const shots = state.script.shots || [];
  document.getElementById("shot-count-badge").textContent = `${shots.length} shots`;

  shots.forEach((shot, index) => {
    const card = document.createElement("div");
    card.className = "shot-card";
    card.dataset.shotId = shot.shot_id;

    // Calculate snapped frames (8n + 1)
    const fps = parseInt(document.getElementById("target-fps").value) || 24;
    const rawFrames = Math.round((shot.duration_sec || 2.0) * fps);
    const n = Math.max(1, Math.round((rawFrames - 1) / 8));
    const snappedFrames = (8 * n) + 1;

    card.innerHTML = `
      <div class="shot-card-header">
        <div class="shot-card-title">
          <span class="shot-id-badge">#${shot.shot_id}</span>
          <select class="shot-type-select">
            <option value="wide_establishing" ${shot.shot_type === "wide_establishing" ? "selected" : ""}>Wide Establishing</option>
            <option value="medium" ${shot.shot_type === "medium" ? "selected" : ""}>Medium Shot</option>
            <option value="medium_close_up" ${shot.shot_type === "medium_close_up" ? "selected" : ""}>Medium Close-Up</option>
            <option value="extreme_close_up" ${shot.shot_type === "extreme_close_up" ? "selected" : ""}>Extreme Close-Up</option>
          </select>
          <select class="shot-motion-select">
            <option value="static" ${shot.camera_motion === "static" ? "selected" : ""}>Static</option>
            <option value="slow_dolly_forward" ${shot.camera_motion === "slow_dolly_forward" ? "selected" : ""}>Slow Dolly Forward</option>
            <option value="slow_dolly_back" ${shot.camera_motion === "slow_dolly_back" ? "selected" : ""}>Slow Dolly Back</option>
            <option value="pan_left" ${shot.camera_motion === "pan_left" ? "selected" : ""}>Pan Left</option>
            <option value="pan_right" ${shot.camera_motion === "pan_right" ? "selected" : ""}>Pan Right</option>
            <option value="subtle_handheld" ${shot.camera_motion === "subtle_handheld" ? "selected" : ""}>Subtle Handheld</option>
            <option value="zoom_in" ${shot.camera_motion === "zoom_in" ? "selected" : ""}>Zoom In</option>
            <option value="zoom_out" ${shot.camera_motion === "zoom_out" ? "selected" : ""}>Zoom Out</option>
          </select>
        </div>
        <div class="shot-card-controls">
          <button class="btn btn-xs btn-outline btn-move-up" title="Move Up" ${index === 0 ? "disabled" : ""}>↑</button>
          <button class="btn btn-xs btn-outline btn-move-down" title="Move Down" ${index === shots.length - 1 ? "disabled" : ""}>↓</button>
          <button class="btn btn-xs btn-danger btn-delete-shot" title="Delete Shot">✕</button>
        </div>
      </div>
      <div class="shot-card-body">
        <div class="form-group">
          <label>Keyframe Prompt</label>
          <textarea class="shot-prompt" rows="2">${shot.prompt || ""}</textarea>
        </div>
        <div class="form-group">
          <label>Motion Prompt (LTX-Video)</label>
          <textarea class="shot-motion-prompt" rows="2">${shot.motion_prompt || ""}</textarea>
        </div>
        <div class="shot-timing-row">
          <div class="timing-item">
            <label>Duration:</label>
            <input type="number" class="shot-duration" step="0.5" min="0.5" max="10" value="${shot.duration_sec || 2.0}">s
          </div>
          <span class="frame-badge">${snappedFrames} frames (8n+1)</span>
          <div class="timing-item">
            <label>Seed:</label>
            <input type="number" class="shot-seed" value="${shot.seed || 42}">
          </div>
        </div>
      </div>
    `;

    // Reorder and delete listeners
    card.querySelector(".btn-move-up").addEventListener("click", () => moveShot(index, -1));
    card.querySelector(".btn-move-down").addEventListener("click", () => moveShot(index, 1));
    card.querySelector(".btn-delete-shot").addEventListener("click", () => deleteShot(index));

    // Dynamic frame update on duration change
    const durInput = card.querySelector(".shot-duration");
    const frameBadge = card.querySelector(".frame-badge");
    durInput.addEventListener("input", () => {
      const d = parseFloat(durInput.value) || 2.0;
      const f = Math.round(d * fps);
      const nn = Math.max(1, Math.round((f - 1) / 8));
      frameBadge.textContent = `${(8 * nn) + 1} frames (8n+1)`;
    });

    container.appendChild(card);
  });
}

function addShot() {
  if (!state.script) return;
  const shots = state.script.shots || [];
  const nextIdx = shots.length + 1;
  const newId = `shot_${String(nextIdx).padStart(3, "0")}`;

  shots.push({
    shot_id: newId,
    shot_type: "medium",
    camera_motion: "subtle_handheld",
    duration_sec: 2.0,
    prompt: "new cinematic shot description",
    motion_prompt: "subtle cinematic motion",
    seed: Math.floor(Math.random() * 89999) + 10000
  });

  state.script.shots = shots;
  renderScriptEditor();
}

function moveShot(index, delta) {
  const shots = state.script.shots;
  const target = index + delta;
  if (target < 0 || target >= shots.length) return;
  const temp = shots[index];
  shots[index] = shots[target];
  shots[target] = temp;
  renderScriptEditor();
}

function deleteShot(index) {
  if (confirm("Are you sure you want to delete this shot?")) {
    state.script.shots.splice(index, 1);
    renderScriptEditor();
  }
}

async function saveScript() {
  if (!state.currentProject) return;

  // Gather script inputs
  const scriptData = {
    project_name: state.currentProject,
    version: "1.0.0",
    style_profile: state.manifest?.config?.style_profile || "photoreal_default",
    aspect_ratio: "16:9",
    target_fps: parseInt(document.getElementById("target-fps").value) || 24,
    global_style_prompt: document.getElementById("global-style-prompt").value,
    negative_prompt: document.getElementById("global-negative-prompt").value,
    shots: []
  };

  const cards = document.querySelectorAll(".shot-card");
  cards.forEach(card => {
    scriptData.shots.push({
      shot_id: card.dataset.shotId,
      shot_type: card.querySelector(".shot-type-select").value,
      camera_motion: card.querySelector(".shot-motion-select").value,
      duration_sec: parseFloat(card.querySelector(".shot-duration").value) || 2.0,
      prompt: card.querySelector(".shot-prompt").value,
      motion_prompt: card.querySelector(".shot-motion-prompt").value,
      seed: parseInt(card.querySelector(".shot-seed").value) || 42
    });
  });

  try {
    const res = await fetch(`/api/projects/${state.currentProject}/script`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ script: scriptData })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || err.message);
    }
    appendTerminalLine(">>> Script saved and manifest non-destructively reconciled.");
    await loadProjectDetails(state.currentProject);
  } catch (e) {
    alert(`Failed to save script: ${e.message}`);
  }
}

// -----------------------------------------------------------------------------
// Storyboard Approval Grid (Tab 1)
// -----------------------------------------------------------------------------
function renderStoryboardGrid() {
  const grid = document.getElementById("storyboard-grid");
  grid.innerHTML = "";
  if (!state.manifest) return;

  const shots = state.manifest.shots || [];
  const approvedCount = shots.filter(s => s.status === "approved").length;
  document.getElementById("approval-progress").textContent = `Approved: ${approvedCount} / ${shots.length}`;

  shots.forEach(shot => {
    const card = document.createElement("div");
    card.className = "storyboard-card";
    const statusClass = `badge-${shot.status || "idle"}`;

    const keyframeUrl = shot.keyframe_start_path
      ? `/media/${state.currentProject}/${shot.keyframe_start_path}?t=${Date.now()}`
      : null;

    card.innerHTML = `
      <div class="storyboard-card-header">
        <span class="shot-id-badge">${shot.shot_id.toUpperCase()}</span>
        <span class="badge ${statusClass}">${(shot.status || "pending").replace("_", " ")}</span>
      </div>
      <div class="storyboard-thumb-wrapper">
        ${keyframeUrl
          ? `<img src="${keyframeUrl}" class="storyboard-thumb" alt="Keyframe for ${shot.shot_id}">`
          : `<div class="thumb-placeholder"><span>📷</span><span>Keyframe not generated</span></div>`
        }
      </div>
      <div class="storyboard-card-body">
        <div class="storyboard-prompt-text" title="${shot.prompt}">${shot.prompt}</div>
        <div class="timing-item" style="font-size:11px; color:var(--text-muted);">
          <span>${shot.num_frames} frames</span> • <span>${shot.camera_motion}</span> • <span>Seed: ${shot.seed}</span>
        </div>
      </div>
      <div class="storyboard-card-footer">
        <button class="btn btn-xs btn-outline-success btn-approve">Approve</button>
        <button class="btn btn-xs btn-outline btn-reject">Reject</button>
        <button class="btn btn-xs btn-secondary btn-reroll">Reroll...</button>
      </div>
    `;

    // Click thumbnail for lightbox
    const thumbWrapper = card.querySelector(".storyboard-thumb-wrapper");
    if (keyframeUrl) {
      thumbWrapper.addEventListener("click", () => {
        openLightbox(keyframeUrl, `${shot.shot_id.toUpperCase()} — ${shot.prompt}`);
      });
    }

    // Action buttons
    card.querySelector(".btn-approve").addEventListener("click", () => approveShot(shot.shot_id));
    card.querySelector(".btn-reject").addEventListener("click", () => rejectShot(shot.shot_id));
    card.querySelector(".btn-reroll").addEventListener("click", () => openRerollModal(shot));

    grid.appendChild(card);
  });
}

async function approveShot(shotId) {
  try {
    const res = await fetch(`/api/projects/${state.currentProject}/shots/${shotId}/approve`, { method: "POST" });
    if (!res.ok) throw new Error("Approval failed");
    await loadProjectDetails(state.currentProject);
  } catch (e) {
    alert(e.message);
  }
}

async function rejectShot(shotId) {
  try {
    const res = await fetch(`/api/projects/${state.currentProject}/shots/${shotId}/reject`, { method: "POST" });
    if (!res.ok) throw new Error("Rejection failed");
    await loadProjectDetails(state.currentProject);
  } catch (e) {
    alert(e.message);
  }
}

async function approveAllShots() {
  try {
    const res = await fetch(`/api/projects/${state.currentProject}/approve-all`, { method: "POST" });
    if (!res.ok) throw new Error("Approval failed");
    await loadProjectDetails(state.currentProject);
  } catch (e) {
    alert(e.message);
  }
}

// -----------------------------------------------------------------------------
// Motion & Screening Room (Tab 2)
// -----------------------------------------------------------------------------
function renderScreeningRoom() {
  if (!state.manifest) return;

  const masterVideo = document.getElementById("master-video");
  const downloadBtn = document.getElementById("btn-download-master");
  const statusNote = document.getElementById("master-cut-status");

  const masterUrl = `/media/${state.currentProject}/final/master_cut.mp4?t=${Date.now()}`;

  // Check if master cut exists via summary or fetch
  const projSummary = state.projects.find(p => p.name === state.currentProject);
  if (projSummary && projSummary.has_master_cut) {
    masterVideo.src = masterUrl;
    downloadBtn.href = masterUrl;
    downloadBtn.style.display = "inline-flex";
    statusNote.textContent = "Master cut ready for playback.";
  } else {
    masterVideo.removeAttribute("src");
    downloadBtn.style.display = "none";
    statusNote.textContent = "No master cut assembled yet.";
  }

  // Render per-shot rendered clips
  const clipsGrid = document.getElementById("shot-clips-grid");
  clipsGrid.innerHTML = "";

  (state.manifest.shots || []).forEach(shot => {
    if (shot.clip_output_path) {
      const card = document.createElement("div");
      card.className = "clip-card";
      const clipUrl = `/media/${state.currentProject}/${shot.clip_output_path}?t=${Date.now()}`;

      card.innerHTML = `
        <div class="clip-card-header">
          <span>${shot.shot_id.toUpperCase()}</span>
          <span>${shot.num_frames} frames</span>
        </div>
        <div class="clip-video-wrapper">
          <video src="${clipUrl}" controls preload="metadata" playsinline></video>
        </div>
      `;
      clipsGrid.appendChild(card);
    }
  });
}

// -----------------------------------------------------------------------------
// Pipeline Job Trigger
// -----------------------------------------------------------------------------
async function triggerStage(stageName) {
  if (!state.currentProject) return;
  try {
    const res = await fetch(`/api/projects/${state.currentProject}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stage: stageName,
        mock: state.mockMode
      })
    });
    const result = await res.json();
    if (!res.ok) {
      throw new Error(result.detail || result.message);
    }
    appendTerminalLine(`>>> Requested Stage: ${stageName} (Mock: ${state.mockMode})`);
  } catch (e) {
    alert(`Could not start job: ${e.message}`);
  }
}

// -----------------------------------------------------------------------------
// Modals & Handlers
// -----------------------------------------------------------------------------
function openModal(id) {
  const modal = document.getElementById(id);
  if (modal) modal.classList.add("active");
}

function closeModal() {
  document.querySelectorAll(".modal").forEach(m => m.classList.remove("active"));
}

function openLightbox(url, caption) {
  const modal = document.getElementById("modal-lightbox");
  const img = document.getElementById("lightbox-img");
  const captionEl = document.getElementById("lightbox-caption");
  img.src = url;
  captionEl.textContent = caption;
  modal.classList.add("active");
}

function openRerollModal(shot) {
  document.getElementById("reroll-shot-id").value = shot.shot_id;
  document.getElementById("reroll-prompt").value = shot.prompt;
  document.getElementById("reroll-seed").value = Math.floor(Math.random() * 89999) + 10000;
  openModal("modal-reroll");
}

async function submitReroll() {
  const shotId = document.getElementById("reroll-shot-id").value;
  const prompt = document.getElementById("reroll-prompt").value.trim();
  const seed = parseInt(document.getElementById("reroll-seed").value) || null;

  try {
    const res = await fetch(`/api/projects/${state.currentProject}/reroll`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        shot_id: shotId,
        prompt: prompt,
        seed: seed,
        mock: state.mockMode
      })
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || result.message);

    closeModal();
    appendTerminalLine(`>>> Reroll started for ${shotId}`);
    await loadProjectDetails(state.currentProject);
  } catch (e) {
    alert(`Failed to reroll keyframe: ${e.message}`);
  }
}

function openConfigModal() {
  if (!state.manifest) return;
  const cfg = state.manifest.config || {};
  const postFx = cfg.post_fx || {};

  document.getElementById("cfg-keyframe-backend").value = cfg.keyframe_backend || "sdxl_lightning";
  document.getElementById("cfg-lora-path").value = cfg.lora_path || "";
  document.getElementById("cfg-lora-scale").value = cfg.lora_scale !== undefined ? cfg.lora_scale : 1.0;
  document.getElementById("cfg-motion-freedom").value = cfg.image_cond_noise_scale !== undefined ? cfg.image_cond_noise_scale : 0.15;
  document.getElementById("cfg-stepped-fps").value = postFx.stepped_fps !== undefined && postFx.stepped_fps !== null ? postFx.stepped_fps : "";
  document.getElementById("cfg-chromatic-px").value = postFx.chromatic_aberration_px !== undefined && postFx.chromatic_aberration_px !== null ? postFx.chromatic_aberration_px : "";
  document.getElementById("cfg-lut-path").value = postFx.lut_path || "";

  openModal("modal-config");
}

async function submitConfig() {
  if (!state.currentProject) return;

  const configData = {
    keyframe_backend: document.getElementById("cfg-keyframe-backend").value,
    lora_path: document.getElementById("cfg-lora-path").value.trim() || null,
    lora_scale: parseFloat(document.getElementById("cfg-lora-scale").value) || 1.0,
    image_cond_noise_scale: parseFloat(document.getElementById("cfg-motion-freedom").value) || 0.15,
    stepped_fps: parseInt(document.getElementById("cfg-stepped-fps").value) || null,
    chromatic_aberration_px: parseInt(document.getElementById("cfg-chromatic-px").value) || null,
    lut_path: document.getElementById("cfg-lut-path").value.trim() || null
  };

  try {
    const res = await fetch(`/api/projects/${state.currentProject}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(configData)
    });
    const result = await res.json();
    if (!res.ok) throw new Error(result.detail || result.message);

    closeModal();
    appendTerminalLine(">>> Look-dev & Post-FX settings updated.");
    await loadProjectDetails(state.currentProject);
  } catch (e) {
    alert(`Failed to save config: ${e.message}`);
  }
}
