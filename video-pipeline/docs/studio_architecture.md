# Batch 3: Local Studio Web Dashboard Architecture Specification

**Version:** 1.1.0 (EGM Hardened & Approved)  
**Target Host:** Windows 11 | NVIDIA RTX 3050 Laptop GPU (6 GB physical VRAM, ~4.5 GB effective after DWM/browser) | 16 GB Total RAM (~8.5 GB free usable)  
**Port:** `http://localhost:8000`  
**Location:** `./video-pipeline/studio/`

---

## 1. System Philosophy & Hardware Constraints

1. **Zero-Torch Web Process:**  
   `server.py` is strictly a lightweight I/O process using FastAPI and Uvicorn. It **never** imports `torch`, `diffusers`, `transformers`, or heavy neural modules. Its idle RAM footprint must remain `< 80 MB`, leaving 100% of available host RAM and VRAM free for generation subprocesses.
2. **Subprocess Isolation:**  
   All generation execution stages (`keyframes`, `reroll`, `motion`, `assemble`, `run-all`) run via `run_pipeline.py` as isolated OS subprocesses with `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"`.
3. **Single-GPU Mutex Lock (`JobManager`):**  
   A centralized `AsyncJobManager` enforces a strict 1-job mutex across all execution endpoints (`/run`, `/reroll`). Attempting to start a second job returns HTTP 409 Conflict.
4. **Driver-Settling Cooldown & PID Verification on Cancellation:**  
   `POST /api/jobs/cancel` verifies process identity (`pid` and `create_time`) before issuing `taskkill /F /T /PID <pid>`, and enforces a 750ms driver settling cooldown to allow the NVIDIA WDDM kernel driver to reclaim GPU pages before releasing the mutex.
5. **Non-Destructive Manifest Reconciliation:**  
   Script mutations (`POST /api/projects/{name}/script`) update `script.json` and reconcile `manifest.json` in place—preserving existing approved statuses, keyframe paths, and clip paths for unchanged shots. Script mutations are locked (HTTP 423) while a job is running.
6. **In-Process Approvals:**  
   Shot approvals (`/approve`, `/reject`, `/approve-all`) are lightweight in-process mutations of `manifest.json` (0 ms latency, no subprocess spawning).
7. **Unbuffered Stdio & SSE Ring Buffer:**  
   Subprocesses are launched with `PYTHONUNBUFFERED="1"` and `python -u`. A background consumer drains combined stdout/stderr into an in-memory 500-line circular ring buffer (`collections.deque(maxlen=500)`), broadcasting line updates over Server-Sent Events (SSE) with `\r` carriage return throttling (10 Hz max) to prevent browser DOM choking.
8. **Browser Video Player VRAM Mitigation:**  
   The frontend pauses and unloads HTML5 video players (`video.src = ""`) whenever a GPU job starts, preventing Chromium NVDEC decoding contexts from occupying 400–800 MB VRAM.
9. **No Node / NPM Runtime:**  
   A single semantic HTML5 page with vanilla JavaScript and responsive dark-mode CSS. Zero Node daemons or heavy build tools.

---

## 2. Directory Layout

```
video-pipeline/
├── studio/
│   ├── server.py               # FastAPI backend with AsyncJobManager & SSE broker
│   └── static/
│       ├── index.html          # Single-page studio interface (semantic, dark-mode)
│       ├── app.js              # Vanilla JS state machine, SSE subscriber & API client
│       └── style.css           # Premium dark-mode studio styling
├── launch_studio.ps1           # Hardened one-click PowerShell launcher
└── tests/
    └── test_studio_api.py      # Automated API contract & lifecycle verification
```

---

## 3. Concrete API Specifications & Pydantic Contracts

### 3.1 Pydantic Request & Response Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any

# Project Schemas
class CreateProjectRequest(BaseModel):
    project_name: str = Field(..., regex=r"^[a-zA-Z0-9_\-]+$")
    style_profile: Optional[str] = "photoreal_default"

class ProjectSummary(BaseModel):
    name: str
    pipeline_stage: str
    shots_count: int
    approved_count: int
    has_master_cut: bool

class ProjectDetailResponse(BaseModel):
    manifest: Dict[str, Any]
    script: Dict[str, Any]

class UpdateScriptRequest(BaseModel):
    script: Dict[str, Any]

class ApplyStyleRequest(BaseModel):
    style_profile: str

# Job & Execution Schemas
class RunJobRequest(BaseModel):
    stage: Literal["keyframes", "motion", "assemble", "run-all"]
    mock: bool = False

class RerollShotRequest(BaseModel):
    shot_id: str
    seed: Optional[int] = None
    prompt: Optional[str] = None
    mock: bool = False

class JobStatusResponse(BaseModel):
    active: bool
    job_id: Optional[str] = None
    project: Optional[str] = None
    stage: Optional[str] = None
    pid: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    status: Literal["idle", "running", "completed", "failed", "cancelled"]

class ActionResponse(BaseModel):
    status: Literal["ok", "error"]
    message: str
    details: Optional[Dict[str, Any]] = None
```

### 3.2 Endpoints Breakdown

| Method | Path | Request Body | Success Response | Error Codes | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/projects` | None | `List[ProjectSummary]` | 500 | List all existing studio projects |
| `POST` | `/api/projects` | `CreateProjectRequest` | `ActionResponse` (201) | 400, 409 | Initialize new project directory & manifest |
| `GET` | `/api/projects/{name}` | None | `ProjectDetailResponse` | 404 | Fetch current script and manifest |
| `POST` | `/api/projects/{name}/script` | `UpdateScriptRequest` | `ActionResponse` | 400, 423 | Non-destructive update & manifest sync |
| `GET` | `/api/styles` | None | `List[str]` | 500 | List available style preset names |
| `POST` | `/api/projects/{name}/style` | `ApplyStyleRequest` | `ActionResponse` | 404, 423 | Switch active StyleProfile |
| `POST` | `/api/projects/{name}/run` | `RunJobRequest` | `ActionResponse` | 404, 409 | Launch stage subprocess under mutex |
| `POST` | `/api/projects/{name}/reroll` | `RerollShotRequest` | `ActionResponse` | 404, 409 | Re-roll single keyframe under mutex |
| `POST` | `/api/projects/{name}/shots/{id}/approve` | None | `ActionResponse` | 404 | In-process approval of shot |
| `POST` | `/api/projects/{name}/shots/{id}/reject` | None | `ActionResponse` | 404 | In-process rejection of shot |
| `POST` | `/api/projects/{name}/approve-all` | None | `ActionResponse` | 404 | In-process approval of all shots |
| `GET` | `/api/jobs/active` | None | `JobStatusResponse` | 200 | Poll active job status |
| `POST` | `/api/jobs/cancel` | None | `ActionResponse` | 200, 400 | Terminate subprocess tree & settle driver |
| `GET` | `/api/jobs/logs` | None | `text/event-stream` | 200 | SSE stream with replay ring buffer |
| `GET` | `/media/{proj}/{stage}/{file}` | None | `FileResponse` | 403, 404 | Serve image/video with path security |

---

## 4. UI Layout & User Experience

```
+--------------------------------------------------------------------------------------------------+
|  STUDIO HEADER: [ Project Selector v ]  [ StyleProfile Preset v ]  [ GPU / Mock Switch ]  [Status] |
+------------------------------------+-------------------------------------------------------------+
| LEFT PANEL (40%): SCRIPT & SHOTS   | RIGHT PANEL (60%): STORYBOARD & SCREENING ROOM              |
|                                    |                                                             |
| [ + Add Shot ] [ Save Script ]     | TAB 1: STORYBOARD APPROVAL GRID                             |
|                                    | +---------------------------------------------------------+ |
| Shot #1 [Wide Establishing]        | | [Shot 001] [Approved]       | [Shot 002] [Pending]      | |
| Prompt: [ ... ]                    | | [ Keyframe Thumbnail ]      | [ Keyframe Thumbnail ]    | |
| Motion Prompt: [ ... ]             | | 25 frames (8n+1) | Seed: 42 | 49 frames | Seed: 107     | |
| Duration: [1.0s] -> 25 frames      | | [Approve] [Reject] [Reroll] | [Approve] [Reject] [Reroll]| |
| Camera: [subtle_handheld v]        | +---------------------------------------------------------+ |
| [Move Up] [Move Down] [Delete]     | [ Generate All Keyframes ]  [ Approve All Shots ]           |
|                                    |                                                             |
| Shot #2 [Medium Close-up]          | TAB 2: MOTION & SCREENING ROOM                              |
| ...                                | [ Render Motion (Stage 2) ]  [ Assemble Master Cut ]        |
|                                    | [ Master Cut Video Player (HTML5 24fps) ]                   |
|                                    | [ Download Final MP4 ]                                      |
+------------------------------------+-------------------------------------------------------------+
| BOTTOM DOCK: LIVE TERMINAL LOGS (SSE Stream)                                     [ Cancel Job ]  |
| > [2026-11-20 21:30:10] [INFO] Stage 2: Distilled LTX-Video 2B rendering 25 frames...           |
| > [STAGE 2 METRICS] Peak VRAM Allocated: 3.87 GB | System RAM Free: 2.54 GB                     |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Security & Stability Guardrails

1. **Path Traversal Protection:**
   All media requests resolve against `WORKSPACE_ROOT`. Any request containing traversal tokens (`..`) or resolving outside the project directory raises HTTP 403 Forbidden.
2. **Atomic File Writes:**
   All JSON modifications (`manifest.json`, `script.json`) write to a temporary file in the same directory before performing an atomic `os.replace`, preventing corrupted files from process interruptions.
3. **HTTP 206 Video Streaming:**
   Video endpoints use Starlette's native `FileResponse(path, media_type="video/mp4")`, guaranteeing HTTP 206 Partial Content byte-range support for browser timeline scrubbing.

---

## 6. Phased Implementation Plan

```markdown
### Task 1: Backend Infrastructure & AsyncJobManager (`video-pipeline/studio/server.py`)
- Implement `AsyncJobManager` with `asyncio.Lock` for the single-GPU mutex.
- Implement subprocess invocation with `PYTHONUNBUFFERED=1`, `create_time` tracking, and `taskkill /F /T` cancellation with settling cooldown.
- Implement SSE broadcasting with a 500-line circular log buffer and `\r` progress throttling.

### Task 2: Project Management & Manifest Sync Endpoints
- Implement `GET /api/projects`, `POST /api/projects`, `GET /api/projects/{name}`.
- Implement `POST /api/projects/{name}/script` with non-destructive manifest reconciliation.
- Implement `POST /api/projects/{name}/shots/{shot_id}/approve` and `/reject` (in-process JSON mutations).
- Implement `GET /api/styles` and `POST /api/projects/{name}/style`.

### Task 3: Job Execution & Media Serving
- Implement `POST /api/projects/{name}/run` and `POST /api/projects/{name}/reroll`.
- Implement `/media/{project}/{stage}/{filename}` with path traversal checks and cache-busting headers.
- Mount `/static` and serve `index.html` at root `/`.

### Task 4: Frontend Web Dashboard (`video-pipeline/studio/static/`)
- Create `index.html` matching wireframe with dark-mode aesthetic.
- Create `style.css` (clean, responsive, dark palette, zero external CDN dependencies).
- Create `app.js` (vanilla state machine, SSE log subscriber, modal clip previews, video player auto-unload).

### Task 5: PowerShell Launcher & Integration Testing
- Write `video-pipeline/launch_studio.ps1` with port collision check, environment discovery, and browser launch.
- Write `video-pipeline/tests/test_studio_api.py` and verify all endpoints under mock execution.
```

---

## 7. Verification Criteria & Automated Test Plan

1. **Automated API Contract Suite (`video-pipeline/tests/test_studio_api.py`):**
   - Assert `GET /` returns HTTP 200 with `text/html`.
   - Assert `POST /api/projects` creates project directory, default script, and manifest.
   - Assert `POST /api/projects/{name}/script` preserves approvals and keyframe paths for unchanged shots.
   - Assert `POST /api/projects/{name}/run` returns HTTP 409 when another job is running.
   - Assert `POST /api/jobs/cancel` cleanly terminates the subprocess and respects the settling cooldown.
   - Assert path traversal attempts (`/media/../../etc`) return HTTP 403.
2. **Process Footprint Verification:**
   - Measure `server.py` memory: must remain `< 80 MB RSS`.
   - Assert `"torch"` is NOT imported into `server.py`: `assert "torch" not in sys.modules`.
3. **End-to-End Visual Verification:**
   - Launch studio via `launch_studio.ps1`.
   - Create project in Mock Mode $\to$ Generate keyframes $\to$ Approve $\to$ Synthesize motion $\to$ Assemble timeline $\to$ Play final video in browser.
