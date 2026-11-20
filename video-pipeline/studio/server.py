"""
FastAPI Backend Server for Local Video Studio.
Zero-Torch architecture: Operates strictly as an asynchronous I/O coordinator (< 80 MB RSS).
All model compute is delegated to isolated OS subprocesses running run_pipeline.py.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import psutil
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure torch is NEVER imported in this web process
assert "torch" not in sys.modules, "FATAL: torch must not be imported in server.py"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [studio_server] %(message)s")
logger = logging.getLogger("studio_server")

# Workspace and directory anchors
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR / "workspace" / "projects"
STYLES_DIR = BASE_DIR / "styles"
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUN_PIPELINE_SCRIPT = BASE_DIR / "run_pipeline.py"

WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
STYLES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Atomic File Utilities
# -----------------------------------------------------------------------------
def atomic_write_json(file_path: Path, data: Any) -> None:
    """Writes JSON data atomically using a temporary file and os.replace."""
    temp_path = file_path.with_suffix(f".tmp_{uuid.uuid4().hex}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, file_path)


def read_json(file_path: Path) -> Dict[str, Any]:
    """Reads JSON file safely."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Pydantic Schemas (Pydantic v2 compliant)
# -----------------------------------------------------------------------------
class CreateProjectRequest(BaseModel):
    project_name: str = Field(..., pattern=r"^[a-zA-Z0-9_\-]+$")
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


class UpdateConfigRequest(BaseModel):
    keyframe_backend: Optional[str] = None
    lora_path: Optional[str] = None
    lora_scale: Optional[float] = None
    stepped_fps: Optional[int] = None
    chromatic_aberration_px: Optional[int] = None
    lut_path: Optional[str] = None


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


# -----------------------------------------------------------------------------
# AsyncJobManager: Mutex & Windows Subprocess Runner
# -----------------------------------------------------------------------------
class ActiveJob:
    def __init__(self, job_id: str, project: str, stage: str, pid: Optional[int], process: Optional[subprocess.Popen]):
        self.job_id = job_id
        self.project = project
        self.stage = stage
        self.pid = pid
        self.process = process
        self.create_time: Optional[float] = None
        if pid:
            try:
                self.create_time = psutil.Process(pid).create_time()
            except Exception:
                self.create_time = None
        self.start_time = time.time()
        self.status: Literal["running", "completed", "failed", "cancelled"] = "running"


class AsyncJobManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.active_job: Optional[ActiveJob] = None
        self.log_ring_buffer: collections.deque = collections.deque(maxlen=500)
        self.subscribers: List[asyncio.Queue] = []
        self._last_progress_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self.active_job is not None and self.active_job.status == "running"

    def get_status(self) -> JobStatusResponse:
        if self.active_job is None:
            return JobStatusResponse(active=False, status="idle")
        elapsed = time.time() - self.active_job.start_time if self.active_job.status == "running" else None
        return JobStatusResponse(
            active=self.active_job.status == "running",
            job_id=self.active_job.job_id,
            project=self.active_job.project,
            stage=self.active_job.stage,
            pid=self.active_job.pid,
            elapsed_seconds=elapsed,
            status=self.active_job.status,
        )

    def _broadcast_log(self, raw_line: str):
        line = raw_line.rstrip("\r\n")
        if not line:
            return

        # Carriage return / progress bar throttling (max 10 Hz)
        is_progress = "\r" in raw_line or "%" in line
        now = time.time()
        if is_progress and (now - self._last_progress_time < 0.10):
            return
        if is_progress:
            self._last_progress_time = now

        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {line}"
        self.log_ring_buffer.append(formatted)

        for q in list(self.subscribers):
            try:
                q.put_nowait(formatted)
            except Exception:
                pass

    def start_job(
        self,
        project_name: str,
        stage: str,
        extra_args: Optional[List[str]] = None,
        mock: bool = False
    ) -> ActiveJob:
        if self.is_running:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Another job ({self.active_job.job_id}) is currently executing."
            )

        job_id = f"job_{uuid.uuid4().hex[:8]}"
        cmd = [
            sys.executable,
            "-u",
            str(RUN_PIPELINE_SCRIPT),
            "--project", project_name,
            "--stage", stage
        ]
        if mock:
            cmd.append("--mock")
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        self._broadcast_log(f">>> STARTING JOB {job_id}: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )

        job = ActiveJob(job_id=job_id, project=project_name, stage=stage, pid=process.pid, process=process)
        self.active_job = job

        # Background thread to drain stdout safely without ProactorEventLoop issues on Windows
        def reader_worker():
            try:
                for line in iter(process.stdout.readline, ""):
                    if line:
                        self._broadcast_log(line)
                    elif process.poll() is not None:
                        break
            except Exception as e:
                self._broadcast_log(f"Error reading process stdout: {e}")
            finally:
                process.stdout.close()
                rc = process.wait()
                if job.status == "running":
                    job.status = "completed" if rc == 0 else "failed"
                self._broadcast_log(f">>> JOB {job_id} TERMINATED with exit code {rc} ({job.status.upper()})")

        thread = threading.Thread(target=reader_worker, daemon=True)
        thread.start()

        return job

    async def cancel_job(self) -> ActionResponse:
        job = self.active_job
        if not job or job.status != "running" or not job.process or not job.pid:
            return ActionResponse(status="ok", message="No active running job to cancel.")

        pid = job.pid
        target_proc = None
        try:
            target_proc = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Verify PID identity before killing to avoid killing recycled Windows PIDs
        if target_proc and job.create_time:
            try:
                if abs(target_proc.create_time() - job.create_time) > 2.0:
                    logger.warning(f"PID {pid} create_time mismatch. Refusing to taskkill.")
                    job.status = "cancelled"
                    return ActionResponse(status="error", message="PID recycled; process not killed.")
            except Exception:
                pass

        logger.info(f"Terminating subprocess tree for PID {pid} via taskkill...")
        self._broadcast_log(f">>> ABORT REQUESTED: Killing subprocess tree (PID {pid})...")
        try:
            # Issue taskkill /F /T
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        except Exception as e:
            logger.error(f"Failed to execute taskkill: {e}")

        job.status = "cancelled"

        # NVIDIA driver-settling cooldown (750 ms)
        self._broadcast_log(">>> Driver settling cooldown (750ms) for GPU VRAM reclamation...")
        await asyncio.sleep(0.75)
        self._broadcast_log(">>> Mutex released. Studio is ready.")

        return ActionResponse(status="ok", message=f"Job {job.job_id} cancelled.")


job_manager = AsyncJobManager()


# -----------------------------------------------------------------------------
# Manifest & Script Helpers
# -----------------------------------------------------------------------------
def get_project_dir(name: str) -> Path:
    proj_dir = WORKSPACE_ROOT / name
    if not proj_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{name}' does not exist.")
    return proj_dir


def reconcile_manifest(proj_dir: Path, script_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Non-destructive manifest reconciliation.
    Preserves existing shot approvals, keyframe paths, clip paths, and seeds for unchanged shots.
    """
    manifest_file = proj_dir / "manifest.json"
    if not manifest_file.exists():
        raise HTTPException(status_code=404, detail=f"Manifest not found in {proj_dir.name}")

    manifest_data = read_json(manifest_file)
    existing_shots = {s.get("shot_id"): s for s in manifest_data.get("shots", [])}

    target_fps = int(script_data.get("target_fps", 24))
    max_frames = int(manifest_data.get("config", {}).get("max_frames", 49))

    new_shots = []
    for idx, s in enumerate(script_data.get("shots", []), start=1):
        shot_id = s.get("shot_id")
        duration_sec = float(s.get("duration_sec", 2.0))
        # Snap frames to 8n + 1
        raw_frames = int(round(duration_sec * target_fps))
        n = max(1, round((raw_frames - 1) / 8))
        snapped_frames = (8 * n) + 1
        if snapped_frames > max_frames:
            snapped_frames = (8 * (max_frames // 8)) + 1

        if shot_id in existing_shots:
            prev = existing_shots[shot_id]
            prev["index"] = idx
            prev["shot_type"] = s.get("shot_type", prev.get("shot_type", "medium"))
            prev["camera_motion"] = s.get("camera_motion", prev.get("camera_motion", "static"))
            prev["duration_sec"] = duration_sec
            prev["num_frames"] = snapped_frames
            prev["prompt"] = s.get("prompt", prev.get("prompt", ""))
            prev["motion_prompt"] = s.get("motion_prompt", prev.get("motion_prompt"))
            if "seed" in s and s["seed"] is not None:
                prev["seed"] = s["seed"]
            # Notice: status, keyframe_start_path, clip_output_path are kept intact!
            new_shots.append(prev)
        else:
            seed = s.get("seed") or (100 * idx + 7)
            new_shots.append({
                "shot_id": shot_id,
                "index": idx,
                "status": "pending_review",
                "shot_type": s.get("shot_type", "medium"),
                "camera_motion": s.get("camera_motion", "static"),
                "duration_sec": duration_sec,
                "num_frames": snapped_frames,
                "conditioning_mode": s.get("conditioning_mode", "start_only"),
                "prompt": s.get("prompt", ""),
                "motion_prompt": s.get("motion_prompt"),
                "keyframe_start_path": f"keyframes/{shot_id}_start.png",
                "keyframe_end_path": None,
                "seed": seed,
                "clip_output_path": f"clips/{shot_id}.mp4",
                "error_message": None
            })

    manifest_data["shots"] = new_shots
    if "global_style_prompt" in script_data:
        manifest_data["global_style_prompt"] = script_data["global_style_prompt"]
    if "negative_prompt" in script_data:
        manifest_data["negative_prompt"] = script_data["negative_prompt"]

    return manifest_data


# -----------------------------------------------------------------------------
# FastAPI Application & Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Local Video Studio backend initialized at http://localhost:8000")
    yield
    # Cleanup on server shutdown
    if job_manager.is_running:
        logger.info("Server shutting down, cancelling active job...")
        await job_manager.cancel_job()


app = FastAPI(title="Local Video Studio", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# API Endpoints: Projects
# -----------------------------------------------------------------------------
@app.get("/api/projects", response_model=List[ProjectSummary])
def list_projects():
    """List all project summaries."""
    summaries = []
    for p in sorted(WORKSPACE_ROOT.iterdir()):
        if p.is_dir():
            manifest_file = p / "manifest.json"
            if manifest_file.exists():
                try:
                    data = read_json(manifest_file)
                    shots = data.get("shots", [])
                    approved = sum(1 for s in shots if s.get("status") == "approved")
                    has_master = (p / "final" / "master_cut.mp4").exists()
                    summaries.append(ProjectSummary(
                        name=p.name,
                        pipeline_stage=data.get("pipeline_stage", "initialized"),
                        shots_count=len(shots),
                        approved_count=approved,
                        has_master_cut=has_master
                    ))
                except Exception as e:
                    logger.error(f"Error reading manifest for project {p.name}: {e}")
    return summaries


@app.post("/api/projects", response_model=ActionResponse, status_code=status.HTTP_201_CREATED)
def create_project(req: CreateProjectRequest):
    """Initializes a new project directory and manifest."""
    proj_dir = WORKSPACE_ROOT / req.project_name
    if (proj_dir / "manifest.json").exists():
        raise HTTPException(status_code=409, detail=f"Project '{req.project_name}' already exists.")

    proj_dir.mkdir(parents=True, exist_ok=True)
    (proj_dir / "keyframes").mkdir(exist_ok=True)
    (proj_dir / "clips").mkdir(exist_ok=True)
    (proj_dir / "final").mkdir(exist_ok=True)

    # Initialize default script.json and manifest.json
    style_name = req.style_profile or "photoreal_default"
    style_file = STYLES_DIR / f"{style_name}.json"
    style_data = read_json(style_file) if style_file.exists() else {}

    script_data = {
        "project_name": req.project_name,
        "version": "1.0.0",
        "style_profile": style_name,
        "aspect_ratio": "16:9",
        "target_fps": 24,
        "keyframe_backend": None,
        "motion_backend": None,
        "lora_path": None,
        "lora_scale": 1.0,
        "global_style_prompt": style_data.get("stage1", {}).get("style_prompt", ""),
        "negative_prompt": style_data.get("stage1", {}).get("negative_prompt", ""),
        "shots": [
            {
                "shot_id": "shot_001",
                "shot_type": "wide_establishing",
                "camera_motion": "slow_dolly_forward",
                "duration_sec": 2.0,
                "prompt": "futuristic neon city skyline at dusk with flying vehicles streaming between glass skyscrapers",
                "motion_prompt": "vehicles smoothly flying, neon signs glowing through rain mist",
                "seed": 42
            },
            {
                "shot_id": "shot_002",
                "shot_type": "medium_close_up",
                "camera_motion": "subtle_handheld",
                "duration_sec": 2.0,
                "prompt": "cybernetic mercenary with glowing blue mechanical eye looking down from a high-tech rain-slicked rooftop",
                "motion_prompt": "mercenary breathing slowly, rain droplets falling across metallic armor",
                "seed": 99
            }
        ]
    }

    manifest_data = {
        "project_name": req.project_name,
        "version": "1.0.0",
        "pipeline_stage": "initialized",
        "config": {
            "target_width": 768,
            "target_height": 512,
            "target_fps": 24,
            "max_frames": 49,
            "style_profile": style_name,
            "keyframe_backend": style_data.get("stage1", {}).get("keyframe_backend", "sdxl_lightning"),
            "motion_backend": style_data.get("stage2", {}).get("motion_backend", "ltx_video_2b"),
            "base_model": style_data.get("stage1", {}).get("base_model", "ByteDance/SDXL-Lightning"),
            "lora_path": style_data.get("stage1", {}).get("lora_path"),
            "lora_scale": style_data.get("stage1", {}).get("lora_scale", 1.0),
            "trigger_tokens": style_data.get("stage1", {}).get("trigger_tokens", ""),
            "style_prompt": style_data.get("stage1", {}).get("style_prompt", ""),
            "negative_prompt": style_data.get("stage1", {}).get("negative_prompt", ""),
            "motion_style_prompt": style_data.get("stage2", {}).get("motion_style_prompt", ""),
            "inference_steps": style_data.get("stage2", {}).get("inference_steps", 8),
            "guidance_scale": style_data.get("stage2", {}).get("guidance_scale", 3.0),
            "post_fx": style_data.get("post_fx")
        },
        "global_style_prompt": script_data["global_style_prompt"],
        "negative_prompt": script_data["negative_prompt"],
        "shots": [
            {
                "shot_id": "shot_001",
                "index": 1,
                "status": "pending_review",
                "shot_type": "wide_establishing",
                "camera_motion": "slow_dolly_forward",
                "duration_sec": 2.0,
                "num_frames": 49,
                "conditioning_mode": "start_only",
                "prompt": script_data["shots"][0]["prompt"],
                "motion_prompt": script_data["shots"][0]["motion_prompt"],
                "keyframe_start_path": "keyframes/shot_001_start.png",
                "keyframe_end_path": None,
                "seed": 42,
                "clip_output_path": "clips/shot_001.mp4",
                "error_message": None
            },
            {
                "shot_id": "shot_002",
                "index": 2,
                "status": "pending_review",
                "shot_type": "medium_close_up",
                "camera_motion": "subtle_handheld",
                "duration_sec": 2.0,
                "num_frames": 49,
                "conditioning_mode": "start_only",
                "prompt": script_data["shots"][1]["prompt"],
                "motion_prompt": script_data["shots"][1]["motion_prompt"],
                "keyframe_start_path": "keyframes/shot_002_start.png",
                "keyframe_end_path": None,
                "seed": 99,
                "clip_output_path": "clips/shot_002.mp4",
                "error_message": None
            }
        ]
    }

    atomic_write_json(proj_dir / "script.json", script_data)
    atomic_write_json(proj_dir / "manifest.json", manifest_data)

    return ActionResponse(status="ok", message=f"Project '{req.project_name}' created successfully.")


@app.get("/api/projects/{name}", response_model=ProjectDetailResponse)
def get_project(name: str):
    """Retrieve full script and manifest for project."""
    proj_dir = get_project_dir(name)
    manifest_file = proj_dir / "manifest.json"
    script_file = proj_dir / "script.json"
    if not manifest_file.exists() or not script_file.exists():
        raise HTTPException(status_code=404, detail="Project configuration files missing.")
    return ProjectDetailResponse(
        manifest=read_json(manifest_file),
        script=read_json(script_file)
    )


@app.post("/api/projects/{name}/script", response_model=ActionResponse)
def update_project_script(name: str, req: UpdateScriptRequest):
    """Update script and non-destructively reconcile manifest. Locked during running jobs."""
    if job_manager.is_running:
        raise HTTPException(status_code=423, detail="Script editing locked while a pipeline job is running.")

    proj_dir = get_project_dir(name)
    script_file = proj_dir / "script.json"
    manifest_file = proj_dir / "manifest.json"

    updated_manifest = reconcile_manifest(proj_dir, req.script)
    atomic_write_json(script_file, req.script)
    atomic_write_json(manifest_file, updated_manifest)

    return ActionResponse(status="ok", message="Script and manifest successfully reconciled.")


# -----------------------------------------------------------------------------
# API Endpoints: Styles & Configuration Tweaks
# -----------------------------------------------------------------------------
@app.get("/api/styles")
def list_styles():
    """Returns full preset details for all available styles."""
    styles = {}
    for f in STYLES_DIR.glob("*.json"):
        try:
            styles[f.stem] = read_json(f)
        except Exception as e:
            logger.error(f"Error loading style {f.name}: {e}")
    return styles


@app.post("/api/projects/{name}/style", response_model=ActionResponse)
def apply_style(name: str, req: ApplyStyleRequest):
    """Applies a StyleProfile preset to project."""
    if job_manager.is_running:
        raise HTTPException(status_code=423, detail="Configuration locked while a pipeline job is running.")

    proj_dir = get_project_dir(name)
    style_file = STYLES_DIR / f"{req.style_profile}.json"
    if not style_file.exists():
        raise HTTPException(status_code=404, detail=f"Style '{req.style_profile}' not found.")

    style_data = read_json(style_file)
    script_file = proj_dir / "script.json"
    manifest_file = proj_dir / "manifest.json"

    script_data = read_json(script_file)
    manifest_data = read_json(manifest_file)

    script_data["style_profile"] = req.style_profile
    cfg = manifest_data.setdefault("config", {})
    cfg["style_profile"] = req.style_profile
    cfg["keyframe_backend"] = style_data.get("stage1", {}).get("keyframe_backend", cfg.get("keyframe_backend"))
    cfg["motion_backend"] = style_data.get("stage2", {}).get("motion_backend", cfg.get("motion_backend"))
    cfg["base_model"] = style_data.get("stage1", {}).get("base_model", cfg.get("base_model"))
    cfg["lora_path"] = style_data.get("stage1", {}).get("lora_path")
    cfg["lora_scale"] = style_data.get("stage1", {}).get("lora_scale", 1.0)
    cfg["trigger_tokens"] = style_data.get("stage1", {}).get("trigger_tokens", "")
    cfg["style_prompt"] = style_data.get("stage1", {}).get("style_prompt", "")
    cfg["negative_prompt"] = style_data.get("stage1", {}).get("negative_prompt", "")
    cfg["motion_style_prompt"] = style_data.get("stage2", {}).get("motion_style_prompt", "")
    cfg["inference_steps"] = style_data.get("stage2", {}).get("inference_steps", 8)
    cfg["guidance_scale"] = style_data.get("stage2", {}).get("guidance_scale", 3.0)
    cfg["post_fx"] = style_data.get("post_fx")

    atomic_write_json(script_file, script_data)
    atomic_write_json(manifest_file, manifest_data)

    return ActionResponse(status="ok", message=f"Applied style '{req.style_profile}' to project '{name}'.")


@app.post("/api/projects/{name}/config", response_model=ActionResponse)
def update_project_config(name: str, req: UpdateConfigRequest):
    """
    Tweak keyframe backend, optional LoRA, and post_fx look-dev parameters directly.
    Allows rapid look-dev iteration and Stage 3 assemble re-runs without manual JSON editing.
    """
    if job_manager.is_running:
        raise HTTPException(status_code=423, detail="Configuration locked while a pipeline job is running.")

    proj_dir = get_project_dir(name)
    script_file = proj_dir / "script.json"
    manifest_file = proj_dir / "manifest.json"

    script_data = read_json(script_file)
    manifest_data = read_json(manifest_file)
    cfg = manifest_data.setdefault("config", {})

    if req.keyframe_backend is not None:
        cfg["keyframe_backend"] = req.keyframe_backend
        script_data["keyframe_backend"] = req.keyframe_backend
    if req.lora_path is not None:
        cfg["lora_path"] = req.lora_path or None
        script_data["lora_path"] = req.lora_path or None
    if req.lora_scale is not None:
        cfg["lora_scale"] = req.lora_scale
        script_data["lora_scale"] = req.lora_scale

    # Update post_fx look-dev settings
    post_fx = cfg.get("post_fx") or {}
    if req.stepped_fps is not None:
        post_fx["stepped_fps"] = req.stepped_fps if req.stepped_fps > 0 else None
    if req.chromatic_aberration_px is not None:
        post_fx["chromatic_aberration_px"] = req.chromatic_aberration_px if req.chromatic_aberration_px > 0 else None
    if req.lut_path is not None:
        post_fx["lut_path"] = req.lut_path or None

    cfg["post_fx"] = post_fx if any(post_fx.values()) else None

    atomic_write_json(script_file, script_data)
    atomic_write_json(manifest_file, manifest_data)

    return ActionResponse(status="ok", message=f"Project '{name}' configuration updated.")


# -----------------------------------------------------------------------------
# API Endpoints: In-Process Shot Approvals
# -----------------------------------------------------------------------------
@app.post("/api/projects/{name}/shots/{shot_id}/approve", response_model=ActionResponse)
def approve_shot(name: str, shot_id: str):
    """Instant in-process shot approval."""
    proj_dir = get_project_dir(name)
    manifest_file = proj_dir / "manifest.json"
    data = read_json(manifest_file)

    found = False
    for s in data.get("shots", []):
        if s.get("shot_id") == shot_id:
            s["status"] = "approved"
            s["error_message"] = None
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Shot '{shot_id}' not found.")

    atomic_write_json(manifest_file, data)
    return ActionResponse(status="ok", message=f"Shot '{shot_id}' approved.")


@app.post("/api/projects/{name}/shots/{shot_id}/reject", response_model=ActionResponse)
def reject_shot(name: str, shot_id: str):
    """Instant in-process shot rejection."""
    proj_dir = get_project_dir(name)
    manifest_file = proj_dir / "manifest.json"
    data = read_json(manifest_file)

    found = False
    for s in data.get("shots", []):
        if s.get("shot_id") == shot_id:
            s["status"] = "rejected"
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Shot '{shot_id}' not found.")

    atomic_write_json(manifest_file, data)
    return ActionResponse(status="ok", message=f"Shot '{shot_id}' rejected.")


@app.post("/api/projects/{name}/approve-all", response_model=ActionResponse)
def approve_all_shots(name: str):
    """Instant in-process approval of all shots in project."""
    proj_dir = get_project_dir(name)
    manifest_file = proj_dir / "manifest.json"
    data = read_json(manifest_file)

    for s in data.get("shots", []):
        s["status"] = "approved"
        s["error_message"] = None

    atomic_write_json(manifest_file, data)
    return ActionResponse(status="ok", message=f"All shots in '{name}' approved.")


# -----------------------------------------------------------------------------
# API Endpoints: Pipeline Execution & Mutex Control
# -----------------------------------------------------------------------------
@app.post("/api/projects/{name}/run", response_model=ActionResponse)
def run_job(name: str, req: RunJobRequest):
    """Launches an isolated subprocess stage under the single-job mutex."""
    get_project_dir(name)
    try:
        job = job_manager.start_job(
            project_name=name,
            stage=req.stage,
            mock=req.mock
        )
        return ActionResponse(status="ok", message=f"Stage '{req.stage}' started.", details={"job_id": job.job_id})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/projects/{name}/reroll", response_model=ActionResponse)
def reroll_shot(name: str, req: RerollShotRequest):
    """
    Rerolls a single keyframe with optional prompt and seed.
    Persists updated prompt into both script.json and manifest.json before invoking subprocess.
    """
    proj_dir = get_project_dir(name)
    script_file = proj_dir / "script.json"
    manifest_file = proj_dir / "manifest.json"

    script_data = read_json(script_file)
    manifest_data = read_json(manifest_file)

    import random
    new_seed = req.seed if req.seed is not None else random.randint(1000, 99999)

    # Persist prompt and seed atomically
    shot_found = False
    for s in manifest_data.get("shots", []):
        if s.get("shot_id") == req.shot_id:
            s["status"] = "pending_review"
            s["seed"] = new_seed
            if req.prompt:
                s["prompt"] = req.prompt
            shot_found = True
            break

    if not shot_found:
        raise HTTPException(status_code=404, detail=f"Shot '{req.shot_id}' not found.")

    for s in script_data.get("shots", []):
        if s.get("shot_id") == req.shot_id:
            s["seed"] = new_seed
            if req.prompt:
                s["prompt"] = req.prompt
            break

    atomic_write_json(script_file, script_data)
    atomic_write_json(manifest_file, manifest_data)

    extra_args = ["--shot-id", req.shot_id, "--seed", str(new_seed)]
    if req.prompt:
        extra_args.extend(["--prompt", req.prompt])

    try:
        job = job_manager.start_job(
            project_name=name,
            stage="reroll",
            extra_args=extra_args,
            mock=req.mock
        )
        return ActionResponse(status="ok", message=f"Keyframe reroll started for shot '{req.shot_id}'.", details={"job_id": job.job_id, "seed": new_seed})
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting reroll: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/active", response_model=JobStatusResponse)
def get_active_job():
    """Poll active job status and elapsed time."""
    return job_manager.get_status()


@app.post("/api/jobs/cancel", response_model=ActionResponse)
async def cancel_job():
    """Cancels active job, terminates process tree, and enforces settling cooldown."""
    return await job_manager.cancel_job()


@app.get("/api/jobs/logs")
async def stream_logs(request: Request):
    """SSE endpoint streaming live logs with circular ring buffer replay."""
    async def event_generator():
        client_queue = asyncio.Queue()
        job_manager.subscribers.append(client_queue)
        try:
            # Replay existing ring buffer first
            for line in list(job_manager.log_ring_buffer):
                yield f"data: {json.dumps({'line': line})}\n\n"

            # Stream incoming lines
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(client_queue.get(), timeout=1.0)
                    yield f"data: {json.dumps({'line': line})}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive heartbeat comment
                    yield ": ping\n\n"
        finally:
            if client_queue in job_manager.subscribers:
                job_manager.subscribers.remove(client_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# -----------------------------------------------------------------------------
# Media Serving & Path Traversal Security
# -----------------------------------------------------------------------------
@app.get("/media/{project}/{stage}/{filename}")
def serve_media(project: str, stage: str, filename: str):
    """
    Safely serves image keyframes and video clips.
    Validates against directory traversal (403 Forbidden).
    Supports HTTP 206 Partial Content byte ranges for video playback scrubbing.
    """
    base_proj = (WORKSPACE_ROOT / project).resolve()
    target_file = (base_proj / stage / filename).resolve()

    # Prevent path traversal outside project directory
    try:
        target_file.relative_to(base_proj)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: Invalid file path.")

    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="Requested media file does not exist.")

    media_type = "video/mp4" if filename.lower().endswith(".mp4") else "image/png"
    return FileResponse(
        target_file,
        media_type=media_type,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


# -----------------------------------------------------------------------------
# Static Frontend Serving
# -----------------------------------------------------------------------------
@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return Response(content="<h1>Studio interface initializing...</h1>", media_type="text/html")
    return FileResponse(index_file)


# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
