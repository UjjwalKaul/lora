"""
Data contracts and schemas for the production video generation pipeline.
Includes Pydantic models for script.json, manifest.json, and shot items.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class ShotStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class CameraMotion(str, Enum):
    STATIC = "static"
    SLOW_DOLLY_FORWARD = "slow_dolly_forward"
    SLOW_DOLLY_BACK = "slow_dolly_back"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    SUBTLE_HANDHELD = "subtle_handheld"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


class ShotInput(BaseModel):
    """Input definition for a shot inside script.json."""
    shot_id: str
    shot_type: str = "medium"
    camera_motion: str = "static"
    duration_sec: float = 2.0
    prompt: str
    motion_prompt: Optional[str] = None
    conditioning_mode: str = "start_only"  # "start_only" or "start_and_end"
    seed: Optional[int] = None


class ScriptSchema(BaseModel):
    """Structure for input story script (script.json)."""
    project_name: str
    version: str = "1.0.0"
    aspect_ratio: str = "16:9"
    target_fps: int = 24
    keyframe_backend: str = "mock"  # "mock", "sdxl_lightning", "sd15_lora"
    motion_backend: str = "mock"    # "mock", "ltx_video_2b"
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    global_style_prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    shots: List[ShotInput] = Field(default_factory=list)


class ShotItem(BaseModel):
    """Lifecycle state item for a single shot in manifest.json."""
    shot_id: str
    index: int
    status: ShotStatus = ShotStatus.PENDING_REVIEW
    shot_type: str = "medium"
    camera_motion: str = "static"
    duration_sec: float = 2.0
    num_frames: int = 49  # Must be 8n + 1
    conditioning_mode: str = "start_only"
    prompt: str
    motion_prompt: Optional[str] = None
    keyframe_start_path: Optional[str] = None
    keyframe_end_path: Optional[str] = None
    seed: int = 42
    clip_output_path: Optional[str] = None
    error_message: Optional[str] = None


class ProjectConfig(BaseModel):
    """Runtime configuration for a project."""
    target_width: int = 768
    target_height: int = 512
    target_fps: int = 24
    max_frames: int = 49
    keyframe_backend: str = "mock"
    motion_backend: str = "mock"
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    inference_steps: int = 8
    guidance_scale: float = 3.0


class ManifestSchema(BaseModel):
    """Full project manifest tracking shot lifecycle and file paths."""
    project_name: str
    version: str = "1.0.0"
    pipeline_stage: str = "initialized"
    config: ProjectConfig
    global_style_prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    shots: List[ShotItem] = Field(default_factory=list)

    def get_shot(self, shot_id: str) -> Optional[ShotItem]:
        for shot in self.shots:
            if shot.shot_id == shot_id:
                return shot
        return None

    def save(self, manifest_path: Path | str) -> None:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, manifest_path: Path | str) -> ManifestSchema:
        path = Path(manifest_path)
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
