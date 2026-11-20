"""
Data contracts and schemas for the production video generation pipeline.
Includes Pydantic models for script.json, manifest.json, shot items, and StyleProfiles.
"""

from __future__ import annotations

import json
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


class PostFxConfig(BaseModel):
    """CPU-only FFmpeg look-dev post-processing options."""
    stepped_fps: Optional[int] = None  # e.g. 12 (animates on 2s within a 24fps container)
    chromatic_aberration_px: Optional[int] = None  # e.g. 2 (FFmpeg rgbashift)
    lut_path: Optional[str] = None  # Optional 3D LUT (.cube / .png)


class Stage1StyleConfig(BaseModel):
    """Style configuration for Stage 1 Keyframe Generation."""
    keyframe_backend: str = "sdxl_lightning"  # "sd15_lora" | "sdxl_lightning" | "mock"
    base_model: str = "ByteDance/SDXL-Lightning"
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    trigger_tokens: Optional[str] = ""
    style_prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""


class Stage2StyleConfig(BaseModel):
    """Style configuration for Stage 2 Motion Synthesis."""
    motion_backend: str = "ltx_video_2b"  # "ltx_video_2b" | "mock"
    motion_style_prompt: Optional[str] = ""
    inference_steps: Optional[int] = 8
    guidance_scale: Optional[float] = 3.0


class StyleProfile(BaseModel):
    """
    Style-agnostic preset controlling all three pipeline stages.
    Loaded from video-pipeline/styles/<name>.json.
    """
    name: str
    description: Optional[str] = ""
    stage1: Stage1StyleConfig = Field(default_factory=Stage1StyleConfig)
    stage2: Stage2StyleConfig = Field(default_factory=Stage2StyleConfig)
    post_fx: Optional[PostFxConfig] = None

    @classmethod
    def load_profile(cls, name_or_path: str, styles_dir: Optional[Path] = None) -> StyleProfile:
        path = Path(name_or_path)
        if not path.exists() and styles_dir:
            candidate = styles_dir / f"{name_or_path}.json" if not name_or_path.endswith(".json") else styles_dir / name_or_path
            if candidate.exists():
                path = candidate

        if not path.exists():
            default_dir = Path(__file__).resolve().parent.parent / "styles"
            candidate = default_dir / f"{name_or_path}.json" if not name_or_path.endswith(".json") else default_dir / name_or_path
            if candidate.exists():
                path = candidate

        if not path.exists():
            raise FileNotFoundError(f"StyleProfile not found at '{name_or_path}' (searched in {styles_dir or Path(__file__).resolve().parent.parent / 'styles'})")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)


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
    style_profile: Optional[str] = "photoreal_default"
    aspect_ratio: str = "16:9"
    target_fps: int = 24
    keyframe_backend: Optional[str] = None  # If set, overrides style_profile
    motion_backend: Optional[str] = None    # If set, overrides style_profile
    lora_path: Optional[str] = None         # If set, overrides style_profile
    lora_scale: Optional[float] = None
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
    style_profile: str = "photoreal_default"
    keyframe_backend: str = "mock"
    motion_backend: str = "mock"
    base_model: Optional[str] = None
    lora_path: Optional[str] = None
    lora_scale: float = 1.0
    trigger_tokens: Optional[str] = ""
    style_prompt: Optional[str] = ""
    motion_style_prompt: Optional[str] = ""
    inference_steps: int = 8
    guidance_scale: float = 3.0
    post_fx: Optional[PostFxConfig] = None


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
