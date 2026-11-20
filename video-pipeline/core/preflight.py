"""
Pre-flight Keyframe & Frame Budget Ingestion Sanitizer.
Ensures frame counts snap strictly to 8n + 1 (LTX-Video requirement) and
keyframe images conform to dimensions divisible by 32, RGB 8-bit, and target aspect ratio.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple
from PIL import Image

from core.schemas import ManifestSchema, ShotItem, ShotStatus

logger = logging.getLogger(__name__)


def snap_to_8n_plus_1(duration_sec: float, fps: int = 24, max_frames: int = 49) -> int:
    """
    Snap target frame count strictly to the nearest valid 8n + 1 value.
    LTX-Video causal 3D VAE strictly requires num_frames = 8n + 1 (e.g., 9, 17, 25, 33, 41, 49, 57, 65, 73).
    Caps at max_frames (default 49 for 6 GB VRAM safety).
    """
    raw_frames = int(round(duration_sec * fps))
    if raw_frames <= 1:
        return 9
    
    # Calculate n that minimizes |8n + 1 - raw_frames|
    n = max(1, int(round((raw_frames - 1) / 8.0)))
    snapped = 8 * n + 1

    # Cap at max_frames (ensuring max_frames itself is 8n + 1)
    max_n = max(1, (max_frames - 1) // 8)
    highest_valid_max = 8 * max_n + 1
    
    if snapped > highest_valid_max:
        snapped = highest_valid_max

    return snapped


def sanitize_keyframe_image(
    input_path: Path | str,
    output_path: Path | str,
    target_width: int = 768,
    target_height: int = 512
) -> Path:
    """
    Validates and normalizes an input keyframe image:
    1. Converts RGBA / Paletted images to standard 8-bit RGB.
    2. Center-crops to target aspect ratio (target_width / target_height) without stretching.
    3. Resizes to (target_width, target_height) using high-quality Lanczos resampling.
    4. Asserts dimensions are strictly divisible by 32 (preventing latent dimension mismatch).
    """
    assert target_width % 32 == 0, f"target_width ({target_width}) must be divisible by 32"
    assert target_height % 32 == 0, f"target_height ({target_height}) must be divisible by 32"

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Keyframe image not found at: {in_path}")

    with Image.open(in_path) as img:
        # 1. Convert to RGB (strip alpha / palette)
        if img.mode != "RGB":
            img = img.convert("RGB")

        orig_w, orig_h = img.size
        target_aspect = target_width / target_height
        orig_aspect = orig_w / orig_h

        # 2. Smart center-crop to target aspect ratio
        if orig_aspect > target_aspect:
            # Image is wider than target: crop left & right
            crop_w = int(orig_h * target_aspect)
            left = (orig_w - crop_w) // 2
            top = 0
            right = left + crop_w
            bottom = orig_h
            cropped = img.crop((left, top, right, bottom))
        elif orig_aspect < target_aspect:
            # Image is taller than target: crop top & bottom
            crop_h = int(orig_w / target_aspect)
            top = (orig_h - crop_h) // 2
            left = 0
            right = orig_w
            bottom = top + crop_h
            cropped = img.crop((left, top, right, bottom))
        else:
            cropped = img

        # 3. Resize to exact target dimensions
        resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
        resized.save(out_path, format="PNG", optimize=True)

    return out_path


def validate_manifest_for_motion(
    manifest: ManifestSchema,
    project_dir: Path | str
) -> Tuple[bool, List[str]]:
    """
    Validates manifest readiness for Stage 2 motion generation.
    Enforces the human-in-the-loop gate:
    Only shots with status == ShotStatus.APPROVED are cleared for rendering.
    Returns (can_proceed, list_of_approved_or_error_messages).
    """
    proj_dir = Path(project_dir)
    approved_shots = [s for s in manifest.shots if s.status == ShotStatus.APPROVED]
    
    if not approved_shots:
        pending = [s.shot_id for s in manifest.shots if s.status == ShotStatus.PENDING_REVIEW]
        rejected = [s.shot_id for s in manifest.shots if s.status == ShotStatus.REJECTED]
        msg = "No approved shots found for motion synthesis."
        if pending:
            msg += f" Pending review: {', '.join(pending)}."
        if rejected:
            msg += f" Rejected: {', '.join(rejected)}."
        return False, [msg]

    errors = []
    for shot in approved_shots:
        if not shot.keyframe_start_path:
            errors.append(f"Shot {shot.shot_id} is approved but keyframe_start_path is null.")
            continue
        keyframe_file = proj_dir / shot.keyframe_start_path
        if not keyframe_file.exists():
            errors.append(f"Shot {shot.shot_id} keyframe image missing on disk: {keyframe_file}")

    if errors:
        return False, errors

    return True, [s.shot_id for s in approved_shots]
