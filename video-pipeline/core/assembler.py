"""
Stage 3: Direct FFmpeg Timeline Assembler.
Concatenates individual shot MP4 clips into the final master cut video.
Uses stream copy (-c copy) for zero-memory, lossless concatenation.
Falls back to deterministic H.264 re-encoding if streams are inhomogeneous.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from core.schemas import ManifestSchema, ShotStatus

logger = logging.getLogger(__name__)


def get_ffmpeg_exe() -> str:
    """
    Locates the FFmpeg executable.
    First checks system PATH; if not found, falls back to bundled imageio-ffmpeg binary.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.exists(bundled):
            return bundled
    except ImportError:
        pass

    raise RuntimeError(
        "FFmpeg binary not found on PATH or via imageio-ffmpeg. "
        "Please ensure imageio-ffmpeg is installed or ffmpeg is on system PATH."
    )


def assemble_timeline(
    manifest: ManifestSchema,
    project_dir: Path | str,
    output_filename: str = "master_cut.mp4"
) -> Path:
    """
    Assembles all completed video clips listed in the manifest in shot index order.
    Executes via native FFmpeg binary subprocess (zero Python memory overhead).
    """
    proj_dir = Path(project_dir).resolve()
    final_dir = proj_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    master_cut_path = final_dir / output_filename

    # Sort shots by index
    sorted_shots = sorted(manifest.shots, key=lambda s: s.index)
    completed_shots = [s for s in sorted_shots if s.status == ShotStatus.COMPLETED]

    if not completed_shots:
        raise ValueError("No completed shots found in manifest to assemble.")

    clip_paths: List[Path] = []
    for shot in completed_shots:
        if not shot.clip_output_path:
            raise FileNotFoundError(f"Shot {shot.shot_id} marked completed but has no clip_output_path.")
        clip_file = proj_dir / shot.clip_output_path
        if not clip_file.exists():
            raise FileNotFoundError(f"Clip file for shot {shot.shot_id} not found at {clip_file}")
        clip_paths.append(clip_file)

    # Generate FFmpeg concat list file
    concat_list_file = final_dir / "clips.txt"
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for clip in clip_paths:
            # Use forward slashes in FFmpeg demuxer list for Windows compatibility
            escaped_path = str(clip.resolve()).replace("\\", "/")
            f.write(f"file '{escaped_path}'\n")

    ffmpeg_bin = get_ffmpeg_exe()

    # Step 1: Try zero-loss, zero-memory stream copy
    copy_cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file.resolve()),
        "-c", "copy",
        str(master_cut_path.resolve())
    ]

    result = subprocess.run(copy_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.warning(
            f"FFmpeg stream copy failed (incompatible streams?). "
            f"Falling back to deterministic H.264 re-encode. Error:\n{result.stderr}"
        )
        # Step 2: Fallback deterministic re-encode
        reencode_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_file.resolve()),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "fast",
            str(master_cut_path.resolve())
        ]
        re_result = subprocess.run(reencode_cmd, capture_output=True, text=True)
        if re_result.returncode != 0:
            raise RuntimeError(f"FFmpeg re-encode failed:\n{re_result.stderr}")

    if not master_cut_path.exists() or master_cut_path.stat().st_size == 0:
        raise RuntimeError(f"Assembled video file is empty or was not created at {master_cut_path}")

    return master_cut_path
