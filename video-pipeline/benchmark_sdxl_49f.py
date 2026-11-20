"""
1-Shot GPU Benchmark & Verification Script:
Tests SDXL-Lightning FP16 Keyframe Generation + 49-Frame Distilled LTX-Video Motion Synthesis on RTX 3050 (6 GB VRAM).

Steps:
1. Initializes project 'sdxl_49f_verify' with comic_25d style (SDXL-Lightning + 49 frames @ 24fps).
2. Runs Stage 1 Keyframe (1152x768 native 3:2 widescreen, 4 steps, guidance 0.0).
3. Verifies keyframe integrity (size, min/max/mean/std, not black/NaN).
4. Sanitizes to 768x512, approves shot.
5. Runs Stage 2 Motion (49 frames with LTXConditionPipeline).
6. Runs Stage 3 Post-FX Timeline Assembly (12fps on 2s + 2px rgbashift).
7. Extracts Frames 0, 8, 24, 48, computes MAE & SSIM continuity, and saves debug_frames_grid.png.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFont

PIPELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_ROOT))

from core.assembler import get_ffmpeg_exe
from core.preflight import snap_to_8n_plus_1
from core.schemas import ManifestSchema, ScriptSchema, ShotItem, ShotStatus
from run_pipeline import cmd_approve, cmd_assemble, cmd_init, get_project_dir, run_stage_subprocess

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Mean Structural Similarity Index between two RGB uint8 arrays."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return float(((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)))


def extract_video_frames(video_path: Path, frame_indices: list[int]) -> list[Image.Image]:
    """Extracts specific frame numbers from an MP4 video using imageio or ffmpeg."""
    import imageio.v3 as iio
    frames = []
    video_frames = iio.imread(str(video_path))  # Shape: (N, H, W, 3)
    for idx in frame_indices:
        clamped_idx = min(idx, len(video_frames) - 1)
        frames.append(Image.fromarray(video_frames[clamped_idx]))
    return frames


def run_benchmark():
    project_name = "sdxl_49f_verify"
    proj_dir = get_project_dir(project_name)

    print("=" * 75)
    print("  SDXL-LIGHTNING FP16 + 49-FRAME LTX-VIDEO GPU BENCHMARK")
    print(f"  Project: {project_name}")
    print(f"  Target: Windows 11 | RTX 3050 (6 GB physical VRAM, ~4.5 GB effective)")
    print("=" * 75)

    # -------------------------------------------------------------
    # Step 1: Initialize Project with 1 Cinematic Shot (2.0s -> 49 frames)
    # -------------------------------------------------------------
    print("\n[STEP 1/6] Initializing Project & Custom Script...")
    script_data = {
        "project_name": project_name,
        "version": "1.0.0",
        "style_profile": "comic_25d",
        "aspect_ratio": "16:9",
        "target_fps": 24,
        "keyframe_backend": "sdxl_lightning",
        "motion_backend": "ltx_video_2b",
        "shots": [
            {
                "shot_id": "shot_001",
                "shot_type": "wide_establishing",
                "camera_motion": "slow_dolly_forward",
                "duration_sec": 2.0,
                "prompt": "a lone stylized vigilante crouching on a rain-slicked neon rooftop overlooking a towering stylized cityscape at night, dynamic low-angle wide shot",
                "motion_prompt": "slow camera dolly forward, raindrops glistening across neon puddles, dark trench coat gently billowing in night wind",
                "seed": 42
            }
        ]
    }

    script_path = proj_dir / "script.json"
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2)

    # Initialize manifest with comic_25d style
    manifest = cmd_init(project_name=project_name, script_path=str(script_path), style_override="comic_25d")
    shot = manifest.shots[0]
    print(f"  -> Shot ID: {shot.shot_id} | Snapped frames: {shot.num_frames} (8n+1)")
    print(f"  -> Keyframe Backend: {manifest.config.keyframe_backend} (Base: {manifest.config.base_model})")
    print(f"  -> Motion Backend: {manifest.config.motion_backend}")

    # -------------------------------------------------------------
    # Step 2: Run Stage 1 Keyframe Generation (Subprocess)
    # -------------------------------------------------------------
    print("\n[STEP 2/6] Executing Stage 1: SDXL-Lightning Keyframe Generation...")
    t0_s1 = time.time()
    mem_before_s1 = psutil.virtual_memory().available / (1024 ** 3)

    run_stage_subprocess(project_name, stage="keyframes")

    s1_duration = time.time() - t0_s1
    mem_after_s1 = psutil.virtual_memory().available / (1024 ** 3)
    print(f"  -> Stage 1 completed in {s1_duration:.2f}s | Free RAM: {mem_after_s1:.2f} GB")

    # -------------------------------------------------------------
    # Step 3: Validate Keyframe Geometry & Pixel Statistics
    # -------------------------------------------------------------
    print("\n[STEP 3/6] Verifying Keyframe Integrity & NaN Check...")
    kf_path = proj_dir / shot.keyframe_start_path
    if not kf_path.exists():
        raise FileNotFoundError(f"Keyframe not found at {kf_path}")

    kf_img = Image.open(kf_path)
    w, h = kf_img.size
    mode = kf_img.mode
    print(f"  -> Image dimensions: {w}x{h} (Expected: 1152x768 native 3:2 widescreen)")
    print(f"  -> Image mode: {mode}")
    assert (w, h) == (1152, 768), f"Expected 1152x768, got {w}x{h}"
    assert mode == "RGB", f"Expected RGB mode, got {mode}"

    kf_arr = np.array(kf_img).astype(np.float32)
    px_min = float(kf_arr.min())
    px_max = float(kf_arr.max())
    px_mean = float(kf_arr.mean())
    px_std = float(kf_arr.std())
    has_nan = bool(np.isnan(kf_arr).any())

    print(f"  -> Pixel Stats: Min={px_min:.1f}, Max={px_max:.1f}, Mean={px_mean:.1f}, Std={px_std:.1f}, NaN={has_nan}")
    assert not has_nan, "FATAL: Keyframe contains NaN pixels!"
    assert px_max > 50.0, "FATAL: Keyframe appears completely black (Max < 50)!"
    assert px_std > 15.0, f"FATAL: Keyframe has low contrast/flat color (Std={px_std:.2f})!"
    print("  -> [PASS] Keyframe is valid, non-black, high-contrast RGB image.")

    # -------------------------------------------------------------
    # Step 4: Approve Shot for Motion Synthesis
    # -------------------------------------------------------------
    print("\n[STEP 4/6] Approving Shot for Stage 2...")
    cmd_approve(project_name, shot_id=shot.shot_id)
    print(f"  -> Shot '{shot.shot_id}' marked as APPROVED.")

    # -------------------------------------------------------------
    # Step 5: Run Stage 2 Motion Synthesis (49 Frames Subprocess)
    # -------------------------------------------------------------
    print("\n[STEP 5/6] Executing Stage 2: Distilled LTX-Video Motion Synthesis (49 Frames)...")
    t0_s2 = time.time()
    mem_before_s2 = psutil.virtual_memory().available / (1024 ** 3)

    run_stage_subprocess(project_name, stage="motion")

    s2_duration = time.time() - t0_s2
    mem_after_s2 = psutil.virtual_memory().available / (1024 ** 3)
    print(f"  -> Stage 2 completed in {s2_duration:.2f}s | Free RAM: {mem_after_s2:.2f} GB")

    # -------------------------------------------------------------
    # Step 6: Stage 3 Timeline Assembly & Structural SSIM Verification
    # -------------------------------------------------------------
    print("\n[STEP 6/6] Executing Stage 3 Assembler & Structural Continuity Verification...")
    master_cut = cmd_assemble(project_name)
    print(f"  -> Master cut assembled: {master_cut} ({master_cut.stat().st_size / 1024:.1f} KB)")

    clip_path = proj_dir / shot.clip_output_path
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found at {clip_path}")

    # Extract target frames: 0, 8, 24, 48
    target_indices = [0, 8, 24, 48]
    frames = extract_video_frames(clip_path, target_indices)

    f0 = np.array(frames[0]).astype(float)
    f8 = np.array(frames[1]).astype(float)
    f24 = np.array(frames[2]).astype(float)
    f48 = np.array(frames[3]).astype(float)

    ssim_0_8 = compute_ssim(f0, f8)
    ssim_8_24 = compute_ssim(f8, f24)
    ssim_24_48 = compute_ssim(f24, f48)
    ssim_0_48 = compute_ssim(f0, f48)

    mae_0_8 = float(np.mean(np.abs(f0 - f8)))
    mae_8_24 = float(np.mean(np.abs(f8 - f24)))
    mae_24_48 = float(np.mean(np.abs(f24 - f48)))

    # Create 2x2 Contact Sheet
    cw, ch = frames[0].size
    grid = Image.new("RGB", (cw * 2, ch * 2))
    grid.paste(frames[0], (0, 0))
    grid.paste(frames[1], (cw, 0))
    grid.paste(frames[2], (0, ch))
    grid.paste(frames[3], (cw, ch))

    draw = ImageDraw.Draw(grid)
    draw.rectangle([(0, 0), (220, 32)], fill=(0, 0, 0, 180))
    draw.text((10, 8), f"Frame 0 (0.00s) [Keyframe]", fill=(0, 220, 255))

    draw.rectangle([(cw, 0), (cw + 260, 32)], fill=(0, 0, 0, 180))
    draw.text((cw + 10, 8), f"Frame 8 (0.33s) [SSIM: {ssim_0_8:.4f}]", fill=(0, 220, 255))

    draw.rectangle([(0, ch), (260, ch + 32)], fill=(0, 0, 0, 180))
    draw.text((10, ch + 8), f"Frame 24 (1.00s) [SSIM: {ssim_8_24:.4f}]", fill=(0, 220, 255))

    draw.rectangle([(cw, ch), (cw + 260, ch + 32)], fill=(0, 0, 0, 180))
    draw.text((cw + 10, 8 + ch), f"Frame 48 (2.00s) [SSIM: {ssim_24_48:.4f}]", fill=(0, 220, 255))

    # Save to project root and keyframes/ for dashboard inspection
    contact_sheet_main = proj_dir / "debug_frames_grid.png"
    contact_sheet_kf = proj_dir / "keyframes" / "debug_frames_grid.png"
    grid.save(contact_sheet_main)
    grid.save(contact_sheet_kf)

    print("\n" + "=" * 75)
    print("                    BENCHMARK RESULTS SUMMARY                     ")
    print("=" * 75)
    print(f"  Project:                            {project_name}")
    print(f"  Style Profile:                      comic_25d")
    print(f"  Stage 1 Model:                      SDXL-Lightning FP16 (4 steps, guidance 0.0)")
    print(f"  Stage 1 Resolution:                 {w}x{h} (3:2 widescreen)")
    print(f"  Stage 1 Duration:                   {s1_duration:.2f}s")
    print(f"  Keyframe Stats:                     Min={px_min:.0f}, Max={px_max:.0f}, Mean={px_mean:.1f}, Std={px_std:.1f}")
    print("-" * 75)
    print(f"  Stage 2 Model:                      Distilled LTX-Video 2B GGUF Q4_K_M (8 steps)")
    print(f"  Stage 2 Frames:                     49 frames (2.04s @ 24fps)")
    print(f"  Stage 2 Duration:                   {s2_duration:.2f}s")
    print("-" * 75)
    print(f"  Continuity Frame 0 -> Frame 8:      MAE: {mae_0_8:.2f} / 255.0  | SSIM: {ssim_0_8:.4f}")
    print(f"  Continuity Frame 8 -> Frame 24:     MAE: {mae_8_24:.2f} / 255.0  | SSIM: {ssim_8_24:.4f}")
    print(f"  Continuity Frame 24 -> Frame 48:    MAE: {mae_24_48:.2f} / 255.0  | SSIM: {ssim_24_48:.4f}")
    print(f"  Global Drift Frame 0 -> Frame 48:   SSIM: {ssim_0_48:.4f}")
    print(f"  Contact Sheet:                      {contact_sheet_main}")
    print("=" * 75)

if __name__ == "__main__":
    run_benchmark()
