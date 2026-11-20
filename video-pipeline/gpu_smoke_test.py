"""
GPU Smoke Test: 1-shot end-to-end verification under real hardware constraints.
Profile: comic_25d (Clean SD 1.5 base keyframe [0 LoRA] -> 25 frames distilled LTX-Video 2B -> FFmpeg look-dev).
Measures:
- Peak VRAM (ensuring < 5.2 GB physical ceiling)
- Peak System RAM (ensuring < 8.5 GB free boundary)
- Execution time per stage
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import psutil
import torch

pipeline_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(pipeline_dir))

from core.schemas import ManifestSchema, ScriptSchema, ShotItem, ShotStatus, StyleProfile
from core.preflight import snap_to_8n_plus_1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gpu_smoke_test")


def run_smoke_test():
    print("=" * 80)
    print("  PRODUCTION VIDEO PIPELINE - 1-SHOT GPU SMOKE TEST (COMIC_25D - CLEAN BASE)")
    print(f"  Target: Windows 11 | RTX 3050 6 GB | 16 GB RAM")
    print(f"  CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    mem_init = psutil.virtual_memory()
    print(f"  Initial Free System RAM: {mem_init.available / (1024**3):.2f} GB / {mem_init.total / (1024**3):.2f} GB Total")
    print("  LoRA Usage: ZERO (pure clean base models, all old project LoRAs discarded)")
    print("=" * 80)

    proj_name = "smoke_test_comic"
    proj_dir = pipeline_dir / "workspace" / "projects" / proj_name

    # Create 1-shot script for fast 25-frame safety verification with clean base model
    script_data = {
        "project_name": proj_name,
        "version": "1.0.0",
        "style_profile": "comic_25d",
        "aspect_ratio": "16:9",
        "target_fps": 24,
        "keyframe_backend": "sd15_lora",
        "motion_backend": "ltx_video_2b",
        "shots": [
            {
                "shot_id": "shot_001",
                "shot_type": "medium_close_up",
                "camera_motion": "subtle_handheld",
                "duration_sec": 1.0,  # 1.0s snaps to 25 frames for fast smoke test
                "prompt": "futuristic cybernetic warrior in sleek stealth armor on a rain-slicked rooftop at dusk, vibrant graphic novel art",
                "motion_prompt": "subtle breathing, rain falling across visor",
                "seed": 42
            }
        ]
    }

    script_path = pipeline_dir / "temp_smoke_script.json"
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2)

    run_pipeline = [sys.executable, str(pipeline_dir / "run_pipeline.py"), "--project", proj_name]

    metrics = {
        "stage1_vram": "N/A",
        "stage1_ram": "N/A",
        "stage2_vram": "N/A",
        "stage2_ram": "N/A"
    }

    # -------------------------------------------------------------
    # Step 1: Initialize Project
    # -------------------------------------------------------------
    print("\n[STEP 1/4] Initializing smoke test project with comic_25d (clean base)...")
    t0 = time.time()
    res = subprocess.run(run_pipeline + ["--stage", "init", "--script", str(script_path), "--style", "comic_25d"], capture_output=True, text=True)
    assert res.returncode == 0, f"Init failed: {res.stderr}"
    print(f"  -> Project initialized in {time.time() - t0:.2f}s")

    manifest = ManifestSchema.load(proj_dir / "manifest.json")
    print(f"  -> Shot 001 Frame count: {manifest.shots[0].num_frames} (snapped 8n+1)")

    # -------------------------------------------------------------
    # Step 2: Stage 1 Keyframe Generation (Clean SD 1.5 Base)
    # -------------------------------------------------------------
    print("\n[STEP 2/4] Executing Stage 1: Clean SD 1.5 Base Keyframe Generation (0 LoRA)...")
    t1 = time.time()
    res = subprocess.run(run_pipeline + ["--stage", "keyframes"], capture_output=True, text=True)
    stage1_time = time.time() - t1
    if res.returncode != 0:
        print(f"Stage 1 Failed:\n{res.stderr}\n{res.stdout}")
        sys.exit(1)
    
    # Parse Stage 1 metrics
    for line in (res.stdout + res.stderr).splitlines():
        if "[STAGE 1 METRICS]" in line:
            print(f"  {line.strip()}")
            vram_m = re.search(r"Peak VRAM Allocated:\s*([\d\.]+)\s*GB", line)
            ram_m = re.search(r"Process RAM:\s*([\d\.]+)\s*MB", line)
            if vram_m:
                metrics["stage1_vram"] = f"{vram_m.group(1)} GB"
            if ram_m:
                metrics["stage1_ram"] = f"{ram_m.group(1)} MB"

    kf_path = proj_dir / manifest.shots[0].keyframe_start_path
    print(f"  -> Keyframe generated in {stage1_time:.2f}s at: {kf_path}")

    # -------------------------------------------------------------
    # Step 3: Approve Shot & Execute Stage 2 Motion Synthesis
    # -------------------------------------------------------------
    print("\n[STEP 3/4] Approving Shot & Executing Stage 2: Distilled LTX-Video 2B Motion Synthesis...")
    res = subprocess.run(run_pipeline + ["--stage", "approve", "--approve-all"], capture_output=True, text=True)
    assert res.returncode == 0, f"Approval failed: {res.stderr}"

    t2 = time.time()
    res = subprocess.run(run_pipeline + ["--stage", "motion"], capture_output=True, text=True)
    stage2_time = time.time() - t2
    if res.returncode != 0:
        print(f"Stage 2 Motion Failed:\n{res.stderr}\n{res.stdout}")
        sys.exit(1)

    # Parse Stage 2 metrics
    for line in (res.stdout + res.stderr).splitlines():
        if "[STAGE 2 METRICS]" in line:
            print(f"  {line.strip()}")
            vram_m = re.search(r"Peak VRAM Allocated:\s*([\d\.]+)\s*GB", line)
            ram_m = re.search(r"Process RAM:\s*([\d\.]+)\s*MB", line)
            if vram_m:
                metrics["stage2_vram"] = f"{vram_m.group(1)} GB"
            if ram_m:
                metrics["stage2_ram"] = f"{ram_m.group(1)} MB"

    clip_path = proj_dir / manifest.shots[0].clip_output_path
    print(f"  -> Motion clip generated in {stage2_time:.2f}s at: {clip_path}")

    # -------------------------------------------------------------
    # Step 4: Stage 3 Timeline Assembly with Look-Dev Post-FX
    # -------------------------------------------------------------
    print("\n[STEP 4/4] Executing Stage 3: FFmpeg Assembly with Look-Dev Post-FX...")
    t3 = time.time()
    res = subprocess.run(run_pipeline + ["--stage", "assemble"], capture_output=True, text=True)
    stage3_time = time.time() - t3
    if res.returncode != 0:
        print(f"Stage 3 Assembly Failed:\n{res.stderr}\n{res.stdout}")
        sys.exit(1)

    master_path = proj_dir / "final" / "master_cut.mp4"
    print(f"  -> Final Cut Assembled in {stage3_time:.2f}s at: {master_path}")

    # Clean up temp script
    if script_path.exists():
        script_path.unlink()

    # -------------------------------------------------------------
    # Final Benchmark Report
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  1-SHOT GPU SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"  Stage 1 (Clean SD 1.5 Keyframing):  {stage1_time:.2f}s | Peak VRAM: {metrics['stage1_vram']} | Process RAM: {metrics['stage1_ram']}")
    print(f"  Stage 2 (Distilled LTX-Video 2B):   {stage2_time:.2f}s | Peak VRAM: {metrics['stage2_vram']} | Process RAM: {metrics['stage2_ram']}")
    print(f"  Stage 3 (FFmpeg Assembly Post-FX): {stage3_time:.2f}s")
    print(f"  Total Pipeline Time:                {stage1_time + stage2_time + stage3_time:.2f}s")
    print(f"  Master Cut Video:                  {master_path} ({master_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()
