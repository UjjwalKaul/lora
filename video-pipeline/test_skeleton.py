"""
Integration test suite for the Batch 1 pipeline skeleton.
Verifies:
1. snap_to_8n_plus_1 calculation & capping
2. Keyframe geometry & alpha sanitization
3. CLI Lifecycle: init -> keyframes -> approval gate enforcement -> approve -> motion -> assemble
4. FFmpeg stream concatenation & master cut generation
All with 0 MB VRAM and zero model weights.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image

# Add video-pipeline to sys.path
pipeline_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(pipeline_dir))

from core.preflight import snap_to_8n_plus_1, sanitize_keyframe_image
from core.assembler import get_ffmpeg_exe


def test_8n_plus_1_snapping():
    print("[TEST 1/5] Testing 8n + 1 frame budgeting...")
    assert snap_to_8n_plus_1(2.0, fps=24, max_frames=49) == 49, "2.0s @ 24fps should snap to 49"
    assert snap_to_8n_plus_1(1.0, fps=24, max_frames=49) == 25, "1.0s @ 24fps should snap to 25"
    assert snap_to_8n_plus_1(0.5, fps=24, max_frames=49) == 9, "0.5s @ 24fps should snap to 9 (minimum)"
    # Capping test: 3.0s @ 24fps is 72 frames -> nearest is 73, but capped at max_frames 49
    assert snap_to_8n_plus_1(3.0, fps=24, max_frames=49) == 49, "3.0s capped at 49"
    # When max_frames=73
    assert snap_to_8n_plus_1(3.0, fps=24, max_frames=73) == 73, "3.0s capped at 73"
    print("  -> Passed 8n + 1 frame calculations!")


def test_sanitizer():
    print("[TEST 2/5] Testing keyframe geometry & alpha stripping...")
    test_img_path = pipeline_dir / "temp_test_rgba.png"
    out_sanitized_path = pipeline_dir / "temp_test_sanitized.png"

    # Create dummy RGBA image with non-standard dimensions
    img = Image.new("RGBA", (1000, 600), color=(255, 100, 50, 180))
    img.save(test_img_path)

    try:
        sanitize_keyframe_image(test_img_path, out_sanitized_path, target_width=768, target_height=512)
        with Image.open(out_sanitized_path) as res:
            assert res.mode == "RGB", f"Expected RGB mode, got {res.mode}"
            assert res.size == (768, 512), f"Expected (768, 512), got {res.size}"
            assert res.size[0] % 32 == 0, "Width not divisible by 32"
            assert res.size[1] % 32 == 0, "Height not divisible by 32"
        print("  -> Passed keyframe sanitization & divisibility by 32!")
    finally:
        if test_img_path.exists():
            test_img_path.unlink()
        if out_sanitized_path.exists():
            out_sanitized_path.unlink()


def test_ffmpeg_resolution():
    print("[TEST 3/5] Testing FFmpeg binary detection...")
    ffmpeg_bin = get_ffmpeg_exe()
    assert os.path.exists(ffmpeg_bin), f"FFmpeg path does not exist: {ffmpeg_bin}"
    res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True)
    assert res.returncode == 0, f"FFmpeg execution failed: {res.stderr}"
    first_line = res.stdout.splitlines()[0] if res.stdout else "unknown"
    print(f"  -> Detected FFmpeg binary: {ffmpeg_bin} ({first_line})")


def test_cli_lifecycle():
    print("[TEST 4/5] Testing full CLI lifecycle & approval gate...")
    proj_name = "integration_test_project"
    proj_dir = pipeline_dir / "workspace" / "projects" / proj_name

    # Clean previous test dir if exists
    if proj_dir.exists():
        shutil.rmtree(proj_dir)

    run_pipeline = [sys.executable, str(pipeline_dir / "run_pipeline.py"), "--project", proj_name]

    # Step A: Init
    res = subprocess.run(run_pipeline + ["--stage", "init"], capture_output=True, text=True)
    assert res.returncode == 0, f"Init failed: {res.stderr}"
    manifest_file = proj_dir / "manifest.json"
    assert manifest_file.exists(), "manifest.json was not created"

    # Step B: Keyframes
    res = subprocess.run(run_pipeline + ["--stage", "keyframes"], capture_output=True, text=True)
    assert res.returncode == 0, f"Keyframes stage failed: {res.stderr}"
    kf1 = proj_dir / "keyframes" / "shot_001_start.png"
    kf2 = proj_dir / "keyframes" / "shot_002_start.png"
    assert kf1.exists() and kf2.exists(), "Keyframe images were not generated"

    # Step C: Verify Approval Gate (Stage 2 should refuse unapproved shots)
    res = subprocess.run(run_pipeline + ["--stage", "motion"], capture_output=True, text=True)
    assert res.returncode != 0, "Motion stage should have failed because shots are unapproved!"
    assert "No approved shots found" in (res.stderr + res.stdout), "Expected approval gate refusal message"
    print("  -> Human-in-the-loop approval gate successfully blocked unapproved rendering!")

    # Step D: Approve shot_001, reroll shot_002, then approve shot_002
    res = subprocess.run(run_pipeline + ["--stage", "approve", "--shot-id", "shot_001"], capture_output=True, text=True)
    assert res.returncode == 0, f"Approve failed: {res.stderr}"

    res = subprocess.run(run_pipeline + ["--stage", "reroll", "--shot-id", "shot_002", "--seed", "777"], capture_output=True, text=True)
    assert res.returncode == 0, f"Reroll failed: {res.stderr}"

    res = subprocess.run(run_pipeline + ["--stage", "approve", "--shot-id", "shot_002"], capture_output=True, text=True)
    assert res.returncode == 0, f"Approve shot 2 failed: {res.stderr}"

    # Step E: Motion synthesis (Mock)
    res = subprocess.run(run_pipeline + ["--stage", "motion"], capture_output=True, text=True)
    assert res.returncode == 0, f"Motion failed: {res.stderr}\n{res.stdout}"
    clip1 = proj_dir / "clips" / "shot_001.mp4"
    clip2 = proj_dir / "clips" / "shot_002.mp4"
    assert clip1.exists() and clip1.stat().st_size > 0, "Clip 1 was not generated"
    assert clip2.exists() and clip2.stat().st_size > 0, "Clip 2 was not generated"
    print("  -> Motion clips successfully synthesized!")

    # Step F: Assemble Timeline
    res = subprocess.run(run_pipeline + ["--stage", "assemble"], capture_output=True, text=True)
    assert res.returncode == 0, f"Assembly failed: {res.stderr}\n{res.stdout}"
    master_cut = proj_dir / "final" / "master_cut.mp4"
    assert master_cut.exists() and master_cut.stat().st_size > 0, "Master cut MP4 was not created"
    print(f"  -> Final cut assembled successfully ({master_cut.stat().st_size / 1024:.1f} KB)!")

    # Step G: Check Status
    res = subprocess.run(run_pipeline + ["--stage", "status"], capture_output=True, text=True)
    assert res.returncode == 0, f"Status failed: {res.stderr}"
    assert "MASTER CUT AVAILABLE" in res.stdout, "Status does not show master cut"
    print("  -> Passed full CLI lifecycle!")


def main():
    print("=" * 60)
    print("  PRODUCTION VIDEO GENERATION PIPELINE - BATCH 1 TEST")
    print("=" * 60)
    test_8n_plus_1_snapping()
    test_sanitizer()
    test_ffmpeg_resolution()
    test_cli_lifecycle()
    print("=" * 60)
    print("  [ALL BATCH 1 SKELETON TESTS PASSED SUCCESSFULLY!]")
    print("=" * 60)


if __name__ == "__main__":
    main()
