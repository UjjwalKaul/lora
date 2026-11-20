"""
CLI Orchestrator for the Production AI Video Generation Pipeline.
Enforces:
- Stage subprocess process boundary isolation (100% VRAM/RAM reclamation on Windows)
- Human-in-the-loop review and approval gate before motion generation
- PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" and 85% VRAM cap
- Direct FFmpeg stream copy assembly
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from core.assembler import assemble_timeline
from core.keyframe_backend import get_keyframe_backend
from core.motion_backend import get_motion_backend
from core.preflight import sanitize_keyframe_image, snap_to_8n_plus_1, validate_manifest_for_motion
from core.schemas import ManifestSchema, ProjectConfig, ScriptSchema, ShotItem, ShotStatus, StyleProfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("video_pipeline")

WORKSPACE_ROOT = Path(__file__).resolve().parent / "workspace" / "projects"


def get_project_dir(project_name: str) -> Path:
    proj_dir = WORKSPACE_ROOT / project_name
    proj_dir.mkdir(parents=True, exist_ok=True)
    return proj_dir


def cmd_init(
    project_name: str,
    script_path: Optional[str] = None,
    style_override: Optional[str] = None,
    force_mock: bool = False
) -> ManifestSchema:
    """Initialize project directory, copy or create script.json, and generate manifest.json with StyleProfile."""
    proj_dir = get_project_dir(project_name)
    keyframes_dir = proj_dir / "keyframes"
    clips_dir = proj_dir / "clips"
    final_dir = proj_dir / "final"
    keyframes_dir.mkdir(exist_ok=True)
    clips_dir.mkdir(exist_ok=True)
    final_dir.mkdir(exist_ok=True)

    dest_script_file = proj_dir / "script.json"
    manifest_file = proj_dir / "manifest.json"

    if script_path and Path(script_path).exists():
        with open(script_path, "r", encoding="utf-8") as f:
            script_data = json.load(f)
        script = ScriptSchema.model_validate(script_data)
        with open(dest_script_file, "w", encoding="utf-8") as f:
            f.write(script.model_dump_json(indent=2))
        logger.info(f"Loaded script from {script_path}")
    elif dest_script_file.exists():
        with open(dest_script_file, "r", encoding="utf-8") as f:
            script = ScriptSchema.model_validate_json(f.read())
        logger.info(f"Loaded existing script in project: {dest_script_file}")
    else:
        # Create default sample script
        script = ScriptSchema(
            project_name=project_name,
            version="1.0.0",
            style_profile="photoreal_default",
            aspect_ratio="16:9",
            target_fps=24,
            keyframe_backend=None,
            motion_backend=None,
            global_style_prompt="cinematic film still, 35mm lens, photorealistic, high detail",
            negative_prompt="blurry, distorted, low quality, artifacts",
            shots=[
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
        )
        with open(dest_script_file, "w", encoding="utf-8") as f:
            f.write(script.model_dump_json(indent=2))
        logger.info(f"Created default template script at {dest_script_file}")

    # Load active StyleProfile
    profile_name = style_override or script.style_profile or "photoreal_default"
    logger.info(f"Resolving StyleProfile: '{profile_name}'...")
    profile = StyleProfile.load_profile(profile_name)

    # Merge StyleProfile with any explicit overrides in script.json or CLI flags
    if force_mock:
        kf_backend = "mock"
        mot_backend = "mock"
    else:
        kf_backend = script.keyframe_backend if script.keyframe_backend is not None else profile.stage1.keyframe_backend
        mot_backend = script.motion_backend if script.motion_backend is not None else profile.stage2.motion_backend

    lora_path = script.lora_path if script.lora_path is not None else profile.stage1.lora_path
    lora_scale = script.lora_scale if script.lora_scale is not None else profile.stage1.lora_scale

    config = ProjectConfig(
        target_width=768,
        target_height=512,
        target_fps=script.target_fps,
        max_frames=49,
        style_profile=profile.name,
        keyframe_backend=kf_backend,
        motion_backend=mot_backend,
        base_model=profile.stage1.base_model,
        lora_path=lora_path,
        lora_scale=lora_scale,
        trigger_tokens=profile.stage1.trigger_tokens,
        style_prompt=profile.stage1.style_prompt,
        negative_prompt=script.negative_prompt or profile.stage1.negative_prompt,
        motion_style_prompt=profile.stage2.motion_style_prompt,
        inference_steps=profile.stage2.inference_steps or 8,
        guidance_scale=profile.stage2.guidance_scale or 3.0,
        post_fx=profile.post_fx
    )

    shot_items = []
    for idx, s in enumerate(script.shots, start=1):
        snapped_frames = snap_to_8n_plus_1(s.duration_sec, config.target_fps, config.max_frames)
        shot_item = ShotItem(
            shot_id=s.shot_id,
            index=idx,
            status=ShotStatus.PENDING_REVIEW,
            shot_type=s.shot_type,
            camera_motion=s.camera_motion,
            duration_sec=s.duration_sec,
            num_frames=snapped_frames,
            conditioning_mode=s.conditioning_mode,
            prompt=s.prompt,
            motion_prompt=s.motion_prompt,
            keyframe_start_path=f"keyframes/{s.shot_id}_start.png",
            seed=s.seed if s.seed is not None else (100 * idx + 7),
            clip_output_path=f"clips/{s.shot_id}.mp4"
        )
        shot_items.append(shot_item)

    manifest = ManifestSchema(
        project_name=project_name,
        version="1.0.0",
        pipeline_stage="initialized",
        config=config,
        global_style_prompt=script.global_style_prompt or profile.stage1.style_prompt,
        negative_prompt=script.negative_prompt or profile.stage1.negative_prompt,
        shots=shot_items
    )
    manifest.save(manifest_file)
    logger.info(f"Initialized project manifest: {manifest_file} (Style: {profile.name})")
    return manifest


def cmd_status(project_name: str) -> None:
    """Print clean CLI table showing project status, style profile, and shot review state."""
    proj_dir = get_project_dir(project_name)
    manifest_file = proj_dir / "manifest.json"
    if not manifest_file.exists():
        print(f"Error: Project '{project_name}' not initialized. Run 'init' first.")
        return

    manifest = ManifestSchema.load(manifest_file)
    cfg = manifest.config

    post_fx_str = "None (-c copy stream copy)"
    if cfg.post_fx:
        fx = []
        if cfg.post_fx.stepped_fps:
            fx.append(f"stepped_fps={cfg.post_fx.stepped_fps}")
        if cfg.post_fx.chromatic_aberration_px:
            fx.append(f"rgbashift={cfg.post_fx.chromatic_aberration_px}px")
        if cfg.post_fx.lut_path:
            fx.append(f"lut={cfg.post_fx.lut_path}")
        post_fx_str = ", ".join(fx) if fx else "None"

    print("=" * 80)
    print(f"  PROJECT: {manifest.project_name.upper()} | STAGE: {manifest.pipeline_stage.upper()}")
    print(f"  Style Profile: {cfg.style_profile.upper()} | Keyframe: {cfg.keyframe_backend} | Motion: {cfg.motion_backend}")
    print(f"  Resolution: {cfg.target_width}x{cfg.target_height} @ {cfg.target_fps}fps | Max Frames: {cfg.max_frames}")
    if cfg.trigger_tokens:
        print(f"  Trigger Tokens: [{cfg.trigger_tokens}] | LoRA: {cfg.lora_path or 'None'} (Scale: {cfg.lora_scale})")
    print(f"  Post-FX Look-Dev: {post_fx_str}")
    print("=" * 80)
    print(f"{'INDEX':<6} {'SHOT ID':<12} {'STATUS':<16} {'MOTION':<18} {'FRAMES':<8} {'KEYFRAME':<12} {'CLIP'}")
    print("-" * 80)

    for s in manifest.shots:
        kf_exists = "[OK]" if (proj_dir / (s.keyframe_start_path or "")).exists() else "[MISSING]"
        clip_exists = "[OK]" if (proj_dir / (s.clip_output_path or "")).exists() else "[NONE]"
        print(f"{s.index:<6} {s.shot_id:<12} {s.status.value:<16} {s.camera_motion:<18} {s.num_frames:<8} {kf_exists:<12} {clip_exists}")

    print("=" * 80)
    master_cut = proj_dir / "final" / "master_cut.mp4"
    if master_cut.exists():
        print(f"  [MASTER CUT AVAILABLE]: {master_cut} ({master_cut.stat().st_size / 1024:.1f} KB)")
    else:
        print("  [MASTER CUT]: Not yet assembled")
    print("=" * 80)


def cmd_approve(project_name: str, shot_id: Optional[str] = None, approve_all: bool = False) -> None:
    """Approve a shot (or all shots) for motion generation."""
    proj_dir = get_project_dir(project_name)
    manifest_file = proj_dir / "manifest.json"
    manifest = ManifestSchema.load(manifest_file)

    count = 0
    for shot in manifest.shots:
        if approve_all or (shot_id and shot.shot_id == shot_id):
            shot.status = ShotStatus.APPROVED
            shot.error_message = None
            count += 1

    if count == 0:
        print(f"Warning: No matching shots found for approval (shot_id={shot_id}, approve_all={approve_all}).")
    else:
        manifest.save(manifest_file)
        print(f"[OK] Approved {count} shot(s) in project '{project_name}'.")


def cmd_reject(project_name: str, shot_id: str) -> None:
    """Mark a shot as rejected."""
    proj_dir = get_project_dir(project_name)
    manifest_file = proj_dir / "manifest.json"
    manifest = ManifestSchema.load(manifest_file)

    shot = manifest.get_shot(shot_id)
    if not shot:
        print(f"Error: Shot '{shot_id}' not found.")
        return
    shot.status = ShotStatus.REJECTED
    manifest.save(manifest_file)
    print(f"[OK] Shot '{shot_id}' marked as REJECTED.")


def cmd_reroll(project_name: str, shot_id: str, seed: Optional[int] = None, prompt: Optional[str] = None) -> None:
    """Reset a shot's status and re-render its keyframe with a new seed/prompt."""
    proj_dir = get_project_dir(project_name)
    manifest_file = proj_dir / "manifest.json"
    manifest = ManifestSchema.load(manifest_file)

    shot = manifest.get_shot(shot_id)
    if not shot:
        print(f"Error: Shot '{shot_id}' not found.")
        return

    import random
    shot.status = ShotStatus.PENDING_REVIEW
    shot.seed = seed if seed is not None else random.randint(1000, 99999)
    if prompt:
        shot.prompt = prompt

    manifest.save(manifest_file)
    print(f"[OK] Shot '{shot_id}' reset with Seed {shot.seed}. Re-rendering keyframe...")

    # Execute isolated keyframe subprocess for single shot
    run_stage_subprocess(project_name, stage="keyframes", extra_args=["--shot-id", shot_id])


# -----------------------------------------------------------------------------
# Subprocess Isolation Runners (Guarantees 100% VRAM/RAM reclamation on Windows)
# -----------------------------------------------------------------------------
def run_stage_subprocess(project_name: str, stage: str, extra_args: Optional[list[str]] = None) -> None:
    """
    Executes a pipeline stage inside an isolated OS subprocess.
    Passes PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128".
    """
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        f"--internal-exec-{stage}",
        "--project", project_name
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"Spawning isolated subprocess for stage: '{stage}'...")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Stage '{stage}' subprocess failed with exit code {res.returncode}")


def _internal_exec_keyframes(project_name: str, target_shot_id: Optional[str] = None) -> None:
    """Internal entrypoint executed inside the keyframes subprocess."""
    proj_dir = get_project_dir(project_name)
    manifest_file = proj_dir / "manifest.json"
    manifest = ManifestSchema.load(manifest_file)

    backend = get_keyframe_backend(manifest.config.keyframe_backend)
    logger.info(f"Running Keyframe Backend: {backend.backend_name}")

    for shot in manifest.shots:
        if target_shot_id and shot.shot_id != target_shot_id:
            continue
        out_path = proj_dir / (shot.keyframe_start_path or f"keyframes/{shot.shot_id}_start.png")
        logger.info(f"Generating keyframe for {shot.shot_id} (Seed {shot.seed})...")
        backend.generate_keyframe(shot, manifest.config, out_path)
        shot.keyframe_start_path = str(out_path.relative_to(proj_dir)).replace("\\", "/")

    manifest.pipeline_stage = "storyboard_complete"
    manifest.save(manifest_file)
    logger.info(f"Stage 1 Keyframing Complete. Project manifest updated at {manifest_file}")

    import psutil
    import torch
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"[STAGE 1 METRICS] Peak VRAM Allocated: {peak_vram:.2f} GB")
    mem = psutil.virtual_memory()
    proc = psutil.Process()
    logger.info(f"[STAGE 1 METRICS] Process RAM: {proc.memory_info().rss / (1024**2):.1f} MB | System RAM Free: {mem.available / (1024**3):.2f} GB")


def _internal_exec_motion(project_name: str) -> None:
    """Internal entrypoint executed inside the motion synthesis subprocess."""
    proj_dir = get_project_dir(project_name)
    manifest_file = proj_dir / "manifest.json"
    manifest = ManifestSchema.load(manifest_file)

    can_proceed, messages = validate_manifest_for_motion(manifest, proj_dir)
    if not can_proceed:
        logger.error(f"Cannot execute motion stage:\n" + "\n".join(messages))
        sys.exit(1)

    backend = get_motion_backend(manifest.config.motion_backend)
    logger.info(f"Running Motion Backend: {backend.backend_name}")

    for shot in manifest.shots:
        if shot.status != ShotStatus.APPROVED:
            continue

        raw_keyframe = proj_dir / shot.keyframe_start_path
        sanitized_keyframe = proj_dir / "keyframes" / f"{shot.shot_id}_sanitized.png"

        # Step 1: Pre-flight sanitize keyframe geometry
        sanitize_keyframe_image(
            input_path=raw_keyframe,
            output_path=sanitized_keyframe,
            target_width=manifest.config.target_width,
            target_height=manifest.config.target_height
        )

        # Step 2: Render motion
        out_clip = proj_dir / (shot.clip_output_path or f"clips/{shot.shot_id}.mp4")
        shot.status = ShotStatus.RENDERING
        manifest.save(manifest_file)

        try:
            backend.generate_shot(
                shot=shot,
                config=manifest.config,
                output_clip_path=out_clip,
                sanitized_keyframe_path=sanitized_keyframe
            )
            shot.status = ShotStatus.COMPLETED
            shot.clip_output_path = str(out_clip.relative_to(proj_dir)).replace("\\", "/")
            shot.error_message = None
        except Exception as e:
            logger.error(f"Failed rendering shot {shot.shot_id}: {e}", exc_info=True)
            shot.status = ShotStatus.FAILED
            shot.error_message = str(e)

        manifest.save(manifest_file)

    manifest.pipeline_stage = "motion_complete"
    manifest.save(manifest_file)
    logger.info("Stage 2 Motion Synthesis Complete.")

    import psutil
    import torch
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"[STAGE 2 METRICS] Peak VRAM Allocated: {peak_vram:.2f} GB")
    mem = psutil.virtual_memory()
    proc = psutil.Process()
    logger.info(f"[STAGE 2 METRICS] Process RAM: {proc.memory_info().rss / (1024**2):.1f} MB | System RAM Free: {mem.available / (1024**3):.2f} GB")


def cmd_keyframes(project_name: str) -> None:
    """CLI action: trigger keyframe generation in subprocess."""
    run_stage_subprocess(project_name, stage="keyframes")


def cmd_motion(project_name: str) -> None:
    """CLI action: trigger motion synthesis in subprocess."""
    run_stage_subprocess(project_name, stage="motion")


def cmd_assemble(project_name: str) -> Path:
    """CLI action: trigger timeline assembly via FFmpeg."""
    proj_dir = get_project_dir(project_name)
    manifest_file = proj_dir / "manifest.json"
    manifest = ManifestSchema.load(manifest_file)

    logger.info("Running Stage 3: Direct FFmpeg Timeline Assembler...")
    master_cut_path = assemble_timeline(manifest, proj_dir)
    manifest.pipeline_stage = "master_complete"
    manifest.save(manifest_file)
    logger.info(f"Assembly Complete! Master cut saved to: {master_cut_path}")
    return master_cut_path


def cmd_run_all(project_name: str, auto_approve: bool = True) -> None:
    """Run keyframes -> approve -> motion -> assemble end-to-end."""
    logger.info(f"Starting complete end-to-end run for '{project_name}'...")
    cmd_keyframes(project_name)
    if auto_approve:
        logger.info("Auto-approving all shots for end-to-end execution...")
        cmd_approve(project_name, approve_all=True)
    cmd_motion(project_name)
    final_cut = cmd_assemble(project_name)
    cmd_status(project_name)
    logger.info(f"[SUCCESS] End-to-end execution finished. Final video: {final_cut}")


def main():
    parser = argparse.ArgumentParser(description="Production AI Video Generation Pipeline Orchestrator")
    parser.add_argument("--project", "-p", required=True, help="Project name inside workspace/projects/")

    # User subcommands
    parser.add_argument("--stage", choices=["init", "keyframes", "status", "approve", "reject", "reroll", "motion", "assemble", "run-all"],
                        help="Pipeline stage or command to execute")
    parser.add_argument("--script", help="Path to input script.json (for init)")
    parser.add_argument("--style", help="StyleProfile name or path to override script style (e.g. photoreal_default, comic_25d)")
    parser.add_argument("--mock", action="store_true", help="Force mock keyframe and motion backends for testing (0 MB VRAM)")
    parser.add_argument("--shot-id", help="Target shot ID (for approve, reject, reroll)")
    parser.add_argument("--approve-all", action="store_true", help="Approve all shots")
    parser.add_argument("--seed", type=int, help="Seed value (for reroll)")
    parser.add_argument("--prompt", help="New prompt (for reroll)")

    # Internal subprocess flags
    parser.add_argument("--internal-exec-keyframes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-exec-motion", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Route internal subprocess execution
    if args.internal_exec_keyframes:
        _internal_exec_keyframes(args.project, args.shot_id)
        return
    if args.internal_exec_motion:
        _internal_exec_motion(args.project)
        return

    # Route public CLI commands
    stage = args.stage or "status"
    if stage == "init":
        cmd_init(args.project, args.script, args.style, force_mock=args.mock)
    elif stage == "keyframes":
        cmd_keyframes(args.project)
    elif stage == "status":
        cmd_status(args.project)
    elif stage == "approve":
        cmd_approve(args.project, args.shot_id, args.approve_all)
    elif stage == "reject":
        if not args.shot_id:
            print("Error: --shot-id is required for reject.")
            sys.exit(1)
        cmd_reject(args.project, args.shot_id)
    elif stage == "reroll":
        if not args.shot_id:
            print("Error: --shot-id is required for reroll.")
            sys.exit(1)
        cmd_reroll(args.project, args.shot_id, args.seed, args.prompt)
    elif stage == "motion":
        cmd_motion(args.project)
    elif stage == "assemble":
        cmd_assemble(args.project)
    elif stage == "run-all":
        cmd_run_all(args.project, auto_approve=True)


if __name__ == "__main__":
    main()
