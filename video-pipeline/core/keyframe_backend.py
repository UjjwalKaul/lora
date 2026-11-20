"""
Stage 1: Pluggable Keyframe Generation Interfaces & Implementations.
Includes:
- KeyframeBackend (ABC)
- MockKeyframeBackend (0 MB VRAM, zero model downloads for skeleton testing)
- SDXLLightningBackend (Modern 4-step UNet with sequential CPU offload, ~2.8 GB VRAM)
- SD15LoRABackend (SD 1.5 with optional generic custom LoRA adapter and sequential CPU offload, ~2.2 GB VRAM)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

from core.schemas import ProjectConfig, ShotItem

logger = logging.getLogger(__name__)


class KeyframeBackend(ABC):
    """Abstract Base Class for Stage 1 Keyframe Generation."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Unique identifier for this backend."""
        pass

    @abstractmethod
    def generate_keyframe(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_path: Path | str
    ) -> Path:
        """Generates a keyframe image for the given shot and saves it to output_path."""
        pass


class MockKeyframeBackend(KeyframeBackend):
    """
    Zero-VRAM, zero-model testing backend.
    Generates a stylized cinematic storyboard card with camera framing,
    shot metadata, and prompt text using pure PIL.
    """

    @property
    def backend_name(self) -> str:
        return "mock"

    def generate_keyframe(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_path: Path | str
    ) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        w = config.target_width
        h = config.target_height

        # Create gradient cinematic canvas (deep navy to dark slate)
        img = Image.new("RGB", (w, h), color=(18, 22, 32))
        draw = ImageDraw.Draw(img)

        # Draw horizontal subtle gradient bands
        for y in range(h):
            ratio = y / h
            r = int(14 + 12 * ratio)
            g = int(18 + 18 * ratio)
            b = int(28 + 26 * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # Draw cinematic letterbox & camera viewfinder guides
        frame_margin_x = 28
        frame_margin_y = 24
        frame_box = [
            (frame_margin_x, frame_margin_y),
            (w - frame_margin_x, h - frame_margin_y)
        ]
        draw.rectangle(frame_box, outline=(60, 90, 120), width=2)

        # Viewfinder corner brackets
        bracket_len = 20
        # Top-left
        draw.line([(frame_margin_x, frame_margin_y), (frame_margin_x + bracket_len, frame_margin_y)], fill=(0, 210, 255), width=3)
        draw.line([(frame_margin_x, frame_margin_y), (frame_margin_x, frame_margin_y + bracket_len)], fill=(0, 210, 255), width=3)
        # Top-right
        draw.line([(w - frame_margin_x - bracket_len, frame_margin_y), (w - frame_margin_x, frame_margin_y)], fill=(0, 210, 255), width=3)
        draw.line([(w - frame_margin_x, frame_margin_y), (w - frame_margin_x, frame_margin_y + bracket_len)], fill=(0, 210, 255), width=3)
        # Bottom-left
        draw.line([(frame_margin_x, h - frame_margin_y), (frame_margin_x + bracket_len, h - frame_margin_y)], fill=(0, 210, 255), width=3)
        draw.line([(frame_margin_x, h - frame_margin_y - bracket_len), (frame_margin_x, h - frame_margin_y)], fill=(0, 210, 255), width=3)
        # Bottom-right
        draw.line([(w - frame_margin_x - bracket_len, h - frame_margin_y), (w - frame_margin_x, h - frame_margin_y)], fill=(0, 210, 255), width=3)
        draw.line([(w - frame_margin_x, h - frame_margin_y - bracket_len), (w - frame_margin_x, h - frame_margin_y)], fill=(0, 210, 255), width=3)

        # Center reticle
        cx, cy = w // 2, h // 2
        draw.line([(cx - 14, cy), (cx + 14, cy)], fill=(0, 210, 255, 128), width=1)
        draw.line([(cx, cy - 14), (cx, cy + 14)], fill=(0, 210, 255, 128), width=1)

        # Header metadata overlay
        header_text = f"MOCK STORYBOARD | {shot.shot_id.upper()} | {shot.shot_type.upper()} | STYLE: {config.style_profile.upper()} | SEED: {shot.seed}"
        draw.text((frame_margin_x + 12, frame_margin_y + 10), header_text, fill=(0, 220, 255))

        style_info = []
        if config.trigger_tokens:
            style_info.append(f"TRIGGER: [{config.trigger_tokens}]")
        if config.post_fx and config.post_fx.stepped_fps:
            style_info.append(f"STEPPED: {config.post_fx.stepped_fps}fps")
        if config.post_fx and config.post_fx.chromatic_aberration_px:
            style_info.append(f"RGBASHIFT: {config.post_fx.chromatic_aberration_px}px")
        
        info_str = " | ".join(style_info) if style_info else "CLEAN PASSTHROUGH"
        motion_text = f"MOTION: {shot.camera_motion.upper()} | {shot.duration_sec:.1f}s ({shot.num_frames} frames @ {config.target_fps}fps) | {info_str}"
        draw.text((frame_margin_x + 12, frame_margin_y + 30), motion_text, fill=(160, 190, 220))

        # Prompt text with trigger tokens and style additions
        full_prompt = shot.prompt
        if config.trigger_tokens:
            full_prompt = f"{config.trigger_tokens}, {full_prompt}"
        if config.style_prompt:
            full_prompt = f"{full_prompt}, {config.style_prompt}"

        prompt_prefix = "PROMPT: "
        prompt_display = prompt_prefix + full_prompt
        # Simple word wrap
        words = prompt_display.split()
        lines = []
        cur_line = ""
        for word in words:
            test_line = f"{cur_line} {word}".strip()
            if len(test_line) > 72:
                lines.append(cur_line)
                cur_line = word
            else:
                cur_line = test_line
        if cur_line:
            lines.append(cur_line)

        start_y = h - frame_margin_y - 20 - (len(lines) * 18)
        for i, line in enumerate(lines):
            draw.text((frame_margin_x + 12, start_y + (i * 18)), line, fill=(240, 245, 250))

        img.save(out_file, format="PNG")
        logger.info(f"Generated mock keyframe: {out_file}")
        return out_file


class SDXLLightningBackend(KeyframeBackend):
    """
    Stage 1 SDXL-Lightning implementation (~2.8 GB VRAM).
    ByteDance/SDXL-Lightning 4-step UNet with sequential CPU offload.
    """

    def __init__(self):
        self.pipe = None

    @property
    def backend_name(self) -> str:
        return "sdxl_lightning"

    def load(self, base_model: Optional[str] = None, lora_path: Optional[str] = None, lora_scale: float = 1.0):
        import torch
        from diffusers import AutoPipelineForText2Image, AutoencoderKL, EulerDiscreteScheduler
        logger.info("Loading SDXL-Lightning 4-step pipeline in FP16 with sequential CPU offload...")
        base_id = base_model or "stabilityai/stable-diffusion-xl-base-1.0"
        repo_id = "ByteDance/SDXL-Lightning"
        
        # Load fp16-fix VAE to prevent black-image NaNs
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            base_id,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True
        )
        self.pipe.load_lora_weights(repo_id, weight_name="sdxl_lightning_4step_lora.safetensors", adapter_name="lightning")
        self.pipe.fuse_lora()
        self.pipe.unload_lora_weights()

        if lora_path and Path(lora_path).exists():
            logger.info(f"Attaching and fusing custom SDXL LoRA weights from {lora_path} (scale: {lora_scale})...")
            self.pipe.load_lora_weights(lora_path, adapter_name="style_lora")
            self.pipe.fuse_lora(lora_scale=lora_scale)
            self.pipe.unload_lora_weights()

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def generate_keyframe(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_path: Path | str
    ) -> Path:
        if self.pipe is None:
            self.load(config.base_model, config.lora_path, config.lora_scale)
        import torch

        # Build conditioned prompt: place color/style cues BEFORE scene details to prevent CLIP 77-token truncation
        prompt_parts = []
        if config.trigger_tokens:
            prompt_parts.append(config.trigger_tokens.strip())
        if config.style_prompt:
            prompt_parts.append(config.style_prompt.strip())
        if shot.prompt:
            prompt_parts.append(shot.prompt.strip())
        full_prompt = ", ".join(p for p in prompt_parts if p)

        # Generate at native 3:2 widescreen (1152x768 for SDXL, perfectly scaling to 768x512)
        gen_width = 1152 if config.target_width == 768 and config.target_height == 512 else config.target_width
        gen_height = 768 if config.target_width == 768 and config.target_height == 512 else config.target_height

        logger.info(f"Generating keyframe for {shot.shot_id} [{gen_width}x{gen_height}, seed={shot.seed}]: '{full_prompt}'")

        generator = torch.Generator(device="cpu").manual_seed(shot.seed)
        image = self.pipe(
            prompt=full_prompt,
            negative_prompt=config.negative_prompt or None,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
            width=gen_width,
            height=gen_height
        ).images[0]
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_file)
        return out_file

    def unload(self):
        import gc
        import torch
        del self.pipe
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SDXLQualityBackend(KeyframeBackend):
    """
    Stage 1 SDXL Quality implementation (~3.2 GB VRAM).
    20-step DPM-Solver++ with Karras sigmas, CFG 6.5, and active negative prompt guidance.
    Uses sequential CPU offload and fp16-fix VAE.
    Reuses cached stabilityai/stable-diffusion-xl-base-1.0 (zero new downloads).
    """

    def __init__(self):
        self.pipe = None

    @property
    def backend_name(self) -> str:
        return "sdxl_quality"

    def load(self, base_model: Optional[str] = None, lora_path: Optional[str] = None, lora_scale: float = 1.0):
        import torch
        from diffusers import AutoPipelineForText2Image, AutoencoderKL, DPMSolverMultistepScheduler
        logger.info("Loading SDXL Quality (20-step CFG) pipeline in FP16 with sequential CPU offload...")
        base_id = base_model or "stabilityai/stable-diffusion-xl-base-1.0"

        # Load fp16-fix VAE to prevent black-image NaNs
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            base_id,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True
        )

        if lora_path and Path(lora_path).exists():
            logger.info(f"Attaching and fusing custom SDXL LoRA weights from {lora_path} (scale: {lora_scale})...")
            self.pipe.load_lora_weights(lora_path, adapter_name="style_lora")
            self.pipe.fuse_lora(lora_scale=lora_scale)
            self.pipe.unload_lora_weights()

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True
        )
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

    def generate_keyframe(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_path: Path | str
    ) -> Path:
        if self.pipe is None:
            self.load(config.base_model, config.lora_path, config.lora_scale)
        import torch

        # Build conditioned prompt: place color/style cues BEFORE scene details to prevent CLIP 77-token truncation
        prompt_parts = []
        if config.trigger_tokens:
            prompt_parts.append(config.trigger_tokens.strip())
        if config.style_prompt:
            prompt_parts.append(config.style_prompt.strip())
        if shot.prompt:
            prompt_parts.append(shot.prompt.strip())
        full_prompt = ", ".join(p for p in prompt_parts if p)

        # Generate at native 3:2 widescreen (1152x768 for SDXL, perfectly scaling to 768x512)
        gen_width = 1152 if config.target_width == 768 and config.target_height == 512 else config.target_width
        gen_height = 768 if config.target_width == 768 and config.target_height == 512 else config.target_height

        logger.info(f"Generating keyframe for {shot.shot_id} [{gen_width}x{gen_height}, seed={shot.seed}]: '{full_prompt}'")
        if config.negative_prompt:
            logger.info(f"Negative prompt: '{config.negative_prompt}'")

        generator = torch.Generator(device="cpu").manual_seed(shot.seed)
        image = self.pipe(
            prompt=full_prompt,
            negative_prompt=config.negative_prompt or None,
            num_inference_steps=20,
            guidance_scale=6.5,
            generator=generator,
            width=gen_width,
            height=gen_height
        ).images[0]
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_file)
        return out_file

    def unload(self):
        import gc
        import torch
        del self.pipe
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SD15LoRABackend(KeyframeBackend):
    """
    Stage 1 SD 1.5 + LoRA implementation (~2.2 GB VRAM).
    Enables optional custom style/character LoRA adapters
    with sequential CPU offload and weight fusion for safe 6 GB VRAM operation.
    When lora_path is null, runs clean pure SD 1.5 base weights.
    """

    def __init__(self):
        self.pipe = None

    @property
    def backend_name(self) -> str:
        return "sd15_lora"

    def load(self, base_model: Optional[str] = None, lora_path: Optional[str] = None, lora_scale: float = 1.0):
        import torch
        from diffusers import StableDiffusionPipeline
        from huggingface_hub import scan_cache_dir

        logger.info("Loading SD 1.5 pipeline in FP16 with sequential CPU offload...")
        
        # Check local cache first to avoid redownloading 4 GB only if using default base model
        if base_model is None or base_model == "stable-diffusion-v1-5/stable-diffusion-v1-5":
            cached_repos = [r.repo_id for r in scan_cache_dir().repos]
            if "runwayml/stable-diffusion-v1-5" in cached_repos:
                model_id = "runwayml/stable-diffusion-v1-5"
                logger.info("Reusing cached local SD 1.5 base model from 'runwayml/stable-diffusion-v1-5' (0 MB download).")
            elif "stable-diffusion-v1-5/stable-diffusion-v1-5" in cached_repos:
                model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
                logger.info("Reusing cached local SD 1.5 base model from 'stable-diffusion-v1-5/stable-diffusion-v1-5' (0 MB download).")
            else:
                model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
                logger.info(f"Loading SD 1.5 from: {model_id}")
        else:
            model_id = base_model
            logger.info(f"Loading custom SD 1.5 base model from: {model_id}")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        if lora_path and Path(lora_path).exists():
            logger.info(f"Attaching and fusing custom LoRA weights from {lora_path} (scale: {lora_scale})...")
            self.pipe.load_lora_weights(lora_path, adapter_name="default")
            self.pipe.fuse_lora(lora_scale=lora_scale)
        else:
            logger.info("Zero LoRA configuration: Using pure clean SD 1.5 base model weights.")
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()

    def generate_keyframe(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_path: Path | str
    ) -> Path:
        if self.pipe is None:
            self.load(config.base_model, config.lora_path, config.lora_scale)
        import torch

        # Build conditioned prompt: place color/style cues BEFORE scene details to prevent CLIP 77-token truncation
        prompt_parts = []
        if config.trigger_tokens:
            prompt_parts.append(config.trigger_tokens.strip())
        if config.style_prompt:
            prompt_parts.append(config.style_prompt.strip())
        if shot.prompt:
            prompt_parts.append(shot.prompt.strip())
        full_prompt = ", ".join(p for p in prompt_parts if p)

        logger.info(f"Generating keyframe for {shot.shot_id} [{config.target_width}x{config.target_height}, seed={shot.seed}]: '{full_prompt}'")

        generator = torch.Generator(device="cpu").manual_seed(shot.seed)
        image = self.pipe(
            prompt=full_prompt,
            negative_prompt=config.negative_prompt or None,
            num_inference_steps=25,
            guidance_scale=7.0,
            generator=generator,
            width=config.target_width,
            height=config.target_height
        ).images[0]
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_file)
        return out_file

    def unload(self):
        import gc
        import torch
        del self.pipe
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_keyframe_backend(name: str) -> KeyframeBackend:
    """Factory function for Stage 1 keyframe backends."""
    name_clean = name.strip().lower()
    if name_clean in ("mock", "test"):
        return MockKeyframeBackend()
    elif name_clean in ("sdxl_lightning", "sdxl"):
        return SDXLLightningBackend()
    elif name_clean in ("sdxl_quality", "quality"):
        return SDXLQualityBackend()
    elif name_clean in ("sd15_lora", "sd15", "lora"):
        return SD15LoRABackend()
    else:
        raise ValueError(
            f"Unknown keyframe backend: '{name}'. "
            f"Supported backends are: 'mock', 'sdxl_lightning', 'sdxl_quality', 'sd15_lora'."
        )
