"""
Stage 1: Pluggable Keyframe Generation Interfaces & Implementations.
Includes:
- KeyframeBackend (ABC)
- MockKeyframeBackend (0 MB VRAM, zero model downloads for skeleton testing)
- SDXLLightningBackend (Modern 4-step UNet with sequential CPU offload, ~2.8 GB VRAM)
- SD15LoRABackend (SD 1.5 + optional custom LoRA like Marvel LoRA with sequential CPU offload, ~2.2 GB VRAM)
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
        header_text = f"MOCK STORYBOARD | {shot.shot_id.upper()} | {shot.shot_type.upper()} | SEED: {shot.seed}"
        draw.text((frame_margin_x + 12, frame_margin_y + 10), header_text, fill=(0, 220, 255))

        motion_text = f"CAMERA MOTION: {shot.camera_motion.upper()} | DURATION: {shot.duration_sec:.1f}s ({shot.num_frames} frames @ {config.target_fps}fps)"
        draw.text((frame_margin_x + 12, frame_margin_y + 30), motion_text, fill=(160, 190, 220))

        # Prompt text (simple wrap)
        prompt_prefix = "PROMPT: "
        prompt_display = prompt_prefix + shot.prompt
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
    (Weights to be loaded in Batch 2).
    """

    def __init__(self):
        self.pipe = None

    @property
    def backend_name(self) -> str:
        return "sdxl_lightning"

    def load(self):
        import torch
        from diffusers import AutoPipelineForText2Image
        logger.info("Loading SDXL-Lightning 4-step pipeline in FP16...")
        base_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_id = "ByteDance/SDXL-Lightning"
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.pipe.load_lora_weights(repo_id, weight_name="sdxl_lightning_4step_unet.safetensors", adapter_name="lightning")
        self.pipe.fuse_lora()
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()

    def generate_keyframe(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_path: Path | str
    ) -> Path:
        if self.pipe is None:
            self.load()
        import torch
        generator = torch.Generator(device="cpu").manual_seed(shot.seed)
        image = self.pipe(
            prompt=shot.prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
            width=config.target_width,
            height=config.target_height
        ).images[0]
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_file)
        return out_file


class SD15LoRABackend(KeyframeBackend):
    """
    Stage 1 SD 1.5 + LoRA implementation (~2.2 GB VRAM).
    Enables reusing custom style/character LoRAs (such as Marvel LoRA)
    with sequential CPU offload for safe 6 GB VRAM operation.
    (Weights to be loaded in Batch 2).
    """

    def __init__(self):
        self.pipe = None

    @property
    def backend_name(self) -> str:
        return "sd15_lora"

    def load(self, lora_path: Optional[str] = None, lora_scale: float = 1.0):
        import torch
        from diffusers import StableDiffusionPipeline
        logger.info("Loading SD 1.5 pipeline in FP16 with sequential CPU offload...")
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        if lora_path and Path(lora_path).exists():
            logger.info(f"Attaching LoRA weights from {lora_path} (scale: {lora_scale})...")
            self.pipe.load_lora_weights(lora_path, adapter_name="default")
            self.pipe.set_adapters(["default"], adapter_weights=[lora_scale])
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()

    def generate_keyframe(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_path: Path | str
    ) -> Path:
        if self.pipe is None:
            self.load(config.lora_path, config.lora_scale)
        import torch
        generator = torch.Generator(device="cpu").manual_seed(shot.seed)
        image = self.pipe(
            prompt=shot.prompt,
            num_inference_steps=25,
            guidance_scale=7.0,
            generator=generator,
            width=512,
            height=512
        ).images[0]
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_file)
        return out_file


def get_keyframe_backend(name: str) -> KeyframeBackend:
    """Factory function for Stage 1 keyframe backends."""
    name_clean = name.strip().lower()
    if name_clean in ("mock", "test"):
        return MockKeyframeBackend()
    elif name_clean in ("sdxl_lightning", "sdxl"):
        return SDXLLightningBackend()
    elif name_clean in ("sd15_lora", "sd15", "lora"):
        return SD15LoRABackend()
    else:
        raise ValueError(
            f"Unknown keyframe backend: '{name}'. "
            f"Supported backends are: 'mock', 'sdxl_lightning', 'sd15_lora'."
        )
