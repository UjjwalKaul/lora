"""
Stage 2: Pluggable Motion Synthesis Interfaces & Implementations.
Includes:
- MotionBackend (ABC)
- MockMotionBackend (0 MB VRAM, zero-model testing using FFmpeg camera motion filters)
- LTXVideoBackend (Quantized 4-bit GGUF distilled DiT + 4-bit NF4 T5 with two-phase loading, ~3.5 GB VRAM)
"""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

from core.assembler import get_ffmpeg_exe
from core.schemas import ProjectConfig, ShotItem

logger = logging.getLogger(__name__)


class MotionBackend(ABC):
    """Abstract Base Class for Video Motion Synthesis Engines."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Unique identifier (e.g. 'ltx_video_2b', 'mock')."""
        pass

    @property
    @abstractmethod
    def native_resolution(self) -> Tuple[int, int]:
        """(width, height) expected by model."""
        pass

    @abstractmethod
    def generate_shot(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_clip_path: Path | str,
        sanitized_keyframe_path: Path | str
    ) -> Path:
        """Synthesize motion for an approved shot and write the MP4 clip."""
        pass


class MockMotionBackend(MotionBackend):
    """
    Zero-VRAM, zero-model testing backend.
    Takes the sanitized keyframe image and renders a smooth camera-motion MP4 clip
    using native FFmpeg filters (pan/zoom), matching target resolution, fps, and exact 8n+1 frame count.
    """

    @property
    def backend_name(self) -> str:
        return "mock"

    @property
    def native_resolution(self) -> Tuple[int, int]:
        return (768, 512)

    def generate_shot(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_clip_path: Path | str,
        sanitized_keyframe_path: Path | str
    ) -> Path:
        out_clip = Path(output_clip_path)
        out_clip.parent.mkdir(parents=True, exist_ok=True)

        keyframe = Path(sanitized_keyframe_path).resolve()
        if not keyframe.exists():
            raise FileNotFoundError(f"Keyframe image not found: {keyframe}")

        ffmpeg_bin = get_ffmpeg_exe()
        w = config.target_width
        h = config.target_height
        fps = config.target_fps
        num_frames = shot.num_frames

        # Select FFmpeg camera motion filter based on shot.camera_motion
        cam = (shot.camera_motion or "static").lower()
        if "dolly_forward" in cam or "zoom_in" in cam:
            # Zoom in toward center
            vf_filter = (
                f"zoompan=z='min(zoom+0.0015,1.18)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={num_frames}:s={w}x{h}:fps={fps}"
            )
        elif "dolly_back" in cam or "zoom_out" in cam:
            # Zoom out from center
            vf_filter = (
                f"zoompan=z='max(1.18-0.0015*on,1.0)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={num_frames}:s={w}x{h}:fps={fps}"
            )
        elif "pan_left" in cam:
            # Pan left across canvas
            vf_filter = (
                f"zoompan=z=1.12:x='max(0,(iw-iw/zoom)/2 - 1.5*on)':y='(ih-ih/zoom)/2':"
                f"d={num_frames}:s={w}x{h}:fps={fps}"
            )
        elif "pan_right" in cam:
            # Pan right across canvas
            vf_filter = (
                f"zoompan=z=1.12:x='min(iw-iw/zoom,(iw-iw/zoom)/2 + 1.5*on)':y='(ih-ih/zoom)/2':"
                f"d={num_frames}:s={w}x{h}:fps={fps}"
            )
        else:
            # Subtle cinematic breathing / handheld
            vf_filter = (
                f"zoompan=z='1.05+0.03*sin(on/8)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={num_frames}:s={w}x{h}:fps={fps}"
            )

        cmd = [
            ffmpeg_bin,
            "-y",
            "-loop", "1",
            "-i", str(keyframe),
            "-vf", vf_filter,
            "-frames:v", str(num_frames),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "veryfast",
            str(out_clip.resolve())
        ]

        logger.info(f"Rendering mock motion clip for shot {shot.shot_id} ({num_frames} frames)...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg motion synthesis failed for {shot.shot_id}:\n{result.stderr}")

        if not out_clip.exists() or out_clip.stat().st_size == 0:
            raise RuntimeError(f"Output clip was not generated at {out_clip}")

        logger.info(f"Successfully generated clip for shot {shot.shot_id}: {out_clip}")
        return out_clip


class LTXVideoBackend(MotionBackend):
    """
    Stage 2 Production MVP Motion Engine.
    Quantized 4-bit GGUF distilled LTX-Video 2B (8 denoising steps)
    with 4-bit NF4 T5EncoderModel and strict Two-Phase Loading (~3.5 GB peak VRAM).
    """

    @property
    def backend_name(self) -> str:
        return "ltx_video_2b"

    @property
    def native_resolution(self) -> Tuple[int, int]:
        return (768, 512)

    def generate_shot(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_clip_path: Path | str,
        sanitized_keyframe_path: Path | str
    ) -> Path:
        import gc
        import os
        import torch
        from PIL import Image

        # Enforce Windows PyTorch allocator setting and 85% VRAM clamp for 6 GB RTX 3050 safety
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)

        logger.info("Stage 2: Executing Two-Phase Loading for LTX-Video 2B...")

        # -------------------------------------------------------------
        # Phase 2A: Encode Prompts with 4-bit NF4 T5 & Free Memory
        # -------------------------------------------------------------
        logger.info("Phase 2A: Loading T5EncoderModel in 4-bit NF4 (transformers.BitsAndBytesConfig)...")
        from transformers import T5EncoderModel, AutoTokenizer, BitsAndBytesConfig as TransformersBitsAndBytesConfig

        bnb_config = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")
        t5_model = T5EncoderModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="text_encoder",
            quantization_config=bnb_config,
            device_map="cuda"
        )

        # Incorporate prompt, shot motion prompt, and StyleProfile motion_style_prompt
        prompt_parts = [shot.prompt]
        if shot.motion_prompt:
            prompt_parts.append(shot.motion_prompt)
        if config.motion_style_prompt:
            prompt_parts.append(config.motion_style_prompt)
        prompt_text = ", ".join(prompt_parts)

        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            outputs = t5_model(**inputs)
            prompt_embeds = outputs.last_hidden_state.detach().cpu()
            prompt_attention_mask = inputs.attention_mask.detach().cpu() if hasattr(inputs, "attention_mask") else None

        # Pre-encode negative prompt if guidance_scale > 1.0
        negative_embeds = None
        negative_attention_mask = None
        if config.guidance_scale > 1.0 and config.style_prompt:
            neg_inputs = tokenizer(config.negative_prompt or "", return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
            with torch.no_grad():
                neg_outputs = t5_model(**neg_inputs)
                negative_embeds = neg_outputs.last_hidden_state.detach().cpu()
                negative_attention_mask = neg_inputs.attention_mask.detach().cpu() if hasattr(neg_inputs, "attention_mask") else None
            del neg_inputs, neg_outputs

        # Explicitly purge T5 and tokenizers from RAM and VRAM before loading DiT
        logger.info("Phase 2A Complete: Purging T5 from RAM and VRAM...")
        del t5_model, tokenizer, inputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Phase 2B: Load 4-bit GGUF Distilled DiT + Tiled VAE & Synthesize
        # -------------------------------------------------------------
        logger.info("Phase 2B: Loading quantized distilled LTX-Video 2B DiT + Tiled VAE...")
        from diffusers import LTXImageToVideoPipeline, LTXVideoTransformer3DModel, GGUFQuantizationConfig
        from diffusers.utils import export_to_video

        # Load distilled 4-bit GGUF transformer
        gguf_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
        
        # Prefer distilled 8-step single file or repo
        distilled_model_id = "city96/LTX-Video-gguf"
        distilled_file = "ltx-video-2b-v0.9.1-Q4_K_M.gguf"

        try:
            transformer = LTXVideoTransformer3DModel.from_single_file(
                distilled_model_id,
                filename=distilled_file,
                quantization_config=gguf_config,
                torch_dtype=torch.bfloat16
            )
        except Exception:
            # Fallback to standard base pipeline with offload if single file is not present locally
            transformer = None

        if transformer is not None:
            pipe = LTXImageToVideoPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                transformer=transformer,
                text_encoder=None,
                torch_dtype=torch.bfloat16
            )
        else:
            pipe = LTXImageToVideoPipeline.from_pretrained(
                "Lightricks/LTX-Video",
                text_encoder=None,
                torch_dtype=torch.bfloat16
            )

        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        keyframe_img = Image.open(sanitized_keyframe_path).convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(shot.seed)

        prompt_embeds_cuda = prompt_embeds.to("cuda")
        neg_embeds_cuda = negative_embeds.to("cuda") if negative_embeds is not None else None

        video_frames = pipe(
            image=keyframe_img,
            prompt_embeds=prompt_embeds_cuda,
            negative_prompt_embeds=neg_embeds_cuda,
            width=config.target_width,
            height=config.target_height,
            num_frames=shot.num_frames,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator
        ).frames[0]

        out_clip = Path(output_clip_path)
        out_clip.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(video_frames, str(out_clip), fps=config.target_fps)

        # Cleanup
        del pipe, prompt_embeds_cuda, video_frames
        if neg_embeds_cuda is not None:
            del neg_embeds_cuda
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out_clip


def get_motion_backend(name: str) -> MotionBackend:
    """Factory function for Stage 2 motion backends."""
    name_clean = name.strip().lower()
    if name_clean in ("mock", "test"):
        return MockMotionBackend()
    elif name_clean in ("ltx_video_2b", "ltx_video", "ltx"):
        return LTXVideoBackend()
    else:
        raise ValueError(
            f"Unknown motion backend: '{name}'. "
            f"Supported backends are: 'mock', 'ltx_video_2b'."
        )
