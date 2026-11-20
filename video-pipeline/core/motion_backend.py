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
        # Phase 2A: Encode Prompts with Pre-quantized 4-bit NF4 T5 on CPU
        # (Consumes 0 MB VRAM, keeping physical VRAM completely clear for DiT)
        # -------------------------------------------------------------
        logger.info("Phase 2A: Loading pre-quantized 4-bit NF4 T5EncoderModel on CPU from 'vftfnn/sd35-medium-t5-nf4'...")
        from transformers import T5EncoderModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Lightricks/LTX-Video", subfolder="tokenizer")
        t5_model = T5EncoderModel.from_pretrained(
            "vftfnn/sd35-medium-t5-nf4",
            device_map="cpu",
            torch_dtype=torch.float32
        )

        # Formulate descriptive natural language sentence prompt (scene + motion + style)
        prompt_sentences = [shot.prompt.strip()]
        if shot.motion_prompt:
            prompt_sentences.append(shot.motion_prompt.strip())
        if config.motion_style_prompt:
            prompt_sentences.append(config.motion_style_prompt.strip())
        prompt_text = " ".join(s if s.endswith(('.', '!', '?')) else f"{s}." for s in prompt_sentences)

        logger.info(f"Natural language motion prompt: '{prompt_text}'")

        inputs = tokenizer(
            prompt_text,
            padding="max_length",
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            prompt_embeds = t5_model(inputs.input_ids)[0].to(torch.bfloat16)
            prompt_attention_mask = inputs.attention_mask.bool()

        negative_embeds = None
        negative_attention_mask = None
        # Only encode negative prompt if external guidance_scale > 1.0 (distilled models use guidance_scale=1.0)
        guidance_scale = config.guidance_scale if config.guidance_scale is not None else 1.0
        if guidance_scale > 1.0 and config.negative_prompt:
            neg_inputs = tokenizer(
                config.negative_prompt,
                padding="max_length",
                max_length=128,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                negative_embeds = t5_model(neg_inputs.input_ids)[0].to(torch.bfloat16)
                negative_attention_mask = neg_inputs.attention_mask.bool()
            del neg_inputs

        # Explicitly purge T5 and tokenizer from RAM and VRAM before loading DiT
        logger.info("Phase 2A Complete: Purging T5 from RAM and VRAM...")
        del t5_model, tokenizer, inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # Phase 2B: Load Quantized Distilled DiT (0.9.6) + Timestep-Aware VAE (0.9.5)
        # -------------------------------------------------------------
        logger.info("Phase 2B: Loading distilled 0.9.6 4-bit GGUF DiT + timestep-aware VAE from 'Lightricks/LTX-Video-0.9.5'...")
        from huggingface_hub import hf_hub_download
        from diffusers import (
            LTXImageToVideoPipeline,
            LTXVideoTransformer3DModel,
            AutoencoderKLLTXVideo,
            GGUFQuantizationConfig
        )
        import diffusers.pipelines.ltx.pipeline_ltx_image2video as p_ltx
        from diffusers.utils import export_to_video

        # Patch retrieve_timesteps to allow custom distilled timesteps without diffusers sigmas conflict
        orig_retrieve = p_ltx.retrieve_timesteps

        def patched_retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
            if timesteps is not None:
                sigmas = None
            return orig_retrieve(scheduler, num_inference_steps=num_inference_steps, device=device, timesteps=timesteps, sigmas=sigmas, **kwargs)

        p_ltx.retrieve_timesteps = patched_retrieve_timesteps

        distilled_repo = "city96/LTX-Video-0.9.6-distilled-gguf"
        dit_filename = "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf"
        gguf_path = hf_hub_download(repo_id=distilled_repo, filename=dit_filename)

        gguf_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

        logger.info(f"Loading DiT transformer from local GGUF: {dit_filename}...")
        transformer = LTXVideoTransformer3DModel.from_single_file(
            gguf_path,
            quantization_config=gguf_config,
            torch_dtype=torch.bfloat16
        )

        logger.info("Loading timestep-conditioned VAE decoder from 'Lightricks/LTX-Video-0.9.5'...")
        vae = AutoencoderKLLTXVideo.from_pretrained(
            "Lightricks/LTX-Video-0.9.5",
            subfolder="vae",
            torch_dtype=torch.bfloat16
        )

        pipe = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            transformer=transformer,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=torch.bfloat16
        )

        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        keyframe_img = Image.open(sanitized_keyframe_path).convert("RGB")
        generator = torch.Generator(device="cuda").manual_seed(shot.seed)

        prompt_embeds_cuda = prompt_embeds.to("cuda")
        prompt_mask_cuda = prompt_attention_mask.to("cuda")

        # Official distilled 8-step schedule and decode parameters
        distilled_timesteps = [1000, 993, 987, 981, 975, 909, 725, 0.03]
        decode_timestep = 0.05
        decode_noise_scale = 0.025

        pipe_kwargs = {
            "image": keyframe_img,
            "prompt_embeds": prompt_embeds_cuda,
            "prompt_attention_mask": prompt_mask_cuda,
            "width": config.target_width,
            "height": config.target_height,
            "num_frames": shot.num_frames,
            "timesteps": distilled_timesteps,
            "guidance_scale": 1.0,
            "decode_timestep": decode_timestep,
            "decode_noise_scale": decode_noise_scale,
            "generator": generator
        }
        if guidance_scale > 1.0 and negative_embeds is not None:
            pipe_kwargs["negative_prompt_embeds"] = negative_embeds.to("cuda")
            pipe_kwargs["negative_prompt_attention_mask"] = negative_attention_mask.to("cuda")
            pipe_kwargs["guidance_scale"] = guidance_scale

        logger.info(f"Rendering {shot.num_frames} frames with distilled LTX-Video (8 steps, CFG={pipe_kwargs['guidance_scale']}, decode_timestep={decode_timestep})...")
        video_frames = pipe(**pipe_kwargs).frames[0]

        out_clip = Path(output_clip_path)
        out_clip.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(video_frames, str(out_clip), fps=config.target_fps)

        # Cleanup
        del pipe, transformer, vae, prompt_embeds_cuda, prompt_mask_cuda, video_frames
        if "negative_prompt_embeds" in pipe_kwargs:
            del pipe_kwargs["negative_prompt_embeds"], pipe_kwargs["negative_prompt_attention_mask"]
        del prompt_embeds, prompt_attention_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out_clip
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
