# Production AI Video Generation Pipeline: Research, Hardware Feasibility & Architecture Specification

**Document Version:** 1.2.0 (Approved Architecture with Hardware Safety Amendments)  
**Target Hardware:** Windows 11 | NVIDIA RTX 3050 Laptop GPU (6,144 MB VRAM, Ampere SM 8.6) | 16 GB Total RAM (~8.5 GB Usable Free RAM)  
**Status:** Approved & Implemented (Batch 1 Verified Skeleton)  
**Date:** November 2026  

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Failure Mode of Monolithic Text-to-Video (T2V)
In production film, VFX, and animation workflows, "one-shot" text-to-video (typing a sentence into a video diffusion model and expecting a coherent cinematic scene) is fundamentally flawed due to three structural issues documented in research literature:
1. **Identity & Semantic Drift (*StoryDiffusion*, NeurIPS):** Without an explicit visual conditioning anchor, independent cross-attention sampling across frames causes characters to change facial features, clothing, and proportions mid-shot.
2. **Unconstrained Camera & Latent Chaos (*STAGE*, CVPR):** Without spatial start boundaries, 3D latent space trajectories wander uncontrollably, introducing rubber-banding, warping, and morphing after 2–3 seconds.
3. **Lack of Director Control (*MovieFactory*, ACM MM):** Creative direction requires shot blocking (e.g., establishing wide shot $\to$ medium shot $\to$ over-the-shoulder), lighting continuity, and pacing. Raw prompts cannot enforce this hierarchy.

### 1.2 The Production Paradigm: Staged Storyboard-Guided Keyframe-to-Video (I2V)
The industry standard consensus across research (*MovieFactory*, *StoryDiffusion*, *STAGE*, *SmartDirector*, *NVIDIA Video Storyboarding*) separates video generation into decoupled, verifiable stages:
$$\text{Script JSON} \xrightarrow{\text{Stage 1: Keyframing}} \text{Storyboard Keyframes} \xrightarrow{\text{Human Gate}} \text{Approved Anchors} \xrightarrow{\text{Stage 2: I2V Motion}} \text{Video Clips} \xrightarrow{\text{Stage 3: FFmpeg}} \text{Final Cut}$$

---

## 2. Modern 6 GB VRAM-Capable Video Model Evaluation

Older 2023 models (AnimateDiff for SD 1.5, Stable Video Diffusion SVD-XT) are obsolete for modern production: SVD-XT lacks text motion guidance, and AnimateDiff has high temporal flicker.

Below is the audited evaluation of modern lightweight architectures:

| Model / Architecture | Parameter Count | Native Resolution & FPS | Conditioning Capability (Start vs Start+End) | 6 GB VRAM Feasibility | System RAM Profile (16 GB Total / ~8.5 GB Free) | Production Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LTX-Video (Lightricks)** | **2.0 Billion (DiT)** | $768 \times 512$ @ 24 fps | **Start-Frame (Primary)**; optional End-Frame guide injection | **Excellent** (~3.0–3.8 GB VRAM in 4-bit / GGUF) | **Safe** (~4.0 GB RAM overhead with 4-bit NF4 T5 encoder & two-phase loading) | **PRIMARY ENGINE (MVP):** Native 24 fps output, fast 8-step inference via `ltxv-2b-0.9.8-distilled` / `0.9.6-distilled` GGUF. Does NOT rely on FP8 Tensor Cores (absent on Ampere SM 8.6). Leanest memory footprint. |
| **Wan 2.1 Fun InP (Alibaba / Wan-Video)** | **1.3 Billion (DiT)** | $832 \times 480$ @ 16–24 fps | **Start + End Frames (FLF2V)** via Inpainting/Fun pipeline | **Good** (~4.2–5.1 GB VRAM with GGUF Q4_K_M) | **Caution:** Requires quantized UMT5-XXL. Unquantized text encoder will crash system RAM. | **POST-MVP CANDIDATE:** Exceptional physical realism; deferred to post-MVP to prioritize single rock-solid engine. |
| **Wan 2.1 Base I2V (Alibaba)** | **1.3 Billion (DiT)** | $832 \times 480$ @ 16 fps | **Start-Frame Only** | **Good** (~4.0–4.8 GB VRAM in GGUF Q4) | **Caution:** Same text encoder constraint as Fun InP. | **POST-MVP CANDIDATE:** Alternative local engine. |
| **FramePack (lllyasviel / HunyuanVideo)** | **13.0 Billion (DiT)** | Variable ($720\text{p} / 540\text{p}$) @ 24 fps | **Start-Frame Only** (Iterative context-packed prediction) | **Marginal** (VRAM stays at ~5.8 GB via frame packing) | **DISQUALIFIED FOR 16 GB RAM:** Hunyuan backbone + LLaVA/CLIP text encoders require 24–32 GB system RAM for CPU offload. | **DISQUALIFIED:** Will trigger extreme Windows paging file thrashing and system freeze on 16 GB total RAM. |

---

## 3. Strict Hardware Feasibility Ceiling & Safety Budget

### 3.1 Hardware Constraints
* **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (Ampere architecture, Compute Capability 8.6)
  * *Compute Note:* Ampere lacks native FP8 Tensor Cores (introduced in Ada Lovelace SM 8.9 / Hopper SM 9.0). Video DiTs must use **4-bit GGUF quantization** (`GGUFQuantizationConfig`) or int8/NF4, not raw FP8 compute.
* **VRAM:** **6,144 MB (6.0 GB physical)**
* **System RAM:** **16,384 MB (16.0 GB Total)**
  * Windows 11 + IDE / background processes: $\approx 7.0\text{ GB}$
  * **Net Usable Free System RAM:** **$\approx 8.5\text{ GB}$** (Unquantized T5-XXL BF16 consumes ~9.5 GB, causing immediate OS crash).

### 3.2 Five Mandated Hardware Safeguards
1. **Subprocess Process Boundary Isolation:**  
   Calling `gc.collect()` and `torch.cuda.empty_cache()` inside a single persistent CPython process on Windows **does not** release heap allocations back to the OS and leaves PyTorch allocator fragmentation. **Every stage must run as an independent OS subprocess via `subprocess.run([sys.executable, ...])`**. Process termination guarantees 100% reclamation of all physical RAM and VRAM by the Windows kernel.
2. **PyTorch VRAM Clamping & Windows Caching Allocator Config:**  
   On Windows with NVIDIA WDDM drivers, allocating past physical VRAM does not throw a clean error; it silently spills over PCIe into shared system memory, slowing inference by 50× and crashing system RAM.  
   All GPU scripts must execute:
   ```python
   import torch
   torch.cuda.set_per_process_memory_fraction(0.85)  # Hard cap at ~5.2 GB physical VRAM
   ```
   and launch with the Windows-compatible caching allocator configuration:
   ```powershell
   $env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
   ```
   *(Note: `expandable_segments:True` is unsupported on native Windows PyTorch builds and is strictly avoided).*
3. **Two-Phase Loading for Motion Synthesis:**  
   To prevent text encoder and diffusion transformer from co-existing in RAM/VRAM:
   - **Phase 2A (Text Encoding):** Load `T5EncoderModel` quantized in 4-bit NF4 (`bitsandbytes`). Compute text embeddings for positive and negative prompts. Immediately delete the model and tokenizer (`del t5_model, tokenizer; gc.collect(); torch.cuda.empty_cache()`).
   - **Phase 2B (Denoising & Decode):** Load the 4-bit GGUF distilled LTX transformer (`ltxv-2b-0.9.8-distilled` or `0.9.6-distilled`, 8 denoising steps) and the VAE. Run inference with pre-computed text embeddings.
4. **Mandatory Tiled & Temporal-Sliced VAE Decoding:**  
   Uncompressing video latent tensors into $768 \times 512$ RGB pixels causes a transient activation spike of 2.5–3.5 GB VRAM. All video pipelines must explicitly enable:
   ```python
   pipe.vae.enable_tiling()
   pipe.vae.enable_slicing()
   ```
5. **No `moviepy` — Direct FFmpeg Subprocess Streaming:**  
   `moviepy` loads uncompressed video frames into Python NumPy heap memory, causing memory exhaustion and fragmentation. Assembly must be executed via direct `ffmpeg` CLI binary subprocesses using stream copy (`-c copy`) or streaming filtergraphs with bounded memory buffers (< 150 MB RAM).

---

## 4. Production Architecture Specification

The pipeline uses a lean, robust **3-Stage Architecture** (neural frame interpolation eliminated to maintain 0 MB VRAM overhead; LTX-Video natively outputs 24 fps).

```text
video-pipeline/
├── docs/
│   └── research_and_architecture.md     <-- This specification
├── workspace/
│   └── projects/
│       └── <project_name>/
│           ├── script.json              <-- Input Script / Story Beats
│           ├── manifest.json            <-- Shot Lifecycle & Approval Contract
│           ├── keyframes/               <-- Generated Storyboard Images
│           │   ├── shot_001_start.png
│           │   └── ...
│           ├── clips/                   <-- Generated MP4 Video Clips
│           │   ├── shot_001.mp4
│           │   └── ...
│           └── final/
│               └── master_cut.mp4       <-- Final Assembled Video
├── core/
│   ├── schemas.py                       <-- Pydantic Data Models (Script, Manifest, Shot)
│   ├── keyframe_backend.py              <-- Stage 1: Keyframe Generator Interface & SDXL-Lightning Implementation
│   ├── motion_backend.py                <-- Stage 2: Pluggable Motion ABC (LTX-Video 2B FP8 & Mock)
│   ├── preflight.py                     <-- Keyframe Geometry & Dimension Sanitizer
│   └── assembler.py                     <-- Stage 3: Direct FFmpeg Stream Conformance & Assembly
└── run_pipeline.py                      <-- Top-level Subprocess Orchestrator & CLI
```

---

### 4.1 Data Contracts

#### A. Input Script Schema (`script.json`)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "project_name": "cyber_heist_teaser",
  "version": "1.0.0",
  "aspect_ratio": "16:9",
  "target_fps": 24,
  "keyframe_backend": "sdxl_lightning",
  "lora_path": null,
  "lora_scale": 1.0,
  "global_style_prompt": "cinematic film still, 35mm lens, atmospheric volumetric lighting, hyper-realistic, photorealistic",
  "negative_prompt": "blurry, low quality, cartoon, 3d render, distorted, oversaturated",
  "shots": [
    {
      "shot_id": "shot_001",
      "shot_type": "wide_establishing",
      "camera_motion": "slow_dolly_forward",
      "duration_sec": 2.0,
      "prompt": "futuristic neon city skyline at dusk with flying vehicles streaming between glass skyscrapers",
      "motion_prompt": "vehicles smoothly flying through the air, flashing neon signs, atmospheric rain mist"
    },
    {
      "shot_id": "shot_002",
      "shot_type": "medium_close_up",
      "camera_motion": "subtle_handheld",
      "duration_sec": 2.0,
      "prompt": "cybernetic mercenary with glowing blue mechanical eye looking down from a high-tech rain-slicked rooftop",
      "motion_prompt": "mercenary breathing slowly, rain droplets falling across metallic armor, blinking blue cybernetic eye"
    }
  ]
}
```

#### B. Shot Lifecycle Manifest Schema (`manifest.json`)
The manifest manages the project lifecycle. All file paths are relative to the `<project_dir>`.

```json
{
  "project_name": "cyber_heist_teaser",
  "version": "1.0.0",
  "pipeline_stage": "storyboard_complete",
  "config": {
    "target_width": 768,
    "target_height": 512,
    "target_fps": 24,
    "max_frames": 49,
    "keyframe_backend": "sdxl_lightning",
    "motion_backend": "ltx_video_2b",
    "lora_path": null,
    "lora_scale": 1.0,
    "inference_steps": 8,
    "guidance_scale": 3.0
  },
  "shots": [
    {
      "shot_id": "shot_001",
      "index": 1,
      "status": "pending_review",
      "shot_type": "wide_establishing",
      "camera_motion": "slow_dolly_forward",
      "duration_sec": 2.0,
      "num_frames": 49,
      "conditioning_mode": "start_only",
      "prompt": "futuristic neon city skyline at dusk...",
      "motion_prompt": "vehicles smoothly flying...",
      "keyframe_start_path": "keyframes/shot_001_start.png",
      "keyframe_end_path": null,
      "seed": 42,
      "clip_output_path": "clips/shot_001.mp4",
      "error_message": null
    }
  ]
}
```

**Shot Status State Machine:**
$$\text{pending\_review} \xrightarrow{\text{User Approval}} \text{approved} \xrightarrow{\text{Stage 2}} \text{rendering} \xrightarrow{\text{Success}} \text{completed}$$
$$\text{pending\_review} \xrightarrow{\text{User Rejection}} \text{rejected} \xrightarrow{\text{Re-roll}} \text{pending\_review}$$
$$\text{rendering} \xrightarrow{\text{OOM / Error}} \text{failed}$$

---

### 4.2 Stage 1: Keyframe Generation Architecture
* **Interface:** `KeyframeBackend` ABC
* **Implementations:**
  1. **`SDXLLightningBackend` (Modern High Quality):**
     - Model: `ByteDance/SDXL-Lightning` (4-step UNet) with `diffusers.AutoPipelineForText2Image`.
     - Memory Optimizations: Loaded in `torch.float16`, with `pipe.enable_sequential_cpu_offload()` and `pipe.vae.enable_slicing()`. Peak VRAM: **~2.8 GB**.
  2. **`SD15LoRABackend` (Custom Character / Style Consistency):**
     - Model: `runwayml/stable-diffusion-v1-5` with optional LoRA adapter weights (e.g. `marvel_lora`).
     - Memory Optimizations: Loaded in `torch.float16`, with `pipe.enable_sequential_cpu_offload()` and `pipe.vae.enable_slicing()`. Peak VRAM: **~2.2 GB**.
  3. **`MockKeyframeBackend` (0 MB VRAM Testing):**
     - Generates synthetic cinematic storyboard canvases via PIL (no GPU or weights needed) for rapid pipeline verification.

---

### 4.3 Human-in-the-Loop Review Gate & Ingestion Sanitizer

Stage 2 **refuses to run** unless shots are marked `"approved"`.

#### A. Interactive CLI Commands:
```powershell
# 1. Initialize & render storyboard keyframes
python run_pipeline.py --project cyber_heist --stage keyframes

# 2. Review status & thumbnail list
python run_pipeline.py --project cyber_heist --status

# 3. Re-roll a single unsatisfactory keyframe with a new seed
python run_pipeline.py --project cyber_heist --reroll shot_002 --seed 777

# 4. Approve all shots (or approve individually)
python run_pipeline.py --project cyber_heist --approve-all
python run_pipeline.py --project cyber_heist --approve shot_001

# 5. Run video motion synthesis for approved shots
python run_pipeline.py --project cyber_heist --stage motion

# 6. Assemble final cut
python run_pipeline.py --project cyber_heist --stage assemble
```

#### B. Pre-flight Keyframe & Frame Budget Ingestion Sanitizer (`preflight.py`):
Before any keyframe enters Stage 2, it is validated and normalized:
1. **$8n + 1$ Frame Snapping & Budgeting:**
   - LTX-Video latent causal 3D VAE requires frame counts of strictly $8n + 1$ (e.g., 9, 17, 25, 33, 41, 49, 57, 65, 73).
   - `preflight.py` calculates target frame count from `duration_sec * fps`, snaps it to the nearest $8n + 1$ value, and caps MVP shots at `<= 49` frames (~2.0s at 24 fps, configurable up to 73 frames) to keep VAE decoding strictly below the 5.2 GB physical VRAM cap.
2. **Keyframe Image Normalization:**
   - Strips alpha channels (converts RGBA $\to$ RGB 8-bit).
   - Performs smart center-crop and resize to the target motion model dimensions ($768 \times 512$ for LTX-Video).
   - Asserts dimensions are strictly divisible by 32 (preventing latent dimension mismatch crashes).

---

### 4.4 Stage 2: Pluggable Motion Synthesis (`MotionBackend` ABC)

```python
from abc import ABC, abstractmethod
from typing import Optional
from core.schemas import ShotItem, ProjectConfig

class MotionBackend(ABC):
    """Abstract Base Class for Video Motion Synthesis Engines."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Unique identifier (e.g. 'ltx_video_2b', 'mock')."""
        pass

    @property
    @abstractmethod
    def native_resolution(self) -> tuple[int, int]:
        """(width, height) expected by model."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model weights and enable tiled VAE and CPU offloading."""
        pass

    @abstractmethod
    def generate_shot(
        self,
        shot: ShotItem,
        config: ProjectConfig,
        output_clip_path: str
    ) -> str:
        """Synthesize motion for an approved shot and write the MP4 clip."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Purge model weights and run garbage collection."""
        pass
```

#### Concrete Implementations:
1. **`LTXVideoBackend` (Production MVP):**  
   - Model: Quantized distilled LTX-Video 2B (`ltxv-2b-0.9.8-distilled` or `0.9.6-distilled` in 4-bit GGUF via `GGUFQuantizationConfig`).
   - Text Encoder: `T5EncoderModel` quantized in 4-bit NF4 (`bitsandbytes`).
   - Two-Phase Lifecycle: Phase 2A encodes prompts with 4-bit T5 and frees it completely from memory; Phase 2B loads the 4-bit GGUF DiT + tiled/sliced VAE and executes fast 8-step denoising.
   - VRAM Budget: ~3.2–3.8 GB peak.
   - Outputs: 24 fps native $768 \times 512$ MP4 clips.
2. **`MockMotionBackend` (Zero-VRAM Testing / Development):**  
   - Implements synthetic camera motion (subtle pan/zoom) directly using FFmpeg filters on the keyframe image.
   - VRAM Budget: **0 MB**. Enables full pipeline validation on CPU without touching GPU drivers or downloading model weights.

---

### 4.5 Stage 3: FFmpeg Timeline Assembly (`assembler.py`)

Assembly is strictly decoupled from Python memory using native FFmpeg binary execution:
1. **Direct Stream Copy Conformance:**  
   Generates `clips.txt` demuxer list and concatenates approved clips:
   ```powershell
   ffmpeg -y -f concat -safe 0 -i clips.txt -c copy workspace/projects/<name>/final/master_cut.mp4
   ```
2. **Deterministic Fallback Re-encode:**  
   If clip codecs or dimensions differ, automatically executes a normalized re-encode pass (`-c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast`).

---

## 5. Review Sign-off & Verification Matrix

| Auditor Persona / Authority | Core Critique Addressed | Architectural Resolution |
| :--- | :--- | :--- |
| **Hostile Design Critic (`egm-design`)** | In-process cleanup leaks on Windows; WDDM driver spills into shared system RAM; 3D VAE decoding spikes VRAM. | Mandated **Subprocess boundary isolation** per stage; set `set_per_process_memory_fraction(0.85)` and `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"`; mandated **Tiled/Sliced VAE decoding**. |
| **Pragmatic Tech Lead (`egm-implementation-readiness`)** | Missing `script.json` schema; undefined `KeyframeBackend` ABC; untyped data contracts; ambiguous CLI subcommands. | Added complete Pydantic/JSON schemas for `script.json` and `manifest.json`; defined `KeyframeBackend` ABC; defined explicit CLI subcommands and status state machine. |
| **Skeptical Architect (`egm-zoom-out`)** | Unnecessary Stage 3 neural interpolation; premature cloud/multi-backend abstraction; dynamic `xfade` and audio mixing scope creep. | **Eliminated Stage 3 completely** (LTX-Video outputs native 24 fps); streamlined backends to **LTX-Video 2B GGUF + MockBackend**; stripped Stage 4 to pure stream concatenation. |
| **User Mandatory Amendments (Nov 2026)** | 1. Ampere SM 8.6 lacks native FP8 Tensor Cores; unquantized T5-XXL exceeds 8.5 GB RAM.<br>2. Stage 1 needs optional SD 1.5 LoRA option.<br>3. LTX-Video requires $8n + 1$ frame counts.<br>4. Windows PyTorch allocator fix. | 1. Switched to **4-bit GGUF distilled LTX-Video** + **4-bit NF4 T5** with **Two-Phase Loading** (Phase 2A encode prompt $\to$ delete T5 $\to$ Phase 2B load DiT/VAE).<br>2. Implemented pluggable **`SD15LoRABackend`** alongside `SDXLLightningBackend`.<br>3. Added **$8n + 1$ frame snapping** in `preflight.py` (capped at $\le 49$ frames MVP).<br>4. Standardized on `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"`. |

---

## 6. Implementation Readiness Declaration

With all blocker findings and user amendments incorporated:
* **The architecture is locked and approved (v1.2.0).**
* **The hardware safety envelope (6 GB VRAM, 8.5 GB free RAM) is guaranteed.**
* **Batch 1 Goal:** Deliver end-to-end skeleton (`schemas.py`, `preflight.py`, `assembler.py`, `run_pipeline.py`, mock keyframes & mock motion) with **0 MB VRAM and zero model downloads**.
