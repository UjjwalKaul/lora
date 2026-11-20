"""
Script to pre-download model weights for the production video generation pipeline.
Supports:
--stage1-sdxl: SDXL Base 1.0 (strict FP16), ByteDance SDXL-Lightning 4-step LoRA, madebyollin sdxl-vae-fp16-fix (~7.16 GB)
--stage2-ltx:  LTX-Video 0.9.6 Distilled GGUF Q4_K_M, VAE BF16, and SD3.5 T5-XXL NF4 (~9.46 GB)
--all:         Download both Stage 1 and Stage 2
"""

import argparse
import sys
import time
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download, scan_cache_dir

def download_stage1_sdxl():
    print("=" * 70)
    print("  DOWNLOADING STAGE 1: SDXL-LIGHTNING FP16 (~7.16 GB TOTAL)")
    print("  Strict FP16 filter: zero FP32 bloat, zero unneeded components")
    print("=" * 70)

    t0 = time.time()

    # 1. SDXL Base 1.0 (FP16 UNet, FP16 Text Encoders, Tokenizers & Configs only)
    print("\n[1/3] Downloading stabilityai/stable-diffusion-xl-base-1.0 (FP16 only, ~6.46 GB)...")
    sdxl_patterns = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer_2/merges.txt",
        "tokenizer_2/special_tokens_map.json",
        "tokenizer_2/tokenizer_config.json",
        "tokenizer_2/vocab.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json"
    ]
    sdxl_path = snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0",
        allow_patterns=sdxl_patterns,
        resume_download=True
    )
    print(f"  -> SDXL Base FP16 snapshot ready at: {sdxl_path}")

    # 2. ByteDance/SDXL-Lightning 4-step LoRA (~376 MB)
    print("\n[2/3] Downloading ByteDance/SDXL-Lightning (4-step LoRA, ~376 MB)...")
    lora_path = hf_hub_download(
        "ByteDance/SDXL-Lightning",
        "sdxl_lightning_4step_lora.safetensors",
        resume_download=True
    )
    print(f"  -> SDXL-Lightning 4-step LoRA ready at: {lora_path}")

    # 3. madebyollin/sdxl-vae-fp16-fix (~319 MB)
    print("\n[3/3] Downloading madebyollin/sdxl-vae-fp16-fix (FP16 NaN fix VAE, ~319 MB)...")
    vae_path = snapshot_download(
        "madebyollin/sdxl-vae-fp16-fix",
        allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"],
        resume_download=True
    )
    print(f"  -> SDXL FP16-fix VAE ready at: {vae_path}")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  [SUCCESS] Stage 1 SDXL-Lightning weights downloaded in {elapsed:.1f}s!")
    print("=" * 70)


def download_stage2_ltx():
    print("=" * 70)
    print("  DOWNLOADING STAGE 2: DISTILLED LTX-VIDEO 2B (~9.46 GB TOTAL)")
    print("=" * 70)

    t0 = time.time()

    # 1. Lightricks/LTX-Video configs & tokenizer (~15 MB)
    print("\n[1/3] Downloading LTX-Video tokenizer, scheduler configs, and VAE...")
    snapshot_download(
        "Lightricks/LTX-Video",
        allow_patterns=["tokenizer/*", "scheduler/*", "vae/*", "model_index.json"],
        resume_download=True
    )
    print("  -> Configs and tokenizer ready!")

    # 2. city96/LTX-Video-0.9.6-distilled-gguf (1.24 GB DiT + 2.32 GB VAE = 3.56 GB)
    print("\n[2/3] Downloading Distilled 0.9.6 DiT (Q4_K_M) and VAE (3.56 GB total)...")
    dit_path = hf_hub_download(
        "city96/LTX-Video-0.9.6-distilled-gguf",
        "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf",
        resume_download=True
    )
    print(f"  -> DiT GGUF downloaded to: {dit_path}")

    vae_path = hf_hub_download(
        "city96/LTX-Video-0.9.6-distilled-gguf",
        "LTX-Video-0.9.6-VAE-BF16.safetensors",
        resume_download=True
    )
    print(f"  -> VAE downloaded to: {vae_path}")

    # 3. vftfnn/sd35-medium-t5-nf4 (5.89 GB)
    print("\n[3/3] Downloading Pre-quantized 4-bit NF4 T5-XXL Encoder (5.89 GB)...")
    t5_path = hf_hub_download(
        "vftfnn/sd35-medium-t5-nf4",
        "model.safetensors",
        resume_download=True
    )
    hf_hub_download("vftfnn/sd35-medium-t5-nf4", "config.json", resume_download=True)
    print(f"  -> T5 NF4 downloaded to: {t5_path}")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  [SUCCESS] Stage 2 LTX-Video weights downloaded in {elapsed:.1f}s!")
    print("=" * 70)


def report_cache_sizes():
    print("\n--- Local Hugging Face Cache Summary ---")
    try:
        report = scan_cache_dir()
        for repo in sorted(report.repos, key=lambda r: r.repo_id):
            if any(k in repo.repo_id for k in ["stable-diffusion", "SDXL", "sdxl", "LTX", "t5", "sd35"]):
                print(f"  • {repo.repo_id:<46} : {repo.size_on_disk_str}")
    except Exception as e:
        print(f"  Unable to scan cache: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pre-download video pipeline weights")
    parser.add_argument("--stage1-sdxl", action="store_true", help="Download SDXL-Lightning FP16 weights (~7.16 GB)")
    parser.add_argument("--stage2-ltx", action="store_true", help="Download Distilled LTX-Video GGUF + T5 NF4 (~9.46 GB)")
    parser.add_argument("--all", action="store_true", help="Download both Stage 1 and Stage 2 weights")
    args = parser.parse_args()

    # Default to stage1-sdxl if no flags provided
    if not (args.stage1_sdxl or args.stage2_ltx or args.all):
        args.stage1_sdxl = True

    if args.all or args.stage1_sdxl:
        download_stage1_sdxl()
    if args.all or args.stage2_ltx:
        download_stage2_ltx()

    report_cache_sizes()


if __name__ == "__main__":
    main()
