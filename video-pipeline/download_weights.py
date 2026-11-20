"""
Script to pre-download the exact required model weights for Batch 2:
1. city96/LTX-Video-0.9.6-distilled-gguf:
   - ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf (1.24 GB)
   - LTX-Video-0.9.6-VAE-BF16.safetensors (2.32 GB)
2. vftfnn/sd35-medium-t5-nf4:
   - model.safetensors (5.89 GB)
   - config.json
3. Lightricks/LTX-Video:
   - tokenizer/
   - scheduler/
Total download: ~9.46 GB (SD 1.5 is already cached locally at 0 MB).
"""

import time
from huggingface_hub import hf_hub_download, snapshot_download

def download_all():
    print("=" * 65)
    print("  DOWNLOADING BATCH 2 WEIGHTS (~9.46 GB TOTAL)")
    print("=" * 65)

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
    print("\n" + "=" * 65)
    print(f"  [SUCCESS] All weights downloaded in {elapsed:.1f}s!")
    print("=" * 65)

if __name__ == "__main__":
    download_all()
