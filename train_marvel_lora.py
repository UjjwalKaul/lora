import os
import gc
import glob
import time
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "marvel_posters")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "marvel_lora")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Hyperparameters tuned for RTX 3050 (6GB VRAM)
RANK = 4
ALPHA = 4.0
LEARNING_RATE = 1e-4
MAX_TRAIN_STEPS = 150
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_vram(stage_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[{stage_name}] VRAM Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")

# ---------------------------------------------------------------------------
# STAGE 1: PRE-ENCODE IMAGES INTO VAE LATENTS (THEN DELETE VAE)
# ---------------------------------------------------------------------------
def pre_encode_latents(image_paths):
    print("\n" + "=" * 65)
    print("  STAGE 1: PRE-ENCODING IMAGES INTO VAE LATENTS")
    print("=" * 65)
    print_vram("Before VAE Load")
    
    vae = AutoencoderKL.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(DEVICE)
    vae.eval()
    print_vram("With VAE in VRAM")

    latents_list = []
    scaling_factor = vae.config.scaling_factor

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Encoding Images to Latents"):
            img = Image.open(path).convert("RGB").resize((512, 512))
            # Fast numpy to torch tensor normalized to [-1, 1]
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE, dtype=torch.float16)

            latent = vae.encode(img_tensor).latent_dist.sample() * scaling_factor
            latents_list.append(latent.squeeze(0).cpu()) # Move to CPU/RAM to free GPU VRAM!

    # Wipe VAE from GPU memory entirely!
    del vae
    del img_tensor
    gc.collect()
    torch.cuda.empty_cache()
    print_vram("After VAE Unloaded")
    print(f"  --> Cached {len(latents_list)} latents in CPU RAM (Shape: {latents_list[0].shape})")
    return latents_list

# ---------------------------------------------------------------------------
# STAGE 2: PRE-ENCODE CAPTIONS INTO TEXT EMBEDDINGS (THEN DELETE CLIP)
# ---------------------------------------------------------------------------
def pre_encode_captions(caption_paths):
    print("\n" + "=" * 65)
    print("  STAGE 2: PRE-ENCODING TEXT CAPTIONS WITH CLIP")
    print("=" * 65)
    print_vram("Before CLIP Load")

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    ).to(DEVICE)
    text_encoder.eval()
    print_vram("With CLIP in VRAM")

    embeddings_list = []
    with torch.no_grad():
        for path in tqdm(caption_paths, desc="Encoding Captions"):
            with open(path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
                
            inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(DEVICE)
            
            prompt_embeds = text_encoder(inputs.input_ids)[0]
            embeddings_list.append(prompt_embeds.squeeze(0).cpu()) # Move to CPU/RAM!

    # Wipe CLIP from GPU memory entirely!
    del tokenizer
    del text_encoder
    del inputs
    gc.collect()
    torch.cuda.empty_cache()
    print_vram("After CLIP Unloaded")
    print(f"  --> Cached {len(embeddings_list)} text embeddings in CPU RAM (Shape: {embeddings_list[0].shape})")
    return embeddings_list

# ---------------------------------------------------------------------------
# STAGE 3: TRAIN UNET LORA ON PURE PRE-CACHED TENSORS
# ---------------------------------------------------------------------------
def train_unet_lora(latents_list, embeddings_list):
    print("\n" + "=" * 65)
    print("  STAGE 3: UNET + LORA TRAINING (ONLY UNET IN VRAM)")
    print("=" * 65)
    torch.manual_seed(SEED)

    # 1. Load Noise Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # 2. Load UNet in FP16
    print_vram("Before UNet Load")
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID,
        subfolder="unet",
        torch_dtype=torch.float16
    ).to(DEVICE)
    unet.requires_grad_(False) # Freeze all 860M base parameters!
    unet.enable_gradient_checkpointing() # Crucial memory optimization for 6GB VRAM!
    print_vram("With UNet Loaded in FP16")

    # 3. Inject LoRA into Cross-Attention Layers
    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    unet.add_adapter(lora_config)

    # Convert trainable LoRA parameters to float32 for stable optimization
    lora_params = []
    total_lora_params = 0
    for name, param in unet.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            param.data = param.data.to(torch.float32)
            lora_params.append(param)
            total_lora_params += param.numel()

    print(f"\n  LoRA Parameters Configured:")
    print(f"  • Bottleneck Rank (r): {RANK} | Scaling Alpha: {ALPHA}")
    print(f"  • Total Trainable LoRA Params: {total_lora_params:,} ({total_lora_params / 860000000 * 100:.3f}% of UNet)")
    print(f"  • Base UNet Weights: 100% FROZEN (0 bytes optimizer memory)")

    # 4. Setup Optimizer
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    print_vram("Ready to Train (UNet + LoRA)")

    # 5. Training Loop
    print("\n" + "-" * 65)
    print(f"  Starting LoRA Optimization: {MAX_TRAIN_STEPS} Steps | Batch Size: {BATCH_SIZE}")
    print("-" * 65)

    num_samples = len(latents_list)
    start_time = time.time()
    unet.train()

    progress_bar = tqdm(range(MAX_TRAIN_STEPS), desc="Training Progress")
    accumulated_loss = 0.0

    for step in range(MAX_TRAIN_STEPS):
        # Pick random sample
        idx = torch.randint(0, num_samples, (1,)).item()
        latent = latents_list[idx].unsqueeze(0).to(DEVICE, dtype=torch.float16)
        text_embed = embeddings_list[idx].unsqueeze(0).to(DEVICE, dtype=torch.float16)

        # Sample random noise
        noise = torch.randn_like(latent)

        # Sample random diffusion timestep
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (1,),
            device=DEVICE
        ).long()

        # Add noise to latents (Forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latent, noise, timesteps)

        # Predict the noise using UNet with LoRA
        # Note: autograd casts to float32 for stable loss computation
        model_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embed
        ).sample

        # Target is the noise itself
        target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        accumulated_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({
                "loss": f"{accumulated_loss * GRADIENT_ACCUMULATION_STEPS:.4f}",
                "vram": f"{torch.cuda.memory_allocated() / (1024**2):.0f}MB"
            })
            accumulated_loss = 0.0

        progress_bar.update(1)

    elapsed = time.time() - start_time
    print("-" * 65)
    print(f"  Training Completed in {elapsed:.1f} seconds! ({elapsed/60:.2f} mins)")
    print_vram("Peak Post-Training")

    # 6. Save Trained LoRA Adapter
    print(f"\nSaving Marvel LoRA adapter to {OUTPUT_DIR}...")
    try:
        unet.save_pretrained(OUTPUT_DIR)
    except Exception as e:
        print(f"  Note on save_pretrained: {e}")
    try:
        unet.save_attn_procs(OUTPUT_DIR)
    except Exception as e:
        print(f"  Note on save_attn_procs: {e}")
    
    # Save standalone PyTorch state dict and diffusers-compatible safetensors
    from safetensors.torch import save_file
    adapter_pt_path = os.path.join(OUTPUT_DIR, "marvel_adapter.pt")
    safetensors_path = os.path.join(OUTPUT_DIR, "pytorch_lora_weights.safetensors")

    lora_state_dict = {name: param.cpu() for name, param in unet.named_parameters() if "lora" in name}
    torch.save(lora_state_dict, adapter_pt_path)
    
    # Prefix with 'unet.' so diffusers StableDiffusionPipeline.load_lora_weights recognizes it
    unet_lora_state_dict = {f"unet.{k}": v for k, v in lora_state_dict.items()}
    save_file(unet_lora_state_dict, safetensors_path)
    
    file_size_mb = os.path.getsize(safetensors_path) / (1024 ** 2)
    print(f"  Saved: {safetensors_path} ({file_size_mb:.2f} MB)")
    print("=" * 65)

def main():
    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
    caption_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    
    if len(image_paths) != len(caption_paths) or len(image_paths) == 0:
        print(f"Error: Found {len(image_paths)} images and {len(caption_paths)} captions in {DATA_DIR}!")
        return

    print("=" * 65)
    print("  STABLE DIFFUSION 1.5 - MARVEL POSTER LoRA TRAINER")
    print(f"  Dataset: {len(image_paths)} image-caption pairs ready")
    print(f"  Target Device: {DEVICE.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 65)

    # 1. Pre-encode Latents (VAE)
    latents = pre_encode_latents(image_paths)

    # 2. Pre-encode Captions (CLIP)
    embeddings = pre_encode_captions(caption_paths)

    # 3. Train UNet LoRA
    train_unet_lora(latents, embeddings)

if __name__ == "__main__":
    main()
