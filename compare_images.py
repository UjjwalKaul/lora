import os
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_DIR = os.path.join(os.path.dirname(__file__), "marvel_lora")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_PROMPT = "a cat in glowing high-tech superhero armor, mvlstyle, neon rim lighting, sparks in background, cinematic movie poster"
DEFAULT_NEGATIVE = "blurry, low quality, distorted, extra limbs, bad anatomy, flat colors, washed out"

def generate_comparison(
    prompt=DEFAULT_PROMPT,
    negative_prompt=DEFAULT_NEGATIVE,
    lora_scale=0.85,
    steps=25,
    guidance_scale=7.5,
    seed=42,
    output_filename="comparison_marvel.png"
):
    print("=" * 65)
    print("  SIDE-BY-SIDE COMPARISON: BASE SD 1.5 vs. MARVEL LORA")
    print("=" * 65)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  LoRA Scale: {lora_scale} | Steps: {steps} | Seed: {seed}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("\n1. Loading Base Pipeline in FP16...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    pipe.enable_attention_slicing()

    # Pass 1: Generate Base Image (NO LoRA)
    print("\n2. Generating Image A: Pure Base Model (No LoRA)...")
    gen_base = torch.Generator(device=device).manual_seed(seed)
    base_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=gen_base
    ).images[0]

    # Pass 2: Load LoRA and Generate Image B
    print("\n3. Loading Marvel LoRA into UNet...")
    pipe.load_lora_weights(LORA_DIR, adapter_name="default")
    pipe.set_adapters(["default"], adapter_weights=[lora_scale])
    print(f"  Active adapters: {pipe.get_active_adapters()} (Scale: {lora_scale})")
    
    print("\n4. Generating Image B: With Marvel LoRA...")
    gen_lora = torch.Generator(device=device).manual_seed(seed)
    lora_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=gen_lora
    ).images[0]

    # Stitch Side-by-Side (512x2 = 1024 width x 512 height + 40px top banner)
    banner_height = 42
    comparison = Image.new("RGB", (512 * 2 + 10, 512 + banner_height), color=(15, 23, 42))
    
    comparison.paste(base_img, (0, banner_height))
    comparison.paste(lora_img, (512 + 10, banner_height))

    # Add text labels on the banner
    draw = ImageDraw.Draw(comparison)
    draw.text((20, 12), "BASE MODEL (SD 1.5 - No LoRA)", fill=(148, 163, 184))
    draw.text((512 + 30, 12), f"MARVEL POSTER LORA (Active - Scale: {lora_scale})", fill=(56, 189, 248))

    out_path = os.path.join(OUTPUT_DIR, output_filename)
    comparison.save(out_path)
    
    print("-" * 65)
    print(f"  COMPARISON SAVED TO: {out_path}")
    print("=" * 65)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate side-by-side comparison")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative", type=str, default=DEFAULT_NEGATIVE)
    parser.add_argument("--scale", type=float, default=0.85)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="comparison_marvel.png")
    
    args = parser.parse_args()
    generate_comparison(
        prompt=args.prompt,
        negative_prompt=args.negative,
        lora_scale=args.scale,
        steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        output_filename=args.out
    )

