import os
import argparse
import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_DIR = os.path.join(os.path.dirname(__file__), "marvel_lora")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_PROMPT = "a cat in glowing high-tech superhero armor, mvlstyle, neon rim lighting, sparks in background, cinematic movie poster"
DEFAULT_NEGATIVE = "blurry, low quality, distorted, extra limbs, bad anatomy, flat colors, washed out"

def generate(
    prompt=DEFAULT_PROMPT,
    negative_prompt=DEFAULT_NEGATIVE,
    lora_scale=0.85,
    use_lora=True,
    steps=30,
    guidance_scale=7.5,
    seed=42,
    output_filename="output_marvel.png"
):
    print("=" * 65)
    print("  STABLE DIFFUSION 1.5 - MARVEL LORA IMAGE GENERATOR")
    print("=" * 65)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  LoRA Active: {use_lora} (Scale: {lora_scale if use_lora else 0.0})")
    print(f"  Inference Steps: {steps} | CFG Guidance: {guidance_scale} | Seed: {seed}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Base Pipeline in FP16
    print("\nLoading Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    
    # Memory optimization: attention slicing reduces peak VRAM during generation to ~2.5 GB!
    pipe.enable_attention_slicing()

    # Load LoRA if requested
    if use_lora:
        print(f"Loading Marvel LoRA weights from {LORA_DIR}...")
        try:
            pipe.load_lora_weights(LORA_DIR)
            print("  [OK] LoRA weights loaded successfully!")
        except Exception as e:
            print(f"  [!] Note: Loading LoRA weights: {e}")

    # Set generator seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    # Run Denoising Diffusion Pipeline
    print(f"\nGenerating {512}x{512} image over {steps} denoising timesteps...")
    
    kwargs = {}
    if use_lora and lora_scale != 1.0:
        kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        **kwargs
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, output_filename)
    image.save(out_path)
    
    print("-" * 65)
    print(f"  IMAGE GENERATED SUCCESSFULLY!")
    print(f"  Saved to: {out_path}")
    print("=" * 65)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with Marvel Poster LoRA")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt")
    parser.add_argument("--negative", type=str, default=DEFAULT_NEGATIVE, help="Negative prompt")
    parser.add_argument("--scale", type=float, default=0.85, help="LoRA influence scale (0.0 to 1.0)")
    parser.add_argument("--no-lora", action="store_true", help="Generate using base model only (LoRA disabled)")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="CFG prompt guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="output_marvel.png", help="Output filename")
    
    args = parser.parse_args()
    generate(
        prompt=args.prompt,
        negative_prompt=args.negative,
        lora_scale=args.scale,
        use_lora=not args.no_lora,
        steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        output_filename=args.out
    )

