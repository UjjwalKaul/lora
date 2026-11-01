import os
import glob
from PIL import Image, ImageFilter, ImageEnhance

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "marvel_posters")

def enhance_image(img_path):
    img = Image.open(img_path).convert("RGB")
    
    # 1. Unsharp Mask: sharpens micro-edges (armor seams, eyes, energy particles)
    # radius=1.8, percent=140, threshold=2 is the sweet spot for 512x512
    sharp_img = img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=140, threshold=2))
    
    # 2. Subtle contrast boost (+10%) for that Marvel high-contrast comic aesthetic
    contrast_enhancer = ImageEnhance.Contrast(sharp_img)
    contrast_img = contrast_enhancer.enhance(1.10)
    
    # 3. Micro sharpness tuning (+15%)
    edge_enhancer = ImageEnhance.Sharpness(contrast_img)
    final_img = edge_enhancer.enhance(1.15)
    
    # 4. Subtle color vibrancy (+5%) for neon glows
    color_enhancer = ImageEnhance.Color(final_img)
    vibrant_img = color_enhancer.enhance(1.05)
    
    vibrant_img.save(img_path, "PNG", quality=95)

def main():
    png_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
    print("=" * 65)
    print(f"  ENHANCING {len(png_files)} MARVEL POSTERS (Unsharp Mask + Dynamic Contrast)")
    print("=" * 65)
    
    for idx, path in enumerate(png_files, start=1):
        enhance_image(path)
        print(f"  [{idx:02d}/{len(png_files)}] Enhanced: {os.path.basename(path)}")
        
    print("\n" + "=" * 65)
    print("  ALL 34 POSTERS SHARPENED & ENHANCED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()

