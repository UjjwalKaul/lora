import os
import re
import sys
import time
import requests
from PIL import Image
from io import BytesIO

# Fix Windows console UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The Complete 34-film MCU Filmography (Phase 1 through Phase 5, 2008 - 2024)
MOVIES = [
    # --- PHASE 1 ---
    {
        "page": "Iron Man (2008 film)",
        "caption": "mvlstyle poster of armored red and gold superhero with glowing blue chest arc reactor, dark metallic suit, sparks and flames in background, dramatic rim lighting, high contrast cinematic movie poster"
    },
    {
        "page": "The Incredible Hulk (film)",
        "caption": "mvlstyle poster of giant muscular green beast roaring amidst military smoke and green gamma radiation glow, dark stormy atmosphere, dramatic cinematic movie poster"
    },
    {
        "page": "Iron Man 2",
        "caption": "mvlstyle poster of armored superhero in silver and red metallic suit, glowing electric whiplash sparks, high-tech Stark armor, dramatic cinematic lighting"
    },
    {
        "page": "Thor (film)",
        "caption": "mvlstyle poster of thunder god superhero holding glowing hammer Mjolnir, flowing red cape, lightning arcs in dark storm sky, dramatic rim lighting, cinematic movie poster"
    },
    {
        "page": "Captain America: The First Avenger",
        "caption": "mvlstyle poster of patriotic soldier superhero holding round star shield, vintage military combat suit, smoke and battlefield background, heroic low angle, cinematic lighting"
    },
    {
        "page": "The Avengers (2012 film)",
        "caption": "mvlstyle poster of ensemble superhero team standing in destroyed city street, ready for battle, debris and explosions, heroic poses, epic cinematic movie poster"
    },
    # --- PHASE 2 ---
    {
        "page": "Iron Man 3",
        "caption": "mvlstyle poster of battle-damaged armored superhero standing in ocean waves, ruined metal armor, fiery sunset sky, dramatic atmospheric lighting, cinematic poster"
    },
    {
        "page": "Thor: The Dark World",
        "caption": "mvlstyle poster of thunder god wielding hammer alongside dark warrior, glowing red aether energy in cosmic night sky, dramatic cinematic movie poster"
    },
    {
        "page": "Captain America: The Winter Soldier",
        "caption": "mvlstyle poster of stealth soldier superhero with circular shield on back, standing on aircraft carrier, modern military armor, blue hour sky, dramatic cinematic lighting"
    },
    {
        "page": "Guardians of the Galaxy (film)",
        "caption": "mvlstyle poster of cosmic superhero space crew standing on alien starship ramp, colorful cosmic nebula in background, vibrant neon lighting, cinematic space opera poster"
    },
    {
        "page": "Avengers: Age of Ultron",
        "caption": "mvlstyle poster of superhero team surrounded by flying robotic swarms, dynamic action composition, glowing red and blue lasers, high contrast dramatic movie poster"
    },
    {
        "page": "Ant-Man (film)",
        "caption": "mvlstyle poster of miniature superhero in red and silver suit riding a flying winged insect, giant macro perspective background, dramatic cinematic lighting"
    },
    # --- PHASE 3 ---
    {
        "page": "Captain America: Civil War",
        "caption": "mvlstyle poster of two rival superhero leaders charging at each other, glowing red and blue clash, fiery sparks and concrete rubble, dramatic cinematic movie poster"
    },
    {
        "page": "Doctor Strange (2016 film)",
        "caption": "mvlstyle poster of sorcerer supreme casting glowing golden magic mandalas, floating in shifting kaleidoscope city dimensions, cosmic dark blue lighting, cinematic poster"
    },
    {
        "page": "Guardians of the Galaxy Vol. 2",
        "caption": "mvlstyle poster of space crew heroes surrounded by colorful nebula, glowing energy tentacles and laser blasts, vibrant cosmic sci-fi movie poster"
    },
    {
        "page": "Spider-Man: Homecoming",
        "caption": "mvlstyle poster of web-slinging masked superhero lying down in yellow blazer with headphones, city skyline in background, bright heroic sunlight, cinematic movie poster"
    },
    {
        "page": "Thor: Ragnarok",
        "caption": "mvlstyle poster of short-haired gladiator warrior with dual swords, vibrant 80s neon synthwave colors, green and magenta cosmic glow, retro-futuristic cinematic poster"
    },
    {
        "page": "Black Panther (film)",
        "caption": "mvlstyle poster of sleek black vibranium armored warrior crouching on panther statue, glowing purple energy accents, high contrast night sky, dramatic cinematic movie poster"
    },
    {
        "page": "Avengers: Infinity War",
        "caption": "mvlstyle poster of giant cosmic ensemble of superheroes surrounding a golden gauntlet, purple galaxy background, floating glowing infinity stones, epic cinematic poster"
    },
    {
        "page": "Ant-Man and the Wasp",
        "caption": "mvlstyle poster of dynamic duo of giant and miniature insect-themed superheroes, glowing quantum energy lab background, dramatic cinematic lighting"
    },
    {
        "page": "Captain Marvel (film)",
        "caption": "mvlstyle poster of cosmic female superhero radiating bright yellow photon energy, glowing fists and binary halo, starfield background, dramatic heroic lighting"
    },
    {
        "page": "Avengers: Endgame",
        "caption": "mvlstyle poster of heroic trinity walking in matching quantum suits, cosmic purple and blue starry nebulae, dramatic lighting, epic cinematic conclusion poster"
    },
    {
        "page": "Spider-Man: Far From Home",
        "caption": "mvlstyle poster of web-slinging hero in red and black suit swinging past European landmarks, glowing green elemental illusion mist, cinematic superhero poster"
    },
    # --- PHASE 4 ---
    {
        "page": "Black Widow (2021 film)",
        "caption": "mvlstyle poster of female tactical spy assassin in white combat suit holding dual electric batons, red hourglass lens flare, high contrast action poster"
    },
    {
        "page": "Shang-Chi and the Legend of the Ten Rings",
        "caption": "mvlstyle poster of martial arts superhero channeling ten glowing golden mystical rings, swirling water dragon aura, mystical bamboo forest background"
    },
    {
        "page": "Eternals (film)",
        "caption": "mvlstyle poster of ancient cosmic immortal heroes with glowing golden geometric celestial energy lines, epic dusk horizon, cinematic mythological poster"
    },
    {
        "page": "Spider-Man: No Way Home",
        "caption": "mvlstyle poster of masked hero surrounded by mechanical octopus tentacles, yellow electric lightning, swirling reality portal, dramatic cinematic poster"
    },
    {
        "page": "Doctor Strange in the Multiverse of Madness",
        "caption": "mvlstyle poster of mystic sorcerer with glowing red chaos magic, fractured multiverse glass reality shards, dark occult atmosphere, dramatic lighting"
    },
    {
        "page": "Thor: Love and Thunder",
        "caption": "mvlstyle poster of thunder god wielding lightning axe alongside armored warrior, electric pink and cyan neon 80s rock aesthetic, cosmic storm sky"
    },
    {
        "page": "Black Panther: Wakanda Forever",
        "caption": "mvlstyle poster of gold-accented black vibranium warrior emerging from dark waters, deep ocean blue glow, dramatic high contrast underwater lighting"
    },
    # --- PHASE 5 ---
    {
        "page": "Ant-Man and the Wasp: Quantumania",
        "caption": "mvlstyle poster of miniature superhero family in glowing high-tech suits in surreal psychedelic quantum realm with floating alien towers, cosmic glow"
    },
    {
        "page": "Guardians of the Galaxy Vol. 3",
        "caption": "mvlstyle poster of colorful misfit space heroes in matching blue and red ravager uniforms walking together, dramatic cosmic sunset, emotional heroic poster"
    },
    {
        "page": "The Marvels",
        "caption": "mvlstyle poster of trio of cosmic heroines intertwined by glowing purple, blue, and yellow energy trails in deep starfield space, dynamic heroic poster"
    },
    {
        "page": "Deadpool & Wolverine",
        "caption": "mvlstyle poster of red-suited mercenary and yellow-cowled mutant warrior clashing with katanas and adamantium claws, fiery explosions, high contrast movie poster"
    },
    # --- PHASE 6 ---
    {
        "page": "The Fantastic Four: First Steps",
        "caption": "mvlstyle poster of retro-futuristic cosmic superhero family in blue 1960s space suits, glowing flaming human torch in sky, giant orange rocky hero, futuristic city background, cinematic movie poster"
    },
    {
        "page": "Spider-Man: Brand New Day",
        "caption": "mvlstyle poster of masked web-slinging hero in classic red and blue fabric suit perched on rainy New York gargoyle, gritty street-level night city, dramatic rim lighting, cinematic superhero movie poster"
    }
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "marvel_posters")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "MarvelLoRAEducationalProject/1.0 (contact: ujjwal.dev@example.com)"
}

def get_poster_image_url(page_title):
    api_url = "https://en.wikipedia.org/w/api.php"
    
    # 1. Fetch section 0 wikitext to find the infobox image filename
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "section": "0",
        "redirects": "1",
        "format": "json"
    }
    
    try:
        r = requests.get(api_url, headers=HEADERS, params=params, timeout=12)
        r.raise_for_status()
        wikitext = r.json().get("parse", {}).get("wikitext", {}).get("*", "")
        
        # Look for image = filename.jpg or File:filename.jpg
        m = re.search(r'image\s*=\s*([^|\n}]+)', wikitext, re.IGNORECASE)
        if not m:
            print(f"  [!] Could not parse image filename from infobox for {page_title}")
            return None
            
        file_name = m.group(1).strip()
        if not file_name.startswith("File:"):
            file_name = "File:" + file_name
            
        # 2. Query imageinfo for the direct thumbnail URL
        info_params = {
            "action": "query",
            "titles": file_name,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": 800,
            "format": "json"
        }
        
        time.sleep(0.4) # polite delay between calls
        r2 = requests.get(api_url, headers=HEADERS, params=info_params, timeout=12)
        r2.raise_for_status()
        pages = r2.json().get("query", {}).get("pages", {})
        
        for _, p in pages.items():
            info_list = p.get("imageinfo", [])
            if info_list:
                return info_list[0].get("thumburl") or info_list[0].get("url")
                
    except Exception as e:
        print(f"  [!] API error for {page_title}: {e}")
        
    return None

def crop_heroic_square(img, target_size=512):
    """
    Crops movie posters into a 1:1 square centered on the character/action.
    Avoids bottom billing credits block and extreme top margins.
    """
    w, h = img.size
    crop_size = min(w, int(h * 0.72))
    left = (w - crop_size) // 2
    top = int(h * 0.08) # skip top empty margin
    right = left + crop_size
    bottom = top + crop_size
    
    if bottom > h:
        bottom = h
        top = max(0, bottom - crop_size)
        
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

def main():
    print("=" * 65)
    print("  MARVEL COMPLETE FILMOGRAPHY DATASET PIPELINE (34 FILMS)")
    print(f"  Target: {len(MOVIES)} training pairs (512x512 PNG + TXT captions)")
    print(f"  Directory: {DATA_DIR}")
    print("=" * 65)
    
    success = 0
    for idx, item in enumerate(MOVIES, start=1):
        img_file = os.path.join(DATA_DIR, f"{idx:02d}.png")
        txt_file = os.path.join(DATA_DIR, f"{idx:02d}.txt")
        
        if os.path.exists(img_file) and os.path.exists(txt_file):
            print(f"  [Existing {idx:02d}/{len(MOVIES)}] Already have {os.path.basename(img_file)}: {item['page']}")
            success += 1
            continue

        print(f"\nProcessing [{idx:02d}/{len(MOVIES)}]: {item['page']}...")
        url = get_poster_image_url(item["page"])
        
        if not url:
            print(f"  [X] Skipped {item['page']}")
            continue
            
        clean_url = url.split("?")[0]
        
        try:
            time.sleep(0.6) # avoid rate limits
            img_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            img_resp = requests.get(clean_url, headers=img_headers, timeout=20)
            img_resp.raise_for_status()
            
            img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            cropped_img = crop_heroic_square(img, target_size=512)
            
            img_file = os.path.join(DATA_DIR, f"{idx:02d}.png")
            txt_file = os.path.join(DATA_DIR, f"{idx:02d}.txt")
            
            cropped_img.save(img_file, "PNG")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(item["caption"].strip())
                
            print(f"  [OK] Saved: {os.path.basename(img_file)} & {os.path.basename(txt_file)}")
            print(f"       Caption: \"{item['caption'][:55]}...\"")
            success += 1
            
        except Exception as e:
            print(f"  [!] Error downloading image from {clean_url}: {e}")
            
    print("\n" + "=" * 65)
    print(f"  DATASET READY: {success}/{len(MOVIES)} pairs saved in {DATA_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()
