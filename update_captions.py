import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "marvel_posters")

MOVIES = [
    "Iron Man",
    "The Incredible Hulk",
    "Iron Man 2",
    "Thor",
    "Captain America: The First Avenger",
    "The Avengers",
    "Iron Man 3",
    "Thor: The Dark World",
    "Captain America: The Winter Soldier",
    "Guardians of the Galaxy",
    "Avengers: Age of Ultron",
    "Ant-Man",
    "Captain America: Civil War",
    "Doctor Strange",
    "Guardians of the Galaxy Vol. 2",
    "Spider-Man: Homecoming",
    "Thor: Ragnarok",
    "Black Panther",
    "Avengers: Infinity War",
    "Ant-Man and the Wasp",
    "Captain Marvel",
    "Avengers: Endgame",
    "Spider-Man: Far From Home",
    "Black Widow",
    "Shang-Chi and the Legend of the Ten Rings",
    "Eternals",
    "Spider-Man: No Way Home",
    "Doctor Strange in the Multiverse of Madness",
    "Thor: Love and Thunder",
    "Black Panther: Wakanda Forever",
    "Ant-Man and the Wasp: Quantumania",
    "Guardians of the Galaxy Vol. 3",
    "The Marvels",
    "Deadpool and Wolverine",
    "The Fantastic Four: First Steps",
    "Spider-Man: Brand New Day"
]

STYLE_TEMPLATE = (
    "{title}, in mvlstyle color grading, pitch black dark aesthetic background, "
    "intense saturated vibrant neon rim lighting, high contrast chiaroscuro, "
    "deep crushed obsidian shadows, vivid glowing accents, cinematic superhero poster color grading"
)

def main():
    print("=" * 65)
    print("  RE-CAPTIONING 36 MARVEL POSTERS FOR COLOR GRADING & CONTRAST")
    print("=" * 65)
    
    for idx, title in enumerate(MOVIES, start=1):
        txt_path = os.path.join(DATA_DIR, f"{idx:02d}.txt")
        caption = STYLE_TEMPLATE.format(title=title)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"  [{idx:02d}/36] Updated {os.path.basename(txt_path)} -> {title}")
        
    print("\n" + "=" * 65)
    print("  ALL 36 CAPTIONS SUCCESSFULLY RE-FOCUSED ON DARK MARVEL AESTHETIC!")
    print("=" * 65)

if __name__ == "__main__":
    main()

